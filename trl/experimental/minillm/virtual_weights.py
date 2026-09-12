# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from collections import defaultdict
from dataclasses import dataclass

import torch
from torch import nn


logger = logging.getLogger(__name__)


@dataclass
class _WeightBinding:
    hf_name: str
    vllm_name: str
    hf_param: nn.Parameter
    vllm_param: nn.Parameter
    offset: int | None = None

    def vllm_view(self) -> torch.Tensor:
        if self.offset is None:
            return self.vllm_param.data
        return self.vllm_param.data.narrow(0, self.offset, self.hf_param.shape[0]).view_as(self.hf_param)


@dataclass
class _ResidentBuffer:
    name: str
    buffer: torch.Tensor
    storage: torch.Tensor


class Qwen35VirtualWeightStore:
    """Share Qwen3.5 parameter storage between Transformers and a colocated native vLLM model.

    This first-stage implementation supports a single, non-quantized TP=1 model. Transformers keeps its original
    parameter namespace for autograd and the optimizer, while vLLM keeps its fused modules. For fused vLLM weights,
    the corresponding Transformers parameters become disjoint views into one packed CUDA allocation.
    """

    _PACKED_RENAMES = (
        (".self_attn.q_proj.", ".self_attn.qkv_proj.", 0),
        (".self_attn.k_proj.", ".self_attn.qkv_proj.", 1),
        (".self_attn.v_proj.", ".self_attn.qkv_proj.", 2),
        (".mlp.gate_proj.", ".mlp.gate_up_proj.", 0),
        (".mlp.up_proj.", ".mlp.gate_up_proj.", 1),
        (".linear_attn.in_proj_qkv.", ".linear_attn.in_proj_qkvz.", 0),
        (".linear_attn.in_proj_z.", ".linear_attn.in_proj_qkvz.", 1),
        (".linear_attn.in_proj_b.", ".linear_attn.in_proj_ba.", 0),
        (".linear_attn.in_proj_a.", ".linear_attn.in_proj_ba.", 1),
    )
    _VLLM_FP32_SUFFIXES = (".linear_attn.A_log",)

    def __init__(
        self,
        arenas: list[torch.Tensor],
        bindings: list[_WeightBinding],
        resident_buffers: list[_ResidentBuffer],
        hf_model: nn.Module,
        vllm_model: nn.Module,
    ):
        self.arenas = arenas
        self.bindings = bindings
        self.resident_buffers = resident_buffers
        self._bindings_by_hf_name = {binding.hf_name: binding for binding in bindings}
        self.device = bindings[0].hf_param.device
        self.hf_model = hf_model
        self.vllm_model = vllm_model
        self._hf_layouts = {binding.hf_name: self._layout(binding.hf_param) for binding in bindings}
        self._vllm_layouts = {binding.vllm_name: self._layout(binding.vllm_param) for binding in bindings}
        self._buffer_layouts = {item.name: self._layout(item.buffer) for item in resident_buffers}

    @staticmethod
    def _layout(tensor: torch.Tensor) -> tuple:
        return tensor.data_ptr(), tensor.shape, tensor.stride(), tensor.dtype, tensor.device

    @classmethod
    def _map_hf_name(cls, name: str) -> tuple[str, int | None]:
        if name.startswith("model.visual."):
            vllm_name = name.removeprefix("model.")
        elif name.startswith("model.language_model."):
            vllm_name = "language_model.model." + name.removeprefix("model.language_model.")
        elif name.startswith("lm_head."):
            vllm_name = "language_model." + name
        else:
            raise ValueError(f"Unsupported Qwen3.5 Transformers parameter name: {name!r}.")

        for source, target, order in cls._PACKED_RENAMES:
            if source in vllm_name:
                return vllm_name.replace(source, target, 1), order
        return vllm_name, None

    @classmethod
    @torch.inference_mode(False)
    def bind(cls, hf_model: nn.Module, vllm_model: nn.Module) -> "Qwen35VirtualWeightStore":
        """Bind loaded models before vLLM profiling/capture, allocating ordinary tensors usable by autograd."""
        config = hf_model.config
        if config.model_type != "qwen3_5":
            raise ValueError(
                "vllm_share_student_weights currently supports only Qwen3.5 models, "
                f"but received model_type={config.model_type!r}."
            )
        if getattr(hf_model, "is_loaded_in_4bit", False) or getattr(hf_model, "is_loaded_in_8bit", False):
            raise ValueError("vllm_share_student_weights does not support quantized Transformers models yet.")

        hf_params = dict(hf_model.named_parameters())
        vllm_params = dict(vllm_model.named_parameters())
        sources_by_target = defaultdict(list)
        unsupported_hf_names = []
        for hf_name, hf_param in hf_params.items():
            try:
                vllm_name, order = cls._map_hf_name(hf_name)
            except ValueError:
                unsupported_hf_names.append(hf_name)
                continue
            if vllm_name not in vllm_params:
                unsupported_hf_names.append(hf_name)
                continue
            sources_by_target[vllm_name].append((order, hf_name, hf_param))

        missing_vllm_names = sorted(set(vllm_params) - set(sources_by_target))
        if unsupported_hf_names or missing_vllm_names:
            details = []
            if unsupported_hf_names:
                details.append(f"unmapped Transformers parameters: {sorted(unsupported_hf_names)[:10]}")
            if missing_vllm_names:
                details.append(f"unmapped vLLM parameters: {missing_vllm_names[:10]}")
            raise ValueError("Cannot build a complete Qwen3.5 virtual weight map; " + "; ".join(details))

        arenas = []
        bindings = []
        with torch.no_grad():
            for vllm_name, vllm_param in vllm_params.items():
                sources = sources_by_target[vllm_name]
                if len(sources) == 1 and sources[0][0] is None:
                    _, hf_name, hf_param = sources[0]
                    if (
                        hf_param.dtype != vllm_param.dtype
                        and vllm_param.dtype == torch.float32
                        and hf_name.endswith(cls._VLLM_FP32_SUFFIXES)
                    ):
                        # Transformers casts A_log with the rest of the checkpoint, while native vLLM intentionally
                        # constructs it in FP32. Keeping this tiny training parameter in FP32 satisfies both kernels.
                        hf_param.data = hf_param.data.to(dtype=torch.float32)
                    cls._validate_compatible(hf_name, hf_param, vllm_name, vllm_param)
                    vllm_param.data = hf_param.data
                    bindings.append(_WeightBinding(hf_name, vllm_name, hf_param, vllm_param))
                    continue

                if any(order is None for order, _, _ in sources):
                    raise ValueError(f"vLLM parameter {vllm_name!r} mixes packed and direct Transformers weights.")
                sources.sort(key=lambda item: item[0])
                cls._validate_packed(vllm_name, vllm_param, sources)

                # Allocate outside vLLM's tagged weight pool. This keeps the shared weights resident when vLLM uses
                # sleep(level=2) to discard its KV cache and other private allocations.
                arena = torch.empty(vllm_param.shape, dtype=vllm_param.dtype, device=vllm_param.device)
                offset = 0
                views = []
                for _, hf_name, hf_param in sources:
                    view = arena.narrow(0, offset, hf_param.shape[0]).view_as(hf_param)
                    view.copy_(hf_param)
                    views.append((hf_name, hf_param, offset, view))
                    offset += hf_param.shape[0]

                vllm_param.data = arena
                for hf_name, hf_param, offset, view in views:
                    hf_param.data = view
                    bindings.append(_WeightBinding(hf_name, vllm_name, hf_param, vllm_param, offset))
                arenas.append(arena)

        # vLLM level-2 sleep also unmaps buffers allocated in its tagged weight pool. Move registered buffers to the
        # regular allocator too, so waking only the KV cache never writes into an unmapped buffer address.
        resident_buffers = []
        with torch.no_grad():
            for name, buffer in vllm_model.named_buffers():
                storage = buffer.clone()
                buffer.data = storage
                resident_buffers.append(_ResidentBuffer(name, buffer, storage))

        store = cls(arenas, bindings, resident_buffers, hf_model, vllm_model)
        store.validate_aliases()
        shared_bytes = sum(param.numel() * param.element_size() for param in vllm_params.values())
        logger.info(
            "Bound %d Transformers parameters to %d vLLM parameters using %d packed arenas (%.2f GiB shared).",
            len(bindings),
            len(vllm_params),
            len(arenas),
            shared_bytes / 1024**3,
        )
        return store

    @staticmethod
    def _validate_compatible(hf_name, hf_param, vllm_name, vllm_param) -> None:
        if hf_param.shape != vllm_param.shape or hf_param.dtype != vllm_param.dtype:
            raise ValueError(
                f"Cannot alias {hf_name!r} to {vllm_name!r}: Transformers has shape/dtype "
                f"{tuple(hf_param.shape)}/{hf_param.dtype}, vLLM has {tuple(vllm_param.shape)}/{vllm_param.dtype}."
            )
        if hf_param.device != vllm_param.device:
            raise ValueError(
                f"Cannot alias {hf_name!r} to {vllm_name!r}: parameters are on different devices "
                f"({hf_param.device} and {vllm_param.device})."
            )

    @staticmethod
    def _validate_packed(vllm_name, vllm_param, sources) -> None:
        first_param = sources[0][2]
        if first_param.ndim == 0 or vllm_param.ndim == 0:
            raise ValueError(f"Packed virtual weight {vllm_name!r} must have at least one dimension.")
        expected_shape = list(first_param.shape)
        expected_shape[0] = sum(param.shape[0] for _, _, param in sources)
        for _, hf_name, hf_param in sources:
            if hf_param.shape[1:] != first_param.shape[1:]:
                raise ValueError(f"Packed source {hf_name!r} has incompatible shape {tuple(hf_param.shape)}.")
            if hf_param.dtype != vllm_param.dtype or hf_param.device != vllm_param.device:
                raise ValueError(f"Packed source {hf_name!r} has a different dtype or device from {vllm_name!r}.")
        if tuple(expected_shape) != tuple(vllm_param.shape):
            raise ValueError(
                f"Packed Transformers parameters produce shape {tuple(expected_shape)}, but {vllm_name!r} has "
                f"shape {tuple(vllm_param.shape)}."
            )

    def validate_aliases(self) -> None:
        """Check current module objects and the addresses/layouts fixed before graph capture."""
        hf_params = dict(self.hf_model.named_parameters())
        vllm_params = dict(self.vllm_model.named_parameters())
        for binding in self.bindings:
            if (
                hf_params.get(binding.hf_name) is not binding.hf_param
                or vllm_params.get(binding.vllm_name) is not binding.vllm_param
                or self._layout(binding.hf_param) != self._hf_layouts[binding.hf_name]
                or self._layout(binding.vllm_param) != self._vllm_layouts[binding.vllm_name]
                or binding.hf_param.data_ptr() != binding.vllm_view().data_ptr()
            ):
                raise RuntimeError(
                    f"Virtual weight alias for {binding.hf_name!r} -> {binding.vllm_name!r} was broken. "
                    "Do not move either model or load a state dict with assign=True after binding."
                )
        buffers = dict(self.vllm_model.named_buffers())
        for resident_buffer in self.resident_buffers:
            if (
                buffers.get(resident_buffer.name) is not resident_buffer.buffer
                or self._layout(resident_buffer.buffer) != self._buffer_layouts[resident_buffer.name]
            ):
                raise RuntimeError(
                    f"Resident vLLM buffer {resident_buffer.name!r} was replaced after virtual-weight binding."
                )

    def publish(self) -> None:
        """Fence optimizer writes before vLLM reads the shared storages."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.validate_aliases()

    def finish_generation(self) -> None:
        """Fence vLLM reads before Transformers starts its next forward/backward pass."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def get_vllm_view(self, hf_name: str) -> torch.Tensor:
        """Return the vLLM view corresponding to a Transformers parameter, primarily for diagnostics."""
        return self._bindings_by_hf_name[hf_name].vllm_view()
