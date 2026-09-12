import atexit
import gc
import logging
import os
import signal
import socket
import subprocess
import sys
from contextlib import nullcontext
from functools import partial

import torch
from accelerate.utils import DistributedType, is_peft_model

from .minillm_config import MiniLLMConfig
from .virtual_weights import Qwen35VirtualWeightStore


logger = logging.getLogger(__name__)


class VirtualWeightVLLMInit:
    """Bind the colocated student after worker loading, before profiling and CUDA Graph capture."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def __enter__(self):
        if not self.enabled:
            return self

        from ...generation import vllm_generation

        self.original_init = vllm_generation.VLLMGeneration._init_vllm

        def init_with_shared_weights(generation):
            from vllm.v1.worker.gpu_worker import Worker

            if generation.mode != "colocate" or generation.tensor_parallel_size != 1:
                raise ValueError("Virtual weights require colocate mode with tensor_parallel_size=1.")
            if (
                generation.accelerator.num_processes != 1
                or generation.accelerator.distributed_type != DistributedType.NO
            ):
                raise ValueError("Virtual weights require one process without DeepSpeed or FSDP.")
            if generation.model_impl != "vllm" or is_peft_model(generation.model):
                raise ValueError("Virtual weights require a native, non-PEFT vLLM model.")
            if next(generation.model.parameters()).device.type != "cuda":
                raise ValueError("Move the Transformers student to CUDA before initializing virtual weights.")

            original_load = Worker.load_model
            original_llm = vllm_generation.LLM

            def load_and_bind(worker, *args, **kwargs):
                before = torch.cuda.memory_allocated(worker.device)
                original_pool_context = worker._maybe_get_memory_pool_context
                # The initial private weights are about to be replaced. Use the ordinary allocator so their
                # unused blocks can be released rather than retained by the long-lived tagged weight pool.
                worker._maybe_get_memory_pool_context = lambda tag: nullcontext()
                try:
                    result = original_load(worker, *args, **kwargs)
                finally:
                    worker._maybe_get_memory_pool_context = original_pool_context
                # Finalize resident storage after loading and before profile_run can compile or capture execution.
                store = Qwen35VirtualWeightStore.bind(generation.model, worker.model_runner.get_model())
                generation.virtual_weight_store = store
                store.publish()
                gc.collect()
                torch.cuda.empty_cache()
                # The HF weights were already resident before vLLM initialization; account only for new storage.
                worker.model_runner.model_memory_usage = max(0, torch.cuda.memory_allocated(worker.device) - before)
                return result

            Worker.load_model = load_and_bind
            vllm_generation.LLM = partial(original_llm, compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"})
            try:
                self.original_init(generation)
                if generation.virtual_weight_store is None:
                    raise RuntimeError("The colocated vLLM worker did not bind virtual student weights.")
                generation.virtual_weight_store.publish()
            finally:
                Worker.load_model = original_load
                vllm_generation.LLM = original_llm

        vllm_generation.VLLMGeneration._init_vllm = init_with_shared_weights
        return self

    def __exit__(self, *args):
        if self.enabled:
            from ...generation import vllm_generation

            vllm_generation.VLLMGeneration._init_vllm = self.original_init


class EagerVLLMInitForDeterminism:
    """Force colocated vLLM initialization to be eager when required by MiniLLM."""

    def __init__(self, enabled: bool, deterministic: bool):
        self.enabled = enabled
        self.deterministic = deterministic
        self.vllm_generation = None
        self.original_llm = None

    def __enter__(self) -> None:
        if not self.enabled or not (self.deterministic or torch.are_deterministic_algorithms_enabled()):
            return

        # VLLMGeneration imports LLM into its own module namespace. Patch that reference only while GRPOTrainer
        # creates the colocated engine, then restore it so this MiniLLM-specific workaround does not affect other
        # trainers.
        from ...generation import vllm_generation

        self.vllm_generation = vllm_generation
        self.original_llm = vllm_generation.LLM
        vllm_generation.LLM = partial(self.original_llm, enforce_eager=True)
        logger.info("PyTorch deterministic algorithms are enabled; initializing colocated vLLM in eager mode.")

    def __exit__(self, *args) -> None:
        if self.vllm_generation is not None:
            self.vllm_generation.LLM = self.original_llm


class TeacherVLLM:
    """Manage the teacher vLLM server and score completion tokens."""

    def __init__(self, model: str, args: MiniLLMConfig):
        from ...generation.vllm_client import VLLMClient

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]

        command = [
            sys.executable,
            "-m",
            "trl.scripts.vllm_serve",
            "--model",
            model,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--gpu-memory-utilization",
            str(args.teacher_vllm_gpu_memory_utilization),
            "--vllm-model-impl",
            args.vllm_model_impl,
        ]
        if args.vllm_max_model_length is not None:
            command.extend(["--max-model-len", str(args.vllm_max_model_length)])
        if args.trust_remote_code:
            command.append("--trust-remote-code")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
            env.pop(name, None)

        base_url = f"http://127.0.0.1:{port}"
        logger.info("Starting teacher vLLM server at %s with model %s", base_url, model)
        self.process = subprocess.Popen(command, env=env, start_new_session=True)
        atexit.register(self.stop)
        try:
            self.client = VLLMClient(
                base_url=base_url,
                connection_timeout=args.teacher_vllm_server_timeout,
            )
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        """Stop the teacher vLLM process group if it is still running."""
        if self.process is None or self.process.poll() is not None:
            return

        logger.info("Stopping teacher vLLM server")
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
            self.process.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            if self.process.poll() is None:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
        finally:
            self.process = None

    def get_log_probs(self, inputs: dict[str, torch.Tensor], temperature: float) -> torch.Tensor:
        """Return the teacher log probability of each sampled completion token."""
        sequences = []
        prompt_lengths = []
        completion_token_ids = []
        for prompt_ids, prompt_mask, completion_ids, completion_mask in zip(
            inputs["prompt_ids"],
            inputs["prompt_mask"],
            inputs["completion_ids"],
            inputs["completion_mask"],
            strict=True,
        ):
            prompt = prompt_ids[prompt_mask.bool()].tolist()
            completion = completion_ids[completion_mask.bool()].tolist()
            sequences.append(prompt + completion)
            prompt_lengths.append(len(prompt))
            completion_token_ids.append(completion)

        result = self.client.get_sequence_logprobs(
            sequences=sequences,
            prompt_lengths=prompt_lengths,
            top_logprobs=1,
            temperature=temperature,
        )

        batch_size, max_completion_length = inputs["completion_ids"].shape
        teacher_log_probs = torch.zeros(
            (batch_size, max_completion_length), dtype=torch.float32, device=inputs["completion_ids"].device
        )

        for i, (expected_ids, log_probs) in enumerate(
            zip(completion_token_ids, result["actual_logprobs"], strict=True)
        ):
            if not expected_ids:
                continue
            values = torch.tensor(log_probs, dtype=torch.float32).squeeze(-1)
            teacher_log_probs[i, : len(expected_ids)] = values.to(teacher_log_probs.device)

        return teacher_log_probs
