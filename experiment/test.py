import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from trl.experimental.minillm import MiniLLMConfig, MiniLLMTrainer

ROOT_DIR = Path(__file__).resolve().parents[2]
TRACE_MAX_ENTRIES = 9223372036854775807
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
ENABLE_MEMORY_VIZ = os.environ.get("ENABLE_MEMORY_VIZ", "0") == "1"
TEACHER_MIXIN_ALPHA = float(os.environ.get("TEACHER_MIXIN_ALPHA", "0.0"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "3"))
TRAIN_SUBSET = int(os.environ.get("TRAIN_SUBSET", "0"))
PER_DEVICE_TRAIN_BATCH_SIZE = int(os.environ.get("PER_DEVICE_TRAIN_BATCH_SIZE", "1"))
NUM_GENERATIONS = int(os.environ.get("NUM_GENERATIONS", "1"))
MAX_COMPLETION_LENGTH = int(os.environ.get("MAX_COMPLETION_LENGTH", "32"))
rollout_snapshot_path = ROOT_DIR / "trl/experiment/memory_snapshot_rollout.pickle"
training_snapshot_path = ROOT_DIR / "trl/experiment/memory_snapshot_training.pickle"


def print_peak_memory(tag: str):
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
    print(f"[peak] {tag}: allocated={peak_allocated:.2f} GiB, reserved={peak_reserved:.2f} GiB")


def dump_snapshot(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.memory._dump_snapshot(str(path))
    print(f"memory snapshot saved to {path.resolve()}")
    print("open it with https://pytorch.org/memory_viz")


class PeakMemoryMiniLLMTrainer(MiniLLMTrainer):
    def _profile_peak(self, tag: str, fn):
        if not torch.cuda.is_available():
            return fn()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        output = fn()
        print_peak_memory(tag)
        return output

    def _profile_memory_viz(self, tag: str, snapshot_path: Path, saved_attr: str, fn):
        if not ENABLE_MEMORY_VIZ or getattr(self, saved_attr, False) or not torch.cuda.is_available():
            return self._profile_peak(tag, fn)

        def run():
            torch.cuda.memory._record_memory_history(
                enabled="all",
                context="all",
                stacks="python",
                max_entries=TRACE_MAX_ENTRIES,
                clear_history=True,
            )
            try:
                return fn()
            finally:
                torch.cuda.synchronize()
                dump_snapshot(snapshot_path)
                torch.cuda.memory._record_memory_history(enabled=None)
                setattr(self, saved_attr, True)

        return self._profile_peak(tag, run)

    def _generate_and_score_completions(self, *args, **kwargs):
        return self._profile_memory_viz(
            "_generate_and_score_completions",
            rollout_snapshot_path,
            "_rollout_profile_saved",
            lambda: super(PeakMemoryMiniLLMTrainer, self)._generate_and_score_completions(*args, **kwargs),
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        return self._profile_memory_viz(
            "compute_loss",
            training_snapshot_path,
            "_training_profile_saved",
            lambda: super(PeakMemoryMiniLLMTrainer, self).compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            ),
        )

dataset = load_dataset(str(ROOT_DIR / "data-tldr"), data_dir="data", split="train")
if TRAIN_SUBSET > 0:
    dataset = dataset.select(range(min(TRAIN_SUBSET, len(dataset))))

processing_class = AutoTokenizer.from_pretrained(str(ROOT_DIR / "models/Qwen3.5-0.8B-Base"), local_files_only=True)
processing_class.padding_side = "left"
if processing_class.pad_token is None:
    processing_class.pad_token = processing_class.eos_token

args_kwargs = {
    "output_dir": str(ROOT_DIR / "Qwen3.5-0.8B-Base-MiniLLM"),
    "bf16": USE_BF16,
    "tf32": True,
    "report_to": "none",
    "save_strategy": "no",
    "logging_steps": 1,
    "max_steps": MAX_STEPS,
    "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
    "num_generations": NUM_GENERATIONS,
    "max_completion_length": MAX_COMPLETION_LENGTH,
    "model_init_kwargs": {"dtype": "bfloat16" if USE_BF16 else "float32"},
    "teacher_model_init_kwargs": {"dtype": "bfloat16" if USE_BF16 else "float32"},
    "teacher_mixin_alpha": TEACHER_MIXIN_ALPHA,
}
if ENABLE_MEMORY_VIZ:
    args_kwargs["max_steps"] = 1

args = MiniLLMConfig(
    **args_kwargs,
)

trainer = PeakMemoryMiniLLMTrainer(
    model=str(ROOT_DIR / "models/Qwen3.5-0.8B-Base"),
    teacher_model=str(ROOT_DIR / "models/Qwen3.5-2B"),
    args=args,
    train_dataset=dataset,
    processing_class=processing_class,
)
trainer.train()
