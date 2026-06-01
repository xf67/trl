import ast
import gc
import json
import os
import random
import re
import signal
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from trl.experimental.minillm import MiniLLMConfig, MiniLLMTrainer


REPO_DIR = Path(__file__).resolve().parents[1]
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
FUNCTION_NAME_RE = re.compile(r"assert\s*\(?\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")
ALLOWED_IMPORTS = {
    "bisect",
    "collections",
    "functools",
    "heapq",
    "itertools",
    "math",
    "operator",
    "re",
    "string",
    "typing",
}

SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "range": range,
    "reversed": reversed,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


@dataclass
class MBPPTask:
    task_id: int
    prompt: str
    test_setup_code: str
    test_list: list[str]


def safe_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
    root_name = name.split(".")[0]
    if root_name not in ALLOWED_IMPORTS:
        raise ImportError(f"Import of '{name}' is not allowed")
    return __import__(name, globals, locals, fromlist, level)


SAFE_BUILTINS["__import__"] = safe_import


def build_processing_class(model_path: Path):
    processing_class = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    processing_class.padding_side = "left"
    if processing_class.pad_token is None:
        processing_class.pad_token = processing_class.eos_token
    return processing_class


def extract_function_name(test_list: list[str]) -> str:
    for test_case in test_list:
        match = FUNCTION_NAME_RE.search(test_case)
        if match is not None:
            return match.group(1)
    raise ValueError(f"Could not infer function name from tests: {test_list!r}")


def build_prompt(text: str, function_name: str) -> str:
    return (
        f"Write a Python function named `{function_name}`.\n"
        "Return only Python code in a single ```python``` block.\n\n"
        "Task:\n"
        f"{text}\n"
    )


def load_mbpp_tasks(seed: int, train_size: int, eval_size: int):
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    dataset = load_dataset("mbpp")

    train_rows = list(dataset["train"])
    eval_rows = list(dataset["test"])
    random.Random(seed).shuffle(train_rows)
    random.Random(seed).shuffle(eval_rows)

    if train_size > len(train_rows):
        raise ValueError(f"Requested {train_size} training tasks, but only {len(train_rows)} MBPP train tasks exist.")
    if eval_size > len(eval_rows):
        raise ValueError(f"Requested {eval_size} eval tasks, but only {len(eval_rows)} MBPP test tasks exist.")

    train_rows = train_rows[:train_size]
    eval_rows = eval_rows[:eval_size]

    train_prompts = []
    for row in train_rows:
        function_name = extract_function_name(row["test_list"])
        train_prompts.append(build_prompt(row["text"], function_name))

    eval_tasks = []
    for row in eval_rows:
        function_name = extract_function_name(row["test_list"])
        eval_tasks.append(
            MBPPTask(
                task_id=row["task_id"],
                prompt=build_prompt(row["text"], function_name),
                test_setup_code=row["test_setup_code"],
                test_list=row["test_list"],
            )
        )

    train_dataset = Dataset.from_dict({"prompt": train_prompts})
    return train_dataset, eval_tasks


def extract_candidate_code(prediction: str) -> str:
    match = CODE_BLOCK_RE.search(prediction)
    if match is not None:
        return match.group(1).strip()

    stripped = prediction.lstrip()
    if stripped.startswith(("def ", "class ", "from ", "import ")):
        return stripped

    def_index = prediction.find("def ")
    if def_index != -1:
        return prediction[def_index:].strip()
    return stripped


def validate_code_ast(code: str):
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    raise ValueError(f"Import of '{alias.name}' is not allowed")
        if isinstance(node, ast.ImportFrom):
            if node.module is None or node.module.split(".")[0] not in ALLOWED_IMPORTS:
                raise ValueError(f"Import from '{node.module}' is not allowed")
        if isinstance(node, ast.Attribute) and isinstance(node.attr, str) and node.attr.startswith("__"):
            raise ValueError("dunder attribute access is not allowed")


class Timeout:
    def __init__(self, seconds: int):
        self.seconds = seconds
        self.previous_handler = None

    def _handle_timeout(self, signum, frame):
        raise TimeoutError("execution timed out")

    def __enter__(self):
        self.previous_handler = signal.signal(signal.SIGALRM, self._handle_timeout)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)

    def __exit__(self, exc_type, exc, tb):
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self.previous_handler)


def run_mbpp_test(task: MBPPTask, prediction: str, timeout_seconds: int):
    code = extract_candidate_code(prediction)
    validate_code_ast(code)

    globals_dict = {"__builtins__": SAFE_BUILTINS}
    with Timeout(timeout_seconds):
        if task.test_setup_code:
            exec(task.test_setup_code, globals_dict, globals_dict)
        exec(code, globals_dict, globals_dict)
        for test_case in task.test_list:
            exec(test_case, globals_dict, globals_dict)
    return code


def generate_predictions(model, processing_class, prompts: list[str], max_new_tokens: int, batch_size: int) -> list[str]:
    device = next(model.parameters()).device
    generations = []
    model.eval()
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        batch_inputs = processing_class(batch_prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            batch_outputs = model.generate(
                **batch_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=processing_class.pad_token_id,
                eos_token_id=processing_class.eos_token_id,
            )
        batch_completions = batch_outputs[:, batch_inputs["input_ids"].shape[1] :]
        generations.extend(processing_class.batch_decode(batch_completions, skip_special_tokens=True))
    return generations


def evaluate_mbpp(
    model,
    processing_class,
    tasks: list[MBPPTask],
    batch_size: int,
    max_new_tokens: int,
    timeout_seconds: int,
):
    predictions = generate_predictions(
        model,
        processing_class,
        [task.prompt for task in tasks],
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )

    valid_code = 0
    functional = 0
    detail = []
    for task, prediction in zip(tasks, predictions, strict=True):
        is_valid = False
        is_functional = False
        code = None
        error = None
        try:
            code = run_mbpp_test(task, prediction, timeout_seconds)
            is_valid = True
            is_functional = True
        except SyntaxError as exc:
            error = f"SyntaxError: {exc}"
        except Exception as exc:  # noqa: BLE001
            try:
                code = extract_candidate_code(prediction)
                validate_code_ast(code)
                is_valid = True
            except Exception:  # noqa: BLE001
                pass
            error = f"{type(exc).__name__}: {exc}"

        valid_code += int(is_valid)
        functional += int(is_functional)
        detail.append(
            {
                "task_id": task.task_id,
                "prediction": prediction,
                "valid_code": is_valid,
                "functional": is_functional,
                "error": error,
            }
        )

    return {
        "valid_code_rate": valid_code / len(tasks),
        "functional_accuracy": functional / len(tasks),
        "detail": detail,
    }


def run_variant(
    *,
    alpha: float,
    model_path: Path,
    teacher_model_path: Path,
    processing_class,
    train_dataset,
    eval_tasks: list[MBPPTask],
    output_dir: Path,
    max_steps: int,
    per_device_train_batch_size: int,
    num_generations: int,
    max_completion_length: int,
    eval_generation_batch_size: int,
    eval_max_new_tokens: int,
    eval_timeout_seconds: int,
    seed: int,
):
    args = MiniLLMConfig(
        output_dir=str(output_dir),
        report_to="none",
        save_strategy="no",
        logging_steps=1,
        max_steps=max_steps,
        seed=seed,
        data_seed=seed,
        bf16=USE_BF16,
        tf32=True,
        per_device_train_batch_size=per_device_train_batch_size,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        teacher_mixin_alpha=alpha,
        model_init_kwargs={"dtype": "bfloat16" if USE_BF16 else "float32"},
        teacher_model_init_kwargs={"dtype": "bfloat16" if USE_BF16 else "float32"},
    )

    trainer = MiniLLMTrainer(
        model=str(model_path),
        teacher_model=str(teacher_model_path),
        args=args,
        train_dataset=train_dataset,
        processing_class=processing_class,
    )

    student_pretrain = evaluate_mbpp(
        trainer.model,
        processing_class,
        eval_tasks,
        batch_size=eval_generation_batch_size,
        max_new_tokens=eval_max_new_tokens,
        timeout_seconds=eval_timeout_seconds,
    )
    teacher_eval = evaluate_mbpp(
        trainer.teacher_model,
        processing_class,
        eval_tasks,
        batch_size=eval_generation_batch_size,
        max_new_tokens=eval_max_new_tokens,
        timeout_seconds=eval_timeout_seconds,
    )

    train_result = trainer.train()
    train_metrics = train_result.metrics

    student_posttrain = evaluate_mbpp(
        trainer.model,
        processing_class,
        eval_tasks,
        batch_size=eval_generation_batch_size,
        max_new_tokens=eval_max_new_tokens,
        timeout_seconds=eval_timeout_seconds,
    )

    result = {
        "teacher_mixin_alpha": alpha,
        "student_pretrain_valid_code_rate": student_pretrain["valid_code_rate"],
        "student_pretrain_functional_accuracy": student_pretrain["functional_accuracy"],
        "teacher_valid_code_rate": teacher_eval["valid_code_rate"],
        "teacher_functional_accuracy": teacher_eval["functional_accuracy"],
        "student_posttrain_valid_code_rate": student_posttrain["valid_code_rate"],
        "student_posttrain_functional_accuracy": student_posttrain["functional_accuracy"],
        "train_loss": train_metrics.get("train_loss"),
        "train_runtime": train_metrics.get("train_runtime"),
    }

    accelerator = trainer.accelerator
    del student_pretrain
    del teacher_eval
    del student_posttrain
    del train_result
    del train_metrics
    del trainer
    gc.collect()
    accelerator.free_memory()
    del accelerator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    return result


def main():
    model_path = Path(os.environ.get("MODEL_PATH", "/home/xxf/models/Qwen3.5-0.8B-Base"))
    teacher_model_path = Path(os.environ.get("TEACHER_MODEL_PATH", "/home/xxf/models/Qwen3.5-4B"))
    output_root = REPO_DIR / "compare_teacher_mixin_mbpp"

    train_size = int(os.environ.get("TRAIN_SIZE", "96"))
    eval_size = int(os.environ.get("EVAL_SIZE", "100"))
    max_steps = int(os.environ.get("MAX_STEPS", "20"))
    per_device_train_batch_size = int(os.environ.get("PER_DEVICE_TRAIN_BATCH_SIZE", "1"))
    num_generations = int(os.environ.get("NUM_GENERATIONS", "1"))
    max_completion_length = int(os.environ.get("MAX_COMPLETION_LENGTH", "160"))
    eval_generation_batch_size = int(os.environ.get("EVAL_GENERATION_BATCH_SIZE", "4"))
    eval_max_new_tokens = int(os.environ.get("EVAL_MAX_NEW_TOKENS", "192"))
    eval_timeout_seconds = int(os.environ.get("EVAL_TIMEOUT_SECONDS", "3"))
    alphas = [float(alpha) for alpha in os.environ.get("TEACHER_MIXIN_ALPHAS", "0.0,0.2").split(",")]
    seed = int(os.environ.get("SEED", "42"))

    print(
        json.dumps(
            {
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
                "hf_endpoint": os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
                "model_path": str(model_path),
                "teacher_model_path": str(teacher_model_path),
                "train_size": train_size,
                "eval_size": eval_size,
                "max_steps": max_steps,
                "per_device_train_batch_size": per_device_train_batch_size,
                "num_generations": num_generations,
                "max_completion_length": max_completion_length,
                "eval_generation_batch_size": eval_generation_batch_size,
                "eval_max_new_tokens": eval_max_new_tokens,
                "eval_timeout_seconds": eval_timeout_seconds,
                "alphas": alphas,
                "seed": seed,
            },
            indent=2,
        )
    )

    processing_class = build_processing_class(model_path)
    train_dataset, eval_tasks = load_mbpp_tasks(seed=seed, train_size=train_size, eval_size=eval_size)

    results = []
    for alpha in alphas:
        output_dir = output_root / f"alpha_{str(alpha).replace('.', '_')}"
        result = run_variant(
            alpha=alpha,
            model_path=model_path,
            teacher_model_path=teacher_model_path,
            processing_class=processing_class,
            train_dataset=train_dataset,
            eval_tasks=eval_tasks,
            output_dir=output_dir,
            max_steps=max_steps,
            per_device_train_batch_size=per_device_train_batch_size,
            num_generations=num_generations,
            max_completion_length=max_completion_length,
            eval_generation_batch_size=eval_generation_batch_size,
            eval_max_new_tokens=eval_max_new_tokens,
            eval_timeout_seconds=eval_timeout_seconds,
            seed=seed,
        )
        results.append(result)
        print(json.dumps(result, indent=2))

    if len(results) == 2:
        summary = {
            "baseline_alpha": results[0]["teacher_mixin_alpha"],
            "compare_alpha": results[1]["teacher_mixin_alpha"],
            "functional_accuracy_delta": (
                results[1]["student_posttrain_functional_accuracy"]
                - results[0]["student_posttrain_functional_accuracy"]
            ),
            "valid_code_rate_delta": (
                results[1]["student_posttrain_valid_code_rate"] - results[0]["student_posttrain_valid_code_rate"]
            ),
        }
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
