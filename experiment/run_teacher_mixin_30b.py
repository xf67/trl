import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = REPO_DIR / "experiment"

SCRIPT_BY_DATASET = {
    "mbpp": EXPERIMENT_DIR / "compare_teacher_mixin_mbpp.py",
    "humaneval": EXPERIMENT_DIR / "compare_teacher_mixin_humaneval.py",
}

DEFAULTS_BY_DATASET = {
    "mbpp": {
        "TRAIN_SIZE": "96",
        "EVAL_SIZE": "100",
        "MAX_STEPS": "20",
        "PER_DEVICE_TRAIN_BATCH_SIZE": "1",
        "NUM_GENERATIONS": "1",
        "MAX_COMPLETION_LENGTH": "160",
        "EVAL_GENERATION_BATCH_SIZE": "1",
        "EVAL_MAX_NEW_TOKENS": "192",
        "EVAL_TIMEOUT_SECONDS": "3",
    },
    "humaneval": {
        "TRAIN_SIZE": "96",
        "EVAL_SIZE": "48",
        "MAX_STEPS": "20",
        "PER_DEVICE_TRAIN_BATCH_SIZE": "1",
        "NUM_GENERATIONS": "1",
        "MAX_COMPLETION_LENGTH": "160",
        "EVAL_GENERATION_BATCH_SIZE": "1",
        "EVAL_MAX_NEW_TOKENS": "192",
        "EVAL_TIMEOUT_SECONDS": "3",
    },
}

SMOKE_DEFAULTS_BY_DATASET = {
    "mbpp": {
        "TRAIN_SIZE": "24",
        "EVAL_SIZE": "8",
        "MAX_STEPS": "1",
    },
    "humaneval": {
        "TRAIN_SIZE": "24",
        "EVAL_SIZE": "12",
        "MAX_STEPS": "1",
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(SCRIPT_BY_DATASET), default=os.environ.get("DATASET", "mbpp"))
    parser.add_argument(
        "--student-model-path",
        default=os.environ.get("MODEL_PATH", "/home/xxf/models/Qwen3.5-0.8B-Base"),
    )
    parser.add_argument(
        "--teacher-model-path",
        default=os.environ.get("TEACHER_MODEL_PATH", "/home/xxf/models/Qwen3-30B-A3B"),
    )
    parser.add_argument("--teacher-mixin-alphas", default=os.environ.get("TEACHER_MIXIN_ALPHAS", "0.0,0.2"))
    parser.add_argument("--seed", default=os.environ.get("SEED", "42"))
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    script_path = SCRIPT_BY_DATASET[args.dataset]

    env = os.environ.copy()
    env["TRL_EXPERIMENTAL_SILENCE"] = env.get("TRL_EXPERIMENTAL_SILENCE", "1")
    env["HF_ENDPOINT"] = env.get("HF_ENDPOINT", "https://hf-mirror.com")
    env["MODEL_PATH"] = args.student_model_path
    env["TEACHER_MODEL_PATH"] = args.teacher_model_path
    env["TEACHER_MIXIN_ALPHAS"] = args.teacher_mixin_alphas
    env["SEED"] = args.seed

    for key, value in DEFAULTS_BY_DATASET[args.dataset].items():
        env[key] = env.get(key, value)

    if args.smoke:
        for key, value in SMOKE_DEFAULTS_BY_DATASET[args.dataset].items():
            env[key] = value

    command = [sys.executable, str(script_path)]
    print("Running command:")
    print(" ".join(command))
    print()
    print("Key environment:")
    for key in [
        "HF_ENDPOINT",
        "MODEL_PATH",
        "TEACHER_MODEL_PATH",
        "TEACHER_MIXIN_ALPHAS",
        "TRAIN_SIZE",
        "EVAL_SIZE",
        "MAX_STEPS",
        "PER_DEVICE_TRAIN_BATCH_SIZE",
        "NUM_GENERATIONS",
        "MAX_COMPLETION_LENGTH",
        "EVAL_GENERATION_BATCH_SIZE",
        "EVAL_MAX_NEW_TOKENS",
        "EVAL_TIMEOUT_SECONDS",
        "SEED",
    ]:
        print(f"{key}={env[key]}")
    print(f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '<inherited-unset>')}")
    print()

    raise SystemExit(subprocess.run(command, cwd=REPO_DIR, env=env).returncode)


if __name__ == "__main__":
    main()
