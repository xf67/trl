# MOPD speed benchmark scripts (single RTX A6000, 2026-11-18)

Used for the TRL rows of the single-GPU MOPD speed study (`fourway-speed-a6000-20261118`).
Workload: MixSFT student, Math/Code/IF teachers routed by `domain`, prompts/responses <= 1024 tokens,
lr 1e-6, reverse KL (beta=1), the Open-MOPD 12k-row parquet in file order.

All scripts take their settings from the environment:
`RUN_DIR`, `BATCH`, `WARMUP_STEPS`, `TOKEN_BUDGET`, `MIN_MEASURED_STEPS` (+ `DATA_PARQUET` where noted).
`bench_recorder.py` writes `RUN_DIR/steps.jsonl` (wall time, prompt/response tokens, rollout seconds) and stops
training after the warm-up steps plus a window of `TOKEN_BUDGET` tokens.

- `gkd_bench_wrap.py`: runs `../gkd_mopd_baseline/train.py` unchanged (pass its normal CLI arguments) and only adds
  the recorder. A6000 setting used: `--batch-size B --max-prompt-length 1024 --max-new-tokens 1024
  --max-length 2048 --loss-chunk-size 1024 --gradient-checkpointing --cpu-adam --cpu-threads 16`.
- `distillation_vllm_mopd.py`: "TRL + vLLM". `trl.DistillationTrainer` with colocated vLLM rollout
  (sleep mode, weights pushed every step), chunked reverse KL, gradient checkpointing; adds per-sample teacher
  routing (batch split by teacher, token-weighted), teacher backbones swapped CPU<->GPU with GPU-resident untied
  `lm_head`s, and `CPUOffloadBF16StochasticAdamW` from `../gkd_mopd_baseline`. Extra env: `DATA_PARQUET`,
  `VLLM_GPU_UTIL` (0.30), `CPU_THREADS` (16). For concurrent runs give each its own `MASTER_PORT`.
  Note: vLLM's Transformers backend decorates the HF `SmolLM3Model` class, so the HF models set
  `do_not_compile = True`.

Python: `/home/zst/miniconda3/envs/minillm_vllm_qwen35` (torch 2.11, vLLM 0.25.1), `PYTHONPATH=<this repo>`.
