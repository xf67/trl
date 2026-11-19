"""TRL + vLLM MOPD: trl.DistillationTrainer with colocated vLLM rollout.

Upstream pieces used unchanged: on-policy generation through a colocated vLLM engine (sleep mode between
phases, student weights pushed into vLLM after each optimizer step), chunked reverse-KL loss (beta=1),
gradient checkpointing.
Added for MOPD (same as the GKD-MOPD fix): each sample is routed to its domain teacher. A micro-batch is
split by teacher_id and each part is scored by its teacher. The per-part losses use the batch-wide token count,
so they sum to the same token mean. Teacher backbones live on the CPU and only the selected one is moved to the
GPU. Each teacher keeps an untied GPU copy of its lm_head (0.25 GB) so the chunked loss's backward never sees
a CPU tensor.
Optimizer: CPUOffloadBF16StochasticAdamW from the TRL GKD-MOPD baseline (BF16 params/grads on GPU, FP32 moments
and update on CPU).
"""
import json
import os
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'gkd_mopd_baseline'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cpu_offload_adamw import CPUOffloadBF16StochasticAdamW  # noqa: E402
from datasets import Dataset  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed  # noqa: E402
from trl import DistillationConfig, DistillationTrainer  # noqa: E402

import bench_recorder as hf_bench  # noqa: E402

BS = int(os.environ['BATCH'])
MODEL_ROOT = Path('/home/xxf/models/OPD')
OUT = Path(os.environ['RUN_DIR'])
VLLM_UTIL = float(os.environ.get('VLLM_GPU_UTIL', '0.30'))
torch.set_num_threads(int(os.environ.get('CPU_THREADS', '16')))
set_seed(42)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ROOT / 'MixSFT', local_files_only=True)
rows = hf_bench.ordered_prompts(tokenizer, os.environ['DATA_PARQUET'], 1024, 6000)
dataset = Dataset.from_list([{'prompt': r['messages'], 'teacher_id': r['teacher_id']} for r in rows])
print('prompts', len(rows), flush=True)

student = AutoModelForCausalLM.from_pretrained(MODEL_ROOT / 'MixSFT', dtype=torch.bfloat16, local_files_only=True)
teachers = {}
for i, name in enumerate(('Math', 'Code', 'IF')):
    t = AutoModelForCausalLM.from_pretrained(MODEL_ROOT / name, dtype=torch.bfloat16, local_files_only=True)
    t.requires_grad_(False).eval()
    # untied, GPU-resident output head; the backbone is swapped between CPU and GPU
    t.lm_head.weight = torch.nn.Parameter(t.lm_head.weight.detach().clone().cuda(), requires_grad=False)
    t.config.tie_word_embeddings = False
    t.model.to('cpu')
    teachers[i] = t


class MOPDDistillationTrainer(DistillationTrainer):
    def _generate(self, prompts):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = super()._generate(prompts)
        torch.cuda.synchronize()
        hf_bench.add_rollout(time.perf_counter() - t0)
        return out

    def _generate_and_score_completions(self, inputs):
        out = super()._generate_and_score_completions(inputs)
        out['teacher_id'] = torch.tensor([x['teacher_id'] for x in inputs], device=out['prompt_ids'].device)
        hf_bench.add_tokens(out['prompt_mask'].sum() + out['completion_mask'].sum(), out['completion_mask'].sum(),
                            out['prompt_ids'].shape[0])
        return out

    def _select(self, teacher_id):
        if getattr(self, '_active', None) == teacher_id:
            return
        if getattr(self, '_active', None) is not None:
            teachers[self._active].model.to('cpu')
        teachers[teacher_id].model.to('cuda')
        self.teacher_model = teachers[teacher_id]
        self._active = teacher_id

    def _compute_loss(self, unwrapped_student, inputs, num_items_in_batch):
        ids = inputs['teacher_id'].view(-1).cpu()
        if num_items_in_batch is None:
            num_items_in_batch = inputs['completion_mask'].sum().clamp(min=1)
        order = sorted(set(ids.tolist()), key=lambda t: (t != getattr(self, '_active', None), t))
        loss = entropy = n_valid = 0
        for teacher_id in order:
            rows_ = (ids == teacher_id).nonzero().view(-1).to(inputs['prompt_ids'].device)
            sub = {k: (v[rows_] if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == ids.numel() else v)
                   for k, v in inputs.items()}
            self._select(teacher_id)
            part, ent, nv = super()._compute_loss(unwrapped_student, sub, num_items_in_batch)
            loss, entropy, n_valid = loss + part, entropy + ent, n_valid + nv
        return loss, entropy, n_valid


config = DistillationConfig(
    output_dir=str(OUT / 'trainer_output'), per_device_train_batch_size=BS, gradient_accumulation_steps=1,
    max_steps=400, learning_rate=1e-6, weight_decay=0.01, max_grad_norm=1.0, lr_scheduler_type='constant',
    warmup_steps=0, beta=1.0, temperature=1.0, top_p=1.0, top_k=0, max_completion_length=1024, bf16=True,
    gradient_checkpointing=True, logging_steps=1, save_strategy='no', report_to='none', seed=42,
    shuffle_dataset=False, remove_unused_columns=False,
    use_vllm=True, vllm_mode='colocate', vllm_enable_sleep_mode=True,
    vllm_gpu_memory_utilization=VLLM_UTIL, vllm_max_model_length=2048,
)
bench = hf_bench.Bench()
trainer = MOPDDistillationTrainer(model=student, teacher_model=teachers[0], args=config, train_dataset=dataset,
                                  processing_class=tokenizer, callbacks=[bench])
teachers[0].model.to('cpu')
trainer._active = None
# vLLM's Transformers backend decorates the shared HF SmolLM3Model class with its compile wrapper;
# opt the HF training/teacher backbones out so they run their normal eager forward.
for m in [student.model] + [t.model for t in teachers.values()]:
    m.do_not_compile = True
trainer.optimizer = CPUOffloadBF16StochasticAdamW(student.parameters(), lr=1e-6, betas=(0.9, 0.999), eps=1e-8,
                                                  weight_decay=0.01, foreach=False)
(OUT / 'config.json').write_text(json.dumps({
    'batch_size': BS, 'vllm_gpu_memory_utilization': VLLM_UTIL, 'trainer': 'DistillationTrainer',
    'rollout': 'vLLM colocate + sleep mode', 'trl_file': sys.modules['trl'].__file__,
    'vllm': __import__('vllm').__version__, 'gpu': torch.cuda.get_device_name(0)}, indent=2))
trainer.train()
bench.finish()
print('COMPLETED', flush=True)
