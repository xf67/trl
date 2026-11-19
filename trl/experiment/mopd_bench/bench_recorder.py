"""Shared step recorder for the HF-Trainer based baselines (TRL, Unsloth).

Writes RUN_DIR/steps.jsonl with the same fields used for the verl runs and stops
training after WARMUP_STEPS + a measured window of TOKEN_BUDGET tokens
(and at least MIN_MEASURED_STEPS steps). Tokens are prompt + generated tokens
that are valid (attention mask) in the trained batch.
"""
import json
import os
from pathlib import Path
import time

import torch
from transformers import TrainerCallback

_STATE = dict(tokens=0, response_tokens=0, sequences=0, rollout_s=0.0)


def add_tokens(total, response, sequences):
    _STATE['tokens'] += int(total)
    _STATE['response_tokens'] += int(response)
    _STATE['sequences'] += int(sequences)


def add_rollout(seconds):
    _STATE['rollout_s'] += float(seconds)


def ordered_prompts(tokenizer, parquet, max_prompt_length, limit):
    """Same stream as verl: dataset order, chat template, prompts > limit filtered."""
    import pandas as pd
    mapping = {'math': 0, 'code': 1, 'if': 2}
    frame = pd.read_parquet(parquet)
    rows = []
    for index, row in enumerate(frame.itertuples(index=False)):
        messages = [dict(m) for m in row.prompt]
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        if hasattr(ids, 'input_ids'):
            ids = ids['input_ids']
        ids = list(ids)
        if not 0 < len(ids) <= max_prompt_length:
            continue
        rows.append(dict(input_ids=ids, teacher_id=mapping[row.domain], sample_id=index, domain=row.domain,
                         messages=messages))
        if len(rows) >= limit:
            break
    return rows


class Bench(TrainerCallback):
    def __init__(self):
        self.out = Path(os.environ['RUN_DIR'])
        self.warmup = int(os.environ['WARMUP_STEPS'])
        self.budget = int(os.environ['TOKEN_BUDGET'])
        self.min_steps = int(os.environ['MIN_MEASURED_STEPS'])
        self.measured_tokens = 0
        self.measured_steps = 0
        self.t_start = None

    def on_train_begin(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        self.t_start = time.time()
        (self.out / 'train-begin.json').write_text(json.dumps({'wall_time': self.t_start}))

    def on_step_begin(self, args, state, control, **kwargs):
        torch.cuda.reset_peak_memory_stats()

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        now = time.time()
        step = state.global_step
        if step > self.warmup:
            self.measured_tokens += _STATE['tokens']
            self.measured_steps += 1
        last = self.measured_tokens >= self.budget and self.measured_steps >= self.min_steps
        row = dict(step=step, wall_time=now, metrics={
            'suite/batch_tokens': _STATE['tokens'],
            'suite/batch_response_tokens': _STATE['response_tokens'],
            'suite/batch_prompt_tokens': _STATE['tokens'] - _STATE['response_tokens'],
            'suite/batch_sequences': _STATE['sequences'],
            'suite/measured_tokens': self.measured_tokens,
            'suite/measured_steps': self.measured_steps,
            'suite/budget_reached': int(last),
            'suite/rollout_s': _STATE['rollout_s'],
            'torch/max_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
            'torch/max_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
        })
        with (self.out / 'steps.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        for key in _STATE:
            _STATE[key] = 0.0 if key == 'rollout_s' else 0
        if last:
            control.should_training_stop = True
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            with (self.out / 'trainer-log.jsonl').open('a') as stream:
                stream.write(json.dumps({'step': state.global_step, **{k: v for k, v in logs.items()
                                                                       if isinstance(v, (int, float))}}) + '\n')

    def finish(self):
        (self.out / 'completed.json').write_text(json.dumps({'finished_at': time.time()}))
