"""Run the TRL GKD-MOPD baseline script (~/Distill/trl working tree, per TRL_READY.md)
unchanged, adding only the shared step recorder (timestamps, prompt+response token
counts, early stop after the measured token budget)."""
import os
from pathlib import Path
import runpy
import sys

HARNESS = Path(__file__).resolve().parent
SCRIPT = Path(__file__).resolve().parents[1] / 'gkd_mopd_baseline/train.py'
sys.path.insert(0, str(HARNESS))
sys.path.insert(0, str(SCRIPT.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import bench_recorder as hf_bench  # noqa: E402
from trl.experimental.gkd import GKDTrainer  # noqa: E402

_generate = GKDTrainer.generate_on_policy_outputs


def generate(model, inputs, generation_config, *args, **kwargs):
    out = _generate(model, inputs, generation_config, *args, **kwargs)
    ids, attention, _ = out
    prompt_len = inputs['prompts'].shape[1]
    hf_bench.add_tokens(attention.sum(), attention[:, prompt_len:].sum(), ids.shape[0])
    return out


GKDTrainer.generate_on_policy_outputs = staticmethod(generate)
BENCH = hf_bench.Bench()
_init = GKDTrainer.__init__


def init(self, *args, **kwargs):
    kwargs['callbacks'] = list(kwargs.get('callbacks') or []) + [BENCH]
    _init(self, *args, **kwargs)


GKDTrainer.__init__ = init
sys.argv = [str(SCRIPT)] + sys.argv[1:]
runpy.run_path(str(SCRIPT), run_name='__main__')
BENCH.finish()
