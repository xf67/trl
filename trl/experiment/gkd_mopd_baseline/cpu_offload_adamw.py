"""BF16StochasticAdamW with its state and update arithmetic on the CPU.

The GPU keeps only the BF16 parameters and gradients. Each step copies a gradient to the host, applies the same
update as `BF16StochasticAdamW` to a pinned BF16 host copy of the parameter (FP32 moments, FP32 update arithmetic,
stochastic BF16 writeback), and copies the result back. The FP32 moments never occupy GPU memory.
"""

import math

import torch

from bf16_stochastic_adamw import BF16StochasticAdamW, _stochastic_copy_bfloat16_


class CPUOffloadBF16StochasticAdamW(BF16StochasticAdamW):
    def load_state_dict(self, state_dict: dict) -> None:
        # Accelerate round-trips the state dict while preparing the optimizer. Optimizer.load_state_dict casts
        # state to each parameter's device and dtype; keep the moments FP32 and everything on the CPU instead.
        torch.optim.Optimizer.load_state_dict(self, state_dict)
        for state in self.state.values():
            for name in ("exp_avg", "exp_avg_sq"):
                if name in state:
                    state[name] = state[name].to(device="cpu", dtype=torch.float32)
            if "param_cpu" in state:
                param_cpu = torch.empty(state["param_cpu"].shape, dtype=torch.bfloat16, pin_memory=True)
                state["param_cpu"] = param_cpu.copy_(state["param_cpu"])
            if isinstance(state.get("step"), torch.Tensor):
                state["step"] = int(state["step"].item())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("CPUOffloadBF16StochasticAdamW does not support sparse gradients")
                if param.dtype != torch.bfloat16:
                    raise RuntimeError(f"CPUOffloadBF16StochasticAdamW requires BF16 parameters, got {param.dtype}")

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["param_cpu"] = torch.empty(param.shape, dtype=torch.bfloat16, pin_memory=True)
                    state["param_cpu"].copy_(param)
                    state["exp_avg"] = torch.zeros(param.shape, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros(param.shape, dtype=torch.float32)

                state["step"] += 1
                step = state["step"]
                param_cpu = state["param_cpu"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                grad_fp32 = grad.to("cpu").float()
                if group["maximize"]:
                    grad_fp32.neg_()

                exp_avg.lerp_(grad_fp32, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=1 - beta2)

                updated_param = param_cpu.float()
                if group["weight_decay"] != 0:
                    updated_param.mul_(1 - group["lr"] * group["weight_decay"])

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                denominator = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(group["eps"])
                updated_param.addcdiv_(exp_avg, denominator, value=-group["lr"] / bias_correction1)
                _stochastic_copy_bfloat16_(param_cpu, updated_param)
                param.copy_(param_cpu, non_blocking=True)
            torch.cuda.synchronize()

        return loss
