import atexit
import logging
import os
import signal
import socket
import subprocess
import sys
from functools import partial

import torch

from .minillm_config import MiniLLMConfig


logger = logging.getLogger(__name__)


class EagerVLLMInitForDeterminism:
    """Force colocated vLLM initialization to be eager while PyTorch determinism is enabled."""

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
