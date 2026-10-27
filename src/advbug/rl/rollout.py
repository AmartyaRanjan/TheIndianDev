"""Sampling backends. VLLMGenerator serves every adapter version from one frozen base (multi-LoRA);
HFGenerator samples from the training model itself and exists for CPU tests and tiny smoke runs.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Protocol

import torch

from advbug.rl.policy import Policy


@dataclass(frozen=True)
class Completion:
    text: str
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class AdapterRef:
    """A saved adapter version. path=None means the plain base model."""

    name: str
    version: int
    path: str | None


AdapterArg = AdapterRef | Sequence[AdapterRef | None] | None


class Generator(Protocol):
    def generate(
        self, prompts: Sequence[Sequence[int]], adapters: AdapterArg, n: int, temperature: float, max_tokens: int
    ) -> list[list[Completion]]: ...


def _per_prompt(adapters: AdapterArg, count: int) -> list[AdapterRef | None]:
    if adapters is None or isinstance(adapters, AdapterRef):
        return [adapters] * count
    assert len(adapters) == count
    return list(adapters)


class VLLMGenerator:
    def __init__(
        self,
        base_model: str,
        max_lora_rank: int,
        gpu_memory_utilization: float = 0.35,
        max_model_len: int = 4096,
        max_loras: int = 4,
        seed: int = 0,
        enforce_eager: bool = False,  # skip CUDA-graph capture: less memory, slower decoding
    ) -> None:
        from vllm import LLM

        self.llm = LLM(
            model=base_model,
            enable_lora=True,
            max_lora_rank=max_lora_rank,
            max_loras=max_loras,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enable_prefix_caching=True,
            dtype="bfloat16",
            seed=seed,
            enforce_eager=enforce_eager,
        )
        self._lora_ids: dict[tuple[str, int], int] = {}

    def _request(self, ref: AdapterRef | None):  # noqa: ANN202 - vllm type imported lazily
        from vllm.lora.request import LoRARequest

        if ref is None or ref.path is None:
            return None
        key = (ref.name, ref.version)
        if key not in self._lora_ids:  # vLLM caches adapters by int id, so every version needs a new one
            self._lora_ids[key] = len(self._lora_ids) + 1
        return LoRARequest(f"{ref.name}-v{ref.version}", self._lora_ids[key], ref.path)

    def generate(
        self, prompts: Sequence[Sequence[int]], adapters: AdapterArg, n: int, temperature: float, max_tokens: int
    ) -> list[list[Completion]]:
        from vllm import SamplingParams, TokensPrompt

        params = SamplingParams(n=n, temperature=temperature, max_tokens=max_tokens)
        requests = [TokensPrompt(prompt_token_ids=list(p)) for p in prompts]
        loras = [self._request(a) for a in _per_prompt(adapters, len(prompts))]
        outputs = self.llm.generate(requests, params, lora_request=loras, use_tqdm=False)
        return [[Completion(o.text, tuple(o.token_ids)) for o in out.outputs] for out in outputs]


class HFGenerator:
    """Samples from the in-memory adapters of `policy` (ignores AdapterRef versions/paths)."""

    def __init__(self, policy: Policy) -> None:
        self.policy = policy

    @torch.no_grad()
    def generate(
        self, prompts: Sequence[Sequence[int]], adapters: AdapterArg, n: int, temperature: float, max_tokens: int
    ) -> list[list[Completion]]:
        tok, model = self.policy.tokenizer, self.policy.model
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        eos = set(_eos_ids(model, tok))
        results: list[list[Completion]] = []
        for prompt, ref in zip(prompts, _per_prompt(adapters, len(prompts))):
            if ref is None or ref.path is None:
                ctx = model.disable_adapter()
            else:
                self.policy.use(ref.name)
                ctx = nullcontext()
            ids = torch.tensor([list(prompt)], device=self.policy.device)
            with ctx:
                out = model.generate(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    do_sample=True,
                    temperature=temperature,
                    max_new_tokens=max_tokens,
                    num_return_sequences=n,
                    pad_token_id=pad,
                )
            group = []
            for row in out[:, ids.shape[1] :].tolist():
                cut = next((i + 1 for i, t in enumerate(row) if t in eos), len(row))
                group.append(Completion(tok.decode(row[:cut], skip_special_tokens=True), tuple(row[:cut])))
            results.append(group)
        return results


def _eos_ids(model: torch.nn.Module, tokenizer) -> list[int]:  # noqa: ANN001
    eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None) or tokenizer.eos_token_id
    return list(eos) if isinstance(eos, (list, tuple)) else [eos]
