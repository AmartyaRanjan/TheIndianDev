"""Supervised fine-tuning of one adapter on (prompt, completion) pairs: the injector format warmup."""

from __future__ import annotations

import random
from collections.abc import Sequence

import torch

from advbug.rl.policy import Policy


def sft(
    policy: Policy,
    adapter: str,
    pairs: Sequence[tuple[Sequence[int], Sequence[int]]],
    lr: float = 1e-4,
    epochs: int = 1,
    batch_size: int = 16,
    micro_batch: int = 4,
    seed: int = 0,
    log_every: int = 10,
) -> list[float]:
    """Token-level cross-entropy on completions only; returns the mean loss of each optimizer step."""
    params = policy.adapter_parameters(adapter)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    rng = random.Random(seed)
    order = list(range(len(pairs)))
    losses: list[float] = []
    policy.use(adapter)
    policy.model.train()
    for _ in range(epochs):
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            batch = [pairs[i] for i in order[start : start + batch_size]]
            total_tokens = sum(len(c) for _, c in batch)
            step_loss = 0.0
            for m in range(0, len(batch), micro_batch):
                prompts, completions = zip(*batch[m : m + micro_batch])
                logp, mask = policy.completion_logprobs(prompts, completions, adapter)
                loss = -(logp * mask).sum() / total_tokens
                loss.backward()
                step_loss += float(loss)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(step_loss)
            if log_every and len(losses) % log_every == 0:
                print(f"  sft step {len(losses)}: loss {sum(losses[-log_every:]) / log_every:.4f}", flush=True)
    policy.model.eval()
    return losses
