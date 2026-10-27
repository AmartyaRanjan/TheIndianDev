"""GRPO advantages and the clipped policy-gradient loss with a KL penalty (plan §4.6).

Loss is token-level: every completion token in the whole step's batch has equal weight, so
micro-batches are normalised by the step's total token count, not their own.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def group_advantages(rewards: torch.Tensor, eps: float = 1e-4, scale: bool = True) -> torch.Tensor:
    """rewards [groups, k] -> advantages [groups, k]: (r - group mean) / (group std + eps).

    Groups whose rewards are all equal get zero advantage (no gradient); the fraction of such
    groups is logged as `zero_variance_groups` and is central to the A5 group-size sweep.
    """
    centered = rewards - rewards.mean(dim=1, keepdim=True)
    if not scale:
        return centered
    return centered / (rewards.std(dim=1, keepdim=True, unbiased=False) + eps)


def zero_variance_fraction(rewards: torch.Tensor, tol: float = 1e-8) -> float:
    return float((rewards.std(dim=1, unbiased=False) < tol).float().mean())


@dataclass(frozen=True)
class LossConfig:
    kl_beta: float = 0.01
    clip_eps: float = 0.2


def policy_loss(
    logp: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    total_tokens: int,
    config: LossConfig,
    old_logp: torch.Tensor | None = None,
    ref_logp: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Clipped surrogate + k3 KL to the reference, summed over tokens / total_tokens.

    logp, old_logp, ref_logp, mask: [batch, tokens]; advantages: [batch] (or [batch, tokens]).
    With old_logp=None the update is fully on-policy (ratio == 1, so the clip is inactive).
    """
    adv = advantages if advantages.dim() == 2 else advantages[:, None].expand_as(logp)
    old = logp.detach() if old_logp is None else old_logp
    ratio = torch.exp(logp - old)
    clipped = torch.clamp(ratio, 1 - config.clip_eps, 1 + config.clip_eps)
    per_token = -torch.minimum(ratio * adv, clipped * adv)
    kl = torch.zeros_like(logp)
    if ref_logp is not None and config.kl_beta > 0:
        delta = ref_logp - logp
        kl = torch.exp(delta) - delta - 1  # k3 estimator: unbiased, always >= 0
        per_token = per_token + config.kl_beta * kl
    mask = mask.to(logp.dtype)
    loss = (per_token * mask).sum() / max(total_tokens, 1)
    with torch.no_grad():
        n = mask.sum().clamp(min=1)
        stats = {
            "kl": float((kl * mask).sum() / n),
            "clip_fraction": float((((ratio - 1).abs() > config.clip_eps).to(logp.dtype) * mask).sum() / n),
        }
    return loss, stats
