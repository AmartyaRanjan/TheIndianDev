"""Significance tests for comparing arms on the same held-out examples (plan §4.8)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import binomtest


def mcnemar(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> float:
    """Exact McNemar p-value: do two systems differ on the same examples?"""
    a, b = np.asarray(a_correct, bool), np.asarray(b_correct, bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    if only_a + only_b == 0:
        return 1.0
    return float(binomtest(only_a, only_a + only_b, 0.5).pvalue)


def paired_bootstrap(
    a_correct: Sequence[bool], b_correct: Sequence[bool], resamples: int = 10_000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean accuracy difference (a - b) with a 95% percentile interval over resampled examples."""
    diff = np.asarray(a_correct, float) - np.asarray(b_correct, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(resamples, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def hierarchical_bootstrap(
    per_seed_correct: Sequence[Sequence[bool]], resamples: int = 10_000, seed: int = 0
) -> tuple[float, float, float]:
    """Accuracy with a 95% interval that resamples seeds, then examples within each seed."""
    runs = np.asarray(per_seed_correct, float)  # [seeds, examples]
    rng = np.random.default_rng(seed)
    n_seeds, n_ex = runs.shape
    stats = np.empty(resamples)
    for r in range(resamples):
        seeds = rng.integers(0, n_seeds, n_seeds)
        examples = rng.integers(0, n_ex, n_ex)
        stats[r] = runs[np.ix_(seeds, examples)].mean()
    return float(runs.mean()), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order."""
    p = np.asarray(p_values, float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adjusted[i] = running
    return adjusted.tolist()
