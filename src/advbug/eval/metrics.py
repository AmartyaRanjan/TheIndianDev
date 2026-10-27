"""Detector metrics over Examples: localization, classification, false positives, calibration.

A prediction localizes a bug if it names any line the bug touches (`Example.bug_lines`).
Classification metrics only use bugs inside the taxonomy (real bugs labelled "other" are
scored for localization only).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np

from advbug.data.schema import Example
from advbug.game.prompts import DetectorPrediction
from advbug.taxonomy import NO_BUG, PRIORITY

OTHER = "other"
LABELS = [c.value for c in PRIORITY] + [NO_BUG]


def line_distance(pred_line: int | None, bug_lines: Sequence[int]) -> float:
    if pred_line is None or not bug_lines:
        return float("inf")
    return min(abs(pred_line - line) for line in bug_lines)


def is_correct(example: Example, pred: DetectorPrediction | None) -> bool:
    """The detector's binary reward criterion: exact line and type, or `no_bug` on clean code."""
    if pred is None:
        return False
    if example.bug_class == NO_BUG:
        return pred.bug_class == NO_BUG
    return pred.bug_class == example.bug_class and line_distance(pred.line, example.bug_lines) == 0


def score(examples: Sequence[Example], preds: Sequence[DetectorPrediction | None]) -> dict[str, float]:
    assert len(examples) == len(preds)
    bugs = [(e, p) for e, p in zip(examples, preds) if e.bug_class != NO_BUG]
    clean = [(e, p) for e, p in zip(examples, preds) if e.bug_class == NO_BUG]
    typed = [(e, p) for e, p in bugs if e.bug_class != OTHER]
    out: dict[str, float] = {"n": len(examples), "n_bugs": len(bugs), "n_clean": len(clean)}
    out["format_failure"] = _mean([p is None for p in preds])
    if bugs:
        dists = [line_distance(p.line if p else None, e.bug_lines) for e, p in bugs]
        out["line_exact"] = _mean([d == 0 for d in dists])
        out["line_within_1"] = _mean([d <= 1 for d in dists])
        out["line_within_3"] = _mean([d <= 3 for d in dists])
        out["flagged_bug"] = _mean([p is not None and p.bug_class != NO_BUG for _, p in bugs])
    if typed:
        out["type_acc"] = _mean([p is not None and p.bug_class == e.bug_class for e, p in typed])
        out["joint_acc"] = _mean([is_correct(e, p) for e, p in typed])
    if clean:
        out["false_positive_rate"] = _mean([p is None or p.bug_class != NO_BUG for _, p in clean])
    labelled = [(e.bug_class, p.bug_class if p else None) for e, p in zip(examples, preds) if e.bug_class != OTHER]
    out.update({f"f1/{k}": v for k, v in per_class_f1(labelled).items()})
    present = [k for k in LABELS if any(t == k for t, _ in labelled)]
    if present:
        out["macro_f1"] = float(np.mean([out[f"f1/{k}"] for k in present]))
    out["accuracy"] = _mean([is_correct(e, p) for e, p in zip(examples, preds) if e.bug_class != OTHER])
    return out


def per_class_f1(pairs: Sequence[tuple[str, str | None]]) -> dict[str, float]:
    """F1 per label over (true, predicted) class pairs; a missing prediction counts as wrong."""
    tp, fp, fn = Counter(), Counter(), Counter()
    for true, pred in pairs:
        if pred == true:
            tp[true] += 1
        else:
            fn[true] += 1
            if pred is not None:
                fp[pred] += 1
    scores = {}
    for label in LABELS:
        if tp[label] + fn[label] == 0:
            continue
        precision = tp[label] / (tp[label] + fp[label]) if tp[label] + fp[label] else 0.0
        recall = tp[label] / (tp[label] + fn[label])
        scores[label] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return scores


def expected_calibration_error(confidence: Sequence[float], correct: Sequence[bool], bins: int = 10) -> float:
    conf, corr = np.asarray(confidence, dtype=float), np.asarray(correct, dtype=float)
    if conf.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(conf[mask].mean() - corr[mask].mean())
    return float(ece)


def _mean(xs: Sequence[bool | float]) -> float:
    return float(np.mean(xs)) if len(xs) else float("nan")
