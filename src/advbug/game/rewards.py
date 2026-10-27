"""Per-sample rewards for both players (plan §4.4).

Detector: binary (exact line + type, or no_bug on clean code) or, in the A6 arm, partial credit.
Injector: -1 if the gate rejects the sample, else fool_rate + λ_div·diversity + λ_sub·subtlety,
where fool_rate is the fraction of the detector's samples on this mutant that were wrong.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from advbug.game.gate import GateResult
from advbug.game.prompts import DetectorPrediction
from advbug.taxonomy import NO_BUG, PRIORITY


@dataclass(frozen=True)
class RewardConfig:
    partial_credit: bool = False  # A6
    invalid_penalty: float = -1.0
    lambda_div: float = 0.1
    lambda_sub: float = 0.05
    max_edit_chars: int = 120  # edit size that scores zero subtlety


def detector_reward(
    bug_class: str, bug_lines: Sequence[int], pred: DetectorPrediction | None, partial_credit: bool = False
) -> float:
    if pred is None:
        return 0.0
    if bug_class == NO_BUG:
        return float(pred.bug_class == NO_BUG)
    distance = min((abs(pred.line - b) for b in bug_lines), default=math.inf) if pred.line else math.inf
    if not partial_credit:
        return float(distance == 0 and pred.bug_class == bug_class)
    line_score = 1.0 if distance == 0 else 0.5 if distance <= 1 else 0.25 if distance <= 3 else 0.0
    return 0.5 * line_score + 0.5 * float(pred.bug_class == bug_class)


def fool_rate(detector_correct: Sequence[bool]) -> float:
    return 1.0 - sum(detector_correct) / len(detector_correct) if detector_correct else 1.0


def subtlety(edit_chars: int, max_edit_chars: int) -> float:
    return max(0.0, 1.0 - edit_chars / max_edit_chars)


class DiversityTracker:
    """Moving-average frequency of the injector's valid bug classes; rarer classes earn a bonus."""

    def __init__(self, classes: Sequence[str] = tuple(c.value for c in PRIORITY), momentum: float = 0.9) -> None:
        self.momentum = momentum
        self.freq = {c: 1.0 / len(classes) for c in classes}

    def bonus(self, bug_class: str) -> float:
        """-log p̂(class) scaled so a class at the uniform rate scores 1; clipped to [0, 1]."""
        p = max(self.freq.get(bug_class, 0.0), 1e-6)
        return min(1.0, max(0.0, -math.log(p) / math.log(len(self.freq))))

    def update(self, valid_classes: Sequence[str]) -> None:
        if not valid_classes:
            return
        for c in self.freq:
            batch = sum(1 for v in valid_classes if v == c) / len(valid_classes)
            self.freq[c] = self.momentum * self.freq[c] + (1 - self.momentum) * batch

    def entropy(self) -> float:
        return -sum(p * math.log(p) for p in self.freq.values() if p > 0)


def injector_reward(
    gate: GateResult, fool: float, diversity_bonus: float, config: RewardConfig, fool_cap: float | None = None
) -> float:
    if not gate.ok:
        return config.invalid_penalty
    fooled = min(fool, fool_cap) if fool_cap is not None else fool
    return fooled + config.lambda_div * diversity_bonus + config.lambda_sub * subtlety(gate.edit_chars, config.max_edit_chars)
