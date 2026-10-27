"""AST-engine mutants that pass the gate: the A1-uniform data source and the synthetic-val set."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence

from advbug.data.schema import CleanFunction, Example
from advbug.game.gate import Gate, GateResult
from advbug.mutation.operators import mutate
from advbug.taxonomy import NO_BUG, PRIORITY, BugClass


def uniform_mutants(
    functions: Sequence[CleanFunction], gate: Gate, rng: random.Random, balance: bool = False
) -> list[tuple[CleanFunction, GateResult]]:
    """One gated mutant per function where possible.

    balance=False: the class is drawn uniformly at random, falling back to other classes when it
    has no valid site (A1-uniform). balance=True: each function gets the least-used class that
    applies to it, so class counts come out close to equal (synthetic-val).
    """
    counts: Counter[BugClass] = Counter()
    candidates: list[tuple[CleanFunction, str, str]] = []
    for fn in functions:
        order = list(PRIORITY)
        rng.shuffle(order)
        if balance:
            order.sort(key=lambda c: counts[c])
        for cls in order:
            m = mutate(fn.code, cls, rng)
            if m:
                candidates.append((fn, m.mutant, cls.value))
                counts[cls] += 1
                break
    results = gate.check_mutants(candidates)
    return [(fn, r) for (fn, _, _), r in zip(candidates, results) if r.ok]


def to_examples(pairs: Sequence[tuple[CleanFunction, GateResult]], source: str, with_clean: bool = True) -> list[Example]:
    out = []
    for fn, r in pairs:
        out.append(Example(f"{fn.id}-bug", source, r.mutant, r.bug_lines, r.claimed_class, r.location, fn.id))
        if with_clean:
            out.append(Example(f"{fn.id}-clean", source, fn.code, (), NO_BUG, None, fn.id))
    return out
