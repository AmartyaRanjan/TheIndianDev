"""Audit (CPU only): how G6 reaches its verdicts, and how repeatable they are.

Re-runs the unit tests of every AST-engine candidate that reached G6 (passed G2-G5) in the
synthetic-val replay, twice, and records whether behaviour change was shown by failing
assertions, by an exception, or by a timeout, plus run-to-run agreement.

    uv run python scripts/audit_g6.py
Writes logs/audit_g6.json.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from advbug.data.schema import CleanFunction, read_jsonl
from advbug.game.gate import Gate, GateConfig
from advbug.sandbox import RunResult, SandboxPool
from scripts.audit_gate import replay


def outcome(r: RunResult) -> str:
    if r.timed_out:
        return "timeout"
    if r.error:
        return "error_on_load"
    if r.failed:
        return "failing_tests"
    return "all_passed"  # the G6 rejection case


def main() -> None:
    val = list(read_jsonl("data/corpus/val.jsonl", CleanFunction))
    with SandboxPool(workers=6) as sandbox:
        rows = replay(val, Gate(GateConfig(), sandbox), random.Random(0))
        reached = [(fn, m) for fn, m, _, r in rows if r.ok or r.failed_check == "G6_equivalent"]
        first = sandbox.run([(m, fn.tests) for fn, m in reached])
        second = sandbox.run([(m, fn.tests) for fn, m in reached])
    a, b = [outcome(r) for r in first], [outcome(r) for r in second]
    verdict = lambda o: "reject" if o == "all_passed" else "accept"  # noqa: E731
    flips = [i for i in range(len(a)) if verdict(a[i]) != verdict(b[i])]
    report = {
        "candidates_reaching_G6": len(reached),
        "run1_outcomes": dict(Counter(a).most_common()),
        "run2_outcomes": dict(Counter(b).most_common()),
        "verdict_agreement": 1 - len(flips) / len(a),
        "flipped": [{"function": reached[i][0].id, "run1": a[i], "run2": b[i]} for i in flips],
        "failing_tests_per_accepted_mutant": dict(
            sorted(Counter(r.failed for r, o in zip(first, a) if o == "failing_tests").items())
        ),
    }
    Path("logs/audit_g6.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
