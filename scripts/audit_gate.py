"""Audit (CPU only): replay the seeded AST-engine pipelines behind synthetic-val and the injector
warmup, keeping every gate verdict, and dump representative accepted/rejected examples with
their unit-test outcomes.

    uv run python scripts/audit_gate.py
Writes logs/audit_gate_funnel.json and logs/audit_gate_examples.txt. The replay is valid only if
the accepted counts match the manifests (836 and 2,628); differences are reported, and each
flipped G6 verdict is re-tested 5 times to show whether a flaky unit test caused it.
"""

from __future__ import annotations

import difflib
import json
import random
from collections import Counter
from pathlib import Path

from advbug.data.schema import CleanFunction, read_jsonl
from advbug.game.gate import CHECKS, Gate, GateConfig, GateResult
from advbug.mutation.classify import classify
from advbug.mutation.operators import mutate
from advbug.sandbox import SandboxPool
from advbug.taxonomy import PRIORITY, BugClass


def replay(functions: list[CleanFunction], gate: Gate, rng: random.Random) -> list[tuple[CleanFunction, str, str, GateResult]]:
    """Same logic and RNG consumption as data.synthetic.uniform_mutants(balance=True), all verdicts kept."""
    counts: Counter[BugClass] = Counter()
    candidates: list[tuple[CleanFunction, str, str]] = []
    for fn in functions:
        order = list(PRIORITY)
        rng.shuffle(order)
        order.sort(key=lambda c: counts[c])
        for cls in order:
            m = mutate(fn.code, cls, rng)
            if m:
                candidates.append((fn, m.mutant, cls.value))
                counts[cls] += 1
                break
    results = gate.check_mutants(candidates)
    return [(fn, mutant, cls, r) for (fn, mutant, cls), r in zip(candidates, results)]


def funnel(rows: list[tuple[CleanFunction, str, str, GateResult]], n_functions: int) -> dict[str, object]:
    failed = Counter(r.failed_check for *_, r in rows if not r.ok)
    rejected_by_class = Counter(cls for _, _, cls, r in rows if not r.ok)
    return {
        "functions": n_functions,
        "functions_with_a_candidate": len(rows),
        "accepted": sum(r.ok for *_, r in rows),
        "rejected": {c: failed[c] for c in CHECKS},
        "rejected_by_claimed_class": dict(rejected_by_class.most_common()),
    }


def diff_lines(a: str, b: str) -> tuple[list[str], list[str]]:
    old, new = [], []
    for line in difflib.unified_diff(a.splitlines(), b.splitlines(), n=0, lineterm=""):
        if line.startswith("-") and not line.startswith("---"):
            old.append(line[1:].strip())
        elif line.startswith("+") and not line.startswith("+++"):
            new.append(line[1:].strip())
    return old, new


def main() -> None:
    val = list(read_jsonl("data/corpus/val.jsonl", CleanFunction))
    train = list(read_jsonl("data/corpus/train.jsonl", CleanFunction))
    out: dict[str, object] = {}
    with SandboxPool(workers=6) as sandbox:
        gate = Gate(GateConfig(), sandbox)
        synthetic = replay(val, gate, random.Random(0))  # scripts/build_synthetic.py, seed 0
        out["synthetic_val"] = funnel(synthetic, len(val))
        rng = random.Random(0)  # scripts/warmup_injector.py, seed 0
        warm_fns = rng.sample(train, 3000)
        warmup = replay(warm_fns, gate, rng)
        out["injector_warmup"] = funnel(warmup, len(warm_fns))
        # replication check against the manifests; G6 runs real tests, so a flaky test can flip a verdict
        built = {(e["pair_id"], e["code"]) for e in read_jsonl("data/synthetic/val.jsonl") if e["bug_class"] != "no_bug"}
        replayed = {(fn.id, m) for fn, m, _, r in synthetic if r.ok}
        flips = sorted(built ^ replayed)
        out["replication"] = {
            "synthetic_val": {"manifest": 836, "replayed": len(replayed), "verdicts_differing": len(flips)},
            "injector_warmup": {"manifest": 2628, "replayed": out["injector_warmup"]["accepted"]},
        }
        by_key = {(fn.id, m): fn for fn, m, *_ in synthetic}
        flip_runs = []
        for fid, mutant in flips:  # re-run each flipped mutant's tests 5 times
            fn = by_key[(fid, mutant)]
            results = sandbox.run([(mutant, fn.tests)] * 5)
            flip_runs.append({"function": fid, "in_built_set": (fid, mutant) in built,
                              "runs": [[r.passed, r.failed, r.timed_out] for r in results]})
        out["replication"]["flipped_verdicts"] = flip_runs

        # representative examples: one accepted per class, plus every rejection type that occurred
        picks: list[tuple[CleanFunction, str, str, GateResult]] = []
        seen: set[str] = set()
        for row in synthetic + warmup:
            key = row[2] if row[3].ok else f"REJECT:{row[3].failed_check}"
            if key not in seen:
                seen.add(key)
                picks.append(row)
        runs = sandbox.run([(fn.code, fn.tests) for fn, *_ in picks] + [(m, fn.tests) for fn, m, *_ in picks])
    originals, mutants = runs[: len(picks)], runs[len(picks) :]

    Path("logs").mkdir(exist_ok=True)
    Path("logs/audit_gate_funnel.json").write_text(json.dumps(out, indent=2) + "\n")
    lines = []
    for (fn, mutant, claimed, r), o, m in zip(picks, originals, mutants):
        old, new = diff_lines(fn.code, mutant)
        lines += [
            f"== {fn.id} | claimed={claimed} | AST classifier={classify(fn.code, mutant)} | "
            f"gate={'ACCEPTED' if r.ok else 'REJECTED ' + str(r.failed_check)} | location={r.location}",
            f"   original: {' / '.join(old)[:160]}",
            f"   mutant:   {' / '.join(new)[:160] or '(lines deleted)'}",
            f"   tests on original: {o.passed} passed, {o.failed} failed | on mutant: {m.passed} passed, {m.failed} failed"
            f"{', timed out' if m.timed_out else ''}{', error: ' + m.error[:60] if m.error else ''}",
        ]
    Path("logs/audit_gate_examples.txt").write_text("\n".join(lines) + "\n")
    print(json.dumps(out, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
