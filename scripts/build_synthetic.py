"""Build synthetic-val: class-balanced AST-engine mutants of the val corpus (+ the clean originals).

    uv run python scripts/build_synthetic.py
Writes data/synthetic/val.jsonl and data/synthetic/manifest.json.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from advbug.data.schema import CleanFunction, file_sha256, read_jsonl, write_jsonl
from advbug.data.synthetic import to_examples, uniform_mutants
from advbug.game.gate import Gate, GateConfig
from advbug.sandbox import SandboxPool
from advbug.taxonomy import NO_BUG, TAXONOMY_FINGERPRINT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", default="data/corpus/val.jsonl")
    parser.add_argument("--out", default="data/synthetic")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    functions = list(read_jsonl(args.corpus, CleanFunction))
    with SandboxPool(workers=6) as sandbox:
        pairs = uniform_mutants(functions, Gate(GateConfig(), sandbox), random.Random(args.seed), balance=True)
    examples = to_examples(pairs, source="synthetic/uniform")
    out = Path(args.out)
    write_jsonl(out / "val.jsonl", examples)
    bugs = [e for e in examples if e.bug_class != NO_BUG]
    manifest = {
        "taxonomy_fingerprint": TAXONOMY_FINGERPRINT,
        "functions": len(functions),
        "mutants": len(bugs),
        "classes": dict(Counter(e.bug_class for e in bugs).most_common()),
        "locations": dict(Counter(e.location for e in bugs).most_common()),
        "sha256": file_sha256(out / "val.jsonl"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
