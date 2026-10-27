"""Build the clean training corpus from downloaded KodCode parquet shards.

    uv run python scripts/build_corpus.py --target 25000
Writes data/corpus/{train,val}.jsonl and data/corpus/manifest.json.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from advbug.data.corpus import build_corpus, split
from advbug.data.schema import file_sha256, write_jsonl
from advbug.taxonomy import TAXONOMY_FINGERPRINT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shards", default="data/raw/kodcode/default/train/*.parquet")
    parser.add_argument("--out", default="data/corpus")
    parser.add_argument("--target", type=int, default=25000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    start = time.time()
    items, stats = build_corpus(args.shards, target=args.target, seed=args.seed, workers=args.workers)
    train, val = split(items)
    out = Path(args.out)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "val.jsonl", val)
    manifest = {
        "source": "KodCode/KodCode-V1 (refs/convert/parquet)",
        "seed": args.seed,
        "target": args.target,
        "taxonomy_fingerprint": TAXONOMY_FINGERPRINT,
        "counts": {"train": len(train), "val": len(val)},
        "subsets": dict(Counter(c.source for c in items).most_common()),
        "stats": stats.as_dict(),
        "sha256": {name: file_sha256(out / f"{name}.jsonl") for name in ("train", "val")},
        "build_seconds": round(time.time() - start),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
