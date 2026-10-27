"""Build the held-out real-bug set (real-dev / real-test). Never used by the training loop.

    uv run python scripts/build_heldout.py --pypibugs data/raw/pypibugs/PyPiBugs.csv \
        --tssb data/raw/tssb/tssb_data_3M --tssb-per-class 400
Writes data/heldout/{dev,test}.jsonl and data/heldout/manifest.json.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from datasketch import MinHashLSH

from advbug.data.corpus import _minhash
from advbug.data.heldout import FixRecord, assign_splits, fetch_all, read_pypibugs, read_tssb, to_examples
from advbug.data.schema import CleanFunction, Example, file_sha256, read_jsonl, write_jsonl
from advbug.taxonomy import NO_BUG, TAXONOMY_FINGERPRINT


def corpus_index(corpus_dir: Path) -> MinHashLSH | None:
    files = [corpus_dir / "train.jsonl", corpus_dir / "val.jsonl"]
    if not all(f.exists() for f in files):
        return None
    lsh = MinHashLSH(threshold=0.8, num_perm=64)
    for f in files:
        for item in read_jsonl(f, CleanFunction):
            lsh.insert(item.id, _minhash(item.code))
    return lsh


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pypibugs", default="data/raw/pypibugs/PyPiBugs.csv")
    parser.add_argument("--tssb", default="data/raw/tssb/tssb_data_3M")
    parser.add_argument("--tssb-per-class", type=int, default=400)
    parser.add_argument("--cache", default="data/raw/files")
    parser.add_argument("--corpus", default="data/corpus")
    parser.add_argument("--out", default="data/heldout")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    records: list[FixRecord] = list(read_pypibugs(args.pypibugs))
    if args.tssb and args.tssb_per_class > 0:
        print("scanning TSSB-3M for in-taxonomy fixes ...", flush=True)
        records += read_tssb(args.tssb, per_class=args.tssb_per_class)
    print(f"{len(records)} fix records; fetching files ...", flush=True)
    files = fetch_all(records, Path(args.cache), threads=args.threads)

    dropped: Counter[str] = Counter()
    examples: list[Example] = []
    seen_buggy: set[str] = set()  # the same bug can appear in both sources: keep the first
    for record, text in zip(records, files):
        result = to_examples(record, text)
        if isinstance(result, str):
            dropped[f"{record.source}:{result}"] += 1
        elif result[0].code in seen_buggy:
            dropped[f"{record.source}:duplicate"] += 1
        else:
            seen_buggy.add(result[0].code)
            examples.extend(result)

    lsh = corpus_index(Path(args.corpus))
    if lsh is None:
        print("warning: corpus not built yet; skipping held-out vs corpus near-duplicate check")
    else:
        leaked = {ex.pair_id for ex in examples if lsh.query(_minhash(ex.code))}
        dropped["near_duplicate_of_corpus"] += len(leaked)
        examples = [ex for ex in examples if ex.pair_id not in leaked]

    examples = assign_splits(examples)
    out = Path(args.out)
    for split in ("dev", "test"):
        write_jsonl(out / f"{split}.jsonl", [ex for ex in examples if ex.split == split])
    bugs = [ex for ex in examples if ex.bug_class != NO_BUG]
    manifest = {
        "taxonomy_fingerprint": TAXONOMY_FINGERPRINT,
        "pairs": len(bugs),
        "pairs_by_split": dict(Counter(ex.split for ex in bugs)),
        "classes": dict(Counter(ex.bug_class for ex in bugs).most_common()),
        "locations": dict(Counter(ex.location for ex in bugs if ex.location).most_common()),
        "sources": dict(Counter(ex.source for ex in bugs)),
        "classes_by_source": {
            src: dict(Counter(ex.bug_class for ex in bugs if ex.source == src).most_common())
            for src in sorted({ex.source for ex in bugs})
        },
        "dropped": dict(dropped.most_common()),
        "corpus_dedup_checked": lsh is not None,
        "sha256": {s: file_sha256(out / f"{s}.jsonl") for s in ("dev", "test")},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
