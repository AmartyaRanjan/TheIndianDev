"""Record types shared by the corpus, the held-out set and evaluation, plus JSONL helpers."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class CleanFunction:
    """A training-corpus snippet with unit tests that pass on it."""

    id: str
    source: str  # e.g. "kodcode/Leetcode"
    code: str
    tests: str


@dataclass(frozen=True)
class Example:
    """One detector input: buggy (bug_lines non-empty) or clean (bug_class == NO_BUG)."""

    id: str
    source: str  # e.g. "pypibugs", "tssb", "synthetic/uniform"
    code: str
    bug_lines: tuple[int, ...]  # 1-based lines of `code`; empty when clean
    bug_class: str  # a BugClass value, "other" (real bug outside the taxonomy) or NO_BUG
    location: str | None = None
    pair_id: str | None = None  # links the buggy and fixed versions of one real bug
    split: str = ""


def write_jsonl(path: str | Path, records: Iterable[Any]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(dataclasses.asdict(r) if dataclasses.is_dataclass(r) else r) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path, cls: type[T] | None = None) -> Iterator[T]:
    with Path(path).open() as f:
        for line in f:
            row = json.loads(line)
            if cls is Example:
                row["bug_lines"] = tuple(row["bug_lines"])
            yield cls(**row) if cls else row


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable_bucket(key: str, buckets: int) -> int:
    """Deterministic bucket for split assignment (independent of PYTHONHASHSEED)."""
    return int(hashlib.sha1(key.encode()).hexdigest(), 16) % buckets
