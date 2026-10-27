"""Held-out real bugs (never used in training): mined bug-fix commits -> function-level Examples.

Each source row gives a repository, a fix commit, a file path and the fix as a unified diff.
We fetch the fixed file at the fix commit, reverse-apply the diff to recover the buggy file, and
cut out the innermost enclosing function in both versions. The buggy function becomes a bug
Example (bug lines = lines the fix touched); the fixed function becomes its paired clean Example.
The bug class is `classify(fixed, buggy)`, the same classifier the training gate uses, or
"other" when the real bug falls outside the taxonomy (scored for localization only).
"""

from __future__ import annotations

import ast
import hashlib
import re
import textwrap
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from advbug.data.schema import Example, stable_bucket
from advbug.mutation.classify import diagnose
from advbug.taxonomy import NO_BUG

MAX_FUNCTION_LINES = 80
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_GITHUB = re.compile(r"github\.com[/:]([^/]+)/([^/]+?)(?:\.git)?/?$")


@dataclass(frozen=True)
class FixRecord:
    source: str  # "pypibugs" | "tssb"
    repo_url: str
    fix_sha: str
    path: str
    diff: str
    source_label: str = ""  # the dataset's own bug-type annotation, kept for analysis

    @property
    def id(self) -> str:
        key = f"{self.repo_url}|{self.fix_sha}|{self.path}|{self.diff}"
        return f"{self.source}-{hashlib.sha1(key.encode()).hexdigest()[:12]}"


# --- fetching --------------------------------------------------------------------------------


def github_slug(repo_url: str) -> tuple[str, str] | None:
    m = _GITHUB.search(repo_url.strip())
    return (m.group(1), m.group(2)) if m else None


def fetch_file(record: FixRecord, cache_dir: Path, timeout: float = 20.0) -> str | None:
    """The file at the fix commit (i.e. the fixed version), cached on disk; None if unavailable."""
    slug = github_slug(record.repo_url)
    if slug is None:
        return None
    cache = cache_dir / f"{record.id}.py"
    missing = cache.with_suffix(".missing")
    if cache.exists():
        return cache.read_text(encoding="utf-8", errors="replace")
    if missing.exists():
        return None
    quoted = urllib.parse.quote(f"{slug[0]}/{slug[1]}/{record.fix_sha}/{record.path}", safe="/")
    url = f"https://raw.githubusercontent.com/{quoted}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            missing.parent.mkdir(parents=True, exist_ok=True)
            missing.touch()
        return None
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):  # ValueError: malformed URL
        return None
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(text, encoding="utf-8")
    return text


def fetch_all(records: list[FixRecord], cache_dir: Path, threads: int = 8) -> list[str | None]:
    with ThreadPoolExecutor(threads) as pool:
        return list(pool.map(lambda r: fetch_file(r, cache_dir), records))


# --- diff handling ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Hunk:
    old_start: int
    new_start: int
    lines: list[str]  # with their ' ', '-', '+' prefixes


def parse_hunks(diff: str) -> list[_Hunk]:
    hunks: list[_Hunk] = []
    current: _Hunk | None = None
    for line in diff.splitlines():
        m = _HUNK.match(line)
        if m:
            current = _Hunk(int(m.group(1)), int(m.group(3)), [])
            hunks.append(current)
        elif current is not None and line[:1] in (" ", "-", "+"):
            current.lines.append(line)
        elif current is not None and line == "":
            current.lines.append(" ")  # some exports strip the space of empty context lines
    return hunks


def reverse_apply(fixed: str, hunk: _Hunk) -> tuple[str, list[int]] | None:
    """Buggy file text and the 1-based buggy-file lines the fix touched; None if it doesn't apply."""
    fixed_lines = fixed.splitlines()
    new_side = [line[1:] for line in hunk.lines if line[0] in " +"]
    old_side = [line[1:] for line in hunk.lines if line[0] in " -"]
    start = _locate(fixed_lines, new_side, hunk.new_start - 1)
    if start is None:
        return None
    buggy_lines = fixed_lines[:start] + old_side + fixed_lines[start + len(new_side) :]
    touched: set[int] = set()
    old_line = start + 1  # current line number in the buggy file
    prev = " "
    for line in hunk.lines:
        tag = line[0]
        if tag == "-":  # present only in the buggy version
            touched.add(old_line)
            old_line += 1
        elif tag == " ":
            old_line += 1
        elif prev == " ":  # a pure insertion by the fix: point at the buggy line where it goes
            touched.add(min(old_line, len(buggy_lines)))
        prev = tag
    trailing = "\n" if fixed.endswith("\n") else ""
    return "\n".join(buggy_lines) + trailing, sorted(touched)


def _locate(haystack: list[str], needle: list[str], hint: int) -> int | None:
    """Where `needle` occurs in `haystack`, preferring the position closest to `hint`."""
    if not needle:
        return hint if 0 <= hint <= len(haystack) else None
    stripped = [h.rstrip() for h in haystack]
    target = [n.rstrip() for n in needle]
    best = None
    for i in range(len(haystack) - len(needle) + 1):
        if stripped[i : i + len(needle)] == target and (best is None or abs(i - hint) < abs(best - hint)):
            best = i
    return best


# --- function extraction ---------------------------------------------------------------------


def _function_spans(text: str) -> dict[str, tuple[int, int]] | None:
    """qualname -> (first line incl. decorators, last line), 1-based; None if the file doesn't parse."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return None
    spans: dict[str, tuple[int, int]] = {}

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}{child.name}"
                if not isinstance(child, ast.ClassDef):
                    first = min([d.lineno for d in child.decorator_list] + [child.lineno])
                    spans[name] = (first, child.end_lineno or child.lineno)
                visit(child, name + ".")
            else:
                visit(child, prefix)

    visit(tree, "")
    return spans


def _innermost(spans: dict[str, tuple[int, int]], lines: list[int]) -> str | None:
    containing = [(e - s, q) for q, (s, e) in spans.items() if all(s <= ln <= e for ln in lines)]
    return min(containing)[1] if containing else None


def _cut(text: str, span: tuple[int, int]) -> str:
    return textwrap.dedent("\n".join(text.splitlines()[span[0] - 1 : span[1]])) + "\n"


def to_examples(record: FixRecord, fixed_file: str | None) -> tuple[Example, Example] | str:
    """(buggy, fixed) Examples, or the reason this record was dropped."""
    if fixed_file is None:
        return "unavailable"
    hunks = parse_hunks(record.diff)
    if len(hunks) != 1:
        return "multi_hunk"
    applied = reverse_apply(fixed_file, hunks[0])
    if applied is None:
        return "patch_mismatch"
    buggy_file, touched = applied
    buggy_spans, fixed_spans = _function_spans(buggy_file), _function_spans(fixed_file)
    if buggy_spans is None or fixed_spans is None:
        return "unparsable"
    qualname = _innermost(buggy_spans, touched)
    if qualname is None or qualname not in fixed_spans:
        return "outside_function"
    span = buggy_spans[qualname]
    if span[1] - span[0] + 1 > MAX_FUNCTION_LINES:
        return "too_long"
    buggy, fixed = _cut(buggy_file, span), _cut(fixed_file, fixed_spans[qualname])
    if buggy == fixed:
        return "no_change"
    diagnosis = diagnose(fixed, buggy)
    bug_lines = tuple(ln - span[0] + 1 for ln in touched)
    bug = Example(
        id=f"{record.id}-bug",
        source=record.source,
        code=buggy,
        bug_lines=bug_lines,
        bug_class=diagnosis.bug_class.value if diagnosis else "other",
        location=diagnosis.location if diagnosis else None,
        pair_id=record.id,
    )
    clean = Example(id=f"{record.id}-fixed", source=record.source, code=fixed, bug_lines=(), bug_class=NO_BUG, pair_id=record.id)
    return bug, clean


# --- source readers --------------------------------------------------------------------------


def read_pypibugs(csv_path: str | Path) -> Iterator[FixRecord]:
    import pandas as pd

    for row in pd.read_csv(csv_path).itertuples():
        yield FixRecord("pypibugs", row.repo, row.hash, row.old_path, row.diff, str(row.rewrite).split("(")[0])


def _tssb_candidates(path: str) -> list[tuple[str, dict[str, str]]]:
    """(statement-level class, row) for in-taxonomy single-statement fixes in one TSSB file."""
    import gzip
    import json
    import warnings

    from advbug.mutation.classify import classify

    out = []
    with gzip.open(path, "rt") as f, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for line in f:
            row = json.loads(line)
            if not row.get("in_function") or row.get("comodified"):
                continue
            # `before`/`after` are space-joined tokens of the buggy/fixed statement, still valid Python
            cls = classify(row["after"], row["before"])
            if cls is not None:
                keep = ("project_url", "commit_sha", "file_path", "diff", "sstub_pattern")
                out.append((cls.value, {k: row[k] for k in keep}))
    return out


def read_tssb(data_dir: str | Path, per_class: int, seed: int = 0, workers: int = 6) -> list[FixRecord]:
    """A class-stratified sample of TSSB-3M fixes whose statement-level change is in the taxonomy."""
    import multiprocessing as mp
    import random

    files = sorted(str(p) for p in Path(data_dir).glob("*.jsonl.gz"))
    by_class: dict[str, list[dict[str, str]]] = {}
    with mp.get_context("spawn").Pool(workers) as pool:
        for found in pool.imap_unordered(_tssb_candidates, files):
            for cls, row in found:
                by_class.setdefault(cls, []).append(row)
    rng = random.Random(seed)
    records = []
    for cls in sorted(by_class):
        rows = sorted(by_class[cls], key=lambda r: (r["project_url"], r["commit_sha"], r["file_path"]))
        rng.shuffle(rows)
        for row in rows[:per_class]:
            records.append(
                FixRecord("tssb", row["project_url"], row["commit_sha"], row["file_path"], row["diff"], row["sstub_pattern"])
            )
    return records


def assign_splits(examples: Iterable[Example], dev_buckets: int = 5) -> list[Example]:
    """~1/dev_buckets of pairs go to real-dev; the rest to real-test. Pairs never straddle splits."""
    out = []
    for ex in examples:
        split = "dev" if stable_bucket(ex.pair_id or ex.id, dev_buckets) == 0 else "test"
        out.append(Example(**{**ex.__dict__, "split": split}))
    return out
