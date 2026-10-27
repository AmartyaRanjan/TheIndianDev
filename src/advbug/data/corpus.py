"""Clean training corpus: KodCode solutions whose own unit tests pass in the sandbox.

Order of work, cheapest first: static filters over every row -> exact dedupe -> seeded shuffle ->
in batches: near-duplicate check (MinHash), mutation-site diversity, tests pass twice in the
sandbox -> stop once the target size is reached -> deterministic train/val split by id hash.
"""

from __future__ import annotations

import ast
import glob
import hashlib
import multiprocessing as mp
import random
import re
import sys
import warnings
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field

import pyarrow.parquet as pq
from datasketch import MinHash, MinHashLSH

from advbug.data.schema import CleanFunction, stable_bucket
from advbug.mutation.operators import site_counts
from advbug.sandbox import SandboxPool

MIN_LINES, MAX_LINES = 8, 40  # non-blank lines of the solution
MAX_LINE_CHARS = 120
MAX_FUNCTIONS = 3
MIN_TESTS = 3
MAX_TEST_CHARS = 6000
MIN_MUTABLE_CLASSES = 3  # classes with at least one candidate site
NEAR_DUP_THRESHOLD = 0.8
_ALLOWED_TEST_IMPORTS = {"solution", "pytest"}
_STDLIB = set(sys.stdlib_module_names)
_TOKEN = re.compile(r"\w+|[^\w\s]")


@dataclass
class BuildStats:
    rows: int = 0
    rejected: Counter[str] = field(default_factory=Counter)
    exact_duplicates: int = 0
    near_duplicates: int = 0
    low_diversity: int = 0
    subset_capped: int = 0
    failed_tests: int = 0
    flaky: int = 0
    kept: int = 0

    def as_dict(self) -> dict[str, object]:
        return {k: dict(v) if isinstance(v, Counter) else v for k, v in vars(self).items()}


def iter_kodcode(pattern: str) -> Iterator[dict[str, str]]:
    for path in sorted(glob.glob(pattern)):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=2048, columns=["question_id", "subset", "solution", "test"]):
            yield from batch.to_pylist()


def normalize(code: str) -> str:
    lines = [line.rstrip() for line in code.strip("\n").splitlines()]
    return "\n".join(lines) + "\n"


def static_reject_reason(code: str, tests: str) -> str | None:
    """Why this (solution, tests) pair is unusable, or None if it passes the cheap filters."""
    if "\t" in code:
        return "tabs"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", SyntaxWarning)  # e.g. invalid escapes: errors in future Pythons
            tree = ast.parse(code)
            test_tree = ast.parse(tests)
    except (SyntaxError, SyntaxWarning, ValueError):
        return "syntax"
    non_blank = [line for line in code.splitlines() if line.strip()]
    if not MIN_LINES <= len(non_blank) <= MAX_LINES:
        return "length"
    if any(len(line) > MAX_LINE_CHARS for line in non_blank):
        return "long_line"
    functions = 0
    for i, node in enumerate(tree.body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions += 1
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if not _imports_ok(node, _STDLIB):
                return "non_stdlib_import"
        elif not (i == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)):
            return "top_level_statement"
    if not 1 <= functions <= MAX_FUNCTIONS:
        return "function_count"
    if len(tests) > MAX_TEST_CHARS:
        return "tests_too_long"
    for node in ast.walk(test_tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)) and not _imports_ok(node, _STDLIB | _ALLOWED_TEST_IMPORTS):
            return "test_import"
    zero_arg_tests = [
        n
        for n in test_tree.body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("test") and not n.args.args and not n.decorator_list
    ]
    if len(zero_arg_tests) < MIN_TESTS:
        return "few_tests"
    return None


def _imports_ok(node: ast.Import | ast.ImportFrom, allowed: set[str]) -> bool:
    if isinstance(node, ast.ImportFrom):
        return node.level == 0 and (node.module or "").split(".")[0] in allowed
    return all(alias.name.split(".")[0] in allowed for alias in node.names)


def _static_job(row: dict[str, str]) -> tuple[str, CleanFunction | None]:
    code, tests = normalize(row["solution"] or ""), (row["test"] or "").strip() + "\n"
    reason = static_reject_reason(code, tests)
    if reason:
        return reason, None
    return "ok", CleanFunction(id=row["question_id"], source=f"kodcode/{row['subset']}", code=code, tests=tests)


def _diversity_job(code: str) -> int:
    return sum(1 for n in site_counts(code).values() if n > 0)


def _minhash(code: str) -> MinHash:
    tokens = _TOKEN.findall(code)
    m = MinHash(num_perm=64, seed=1)
    for i in range(max(len(tokens) - 2, 1)):
        m.update(" ".join(tokens[i : i + 3]).encode())
    return m


def build_corpus(
    pattern: str,
    target: int,
    seed: int = 0,
    workers: int = 6,
    max_subset_fraction: float = 0.3,
    batch_size: int = 1500,
    log: bool = True,
) -> tuple[list[CleanFunction], BuildStats]:
    stats = BuildStats()
    ctx = mp.get_context("spawn")
    candidates: dict[str, CleanFunction] = {}  # keyed by an AST hash: exact dedupe
    with ctx.Pool(workers) as pool:
        for reason, item in pool.imap_unordered(_static_job, iter_kodcode(pattern), chunksize=256):
            stats.rows += 1
            if item is None:
                stats.rejected[reason] += 1
                continue
            key = hashlib.sha1(ast.dump(ast.parse(item.code)).encode()).hexdigest()
            if key in candidates:
                stats.exact_duplicates += 1
            else:
                candidates[key] = item
    pool_items = sorted(candidates.values(), key=lambda c: c.id)
    random.Random(seed).shuffle(pool_items)
    if log:
        print(f"static: {stats.rows} rows -> {len(pool_items)} unique candidates; rejected {dict(stats.rejected)}")

    lsh = MinHashLSH(threshold=NEAR_DUP_THRESHOLD, num_perm=64)
    per_subset: Counter[str] = Counter()
    cap = max(1, int(max_subset_fraction * target))
    kept: list[CleanFunction] = []
    with ctx.Pool(workers) as pool, SandboxPool(workers=workers, timeout=3.0) as sandbox:
        for start in range(0, len(pool_items), batch_size):
            batch = []
            for item in pool_items[start : start + batch_size]:
                if per_subset[item.source] >= cap:
                    stats.subset_capped += 1
                    continue
                mh = _minhash(item.code)
                if lsh.query(mh):
                    stats.near_duplicates += 1
                    continue
                lsh.insert(item.id, mh)
                batch.append(item)
            diverse = []
            for item, n_classes in zip(batch, pool.map(_diversity_job, [b.code for b in batch], chunksize=16)):
                if n_classes >= MIN_MUTABLE_CLASSES:
                    diverse.append(item)
                else:
                    stats.low_diversity += 1
            batch = diverse
            jobs = [(b.code, b.tests) for b in batch]
            first, second = sandbox.run(jobs), sandbox.run(jobs)
            for item, r1, r2 in zip(batch, first, second):
                if not r1.all_passed:
                    stats.failed_tests += 1
                elif not r2.all_passed or r1.passed != r2.passed:
                    stats.flaky += 1
                elif per_subset[item.source] < cap and len(kept) < target:
                    kept.append(item)
                    per_subset[item.source] += 1
            if log:
                print(f"  scanned {min(start + batch_size, len(pool_items))}: kept {len(kept)}/{target}", flush=True)
            if len(kept) >= target:
                break
    stats.kept = len(kept)
    return kept, stats


def split(items: list[CleanFunction], val_buckets: int = 25) -> tuple[list[CleanFunction], list[CleanFunction]]:
    """~1/val_buckets of items go to synthetic-val, chosen by a stable hash of the id."""
    train = [c for c in items if stable_bucket(c.id, val_buckets) != 0]
    val = [c for c in items if stable_bucket(c.id, val_buckets) == 0]
    return sorted(train, key=lambda c: c.id), sorted(val, key=lambda c: c.id)
