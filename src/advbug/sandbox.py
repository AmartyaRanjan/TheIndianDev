"""Run a snippet's unit tests in a throwaway child process.

Used when building the corpus (originals must pass) and by gate check G6 (a mutant must fail at
least one test, otherwise it is an equivalent mutant). Each job runs in a fresh fork of a warm,
single-threaded worker with CPU/memory/file-size rlimits, a private temp directory, a wall-clock
timeout, and network/subprocess/delete calls stubbed out.

This is defence in depth, not a security boundary (unprivileged network namespaces are not
available on this host). The real guarantee is upstream: G6 only executes code after G5 has
confirmed it is a single taxonomy edit of corpus code.
"""

from __future__ import annotations

import inspect
import json
import multiprocessing as mp
import os
import resource
import select
import shutil
import signal
import sys
import tempfile
import time
import types
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

_EXTRA_MEMORY = 1 << 30  # address space allowed on top of what the worker already maps
_FILE_SIZE_LIMIT = 16 << 20
_ERROR_CHARS = 300


@dataclass(frozen=True)
class RunResult:
    passed: int
    failed: int
    timed_out: bool = False
    error: str | None = None  # solution or tests failed to load, or the child crashed

    @property
    def all_passed(self) -> bool:
        return self.passed > 0 and self.failed == 0 and not self.timed_out and self.error is None


def run_tests(solution: str, tests: str, timeout: float = 2.0) -> RunResult:
    """Execute `solution` as module `solution`, then every zero-argument test in `tests`.

    Forks the calling process, so call it from a light single-threaded process (SandboxPool
    workers), never from a process holding CUDA state or threads.
    """
    tmp = tempfile.mkdtemp(prefix="advbug-sbx-")
    rfd, wfd = os.pipe()
    pid = os.fork()
    if pid == 0:  # child: never returns
        os.close(rfd)
        _child(solution, tests, wfd, tmp, timeout)
    os.close(wfd)
    chunks: list[bytes] = []
    timed_out = False
    deadline = time.monotonic() + timeout
    try:
        while True:
            remaining = deadline - time.monotonic()
            ready = select.select([rfd], [], [], max(remaining, 0))[0] if remaining > 0 else []
            if not ready:
                timed_out = True
                break
            data = os.read(rfd, 1 << 16)
            if not data:
                break
            chunks.append(data)
    finally:
        if timed_out:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        _, status = os.waitpid(pid, 0)
        os.close(rfd)
        shutil.rmtree(tmp, ignore_errors=True)
    if timed_out:
        return RunResult(0, 0, timed_out=True)
    try:
        payload = json.loads(b"".join(chunks))
    except json.JSONDecodeError:
        return RunResult(0, 0, error=f"child died (status {status})")
    return RunResult(payload["passed"], payload["failed"], error=payload["error"])


def _child(solution: str, tests: str, wfd: int, tmp: str, timeout: float) -> None:
    try:
        os.setsid()
        devnull = os.open(os.devnull, os.O_RDWR)
        for fd in (0, 1, 2):
            os.dup2(devnull, fd)
        os.chdir(tmp)
        cpu = int(timeout) + 1
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
        resource.setrlimit(resource.RLIMIT_AS, (_mapped_bytes() + _EXTRA_MEMORY,) * 2)
        resource.setrlimit(resource.RLIMIT_FSIZE, (_FILE_SIZE_LIMIT,) * 2)
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        _disable_escape_hatches()
        result = _execute(solution, tests)
    except BaseException as e:  # noqa: BLE001 - anything at all must still produce a result
        result = {"passed": 0, "failed": 0, "error": f"sandbox: {type(e).__name__}"}
    try:
        os.write(wfd, json.dumps(result).encode())
    finally:
        os._exit(0)


def _mapped_bytes() -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmSize:"):
                return int(line.split()[1]) * 1024
    return 0


def _blocked(*_args: object, **_kwargs: object) -> None:
    raise PermissionError("disabled inside the advbug sandbox")


def _disable_escape_hatches() -> None:
    import socket
    import subprocess

    for name in ("socket", "create_connection", "getaddrinfo", "socketpair"):
        setattr(socket, name, _blocked)
    for name in ("Popen", "run", "call", "check_call", "check_output"):
        setattr(subprocess, name, _blocked)
    for name in (
        "system", "popen", "fork", "forkpty", "kill", "killpg", "execv", "execve", "execvp", "execvpe",
        "spawnv", "spawnve", "spawnvp", "posix_spawn", "posix_spawnp",
        "remove", "unlink", "rmdir", "removedirs", "rename", "renames", "replace", "chmod", "chown",
    ):  # fmt: skip
        if hasattr(os, name):
            setattr(os, name, _blocked)
    shutil.rmtree = _blocked
    shutil.move = _blocked


def _execute(solution: str, tests: str) -> dict[str, object]:
    module = types.ModuleType("solution")
    module.__file__ = "solution.py"
    sys.modules["solution"] = module
    try:
        exec(compile(solution, "solution.py", "exec"), module.__dict__)
    except BaseException as e:  # noqa: BLE001
        return {"passed": 0, "failed": 0, "error": f"solution: {type(e).__name__}: {e}"[:_ERROR_CHARS]}
    # Some KodCode test files omit `from solution import ...`; behave as if it were `import *`.
    namespace: dict[str, object] = {k: v for k, v in module.__dict__.items() if not k.startswith("__")}
    namespace["__name__"] = "test_solution"
    try:
        exec(compile(tests, "test_solution.py", "exec"), namespace)
    except BaseException as e:  # noqa: BLE001
        return {"passed": 0, "failed": 0, "error": f"tests: {type(e).__name__}: {e}"[:_ERROR_CHARS]}
    passed = failed = 0
    for test in _collect_tests(namespace):
        try:
            test()
            passed += 1
        except BaseException as e:  # noqa: BLE001 - assertion errors, crashes, recursion, ...
            if type(e).__name__ != "Skipped":
                failed += 1
    return {"passed": passed, "failed": failed, "error": None}


def _collect_tests(namespace: dict[str, object]) -> list[object]:
    """Zero-argument `test*` functions and `Test*` class methods; fixtures/parametrize are skipped."""
    tests: list[object] = []
    for name, obj in namespace.items():
        if name.startswith("test") and inspect.isfunction(obj) and _no_required_args(obj):
            tests.append(obj)
        elif name.startswith("Test") and inspect.isclass(obj):
            try:
                instance = obj()
            except Exception:  # noqa: BLE001
                continue
            for attr, method in inspect.getmembers(instance, inspect.ismethod):
                if attr.startswith("test") and _no_required_args(method):
                    tests.append(method)
    return tests


def _no_required_args(fn: object) -> bool:
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return all(p.default is not p.empty or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in params)


# --- worker pool -----------------------------------------------------------------------------


def _warm_worker() -> None:
    import pytest  # noqa: F401 - preloaded so forked children import tests quickly


def _run_job(job: tuple[str, str], timeout: float) -> RunResult:
    return run_tests(job[0], job[1], timeout)


class SandboxPool:
    """Long-lived spawn-context workers; each job is a fork of a warm worker."""

    def __init__(self, workers: int = 4, timeout: float = 2.0) -> None:
        self.timeout = timeout
        self._pool = ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context("spawn"), initializer=_warm_worker
        )

    def run(self, jobs: Iterable[tuple[str, str]]) -> list[RunResult]:
        jobs = list(jobs)
        return list(self._pool.map(_run_job, jobs, [self.timeout] * len(jobs), chunksize=4))

    def close(self) -> None:
        # Terminate rather than wait: a blocking shutdown was seen to hang and leak idle workers.
        processes = list(getattr(self._pool, "_processes", {}).values())
        self._pool.shutdown(wait=False, cancel_futures=True)
        for p in processes:
            if p.is_alive():
                p.terminate()
        for p in processes:
            p.join(timeout=5)

    def __enter__(self) -> SandboxPool:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
