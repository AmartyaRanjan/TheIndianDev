"""Validity gate for injector outputs (plan §4.4). Each check is reported separately.

G1 format     output parses, class is in the taxonomy, line range is valid
G2 size       non-empty edit of at most 3 lines and a bounded number of characters
G3 compile    the mutant compiles
G4 artifacts  no new/changed comments, tabs, trailing whitespace or overlong lines (detector shortcuts)
G5 label      `classify(original, mutant)` equals the claimed class
G6 behaviour  the mutant fails at least one unit test (rejects equivalent mutants)

G6 runs only after G1-G5 pass, so code the injector wrote is executed only once it is known to be
a single taxonomy edit of corpus code. The A3 arm disables G2-G6; G1 always applies because the
detector has to be scored against a parseable claim.
"""

from __future__ import annotations

import difflib
import io
import tokenize
from collections import Counter
from dataclasses import dataclass

from advbug.data.schema import CleanFunction
from advbug.game.prompts import apply_action, parse_injector
from advbug.mutation.classify import diagnose
from advbug.mutation.lines import changed_span, edited_line_count
from advbug.sandbox import SandboxPool
from advbug.taxonomy import BugClass

CHECKS = ("G1_format", "G2_size", "G3_compile", "G4_artifact", "G5_label", "G6_equivalent")
_CLASS_NAMES = {c.value for c in BugClass}


@dataclass(frozen=True)
class GateConfig:
    enabled: bool = True  # False = A3 arm (only G1 enforced)
    max_edit_lines: int = 3
    max_edit_chars: int = 120
    max_line_chars: int = 120


@dataclass(frozen=True)
class GateResult:
    ok: bool
    failed_check: str | None
    mutant: str | None = None
    claimed_class: str | None = None
    bug_lines: tuple[int, ...] = ()  # where the detector should point, in mutant coordinates
    location: str | None = None
    edit_chars: int = 0


def edit_chars(original: str, mutant: str) -> int:
    """Characters inserted plus characters deleted (a token-level edit-size proxy)."""
    ops = difflib.SequenceMatcher(a=original, b=mutant, autojunk=False).get_opcodes()
    return sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in ops if tag != "equal")


def _comments(code: str) -> Counter[str] | None:
    try:
        tokens = tokenize.generate_tokens(io.StringIO(code).readline)
        return Counter(t.string for t in tokens if t.type == tokenize.COMMENT)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return None


def _artifact(original: str, mutant: str, max_line_chars: int) -> bool:
    before, after = _comments(original), _comments(mutant)
    if before is None or after is None or after - before:
        return True  # a comment was added or changed
    old_lines = set(original.splitlines())
    for line in mutant.splitlines():
        if line in old_lines:
            continue
        if "\t" in line or line != line.rstrip() or len(line) > max_line_chars:
            return True
    return False


def static_checks(original: str, mutant: str, claimed: str, config: GateConfig) -> GateResult:
    """G2-G5 on an already-applied edit (also used for AST-engine mutants in the A1 arm)."""
    span = changed_span(original, mutant)
    lines = tuple(range(span[0], span[1] + 1)) if span else ()
    chars = edit_chars(original, mutant)
    base = dict(mutant=mutant, claimed_class=claimed, bug_lines=lines, edit_chars=chars)
    if not span or edited_line_count(original, mutant) > config.max_edit_lines or chars > config.max_edit_chars:
        return GateResult(False, "G2_size", **base)
    try:
        compile(mutant, "<mutant>", "exec")
    except (SyntaxError, ValueError):
        return GateResult(False, "G3_compile", **base)
    if _artifact(original, mutant, config.max_line_chars):
        return GateResult(False, "G4_artifact", **base)
    diagnosis = diagnose(original, mutant)
    if diagnosis is None or diagnosis.bug_class != claimed:
        return GateResult(False, "G5_label", **base)
    return GateResult(True, None, location=diagnosis.location, **base)


class Gate:
    def __init__(self, config: GateConfig, sandbox: SandboxPool | None) -> None:
        if config.enabled and sandbox is None:
            raise ValueError("an enabled gate needs a sandbox for G6")
        self.config = config
        self.sandbox = sandbox

    def check(self, items: list[tuple[CleanFunction, str]]) -> list[GateResult]:
        """Gate a batch of (clean function, raw injector output) pairs."""
        results = [self._parse(fn, text) for fn, text in items]
        return self._behaviour([fn for fn, _ in items], results)

    def check_mutants(self, items: list[tuple[CleanFunction, str, str]]) -> list[GateResult]:
        """Gate already-applied (function, mutant, claimed class) triples, e.g. AST-engine mutants."""
        results = [static_checks(fn.code, mutant, claimed, self.config) for fn, mutant, claimed in items]
        return self._behaviour([fn for fn, _, _ in items], results)

    def _parse(self, fn: CleanFunction, text: str) -> GateResult:
        action = parse_injector(text)
        mutant = apply_action(fn.code, action) if action else None
        if action is None or mutant is None or action.bug_class not in _CLASS_NAMES:
            return GateResult(False, "G1_format")
        if not self.config.enabled:
            span = changed_span(fn.code, mutant)
            lines = tuple(range(span[0], span[1] + 1)) if span else (action.start,)
            return GateResult(True, None, mutant, action.bug_class, lines, edit_chars=edit_chars(fn.code, mutant))
        return static_checks(fn.code, mutant, action.bug_class, self.config)

    def _behaviour(self, fns: list[CleanFunction], results: list[GateResult]) -> list[GateResult]:
        if not self.config.enabled:
            return results
        pending = [i for i, r in enumerate(results) if r.ok]
        runs = self.sandbox.run([(results[i].mutant, fns[i].tests) for i in pending]) if pending else []
        out = list(results)
        for i, run in zip(pending, runs):
            if run.all_passed:  # behaviour unchanged under the tests: an equivalent mutant
                out[i] = GateResult(**{**results[i].__dict__, "ok": False, "failed_check": "G6_equivalent"})
        return out
