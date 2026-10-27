"""Text-level helpers: where did an edit land, in the mutant's own line numbering?"""

from __future__ import annotations

import difflib


def changed_span(original: str, mutant: str) -> tuple[int, int] | None:
    """1-based inclusive (first, last) changed line in `mutant`, or None if the texts are equal.

    For a pure deletion the span is the single line that now sits where the deleted lines were,
    which is what a detector should point at.
    """
    a, b = original.splitlines(), mutant.splitlines()
    opcodes = [op for op in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes() if op[0] != "equal"]
    if not opcodes:
        return None
    first = opcodes[0][3] + 1
    last = max(opcodes[-1][4], first)
    if not b:
        return None
    return min(first, len(b)), min(last, len(b))


def edited_line_count(original: str, mutant: str) -> int:
    """Number of original lines replaced or deleted plus mutant lines inserted (edit size for G2)."""
    a, b = original.splitlines(), mutant.splitlines()
    total = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag != "equal":
            total += max(i2 - i1, j2 - j1)
    return total
