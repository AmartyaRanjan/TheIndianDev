"""Injector/detector prompts and strict output parsers.

Both players see code with line-number prefixes (` 7| ...`). Outputs use small XML-ish tags so
parsing is unambiguous and format failures are countable.

Injector:  <type>CLASS</type> <lines>A-B</lines> <new>replacement lines</new>
Detector:  <line>N</line><type>CLASS</type>   or   <type>no_bug</type>
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from advbug.taxonomy import DESCRIPTIONS, NO_BUG, PRIORITY, BugClass

Messages = list[dict[str, str]]

_CLASS_NAMES = {c.value for c in BugClass}
_TYPE = re.compile(r"<type>\s*([a-z_]+)\s*</type>")
_LINE = re.compile(r"<line>\s*(\d+)\s*</line>")
_LINES = re.compile(r"<lines>\s*(\d+)\s*(?:-\s*(\d+)\s*)?</lines>")
_NEW = re.compile(r"<new>(.*?)</new>", re.DOTALL)


@dataclass(frozen=True)
class InjectorAction:
    bug_class: str  # as written by the model; may be outside the taxonomy (the gate checks it)
    start: int
    end: int
    replacement: tuple[str, ...]


@dataclass(frozen=True)
class DetectorPrediction:
    bug_class: str  # a BugClass value or NO_BUG
    line: int | None  # None when bug_class == NO_BUG


def number_lines(code: str) -> str:
    lines = code.splitlines()
    width = len(str(len(lines)))
    return "\n".join(f"{i:>{width}}| {line}" for i, line in enumerate(lines, 1))


def _taxonomy_block() -> str:
    return "\n".join(f"- {c.value}: {DESCRIPTIONS[c]}" for c in PRIORITY)


_SYSTEM = "You are an expert Python programmer and code reviewer."

_DETECTOR_TASK = """The Python code below contains at most one bug. If it has a bug, the bug is of exactly one of these types:
{taxonomy}

```python
{code}
```

If there is a bug, reply with one line: <line>N</line><type>TYPE</type>
where N is the line number of the bug and TYPE is one of the types above.
If the code is correct, reply with: <type>no_bug</type>
Reply with nothing else."""

_INJECTOR_TASK = """Below is a correct Python function with line numbers. Introduce exactly one realistic, subtle bug of one of these types:
{taxonomy}

Rules:
- Edit at most 3 consecutive lines and change as little as possible.
- Keep the indentation; do not add, remove or change comments or docstrings.
- The code must still run; the bug must change its behaviour.

```python
{code}
```

Reply in exactly this format and nothing else:
<type>TYPE</type>
<lines>A-B</lines>
<new>
the new source for lines A to B, with indentation (leave empty to delete them)
</new>"""

_FEW_SHOT = [
    (
        "def last_index(xs, target):\n    for i in range(len(xs) - 1, 0, -1):\n        if xs[i] == target:\n"
        "            return i\n    return -1\n",
        "<line>2</line><type>boundary_shift</type>",
    ),
    ("def mean(values):\n    if not values:\n        return 0.0\n    return sum(values) / len(values)\n", "<type>no_bug</type>"),
]


def detector_messages(code: str, few_shot: bool = False) -> Messages:
    messages: Messages = [{"role": "system", "content": _SYSTEM}]
    if few_shot:
        for example, answer in _FEW_SHOT:
            messages.append({"role": "user", "content": _detector_user(example)})
            messages.append({"role": "assistant", "content": answer})
    messages.append({"role": "user", "content": _detector_user(code)})
    return messages


def _detector_user(code: str) -> str:
    return _DETECTOR_TASK.format(taxonomy=_taxonomy_block(), code=number_lines(code))


def injector_messages(code: str) -> Messages:
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _INJECTOR_TASK.format(taxonomy=_taxonomy_block(), code=number_lines(code))},
    ]


def parse_detector(text: str) -> DetectorPrediction | None:
    """The first well-formed answer in `text`, or None (a format failure)."""
    type_match = _TYPE.search(text)
    if not type_match:
        return None
    bug_class = type_match.group(1)
    if bug_class == NO_BUG:
        return DetectorPrediction(NO_BUG, None)
    line_match = _LINE.search(text)
    if bug_class not in _CLASS_NAMES or not line_match:
        return None
    return DetectorPrediction(bug_class, int(line_match.group(1)))


def parse_injector(text: str) -> InjectorAction | None:
    type_match, lines_match, new_match = _TYPE.search(text), _LINES.search(text), _NEW.search(text)
    if not (type_match and lines_match and new_match):
        return None
    start = int(lines_match.group(1))
    end = int(lines_match.group(2) or start)
    body = new_match.group(1)
    body = body[1:] if body.startswith("\n") else body
    body = body[:-1] if body.endswith("\n") else body
    replacement = tuple(body.split("\n")) if body.strip() else ()
    return InjectorAction(type_match.group(1), start, end, replacement)


def apply_action(code: str, action: InjectorAction) -> str | None:
    """The edited code, or None if the line range is invalid.

    If the model dropped the indentation of the whole block (a common small-model slip), the
    original first line's indentation is restored. Applied identically in every ablation arm.
    """
    lines = code.splitlines()
    if not 1 <= action.start <= action.end <= len(lines):
        return None
    replacement = list(action.replacement)
    original = lines[action.start - 1]
    indent = original[: len(original) - len(original.lstrip())]
    non_blank = [r for r in replacement if r.strip()]
    if indent and non_blank and min(len(r) - len(r.lstrip()) for r in non_blank) == 0:
        replacement = [indent + r if r.strip() else r for r in replacement]
    return "\n".join(lines[: action.start - 1] + replacement + lines[action.end :]) + "\n"


def format_injector_output(original: str, mutant: str, bug_class: str) -> str | None:
    """The injector answer that turns `original` into `mutant` (used for the format warmup)."""
    import difflib

    a, b = original.splitlines(), mutant.splitlines()
    ops = [op for op in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes() if op[0] != "equal"]
    if not ops:
        return None
    i1, i2, j1, j2 = ops[0][1], ops[-1][2], ops[0][3], ops[-1][4]
    if i2 == i1:  # a pure insertion: anchor it on the line it goes before
        i2 = i1 + 1
        j2 += 1
    new = "\n".join(b[j1:j2])
    return f"<type>{bug_class}</type>\n<lines>{i1 + 1}-{i2}</lines>\n<new>\n{new}\n</new>"
