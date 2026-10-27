import pytest

from advbug.mutation.classify import classify, diagnose
from advbug.mutation.lines import changed_span, edited_line_count
from advbug.taxonomy import BugClass as B

BASE = """def f(xs, n, key=None):
    total = 0
    for i in range(n):
        if xs[i] < n and key is not None:
            total += xs[i] * 2
    if key is None:
        return 0
    return int(total) if total else total
"""


def edit(old: str, new: str, src: str = BASE) -> str:
    assert src.count(old) == 1, old
    return src.replace(old, new)


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        ("key is not None", "key is None", B.NEGATION),
        ("if xs[i] < n and", "if not xs[i] < n and", B.NEGATION),
        ("xs[i] < n", "xs[i] <= n", B.BOUNDARY_SHIFT),
        ("range(n)", "range(n - 1)", B.BOUNDARY_SHIFT),
        ("* 2", "* 3", B.BOUNDARY_SHIFT),
        ("total = 0", "total = 1", B.BOUNDARY_SHIFT),
        ("xs[i] < n", "xs[i] > n", B.COMPARISON_OP),
        (" and key", " or key", B.BOOLEAN_OP),
        ("* 2", "+ 2", B.ARITHMETIC_OP),
        ("total += ", "total -= ", B.ARITHMETIC_OP),
        ("xs[i] < n and key is not None", "xs[i] < n", B.NONE_GUARD_OMISSION),
        ("    if key is None:\n        return 0\n", "", B.NONE_GUARD_OMISSION),
        ("int(total)", "total", B.TYPE_CONVERSION),
        ("int(total)", "float(total)", B.TYPE_CONVERSION),
        ("return 0\n", "return None\n", B.WRONG_RETURN),
        ("return 0\n", "return\n", B.WRONG_RETURN),
        ("xs[i] * 2", "xs[n] * 2", B.VARIABLE_MISUSE),
    ],
)
def test_single_cause_edits(old, new, expected):
    assert classify(BASE, edit(old, new)) == expected


def test_argument_swap():
    src = "def g(a, b):\n    return pow(a + 1, b)\n"
    assert classify(src, "def g(a, b):\n    return pow(b, a + 1)\n") == B.ARGUMENT_SWAP
    kw = "def g(a, b):\n    return h(x=a, y=b)\n"
    assert classify(kw, "def g(a, b):\n    return h(x=b, y=a)\n") == B.ARGUMENT_SWAP


def test_none_guard_value_fallback_only_outside_conditions():
    src = "def g(x):\n    y = x or []\n    if x or y:\n        return y\n    return x\n"
    assert classify(src, src.replace("x or []", "x")) == B.NONE_GUARD_OMISSION
    assert classify(src, src.replace("if x or y", "if x")) is None


def test_true_false_flip_is_negation():
    src = "def g():\n    return True\n"
    assert classify(src, "def g():\n    return False\n") == B.NEGATION


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("total = 0", "total = 0  # changed"),  # comment only: no AST change
        ("    total = 0\n", "    total  =  0\n"),  # formatting only
        ("xs[i] < n", "xs[i] <= n + 1"),  # two changes in one expression
        ("for i in range(n)", "for i in range(len(xs))"),  # not a taxonomy pattern
        ("xs[i] * 2", "xs[zzz] * 2"),  # name not in scope
        ("total += xs[i] * 2", "total += xs[i] * 2\n            total -= 1"),  # added statement
    ],
)
def test_rejected_edits(old, new):
    assert classify(BASE, edit(old, new)) is None


def test_two_independent_edits_rejected():
    mutant = edit("xs[i] < n", "xs[i] <= n", edit("int(total)", "total"))
    assert classify(BASE, mutant) is None


def test_unparsable_mutant_rejected():
    assert classify(BASE, BASE.replace("return 0", "return 0 +")) is None


def test_priority_return_of_off_by_one_is_boundary():
    src = "def g(n):\n    return n - 1\n"
    assert classify(src, "def g(n):\n    return n\n") == B.BOUNDARY_SHIFT


def test_location_tags():
    assert diagnose(BASE, edit("range(n)", "range(n - 1)")).location == "loop_header"
    assert diagnose(BASE, edit("xs[i] < n", "xs[i] <= n")).location == "condition"
    assert diagnose(BASE, edit("return 0\n", "return None\n")).location == "return"


def test_changed_span_replacement_and_deletion():
    mutant = edit("xs[i] < n", "xs[i] <= n")
    assert changed_span(BASE, mutant) == (4, 4)
    deleted = edit("    if key is None:\n        return 0\n", "")
    # the line now sitting where the guard was is the `return int(...)` line
    assert changed_span(BASE, deleted) == (6, 6)
    assert deleted.splitlines()[5].strip().startswith("return int")
    assert edited_line_count(BASE, deleted) == 2
    assert changed_span(BASE, BASE) is None
