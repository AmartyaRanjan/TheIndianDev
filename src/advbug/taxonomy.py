"""Bug taxonomy v1: the closed set of bug classes every component and metric is defined over.

Classes are disjoint by construction: `mutation.classify` checks them in PRIORITY order and the
first match wins. Changing anything here changes TAXONOMY_FINGERPRINT, which every run logs.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum

TAXONOMY_VERSION = "1.0"


class BugClass(StrEnum):
    NEGATION = "negation"
    BOUNDARY_SHIFT = "boundary_shift"
    COMPARISON_OP = "comparison_op"
    BOOLEAN_OP = "boolean_op"
    ARITHMETIC_OP = "arithmetic_op"
    ARGUMENT_SWAP = "argument_swap"
    NONE_GUARD_OMISSION = "none_guard_omission"
    TYPE_CONVERSION = "type_conversion"
    WRONG_RETURN = "wrong_return"
    VARIABLE_MISUSE = "variable_misuse"


NO_BUG = "no_bug"

PRIORITY: tuple[BugClass, ...] = tuple(BugClass)

DESCRIPTIONS: dict[BugClass, str] = {
    BugClass.NEGATION: "a condition is logically inverted: `not` added or dropped, `==`/`!=`, "
    "`is`/`is not` or `in`/`not in` flipped, or True/False swapped",
    BugClass.BOUNDARY_SHIFT: "off-by-one: `<` vs `<=` (or `>` vs `>=`), an integer constant changed by 1, "
    "or a `+ 1`/`- 1` added or removed",
    BugClass.COMPARISON_OP: "a comparison uses the wrong operator, e.g. `<` instead of `>`",
    BugClass.BOOLEAN_OP: "`and` used instead of `or`, or vice versa",
    BugClass.ARITHMETIC_OP: "wrong arithmetic operator, e.g. `+` instead of `-` or `/` instead of `//`",
    BugClass.ARGUMENT_SWAP: "two arguments of a call are passed in the wrong order",
    BugClass.NONE_GUARD_OMISSION: "a check for None, or a fallback such as `x or []`, is missing",
    BugClass.TYPE_CONVERSION: "a type conversion such as int(), str(), float() or list() is missing, extra or wrong",
    BugClass.WRONG_RETURN: "a return statement returns the wrong expression, or nothing/None",
    BugClass.VARIABLE_MISUSE: "a variable is used where a different variable from the same scope was intended",
}

# Builtins whose addition, removal or substitution counts as a type_conversion bug.
CONVERSION_FUNCS = frozenset({"int", "float", "str", "bool", "list", "tuple", "set", "frozenset", "dict"})

# Secondary location tags (analysis only, not classes), most specific first.
LOCATIONS = ("loop_header", "condition", "return", "call_arg", "statement")


def _fingerprint() -> str:
    payload = TAXONOMY_VERSION + "|" + "|".join(f"{c.value}:{DESCRIPTIONS[c]}" for c in PRIORITY)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


TAXONOMY_FINGERPRINT = _fingerprint()
