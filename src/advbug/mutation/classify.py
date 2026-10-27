"""AST-diff classifier: which taxonomy class turns `original` into `mutant`?

Used by the validity gate (G5) and to label real held-out bugs as `classify(fixed, buggy)`.
The two trees are diffed structurally (positions and formatting ignored); a change is accepted
only if it is a single cause: one differing site, or two arguments of one call swapped.
Classes are then tested in taxonomy PRIORITY order and the first match wins.
"""

from __future__ import annotations

import ast
import builtins
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from advbug.taxonomy import CONVERSION_FUNCS, PRIORITY, BugClass

Path = tuple[str | int, ...]
Chain = list[tuple[ast.AST, str]]  # (ancestor, field leading towards the diff), outermost first

_STMT_LIST_FIELDS = frozenset({"body", "orelse", "finalbody"})
_NEG_PAIRS = {frozenset({ast.Eq, ast.NotEq}), frozenset({ast.Is, ast.IsNot}), frozenset({ast.In, ast.NotIn})}
_BOUNDARY_PAIRS = {frozenset({ast.Lt, ast.LtE}), frozenset({ast.Gt, ast.GtE})}
_BUILTINS = frozenset(dir(builtins))


@dataclass(frozen=True)
class Diagnosis:
    bug_class: BugClass
    location: str


@dataclass(frozen=True)
class _Diff:
    kind: str  # "node": node types differ | "field": scalar differs | "child": optional child added/removed | "list": list length differs
    path: Path  # path to the differing node ("node") or to the node owning the field (other kinds)
    orig: Any
    mut: Any
    field: str | None = None


def classify(original: str, mutant: str) -> BugClass | None:
    diagnosis = diagnose(original, mutant)
    return diagnosis.bug_class if diagnosis else None


def diagnose(original: str, mutant: str) -> Diagnosis | None:
    try:
        a, b = ast.parse(original), ast.parse(mutant)
    except (SyntaxError, ValueError):
        return None
    diffs = _diffs(a, b, ())
    if len(diffs) == 1:
        d = diffs[0]
        chain = _chain(a, d.path)
        for cls in PRIORITY:
            rule = _RULES.get(cls)
            if rule and rule(d, chain, a):
                return Diagnosis(cls, _location(chain, d))
        return None
    if len(diffs) > 1 and _is_argument_swap(diffs, a, b):
        return Diagnosis(BugClass.ARGUMENT_SWAP, "call_arg")
    return None


# --- structural diff -------------------------------------------------------------------------


def _diffs(a: ast.AST, b: ast.AST, path: Path) -> list[_Diff]:
    if type(a) is not type(b):
        return [_Diff("node", path, a, b)]
    out: list[_Diff] = []
    for field in a._fields:
        va, vb = getattr(a, field, None), getattr(b, field, None)
        if isinstance(va, list) and isinstance(vb, list):
            if len(va) != len(vb):
                out.append(_Diff("list", path, a, b, field))
                continue
            for i, (x, y) in enumerate(zip(va, vb)):
                out += _diff_slot(x, y, a, b, path, field, path + (field, i))
        else:
            out += _diff_slot(va, vb, a, b, path, field, path + (field,))
    return out


def _diff_slot(x: Any, y: Any, a: ast.AST, b: ast.AST, owner: Path, field: str, path: Path) -> list[_Diff]:
    if isinstance(x, ast.AST) and isinstance(y, ast.AST):
        return _diffs(x, y, path)
    if isinstance(x, ast.AST) or isinstance(y, ast.AST):
        return [_Diff("child", owner, a, b, field)]
    if type(x) is not type(y) or x != y:
        return [_Diff("field", owner, a, b, field)]
    return []


def _node_at(tree: ast.AST, path: Path) -> Any:
    node: Any = tree
    for step in path:
        node = node[step] if isinstance(step, int) else getattr(node, step)
    return node


def _chain(tree: ast.AST, path: Path) -> Chain:
    chain: Chain = []
    node: Any = tree
    for step in path:
        if isinstance(step, str):
            chain.append((node, step))
            node = getattr(node, step)
        else:
            node = node[step]
    return chain


def _same(x: Any, y: Any) -> bool:
    if isinstance(x, ast.AST) and isinstance(y, ast.AST):
        return ast.dump(x) == ast.dump(y)
    return False


# --- predicates ------------------------------------------------------------------------------


def _op_pair(d: _Diff, base: type) -> frozenset[type] | None:
    if d.kind == "node" and isinstance(d.orig, base) and isinstance(d.mut, base):
        return frozenset({type(d.orig), type(d.mut)})
    return None


def _int_value(node: Any) -> int | None:
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.USub, ast.UAdd))
        and isinstance(node.operand, ast.Constant)
        and type(node.operand.value) is int
    ):
        return -node.operand.value if isinstance(node.op, ast.USub) else node.operand.value
    return None


def _wraps_not(x: Any, y: Any) -> bool:
    return isinstance(x, ast.UnaryOp) and isinstance(x.op, ast.Not) and _same(x.operand, y)


def _plus_minus_one(x: Any, y: Any) -> bool:
    """x is `y + 1`, `y - 1` or `1 + y`."""
    if not (isinstance(x, ast.BinOp) and isinstance(x.op, (ast.Add, ast.Sub))):
        return False
    if _int_value(x.right) == 1 and _same(x.left, y):
        return True
    return isinstance(x.op, ast.Add) and _int_value(x.left) == 1 and _same(x.right, y)


def _is_none(node: Any) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def is_none_check(node: Any) -> bool:
    """`x is None`, `x is not None`, `x == None`, `x != None` (either side), or a not/and/or of one."""
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        return isinstance(node.ops[0], (ast.Is, ast.IsNot, ast.Eq, ast.NotEq)) and (
            _is_none(node.left) or _is_none(node.comparators[0])
        )
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return is_none_check(node.operand)
    if isinstance(node, ast.BoolOp):
        return any(is_none_check(v) for v in node.values)
    return False


def _is_guard(stmt: Any) -> bool:
    return isinstance(stmt, ast.If) and not stmt.orelse and len(stmt.body) <= 2 and is_none_check(stmt.test)


def _single_removal(xs: list[Any], ys: list[Any]) -> Any | None:
    """The element of xs whose removal yields ys, if exactly one does."""
    if len(xs) != len(ys) + 1:
        return None
    ys_dump = [ast.dump(y) for y in ys]
    for i, x in enumerate(xs):
        if [ast.dump(v) for v in xs[:i] + xs[i + 1 :]] == ys_dump:
            return x
    return None


def _in_test_position(chain: Chain) -> bool:
    if not chain:
        return False
    parent, field = chain[-1]
    if field == "test" and isinstance(parent, (ast.If, ast.While, ast.IfExp, ast.Assert)):
        return True
    return field == "ifs" and isinstance(parent, ast.comprehension)


def _wraps_conversion(x: Any, y: Any) -> bool:
    return (
        isinstance(x, ast.Call)
        and isinstance(x.func, ast.Name)
        and x.func.id in CONVERSION_FUNCS
        and len(x.args) == 1
        and not x.keywords
        and _same(x.args[0], y)
    )


def _empty_container(node: Any) -> bool:
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return not node.elts
    if isinstance(node, ast.Dict):
        return not node.keys
    return False


def _scope_names(chain: Chain, tree: ast.AST) -> set[str]:
    """Names a variable could plausibly be confused with: params and bindings of the enclosing
    function, plus non-builtin names it reads (module constants, helpers)."""
    scope: ast.AST = tree
    for node, _ in chain:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            scope = node
    names: set[str] = set()
    for node in ast.walk(scope):
        if isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.Name) and (not isinstance(node.ctx, ast.Load) or node.id not in _BUILTINS):
            names.add(node.id)
    return names


# --- class rules (single diff) ---------------------------------------------------------------


def _negation(d: _Diff, chain: Chain, tree: ast.AST) -> bool:
    if _op_pair(d, ast.cmpop) in _NEG_PAIRS:
        return True
    if d.kind == "node" and (_wraps_not(d.orig, d.mut) or _wraps_not(d.mut, d.orig)):
        return True
    return (
        d.kind == "field"
        and d.field == "value"
        and isinstance(d.orig, ast.Constant)
        and type(d.orig.value) is bool
        and type(d.mut.value) is bool
    )


def _boundary_shift(d: _Diff, chain: Chain, tree: ast.AST) -> bool:
    if _op_pair(d, ast.cmpop) in _BOUNDARY_PAIRS:
        return True
    if d.kind in ("node", "field"):
        x, y = _int_value(d.orig), _int_value(d.mut)
        if x is not None and y is not None and abs(x - y) == 1:
            return True
    return d.kind == "node" and (_plus_minus_one(d.orig, d.mut) or _plus_minus_one(d.mut, d.orig))


def _comparison_op(d: _Diff, chain: Chain, tree: ast.AST) -> bool:
    return _op_pair(d, ast.cmpop) is not None


def _boolean_op(d: _Diff, chain: Chain, tree: ast.AST) -> bool:
    return _op_pair(d, ast.boolop) is not None


def _arithmetic_op(d: _Diff, chain: Chain, tree: ast.AST) -> bool:
    return _op_pair(d, ast.operator) is not None or _op_pair(d, ast.unaryop) == frozenset({ast.UAdd, ast.USub})


def _none_guard_omission(d: _Diff, chain: Chain, tree: ast.AST) -> bool:
    if d.kind == "list" and d.field in _STMT_LIST_FIELDS:
        removed = _single_removal(getattr(d.orig, d.field), getattr(d.mut, d.field))
        return removed is not None and _is_guard(removed)
    if d.kind == "list" and isinstance(d.orig, ast.BoolOp) and d.field == "values":
        removed = _single_removal(d.orig.values, d.mut.values)
        return removed is not None and is_none_check(removed)
    if d.kind != "node":
        return False
    o, m = d.orig, d.mut
    if isinstance(o, ast.BoolOp) and len(o.values) == 2:
        if (_same(o.values[0], m) and is_none_check(o.values[1])) or (_same(o.values[1], m) and is_none_check(o.values[0])):
            return True
        # `x or default` -> `x`, only where the expression is a value rather than a condition
        if isinstance(o.op, ast.Or) and _same(o.values[0], m) and not _in_test_position(chain):
            return True
    return isinstance(o, ast.IfExp) and is_none_check(o.test) and (_same(o.body, m) or _same(o.orelse, m))


def _type_conversion(d: _Diff, chain: Chain, tree: ast.AST) -> bool:
    if d.kind == "node":
        if _wraps_conversion(d.orig, d.mut) or _wraps_conversion(d.mut, d.orig):
            return True
        return _empty_container(d.orig) and _empty_container(d.mut)
    if d.kind == "field" and isinstance(d.orig, ast.Name) and d.field == "id":
        on_call = bool(chain) and isinstance(chain[-1][0], ast.Call) and chain[-1][1] == "func"
        return on_call and d.orig.id in CONVERSION_FUNCS and d.mut.id in CONVERSION_FUNCS
    return False


def _wrong_return(d: _Diff, chain: Chain, tree: ast.AST) -> bool:
    if d.kind == "child" and isinstance(d.orig, ast.Return):
        return True  # `return x` <-> `return`
    returned = bool(chain) and isinstance(chain[-1][0], ast.Return)
    # the whole returned expression replaced, or a returned constant changed (`return 0` -> `return None`)
    return returned and (d.kind == "node" or (d.kind == "field" and isinstance(d.orig, ast.Constant)))


def _variable_misuse(d: _Diff, chain: Chain, tree: ast.AST) -> bool:
    if not (d.kind == "field" and d.field == "id" and isinstance(d.orig, ast.Name) and isinstance(d.orig.ctx, ast.Load)):
        return False
    if chain and isinstance(chain[-1][0], ast.Call) and chain[-1][1] == "func":
        return False  # calling the wrong function is not a variable misuse
    return d.mut.id in _scope_names(chain, tree) - {d.orig.id}


_RULES: dict[BugClass, Callable[[_Diff, Chain, ast.AST], bool]] = {
    BugClass.NEGATION: _negation,
    BugClass.BOUNDARY_SHIFT: _boundary_shift,
    BugClass.COMPARISON_OP: _comparison_op,
    BugClass.BOOLEAN_OP: _boolean_op,
    BugClass.ARITHMETIC_OP: _arithmetic_op,
    BugClass.NONE_GUARD_OMISSION: _none_guard_omission,
    BugClass.TYPE_CONVERSION: _type_conversion,
    BugClass.WRONG_RETURN: _wrong_return,
    BugClass.VARIABLE_MISUSE: _variable_misuse,
}


# --- argument swap (two diffs) ---------------------------------------------------------------


def _is_argument_swap(diffs: list[_Diff], a: ast.AST, b: ast.AST) -> bool:
    prefix = list(diffs[0].path)
    for d in diffs[1:]:
        n = 0
        while n < min(len(prefix), len(d.path)) and prefix[n] == d.path[n]:
            n += 1
        prefix = prefix[:n]
    if not prefix or prefix[-1] not in ("args", "keywords"):
        return False
    ca, cb = _node_at(a, tuple(prefix[:-1])), _node_at(b, tuple(prefix[:-1]))
    if not (isinstance(ca, ast.Call) and isinstance(cb, ast.Call) and _same(ca.func, cb.func)):
        return False
    if prefix[-1] == "args":
        if not _same_list(ca.keywords, cb.keywords):
            return False
        xs, ys = ca.args, cb.args
    else:
        if not _same_list(ca.args, cb.args) or [k.arg for k in ca.keywords] != [k.arg for k in cb.keywords]:
            return False
        xs, ys = [k.value for k in ca.keywords], [k.value for k in cb.keywords]
    if len(xs) != len(ys) or any(isinstance(x, ast.Starred) for x in xs):
        return False
    changed = [i for i, (x, y) in enumerate(zip(xs, ys)) if not _same(x, y)]
    if len(changed) != 2:
        return False
    i, j = changed
    return _same(xs[i], ys[j]) and _same(xs[j], ys[i])


def _same_list(xs: list[Any], ys: list[Any]) -> bool:
    return len(xs) == len(ys) and all(_same(x, y) for x, y in zip(xs, ys))


# --- location tag ----------------------------------------------------------------------------


def _location(chain: Chain, d: _Diff) -> str:
    steps = [(type(node), field) for node, field in chain]
    if d.kind != "node":  # the diff sits on the node itself, e.g. a list owned by a For
        steps.append((type(d.orig), d.field or ""))
    if any(t in (ast.For, ast.AsyncFor) and f in ("iter", "target") or t is ast.While and f == "test" for t, f in steps):
        return "loop_header"
    if any(f == "test" and t in (ast.If, ast.IfExp, ast.Assert) or t is ast.comprehension and f == "ifs" for t, f in steps):
        return "condition"
    if any(t is ast.Return for t, _ in steps):
        return "return"
    if any(t is ast.Call and f in ("args", "keywords") for t, f in steps):
        return "call_arg"
    return "statement"
