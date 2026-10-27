"""Format-preserving mutators: one family of edit sites per taxonomy class, applied with libcst.

`mutate(code, cls, rng)` lists every site where `cls` applies, tries them in random order and
returns the first mutant that `classify` labels `cls`. Every generated label therefore agrees
with the gate's label check (G5) by construction. libcst is used instead of `ast.unparse` so that
the rest of the function keeps its exact formatting and comments; a reformatted function would
let the detector spot mutants without understanding them.
"""

from __future__ import annotations

import ast
import dataclasses
import random
from collections.abc import Iterable
from dataclasses import dataclass

import libcst as cst
from libcst.metadata import ExpressionContext, ExpressionContextProvider, MetadataWrapper, ParentNodeProvider

from advbug.mutation.classify import diagnose
from advbug.mutation.lines import changed_span
from advbug.taxonomy import CONVERSION_FUNCS, PRIORITY, BugClass

Replacement = cst.CSTNode | cst.RemovalSentinel


@dataclass(frozen=True)
class Mutation:
    bug_class: BugClass
    mutant: str
    lines: tuple[int, int]  # changed span in the mutant, 1-based inclusive
    location: str


@dataclass(frozen=True)
class _Site:
    target: cst.CSTNode
    replacement: Replacement


_NEG_FLIP: dict[type, type] = {
    cst.Equal: cst.NotEqual,
    cst.NotEqual: cst.Equal,
    cst.Is: cst.IsNot,
    cst.IsNot: cst.Is,
    cst.In: cst.NotIn,
    cst.NotIn: cst.In,
}
_BOUNDARY_FLIP: dict[type, type] = {
    cst.LessThan: cst.LessThanEqual,
    cst.LessThanEqual: cst.LessThan,
    cst.GreaterThan: cst.GreaterThanEqual,
    cst.GreaterThanEqual: cst.GreaterThan,
}
_ORDER_OPS = (cst.LessThan, cst.LessThanEqual, cst.GreaterThan, cst.GreaterThanEqual, cst.Equal, cst.NotEqual)
_COMPARISON_SWAPS: dict[type, list[type]] = {
    op: [o for o in _ORDER_OPS if o not in (op, _NEG_FLIP.get(op), _BOUNDARY_FLIP.get(op))] for op in _ORDER_OPS
}
# Swaps stay within one precedence level so the tree shape (and hence the classifier's view) is unchanged.
_ARITH_SWAPS: dict[type, list[type]] = {
    cst.Add: [cst.Subtract],
    cst.Subtract: [cst.Add],
    cst.Multiply: [cst.Divide, cst.FloorDivide],
    cst.Divide: [cst.Multiply, cst.FloorDivide],
    cst.FloorDivide: [cst.Divide, cst.Modulo],
    cst.Modulo: [cst.FloorDivide],
}
_AUG_SWAPS: dict[type, list[type]] = {
    cst.AddAssign: [cst.SubtractAssign],
    cst.SubtractAssign: [cst.AddAssign],
    cst.MultiplyAssign: [cst.DivideAssign, cst.FloorDivideAssign],
    cst.DivideAssign: [cst.MultiplyAssign, cst.FloorDivideAssign],
    cst.FloorDivideAssign: [cst.DivideAssign],
}
_CONVERSION_SWAPS: dict[str, list[str]] = {
    "int": ["float", "str"],
    "float": ["int"],
    "str": ["int"],
    "list": ["tuple", "set"],
    "tuple": ["list"],
    "set": ["list"],
}
_ATOMS = (
    cst.Name,
    cst.Attribute,
    cst.Call,
    cst.Subscript,
    cst.Integer,
    cst.Float,
    cst.Imaginary,
    cst.SimpleString,
    cst.ConcatenatedString,
    cst.FormattedString,
    cst.List,
    cst.Dict,
    cst.Set,
    cst.ListComp,
    cst.SetComp,
    cst.DictComp,
    cst.GeneratorExp,
)
_STRINGY = (cst.SimpleString, cst.ConcatenatedString, cst.FormattedString, cst.List, cst.ListComp)
_CONSTANT_NAMES = frozenset({"True", "False", "None"})


# --- small CST helpers -----------------------------------------------------------------------


def _op(new_type: type, old: cst.CSTNode) -> cst.CSTNode:
    """Instantiate operator `new_type`, keeping the old operator's surrounding whitespace."""
    names = {f.name for f in dataclasses.fields(new_type)}
    kwargs = {k: getattr(old, k) for k in ("whitespace_before", "whitespace_after") if k in names and hasattr(old, k)}
    return new_type(**kwargs)


def _keep_parens(outer: cst.BaseExpression, inner: cst.BaseExpression) -> cst.BaseExpression:
    """Replace `outer` by its sub-expression `inner`, carrying over outer's parentheses."""
    if outer.lpar and not inner.lpar:
        return inner.with_changes(lpar=outer.lpar, rpar=outer.rpar)
    return inner


def _parenthesize(expr: cst.BaseExpression) -> cst.BaseExpression:
    if expr.lpar or isinstance(expr, _ATOMS):
        return expr
    return expr.with_changes(lpar=[cst.LeftParen()], rpar=[cst.RightParen()])


def _is_offsettable(expr: cst.BaseExpression) -> bool:
    """Expressions that read naturally with `+ 1` / `- 1` appended."""
    if isinstance(expr, cst.Name):
        return expr.value not in _CONSTANT_NAMES
    return isinstance(expr, (cst.Attribute, cst.Call, cst.Subscript))


def _offsets(expr: cst.BaseExpression) -> list[cst.BaseExpression]:
    return [cst.BinaryOperation(left=expr, operator=op(), right=cst.Integer("1")) for op in (cst.Add, cst.Subtract)]


def _is_none_name(expr: cst.CSTNode) -> bool:
    return isinstance(expr, cst.Name) and expr.value == "None"


def _is_none_check(expr: cst.CSTNode) -> bool:
    if isinstance(expr, cst.Comparison) and len(expr.comparisons) == 1:
        target = expr.comparisons[0]
        return isinstance(target.operator, (cst.Is, cst.IsNot, cst.Equal, cst.NotEqual)) and (
            _is_none_name(expr.left) or _is_none_name(target.comparator)
        )
    if isinstance(expr, cst.UnaryOperation) and isinstance(expr.operator, cst.Not):
        return _is_none_check(expr.expression)
    if isinstance(expr, cst.BooleanOperation):
        return _is_none_check(expr.left) or _is_none_check(expr.right)
    return False


def _negated(test: cst.BaseExpression) -> cst.BaseExpression:
    if isinstance(test, (cst.BooleanOperation, cst.IfExp, cst.Lambda, cst.NamedExpr)) and not test.lpar:
        test = test.with_changes(lpar=[cst.LeftParen()], rpar=[cst.RightParen()])
    return cst.UnaryOperation(operator=cst.Not(), expression=test)


def _code(node: cst.CSTNode) -> str:
    return cst.Module(body=[]).code_for_node(node)


def _stmt_count(body: cst.BaseSuite) -> int:
    if isinstance(body, cst.SimpleStatementSuite):
        return len(body.body)
    return sum(len(s.body) if isinstance(s, cst.SimpleStatementLine) else 1 for s in body.body)


# --- site collection -------------------------------------------------------------------------


class _SiteCollector(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (ParentNodeProvider, ExpressionContextProvider)

    def __init__(self) -> None:
        super().__init__()
        self.sites: dict[BugClass, list[_Site]] = {c: [] for c in BugClass}
        self._bound: set[str] = set()
        self._loads: list[cst.Name] = []

    def _add(self, cls: BugClass, target: cst.CSTNode, replacement: Replacement) -> None:
        self.sites[cls].append(_Site(target, replacement))

    def _parent(self, node: cst.CSTNode) -> cst.CSTNode | None:
        return self.get_metadata(ParentNodeProvider, node, None)

    def _negate_test(self, test: cst.BaseExpression) -> None:
        if not (isinstance(test, cst.UnaryOperation) and isinstance(test.operator, cst.Not)):
            self._add(BugClass.NEGATION, test, _negated(test))

    # comparisons: negation / boundary / wrong operator, and +-1 on the right operand
    def visit_Comparison(self, node: cst.Comparison) -> None:
        for target in node.comparisons:
            op = type(target.operator)
            if op in _NEG_FLIP:
                self._add(BugClass.NEGATION, target, target.with_changes(operator=_op(_NEG_FLIP[op], target.operator)))
            if op in _BOUNDARY_FLIP:
                new = target.with_changes(operator=_op(_BOUNDARY_FLIP[op], target.operator))
                self._add(BugClass.BOUNDARY_SHIFT, target, new)
            for swap in _COMPARISON_SWAPS.get(op, []):
                self._add(BugClass.COMPARISON_OP, target, target.with_changes(operator=_op(swap, target.operator)))
        if len(node.comparisons) == 1 and type(node.comparisons[0].operator) in _BOUNDARY_FLIP:
            comparator = node.comparisons[0].comparator
            if _is_offsettable(comparator):
                for new in _offsets(comparator):
                    self._add(BugClass.BOUNDARY_SHIFT, comparator, new)

    def visit_BooleanOperation(self, node: cst.BooleanOperation) -> None:
        swap = cst.Or if isinstance(node.operator, cst.And) else cst.And
        self._add(BugClass.BOOLEAN_OP, node, node.with_changes(operator=_op(swap, node.operator)))
        if _is_none_check(node.left):
            self._add(BugClass.NONE_GUARD_OMISSION, node, _keep_parens(node, node.right))
        if _is_none_check(node.right):
            self._add(BugClass.NONE_GUARD_OMISSION, node, _keep_parens(node, node.left))
        parent = self._parent(node)
        in_test = isinstance(parent, (cst.If, cst.While, cst.IfExp, cst.Assert)) and getattr(parent, "test", None) is node
        if isinstance(node.operator, cst.Or) and not in_test:
            self._add(BugClass.NONE_GUARD_OMISSION, node, _keep_parens(node, node.left))

    def visit_BinaryOperation(self, node: cst.BinaryOperation) -> None:
        op = type(node.operator)
        if not isinstance(node.left, _STRINGY) and not isinstance(node.right, _STRINGY):
            for swap in _ARITH_SWAPS.get(op, []):
                self._add(BugClass.ARITHMETIC_OP, node, node.with_changes(operator=_op(swap, node.operator)))
        if op in (cst.Add, cst.Subtract) and isinstance(node.right, cst.Integer) and node.right.value == "1":
            self._add(BugClass.BOUNDARY_SHIFT, node, _keep_parens(node, node.left))
        if op is cst.Add and isinstance(node.left, cst.Integer) and node.left.value == "1":
            self._add(BugClass.BOUNDARY_SHIFT, node, _keep_parens(node, node.right))
        if op is cst.Divide:
            bare = node.with_changes(lpar=[], rpar=[])
            self._add(BugClass.TYPE_CONVERSION, node, cst.Call(func=cst.Name("int"), args=[cst.Arg(value=bare)]))

    def visit_AugAssign(self, node: cst.AugAssign) -> None:
        for swap in _AUG_SWAPS.get(type(node.operator), []):
            self._add(BugClass.ARITHMETIC_OP, node, node.with_changes(operator=_op(swap, node.operator)))

    def visit_UnaryOperation(self, node: cst.UnaryOperation) -> None:
        if isinstance(node.operator, cst.Not):
            self._add(BugClass.NEGATION, node, _keep_parens(node, node.expression))

    def visit_If(self, node: cst.If) -> None:
        self._negate_test(node.test)
        parent = self._parent(node)
        removable = isinstance(parent, (cst.IndentedBlock, cst.Module)) and len(parent.body) >= 2
        if removable and node.orelse is None and _stmt_count(node.body) <= 2 and _is_none_check(node.test):
            self._add(BugClass.NONE_GUARD_OMISSION, node, cst.RemovalSentinel.REMOVE)

    def visit_While(self, node: cst.While) -> None:
        self._negate_test(node.test)

    def visit_IfExp(self, node: cst.IfExp) -> None:
        self._negate_test(node.test)
        if _is_none_check(node.test):
            self._add(BugClass.NONE_GUARD_OMISSION, node, _keep_parens(node, node.body))
            self._add(BugClass.NONE_GUARD_OMISSION, node, _keep_parens(node, node.orelse))

    def visit_Integer(self, node: cst.Integer) -> None:
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            return
        for new in (value + 1, value - 1):
            if new >= 0:
                self._add(BugClass.BOUNDARY_SHIFT, node, node.with_changes(value=str(new)))

    def visit_SubscriptElement(self, node: cst.SubscriptElement) -> None:
        if isinstance(node.slice, cst.Index) and _is_offsettable(node.slice.value):
            for new in _offsets(node.slice.value):
                self._add(BugClass.BOUNDARY_SHIFT, node.slice.value, new)

    def visit_Call(self, node: cst.Call) -> None:
        positional = [i for i, a in enumerate(node.args) if a.keyword is None and not a.star]
        for i, j in zip(positional, positional[1:]):
            a, b = node.args[i], node.args[j]
            if _code(a.value) != _code(b.value):
                args = list(node.args)
                args[i], args[j] = a.with_changes(value=b.value), b.with_changes(value=a.value)
                self._add(BugClass.ARGUMENT_SWAP, node, node.with_changes(args=args))
        if not isinstance(node.func, cst.Name):
            return
        name = node.func.value
        if name in CONVERSION_FUNCS and len(node.args) == 1 and positional == [0]:
            self._add(BugClass.TYPE_CONVERSION, node, _keep_parens(node, _parenthesize(node.args[0].value)))
            for swap in _CONVERSION_SWAPS.get(name, []):
                self._add(BugClass.TYPE_CONVERSION, node, node.with_changes(func=node.func.with_changes(value=swap)))
        if name == "range":
            for i in positional:
                if _is_offsettable(node.args[i].value):
                    for new in _offsets(node.args[i].value):
                        self._add(BugClass.BOUNDARY_SHIFT, node.args[i].value, new)

    def visit_Return(self, node: cst.Return) -> None:
        value = node.value
        if value is None or _is_none_name(value):
            return
        self._add(BugClass.WRONG_RETURN, value, cst.Name("None"))
        parts: list[cst.BaseExpression] = []
        if isinstance(value, cst.BinaryOperation):
            parts = [value.left, value.right]
        elif isinstance(value, cst.Call):
            parts = [a.value for a in value.args if a.keyword is None and not a.star]
        elif isinstance(value, (cst.Subscript, cst.Attribute)):
            parts = [value.value]
        for part in parts:
            if isinstance(part, (cst.Name, cst.Attribute, cst.Subscript, cst.Call)):
                self._add(BugClass.WRONG_RETURN, value, part.with_changes(lpar=[], rpar=[]))

    def visit_Name(self, node: cst.Name) -> None:
        if node.value in ("True", "False"):
            flipped = "False" if node.value == "True" else "True"
            self._add(BugClass.NEGATION, node, node.with_changes(value=flipped))
            return
        ctx = self.get_metadata(ExpressionContextProvider, node, None)
        parent = self._parent(node)
        if ctx == ExpressionContext.STORE and not isinstance(parent, (cst.FunctionDef, cst.ClassDef)):
            self._bound.add(node.value)
        elif ctx == ExpressionContext.LOAD:
            is_func = isinstance(parent, cst.Call) and parent.func is node
            is_keyword = isinstance(parent, cst.Arg) and parent.keyword is node
            if not (is_func or is_keyword):
                self._loads.append(node)

    def finish(self) -> None:
        for node in self._loads:
            if node.value in self._bound:
                for other in sorted(self._bound - {node.value}):
                    self._add(BugClass.VARIABLE_MISUSE, node, node.with_changes(value=other))


class _Replace(cst.CSTTransformer):
    def __init__(self, site: _Site) -> None:
        super().__init__()
        self._site = site

    def on_leave(self, original_node: cst.CSTNode, updated_node: cst.CSTNode) -> Replacement:
        return self._site.replacement if original_node is self._site.target else updated_node


# --- public API ------------------------------------------------------------------------------


def _collect(code: str) -> tuple[cst.Module, dict[BugClass, list[_Site]]] | None:
    try:
        wrapper = MetadataWrapper(cst.parse_module(code))
    except cst.ParserSyntaxError:
        return None
    collector = _SiteCollector()
    wrapper.visit(collector)
    collector.finish()
    return wrapper.module, collector.sites


def site_counts(code: str) -> dict[BugClass, int]:
    """Number of candidate edit sites per class (before the classifier self-check)."""
    collected = _collect(code)
    return {c: len(s) for c, s in collected[1].items()} if collected else {}


def _try_sites(
    code: str, module: cst.Module, sites: list[_Site], cls: BugClass, rng: random.Random, max_tries: int
) -> Mutation | None:
    order = list(sites)
    rng.shuffle(order)
    for site in order[:max_tries]:
        try:
            mutant = module.visit(_Replace(site)).code
        except (cst.CSTValidationError, TypeError, ValueError):
            continue
        diagnosis = diagnose(code, mutant)
        span = changed_span(code, mutant)
        if diagnosis and diagnosis.bug_class == cls and span:
            return Mutation(cls, mutant, span, diagnosis.location)
    return None


def mutate(code: str, cls: BugClass, rng: random.Random, max_tries: int = 25) -> Mutation | None:
    """Apply one random `cls` mutation to `code`; None if the class has no valid site."""
    collected = _collect(code)
    if collected is None:
        return None
    module, sites = collected
    return _try_sites(code, module, sites[cls], cls, rng, max_tries)


def mutate_any(
    code: str, rng: random.Random, classes: Iterable[BugClass] = PRIORITY, max_tries: int = 25
) -> Mutation | None:
    """Pick classes in random order and return the first one that applies to `code`."""
    collected = _collect(code)
    if collected is None:
        return None
    module, sites = collected
    order = list(classes)
    rng.shuffle(order)
    for cls in order:
        mutation = _try_sites(code, module, sites[cls], cls, rng, max_tries)
        if mutation:
            return mutation
    return None
