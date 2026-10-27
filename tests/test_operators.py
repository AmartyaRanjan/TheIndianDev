import random

import pytest

from advbug.mutation.classify import classify
from advbug.mutation.lines import changed_span, edited_line_count
from advbug.mutation.operators import mutate, mutate_any, site_counts
from advbug.taxonomy import BugClass
from tests.samples import SAMPLES


def _untouched_lines_identical(original: str, mutant: str, span: tuple[int, int]) -> bool:
    """Every line outside the edit is byte-identical (formatting and comments preserved)."""
    a, b = original.splitlines(), mutant.splitlines()
    first, last = span
    shift = len(b) - len(a)
    before_ok = a[: first - 1] == b[: first - 1]
    after_a = a[last - shift :] if last - shift >= 0 else a
    return before_ok and after_a == b[last:]


@pytest.mark.parametrize("cls", list(BugClass))
def test_every_class_applies_somewhere(cls):
    rng = random.Random(0)
    hits = [m for code in SAMPLES if (m := mutate(code, cls, rng))]
    assert hits, f"no sample admits a {cls} mutation"


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("sample", range(len(SAMPLES)))
def test_mutations_are_labelled_minimal_and_format_preserving(sample, seed):
    code = SAMPLES[sample]
    rng = random.Random(seed)
    for cls in BugClass:
        m = mutate(code, cls, rng)
        if m is None:
            continue
        assert m.mutant != code
        compile(m.mutant, "<mutant>", "exec")
        assert classify(code, m.mutant) == cls
        assert m.lines == changed_span(code, m.mutant)
        assert edited_line_count(code, m.mutant) <= 3
        assert _untouched_lines_identical(code, m.mutant, m.lines), (cls, m.mutant)
        assert "# skip missing" in m.mutant or "# skip missing" not in code  # comments survive


def test_mutate_any_and_determinism():
    code = SAMPLES[2]
    first = mutate_any(code, random.Random(7))
    second = mutate_any(code, random.Random(7))
    assert first is not None and first == second


def test_site_counts_and_unparsable_input():
    assert site_counts(SAMPLES[0])[BugClass.BOUNDARY_SHIFT] > 0
    assert mutate("def broken(:\n", BugClass.NEGATION, random.Random(0)) is None
    assert site_counts("def broken(:\n") == {}
