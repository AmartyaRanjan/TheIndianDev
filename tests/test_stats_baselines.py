import math

from advbug.data.schema import Example
from advbug.eval.baselines import LINTER_CLASS, AnswerCache, linter_predictions
from advbug.eval.stats import hierarchical_bootstrap, holm, mcnemar, paired_bootstrap
from advbug.taxonomy import NO_BUG


def test_mcnemar_and_bootstrap():
    a = [True] * 30 + [False] * 10
    b = [True] * 10 + [False] * 30
    assert mcnemar(a, b) < 1e-4
    assert mcnemar(a, a) == 1.0
    mean, lo, hi = paired_bootstrap(a, b, resamples=2000)
    assert math.isclose(mean, 0.5) and lo < 0.5 < hi and lo > 0


def test_hierarchical_bootstrap_and_holm():
    mean, lo, hi = hierarchical_bootstrap([[True, False] * 10, [True, True] * 10], resamples=500)
    assert math.isclose(mean, 0.75) and lo <= mean <= hi
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_linter_flags_undefined_name_line():
    bug = Example("b", "t", "def f(x):\n    y = x + 1\n    return z + y\n", (3,), "variable_misuse")
    clean = Example("c", "t", "def f(x):\n    return x\n", (), NO_BUG)
    p_bug, p_clean = linter_predictions([bug, clean])
    assert p_bug.bug_class == LINTER_CLASS and p_bug.line == 3
    assert p_clean.bug_class == NO_BUG


def test_claude_cache_roundtrip(tmp_path):
    cache = AnswerCache(tmp_path / "c.jsonl")
    cache.add({"id": "x", "text": "<type>no_bug</type>"})
    assert AnswerCache(tmp_path / "c.jsonl").rows["x"]["text"] == "<type>no_bug</type>"
