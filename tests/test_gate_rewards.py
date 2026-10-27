import math

import pytest

from advbug.data.schema import CleanFunction
from advbug.game.gate import Gate, GateConfig, GateResult
from advbug.game.prompts import DetectorPrediction
from advbug.game.rewards import DiversityTracker, RewardConfig, detector_reward, fool_rate, injector_reward
from advbug.sandbox import SandboxPool
from advbug.taxonomy import NO_BUG

FN = CleanFunction(
    id="t1",
    source="test",
    code=(
        "def total(xs, scale=1):\n"
        '    """Sum of xs times scale."""\n'
        "    s = 0\n"
        "    for x in xs:  # accumulate\n"
        "        s += x\n"
        "    return s * scale\n"
    ),
    tests=(
        "def test_basic():\n    assert total([1, 2, 3]) == 6\n"
        "def test_scale():\n    assert total([1, 2], 3) == 9\n"
        "def test_empty():\n    assert total([]) == 0\n"
    ),
)


def out(cls: str, lines: str, new: str) -> str:
    return f"<type>{cls}</type>\n<lines>{lines}</lines>\n<new>\n{new}\n</new>"


@pytest.fixture(scope="module")
def gate():
    with SandboxPool(workers=2) as sandbox:
        yield Gate(GateConfig(), sandbox)


@pytest.mark.parametrize(
    ("text", "failed"),
    [
        (out("arithmetic_op", "5-5", "        s -= x"), None),  # a real bug
        ("no tags here", "G1_format"),
        (out("not_a_class", "5-5", "        s -= x"), "G1_format"),
        (out("arithmetic_op", "9-9", "        s -= x"), "G1_format"),  # line out of range
        (out("arithmetic_op", "5-5", "        s += x"), "G2_size"),  # no-op edit
        (out("arithmetic_op", "5-5", "        s -= x +"), "G3_compile"),
        (out("arithmetic_op", "5-5", "        s -= x  # bug"), "G4_artifact"),  # comment leaks the answer
        (out("arithmetic_op", "4-4", "    for x in xs:  # accumulate!"), "G4_artifact"),  # changed comment
        (out("negation", "5-5", "        s -= x"), "G5_label"),  # wrong claimed class
        (out("arithmetic_op", "6-6", "    return scale * s"), "G5_label"),  # operand swap is not in the taxonomy
        (out("boundary_shift", "3-3", "    s = 0 * 1"), "G5_label"),
        (out("boundary_shift", "6-6", "    return s * scale * 1"), "G5_label"),
        (out("variable_misuse", "6-6", "    return s * s"), None),
    ],
)
def test_gate_checks(gate, text, failed):
    (r,) = gate.check([(FN, text)])
    assert r.failed_check == failed, r
    assert r.ok == (failed is None)


def test_equivalent_mutant_rejected_by_tests(gate):
    fn = CleanFunction("t2", "test", "def f(x):\n    y = x * 1\n    return y\n", "def test_a():\n    assert f(2) == 2\n" * 1)
    (r,) = gate.check([(fn, out("boundary_shift", "2-2", "    y = x * 0 + x"))])
    assert r.failed_check == "G5_label"  # not a single +-1 edit
    (r,) = gate.check([(fn, out("arithmetic_op", "2-2", "    y = x // 1"))])
    assert r.failed_check == "G6_equivalent"  # x // 1 == x * 1 for the tested ints


def test_gate_disabled_is_a3_arm():
    gate = Gate(GateConfig(enabled=False), sandbox=None)
    (r,) = gate.check([(FN, out("negation", "5-5", "        s -= x +"))])  # neither compiles nor matches
    assert r.ok and r.bug_lines == (5,)
    (r,) = gate.check([(FN, "garbage")])
    assert r.failed_check == "G1_format"


def test_detector_reward_binary_and_partial():
    assert detector_reward("negation", (4,), DetectorPrediction("negation", 4)) == 1.0
    assert detector_reward("negation", (4,), DetectorPrediction("negation", 5)) == 0.0
    assert detector_reward(NO_BUG, (), DetectorPrediction(NO_BUG, None)) == 1.0
    assert detector_reward(NO_BUG, (), DetectorPrediction("negation", 2)) == 0.0
    assert detector_reward("negation", (4,), None) == 0.0
    assert detector_reward("negation", (4,), DetectorPrediction("negation", 5), partial_credit=True) == 0.75
    assert detector_reward("negation", (4,), DetectorPrediction("arithmetic_op", 4), partial_credit=True) == 0.5
    assert detector_reward("negation", (4,), DetectorPrediction(NO_BUG, None), partial_credit=True) == 0.0


def test_injector_reward_and_diversity():
    cfg = RewardConfig(lambda_div=0.1, lambda_sub=0.0)
    valid = GateResult(True, None, edit_chars=10)
    assert injector_reward(GateResult(False, "G5_label"), 1.0, 1.0, cfg) == -1.0
    assert math.isclose(injector_reward(valid, fool_rate([True, False, False, False]), 0.5, cfg), 0.75 + 0.05)
    assert math.isclose(injector_reward(valid, 1.0, 0.0, cfg, fool_cap=0.5), 0.5)
    tracker = DiversityTracker(momentum=0.5)
    assert math.isclose(tracker.bonus("negation"), 1.0)
    for _ in range(5):
        tracker.update(["negation"] * 9 + ["boolean_op"])
    assert tracker.bonus("negation") < 0.1 < tracker.bonus("boolean_op") <= 1.0
    assert tracker.bonus("argument_swap") == 1.0
    assert tracker.entropy() < math.log(10)
