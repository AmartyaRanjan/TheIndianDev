import math

from advbug.data.schema import Example
from advbug.eval.metrics import expected_calibration_error, is_correct, per_class_f1, score
from advbug.game.prompts import (
    DetectorPrediction,
    InjectorAction,
    apply_action,
    detector_messages,
    injector_messages,
    number_lines,
    parse_detector,
    parse_injector,
)
from advbug.mutation.classify import classify
from advbug.taxonomy import NO_BUG

CODE = "def f(xs):\n    total = 0\n    for x in xs:\n        total += x\n    return total\n"


def test_number_lines_pads_width():
    numbered = number_lines("\n".join(f"x{i} = {i}" for i in range(12)) + "\n")
    assert numbered.splitlines()[0] == " 1| x0 = 0" and numbered.splitlines()[11] == "12| x11 = 11"


def test_messages_contain_code_and_taxonomy():
    user = detector_messages(CODE)[-1]["content"]
    assert "3|     for x in xs:" in user and "boundary_shift" in user and "no_bug" in user
    assert len(detector_messages(CODE, few_shot=True)) == 6
    assert "<lines>A-B</lines>" in injector_messages(CODE)[-1]["content"]


def test_parse_detector():
    assert parse_detector("<line>4</line><type>arithmetic_op</type>") == DetectorPrediction("arithmetic_op", 4)
    assert parse_detector("I think... <type>no_bug</type>") == DetectorPrediction(NO_BUG, None)
    assert parse_detector("<type>made_up</type><line>3</line>") is None
    assert parse_detector("<type>negation</type>") is None  # bug claimed without a line
    assert parse_detector("line 4 looks wrong") is None


def test_parse_and_apply_injector():
    text = "<type>arithmetic_op</type>\n<lines>4-4</lines>\n<new>\n        total -= x\n</new>"
    action = parse_injector(text)
    assert action == InjectorAction("arithmetic_op", 4, 4, ("        total -= x",))
    mutant = apply_action(CODE, action)
    assert classify(CODE, mutant) == "arithmetic_op"


def test_apply_restores_dropped_indentation_and_rejects_bad_ranges():
    action = parse_injector("<type>arithmetic_op</type><lines>4</lines><new>total -= x</new>")
    assert apply_action(CODE, action).splitlines()[3] == "        total -= x"
    assert apply_action(CODE, InjectorAction("negation", 0, 1, ())) is None
    assert apply_action(CODE, InjectorAction("negation", 3, 9, ())) is None
    deleted = apply_action(CODE, parse_injector("<type>x</type><lines>2-2</lines><new>\n</new>"))
    assert "total = 0" not in deleted


def _ex(bug_class, lines=()):
    return Example(id="e", source="t", code=CODE, bug_lines=lines, bug_class=bug_class)


def test_score_and_correctness():
    examples = [_ex("arithmetic_op", (4,)), _ex("negation", (3,)), _ex(NO_BUG), _ex("other", (5,))]
    preds = [
        DetectorPrediction("arithmetic_op", 4),  # fully correct
        DetectorPrediction("negation", 5),  # right type, line off by 2
        DetectorPrediction("negation", 2),  # false positive
        DetectorPrediction("variable_misuse", 5),  # out-of-taxonomy bug: localization only
    ]
    assert is_correct(examples[0], preds[0]) and not is_correct(examples[1], preds[1])
    s = score(examples, preds)
    assert s["line_exact"] == 2 / 3 and s["line_within_3"] == 1.0
    assert s["type_acc"] == 1.0 and s["joint_acc"] == 0.5
    assert s["false_positive_rate"] == 1.0 and s["format_failure"] == 0.0
    assert math.isclose(s["accuracy"], 1 / 3)


def test_f1_and_ece():
    assert per_class_f1([("made_up", "made_up")]) == {}  # labels outside the taxonomy are ignored
    f1 = per_class_f1([("negation", "negation"), ("negation", NO_BUG), (NO_BUG, NO_BUG)])
    assert math.isclose(f1["negation"], 2 / 3) and math.isclose(f1[NO_BUG], 2 / 3)
    assert math.isclose(expected_calibration_error([1.0, 1.0], [True, True]), 0.0)
    assert math.isclose(expected_calibration_error([0.9, 0.9], [False, False]), 0.9)
