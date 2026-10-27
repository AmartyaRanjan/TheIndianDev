import random

from advbug.game.prompts import apply_action, format_injector_output, parse_injector
from advbug.mutation.operators import mutate
from advbug.taxonomy import BugClass
from tests.samples import SAMPLES


def test_format_roundtrips_every_class():
    rng = random.Random(0)
    checked = 0
    for code in SAMPLES:
        for cls in BugClass:
            m = mutate(code, cls, rng)
            if m is None:
                continue
            text = format_injector_output(code, m.mutant, cls.value)
            action = parse_injector(text)
            assert action is not None and action.bug_class == cls.value
            assert apply_action(code, action) == m.mutant, (cls, text)
            checked += 1
    assert checked > 20
