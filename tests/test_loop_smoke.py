"""End-to-end CPU smoke test of the co-training loop: tiny random model, scripted generations,
real gate + sandbox, real GRPO updates, adapter publishing, pools and checkpoint/resume."""

import difflib
import random
import re
from pathlib import Path

import pytest
import torch

from advbug.config import Config
from advbug.data.schema import CleanFunction
from advbug.game.gate import Gate, GateConfig
from advbug.mutation.operators import mutate_any
from advbug.rl.loop import CoTrainer
from advbug.rl.rollout import Completion, _per_prompt
from advbug.sandbox import SandboxPool
from advbug.taxonomy import PRIORITY
from tests.test_rl_core import tiny_policy

_CODE_BLOCK = re.compile(r"```python\n(.*?)\n```", re.DOTALL)

CORPUS = [
    CleanFunction(
        f"f{i}",
        "test",
        f"def f{i}(xs, k):\n    total = 0\n    for x in xs:\n        if x > k:\n            total += x * {i + 2}\n    return total\n",
        f"def test_a():\n    assert f{i}([1, 5, 9], 4) == {14 * (i + 2)}\n"
        f"def test_b():\n    assert f{i}([], 0) == 0\n"
        f"def test_c():\n    assert f{i}([3, 4], 3) == {4 * (i + 2)}\n",
    )
    for i in range(6)
]


def injector_answer(code: str, rng: random.Random) -> str:
    m = mutate_any(code, rng) if rng.random() < 0.7 else None
    if m is None:
        return "I would rather not."
    a, b = code.splitlines(), m.mutant.splitlines()
    ops = [op for op in difflib.SequenceMatcher(a=a, b=b).get_opcodes() if op[0] != "equal"]
    i1, i2, j1, j2 = ops[0][1], ops[-1][2], ops[0][3], ops[-1][4]
    new = "\n".join(b[j1:j2])
    return f"<type>{m.bug_class.value}</type>\n<lines>{i1 + 1}-{max(i2, i1 + 1)}</lines>\n<new>\n{new}\n</new>"


class ScriptedGenerator:
    """Real-looking outputs so every code path (valid, invalid, varied rewards) is exercised."""

    def __init__(self, tokenizer, seed=0):
        self.tok, self.rng = tokenizer, random.Random(seed)

    def generate(self, prompts, adapters, n, temperature, max_tokens):
        out = []
        for prompt, _ in zip(prompts, _per_prompt(adapters, len(prompts))):
            text = self.tok.decode(prompt)
            code = re.sub(r"(?m)^\s*\d+\| ?", "", _CODE_BLOCK.findall(text)[-1]) + "\n"
            group = []
            for _ in range(n):
                if "Introduce exactly one" in text:
                    answer = injector_answer(code, self.rng)
                else:
                    line = self.rng.randint(1, len(code.splitlines()))
                    answer = self.rng.choice(
                        ["<type>no_bug</type>", f"<line>{line}</line><type>{self.rng.choice(PRIORITY).value}</type>", "?"]
                    )
                ids = self.tok(answer, add_special_tokens=False)["input_ids"][: max_tokens - 1] + [self.tok.eos_token_id]
                group.append(Completion(answer, tuple(ids)))
            out.append(group)
        return out


@pytest.fixture(scope="module")
def sandbox():
    with SandboxPool(workers=2) as pool:
        yield pool


def make_trainer(tokenizer, sandbox, run_dir: Path, **overrides) -> CoTrainer:
    cfg = Config(out_dir=str(run_dir), backend="hf", steps=4, sandbox_workers=2)
    cfg.injector.functions_per_step, cfg.injector.group_size = 3, 4
    cfg.detector.group_size, cfg.detector.train_mutants, cfg.detector.train_clean = 4, 6, 2
    cfg.optim.micro_batch, cfg.optim.lr = 4, 1e-3
    cfg.pool.every, cfg.pool.keep = 2, 2
    for key, value in overrides.items():
        section, _, name = key.partition("__")
        setattr(getattr(cfg, section), name, value)
    policy = tiny_policy(tokenizer)
    gate = Gate(GateConfig(enabled=cfg.gate.enabled), sandbox if cfg.gate.enabled or cfg.injector.mode == "uniform" else None)
    return CoTrainer(cfg, policy, ScriptedGenerator(tokenizer), gate, CORPUS)


def test_adversarial_steps_pools_and_resume(tokenizer, sandbox, tmp_path):
    trainer = make_trainer(tokenizer, sandbox, tmp_path)
    before = [p.detach().clone() for p in trainer.policy.adapter_parameters("injector")]
    history = [trainer.step()[0] for _ in range(4)]
    m = history[-1]
    for key in ("gate/pass_rate", "fool_rate", "injector/class_entropy", "detector/synthetic_acc", "detector/zero_variance_groups"):
        assert key in m, key
    assert any(h.get("injector/trained_sequences", 0) > 0 for h in history)
    assert any(h.get("detector/trained_sequences", 0) > 0 for h in history)
    after = trainer.policy.adapter_parameters("injector")
    assert any(not torch.equal(a, b) for a, b in zip(after, before))  # the injector actually learned
    assert len(trainer.pools["injector"]) == 2 and len(trainer.pools["detector"]) == 2
    on_disk = {p.name for p in (tmp_path / "full-s0" / "adapters" / "detector").iterdir()}
    assert len(on_disk) <= 2 + trainer.cfg.pool.keep  # old versions are cleaned up

    trainer.save_checkpoint()
    resumed = make_trainer(tokenizer, sandbox, tmp_path)
    resumed.load_checkpoint(trainer.cfg.run_dir / "checkpoint.pt")
    assert resumed.step_idx == 4 and resumed.current == trainer.current
    for a, b in zip(resumed.policy.adapter_parameters("detector"), trainer.policy.adapter_parameters("detector")):
        assert torch.allclose(a, b)
    assert resumed.step()[0]["step"] == 4


@pytest.mark.parametrize(
    "overrides",
    [
        {"injector__mode": "uniform", "pool__enabled": False},  # A1-uniform
        {"injector__mode": "frozen", "pool__enabled": False},  # A1-frozen
        {"gate__enabled": False},  # A3
        {"curriculum__enabled": True, "rewards__partial_credit": True},  # A7 + A6
    ],
)
def test_other_arms_run(tokenizer, sandbox, tmp_path, overrides):
    trainer = make_trainer(tokenizer, sandbox, tmp_path, **overrides)
    frozen = [p.detach().clone() for p in trainer.policy.adapter_parameters("injector")]
    for _ in range(2):
        metrics, _ = trainer.step()
    assert "detector/reward_mean" in metrics
    if overrides.get("injector__mode") in ("uniform", "frozen"):
        assert all(torch.equal(a, b) for a, b in zip(trainer.policy.adapter_parameters("injector"), frozen))
