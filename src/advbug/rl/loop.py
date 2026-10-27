"""The co-training step (plan §4.5) and its bookkeeping: adapters, pools, checkpoints, evaluation.

One step:
  1. B clean functions -> K injector samples each -> gate        (A1 arms: frozen or AST-engine mutants)
  2. a past injector adds a share of the detector's mutant inputs  (pools on)
  3. the detector answers K_D times for every valid mutant and some clean functions. Those
     answers are both each mutant's fool rate (injector reward) and the detector's GRPO groups
  4. a past detector (greedy) is blended into the fool rate        (pools on)
  5. GRPO update of the injector (groups = K mutants of a function) and of the detector
  6. publish the new adapters to the generator; snapshot pools every `pool.every` steps
"""

from __future__ import annotations

import json
import math
import random
import shutil
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from advbug.config import Config
from advbug.data.schema import CleanFunction, Example
from advbug.data.synthetic import uniform_mutants
from advbug.eval.metrics import score
from advbug.game.gate import CHECKS, Gate, GateConfig, GateResult
from advbug.game.prompts import detector_messages, injector_messages, parse_detector
from advbug.game.rewards import DiversityTracker, RewardConfig, detector_reward, fool_rate, injector_reward
from advbug.rl.grpo import LossConfig, group_advantages, policy_loss, zero_variance_fraction
from advbug.rl.policy import Policy
from advbug.rl.rollout import AdapterRef, Completion, Generator
from advbug.taxonomy import NO_BUG

PLAYERS = ("injector", "detector")


@dataclass(frozen=True)
class _Mutant:
    fn: CleanFunction
    gate: GateResult  # always ok
    group: int  # row in this step's injector batch; -1 for mutants from a past injector
    sample: int


class CoTrainer:
    def __init__(
        self, cfg: Config, policy: Policy, generator: Generator, gate: Gate, corpus: Sequence[CleanFunction]
    ) -> None:
        self.cfg, self.policy, self.gen, self.gate = cfg, policy, generator, gate
        self.corpus = list(corpus)
        self.rng = random.Random(cfg.seed)
        torch.manual_seed(cfg.seed)
        o = cfg.optim
        self.loss_cfg = LossConfig(kl_beta=o.kl_beta, clip_eps=o.clip_eps)
        self.reward_cfg = RewardConfig(
            cfg.rewards.partial_credit,
            cfg.rewards.invalid_penalty,
            cfg.rewards.lambda_div,
            cfg.rewards.lambda_sub,
            GateConfig().max_edit_chars,
        )
        self.opt = {p: torch.optim.AdamW(policy.adapter_parameters(p), lr=o.lr, weight_decay=0.0) for p in PLAYERS}
        warmup = max(1, o.warmup_steps)
        self.sched = {p: torch.optim.lr_scheduler.LambdaLR(self.opt[p], lambda s: min(1.0, (s + 1) / warmup)) for p in PLAYERS}
        self.diversity = DiversityTracker()
        self.adapter_dir = cfg.run_dir / "adapters"
        self.pools: dict[str, list[AdapterRef]] = {p: [] for p in PLAYERS}
        self.current: dict[str, AdapterRef] = {}
        self._on_disk: dict[str, list[AdapterRef]] = {p: [] for p in PLAYERS}
        self.step_idx = 0
        self.detector_acc: float | None = None  # moving average of on-policy synthetic accuracy
        if cfg.injector.init_adapter:  # shared format warmup (scripts/warmup_injector.py)
            policy.load_adapter_weights("injector", cfg.injector.init_adapter)
        for p in PLAYERS:
            self._publish(p)
        self.initial_injector = self.current["injector"]

    # --- adapters ------------------------------------------------------------------------

    def _publish(self, player: str) -> None:
        root = self.adapter_dir / player / f"v{self.step_idx}"
        path = self.policy.save_adapter(player, root)
        ref = AdapterRef(player, self.step_idx, str(path))
        self.current[player] = ref
        self._on_disk[player].append(ref)
        keep = {r.path for r in self.pools[player]} | {getattr(self, "initial_injector", ref).path}
        while len(self._on_disk[player]) > 2:  # the generator may still be serving the previous version
            old = self._on_disk[player].pop(0)
            if old.path not in keep:
                shutil.rmtree(self.adapter_dir / player / f"v{old.version}", ignore_errors=True)

    def _snapshot_pools(self) -> None:
        for p in PLAYERS:
            if p == "injector" and self.cfg.injector.mode != "adversarial":
                continue
            self.pools[p].append(self.current[p])
            if len(self.pools[p]) > self.cfg.pool.keep:
                evicted = self.pools[p].pop(0)
                if evicted not in self._on_disk[p] and evicted != self.initial_injector:
                    shutil.rmtree(self.adapter_dir / p / f"v{evicted.version}", ignore_errors=True)

    # --- one step ----------------------------------------------------------------------------

    def step(self) -> tuple[dict[str, float], list[dict[str, object]]]:
        cfg, inj, det = self.cfg, self.cfg.injector, self.cfg.detector
        metrics: dict[str, float] = {"step": self.step_idx}
        adversarial = inj.mode == "adversarial"

        # 1. candidate mutants
        inj_prompts: list[list[int]] = []
        inj_outs: list[list[Completion]] = []
        if inj.mode == "uniform":
            fns = self.rng.sample(self.corpus, min(len(self.corpus), inj.functions_per_step * inj.group_size))
            mutants = [_Mutant(fn, r, i, 0) for i, (fn, r) in enumerate(uniform_mutants(fns, self.gate, self.rng))]
            metrics["gate/pass_rate"] = len(mutants) / len(fns)
        else:
            fns = self.rng.sample(self.corpus, inj.functions_per_step)
            ref = self.current["injector"] if adversarial else self.initial_injector
            inj_prompts = [self.policy.prompt_ids(injector_messages(fn.code)) for fn in fns]
            inj_outs = self.gen.generate(inj_prompts, ref, inj.group_size, inj.temperature, inj.max_tokens)
            results = self.gate.check([(fn, c.text) for fn, group in zip(fns, inj_outs) for c in group])
            k = inj.group_size
            mutants = [_Mutant(fns[i // k], r, i // k, i % k) for i, r in enumerate(results) if r.ok]
            metrics["gate/pass_rate"] = len(mutants) / len(results)
            failed = Counter(r.failed_check for r in results if not r.ok)
            metrics.update({f"gate/{c}": failed[c] / len(results) for c in CHECKS})

        # 2. a past injector's mutants for the detector
        pooled: list[_Mutant] = []
        if cfg.pool.enabled and adversarial and self.pools["injector"]:
            extra = self.rng.sample(self.corpus, max(1, round(cfg.pool.detector_data_fraction * det.train_mutants)))
            ref = self.rng.choice(self.pools["injector"])
            prompts = [self.policy.prompt_ids(injector_messages(fn.code)) for fn in extra]
            outs = self.gen.generate(prompts, ref, 1, inj.temperature, inj.max_tokens)
            res = self.gate.check([(fn, o[0].text) for fn, o in zip(extra, outs)])
            pooled = [_Mutant(fn, r, -1, 0) for fn, r in zip(extra, res) if r.ok]

        # 3. detector answers
        clean = self.rng.sample(self.corpus, det.train_clean)
        items = [(m.gate.mutant, m.gate.bug_lines, m.gate.claimed_class) for m in mutants + pooled]
        items += [(fn.code, (), NO_BUG) for fn in clean]
        det_prompts = [self.policy.prompt_ids(detector_messages(code)) for code, _, _ in items]
        det_outs = self.gen.generate(det_prompts, self.current["detector"], det.group_size, det.temperature, det.max_tokens)
        preds = [[parse_detector(c.text) for c in group] for group in det_outs]
        partial = cfg.rewards.partial_credit
        det_rewards = [[detector_reward(cls, lines, p, partial) for p in ps] for (_, lines, cls), ps in zip(items, preds)]
        correct = [[detector_reward(cls, lines, p) == 1.0 for p in ps] for (_, lines, cls), ps in zip(items, preds)]
        n_cur, n_pool = len(mutants), len(pooled)

        # 4. fool rates (injector reward signal)
        fool_now = [fool_rate(correct[i]) for i in range(n_cur)]
        fool = list(fool_now)
        if cfg.pool.enabled and adversarial and self.pools["detector"] and mutants:
            ref = self.rng.choice(self.pools["detector"])
            outs = self.gen.generate(det_prompts[:n_cur], ref, 1, 0.0, det.max_tokens)
            w = cfg.pool.detector_weight
            for i, (o, (_, lines, cls)) in enumerate(zip(outs, items[:n_cur])):
                past_correct = detector_reward(cls, lines, parse_detector(o[0].text)) == 1.0
                fool[i] = (1 - w) * fool_now[i] + w * float(not past_correct)

        # 5. injector rewards [B, K]
        dropped: set[int] = set()
        if adversarial:
            rewards = torch.full((len(fns), inj.group_size), self.reward_cfg.invalid_penalty)
            competent = self.detector_acc is not None and self.detector_acc >= cfg.curriculum.threshold
            cap = cfg.curriculum.cap if cfg.curriculum.enabled and not competent else None
            for i, m in enumerate(mutants):
                if cfg.curriculum.enabled and competent and fool_now[i] == 0.0:
                    dropped.add(i)  # A7 phase B: trivially caught mutants count as invalid
                    continue
                bonus = self.diversity.bonus(m.gate.claimed_class)
                rewards[m.group, m.sample] = injector_reward(m.gate, fool[i], bonus, self.reward_cfg, fool_cap=cap)
            self.diversity.update([m.gate.claimed_class for m in mutants])
            metrics["curriculum/competent"] = float(competent)

        # 6. detector training groups
        current_idx = [i for i in range(n_cur) if i not in dropped]
        self.rng.shuffle(current_idx)
        n_from_pool = min(n_pool, round(cfg.pool.detector_data_fraction * det.train_mutants)) if n_pool else 0
        train_idx = current_idx[: det.train_mutants - n_from_pool] + list(range(n_cur, n_cur + n_from_pool))
        train_idx += list(range(n_cur + n_pool, len(items)))

        # 7. updates
        if adversarial:
            metrics.update(self._update("injector", inj_prompts, inj_outs, rewards))
        det_reward_tensor = torch.tensor([det_rewards[i] for i in train_idx], dtype=torch.float32)
        metrics.update(
            self._update("detector", [det_prompts[i] for i in train_idx], [det_outs[i] for i in train_idx], det_reward_tensor)
        )

        # 8. publish
        self.step_idx += 1
        for p in PLAYERS if adversarial else ("detector",):
            self._publish(p)
        if cfg.pool.enabled and self.step_idx % cfg.pool.every == 0:
            self._snapshot_pools()

        # 9. dynamics (PDF §3.2)
        classes = Counter(m.gate.claimed_class for m in mutants)
        total = sum(classes.values())
        metrics["injector/class_entropy"] = -sum(c / total * math.log(c / total) for c in classes.values()) if total else 0.0
        metrics["injector/classes"] = len(classes)
        metrics["fool_rate"] = sum(fool_now) / n_cur if n_cur else float("nan")
        on_policy = [sum(correct[i]) / len(correct[i]) for i in range(n_cur)]
        if on_policy:
            acc = sum(on_policy) / len(on_policy)
            metrics["detector/synthetic_acc"] = acc
            self.detector_acc = acc if self.detector_acc is None else 0.9 * self.detector_acc + 0.1 * acc
        clean_ok = [sum(correct[i]) / len(correct[i]) for i in range(n_cur + n_pool, len(items))]
        metrics["detector/clean_acc"] = sum(clean_ok) / len(clean_ok) if clean_ok else float("nan")
        metrics["detector/format_failure"] = sum(p is None for ps in preds for p in ps) / max(1, sum(map(len, preds)))
        samples = [
            {"step": self.step_idx, "fn": m.fn.id, "class": m.gate.claimed_class, "fool": fool_now[i], "mutant": m.gate.mutant}
            for i, m in enumerate(mutants[:4])
        ]
        return metrics, samples

    def _update(
        self, player: str, prompts: Sequence[Sequence[int]], outs: Sequence[Sequence[Completion]], rewards: torch.Tensor
    ) -> dict[str, float]:
        adv = group_advantages(rewards)
        stats = {
            f"{player}/reward_mean": float(rewards.mean()),
            f"{player}/reward_std": float(rewards.std(unbiased=False)),
            f"{player}/zero_variance_groups": zero_variance_fraction(rewards),
            f"{player}/advantage_var": float(adv.var(unbiased=False)),
        }
        seqs = [
            (prompts[g], outs[g][k].token_ids, float(adv[g, k]))
            for g in range(len(prompts))
            for k in range(len(outs[g]))
            if outs[g][k].token_ids
        ]
        if self.cfg.optim.skip_zero_advantage:
            seqs = [s for s in seqs if s[2] != 0.0]
        stats[f"{player}/trained_sequences"] = len(seqs)
        if not seqs:
            return stats
        total_tokens = sum(len(s[1]) for s in seqs)
        params = self.policy.adapter_parameters(player)
        self.policy.use(player)
        self.policy.model.train()
        kl_total = 0.0
        micro = self.cfg.optim.micro_batch
        for start in range(0, len(seqs), micro):
            chunk = seqs[start : start + micro]
            p, c, a = zip(*chunk)
            logp, mask = self.policy.completion_logprobs(p, c, player)
            ref = None
            if self.loss_cfg.kl_beta > 0:
                with torch.no_grad():
                    ref, _ = self.policy.completion_logprobs(p, c, None)
            advantages = torch.tensor(a, device=logp.device, dtype=logp.dtype)
            loss, st = policy_loss(logp, advantages, mask, total_tokens, self.loss_cfg, ref_logp=ref)
            loss.backward()
            kl_total += st["kl"] * float(mask.sum())
        stats[f"{player}/grad_norm"] = float(torch.nn.utils.clip_grad_norm_(params, self.cfg.optim.max_grad_norm))
        self.opt[player].step()
        self.sched[player].step()
        self.opt[player].zero_grad(set_to_none=True)
        self.policy.model.eval()
        stats[f"{player}/kl"] = kl_total / total_tokens
        return stats

    # --- evaluation ----------------------------------------------------------------------------

    def evaluate(self, examples: Sequence[Example]) -> dict[str, float]:
        prompts = [self.policy.prompt_ids(detector_messages(e.code)) for e in examples]
        outs = self.gen.generate(prompts, self.current["detector"], 1, 0.0, self.cfg.detector.max_tokens)
        return score(list(examples), [parse_detector(o[0].text) for o in outs])

    # --- checkpoints ---------------------------------------------------------------------------

    def save_checkpoint(self) -> Path:
        path = self.cfg.run_dir / "checkpoint.pt"
        state = {
            "step": self.step_idx,
            "current": {p: r.__dict__ for p, r in self.current.items()},
            "pools": {p: [r.__dict__ for r in refs] for p, refs in self.pools.items()},
            "on_disk": {p: [r.__dict__ for r in refs] for p, refs in self._on_disk.items()},
            "initial_injector": self.initial_injector.__dict__,
            "optim": {p: self.opt[p].state_dict() for p in PLAYERS},
            "sched": {p: self.sched[p].state_dict() for p in PLAYERS},
            "diversity": self.diversity.freq,
            "detector_acc": self.detector_acc,
            "rng": self.rng.getstate(),
            "torch_rng": torch.get_rng_state(),
        }
        tmp = path.with_suffix(".tmp")
        torch.save(state, tmp)
        tmp.replace(path)
        return path

    def load_checkpoint(self, path: Path) -> None:
        state = torch.load(path, weights_only=False)
        self.step_idx = state["step"]
        self.current = {p: AdapterRef(**r) for p, r in state["current"].items()}
        self.pools = {p: [AdapterRef(**r) for r in refs] for p, refs in state["pools"].items()}
        self._on_disk = {p: [AdapterRef(**r) for r in refs] for p, refs in state["on_disk"].items()}
        self.initial_injector = AdapterRef(**state["initial_injector"])
        for p in PLAYERS:
            self.policy.load_adapter_weights(p, self.current[p].path)
            self.opt[p].load_state_dict(state["optim"][p])
            self.sched[p].load_state_dict(state["sched"][p])
        self.diversity.freq = state["diversity"]
        self.detector_acc = state["detector_acc"]
        self.rng.setstate(state["rng"])
        torch.set_rng_state(state["torch_rng"])


class RunLog:
    """Append-only JSONL logs; figures are built from these, W&B is optional viewing."""

    def __init__(self, run_dir: Path, config: dict[str, object], use_wandb: bool = False) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        self.metrics = (run_dir / "metrics.jsonl").open("a")
        self.samples = (run_dir / "samples.jsonl").open("a")
        self.wandb = None
        if use_wandb:
            import wandb

            self.wandb = wandb.init(project="advbug", name=run_dir.name, config=config, dir=str(run_dir))

    def log(self, metrics: dict[str, float], samples: Sequence[dict[str, object]] = ()) -> None:
        self.metrics.write(json.dumps(metrics) + "\n")
        self.metrics.flush()
        for s in samples:
            self.samples.write(json.dumps(s) + "\n")
        self.samples.flush()
        if self.wandb:
            self.wandb.log(metrics, step=int(metrics.get("step", 0)))

    def close(self) -> None:
        self.metrics.close()
        self.samples.close()
        if self.wandb:
            self.wandb.finish()
