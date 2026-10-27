"""Train one ablation arm.

    uv run python scripts/train.py --arm configs/arms/full.yaml seed=1
    uv run python scripts/train.py --arm configs/arms/a1_uniform.yaml --resume
Logs to runs/<arm>-s<seed>/{metrics,samples}.jsonl; checkpoint.pt every `checkpoint_every` steps.
"""

from __future__ import annotations

import argparse
import time

from advbug.config import Config, load_config
from advbug.data.schema import CleanFunction, Example, file_sha256, read_jsonl
from advbug.game.gate import Gate, GateConfig
from advbug.rl.loop import CoTrainer, RunLog
from advbug.rl.policy import LoraSettings, Policy
from advbug.rl.rollout import HFGenerator, VLLMGenerator
from advbug.sandbox import SandboxPool
from advbug.taxonomy import TAXONOMY_FINGERPRINT


def eval_subset(path: str, limit: int) -> list[Example]:
    """A fixed subset that keeps buggy/fixed pairs together (sorted by pair id)."""
    examples = sorted(read_jsonl(path, Example), key=lambda e: (e.pair_id or e.id, e.id))
    return examples[:limit]


def run(cfg: Config, resume: bool) -> None:
    corpus = list(read_jsonl(cfg.corpus, CleanFunction))
    real_dev = eval_subset(cfg.real_dev, cfg.eval.real_dev_limit)
    synthetic = eval_subset(cfg.synthetic_val, cfg.eval.synthetic_limit)
    needs_sandbox = cfg.gate.enabled or cfg.injector.mode == "uniform"
    sandbox = SandboxPool(cfg.sandbox_workers) if needs_sandbox else None  # spawned before CUDA initialises
    gate = Gate(GateConfig(enabled=cfg.gate.enabled), sandbox)
    lora = LoraSettings(r=cfg.lora.r, alpha=cfg.lora.alpha)
    if cfg.backend == "vllm":
        generator = VLLMGenerator(
            cfg.base_model,
            max_lora_rank=cfg.lora.r,
            gpu_memory_utilization=cfg.vllm.gpu_memory_utilization,
            max_model_len=cfg.vllm.max_model_len,
            max_loras=cfg.vllm.max_loras,
            seed=cfg.seed,
            enforce_eager=cfg.vllm.enforce_eager,
        )
        policy = Policy.from_pretrained(cfg.base_model, ["injector", "detector"], lora)
    else:
        policy = Policy.from_pretrained(cfg.base_model, ["injector", "detector"], lora, device="cpu")
        generator = HFGenerator(policy)

    trainer = CoTrainer(cfg, policy, generator, gate, corpus)
    checkpoint = cfg.run_dir / "checkpoint.pt"
    if resume and checkpoint.exists():
        trainer.load_checkpoint(checkpoint)
        print(f"resumed {cfg.run_dir} at step {trainer.step_idx}")
    config = cfg.to_dict() | {"taxonomy_fingerprint": TAXONOMY_FINGERPRINT, "corpus_sha256": file_sha256(cfg.corpus)}
    log = RunLog(cfg.run_dir, config, cfg.wandb)

    def evaluation() -> dict[str, float]:
        out = {f"real_dev/{k}": v for k, v in trainer.evaluate(real_dev).items()}
        out |= {f"synthetic_val/{k}": v for k, v in trainer.evaluate(synthetic).items()}
        out["gap/synthetic_val_minus_real"] = out["synthetic_val/accuracy"] - out["real_dev/accuracy"]
        return out

    try:
        if trainer.step_idx == 0:
            log.log({"step": 0} | evaluation())
        while trainer.step_idx < cfg.steps:
            start = time.time()
            metrics, samples = trainer.step()
            metrics["time/step_seconds"] = time.time() - start
            t = trainer.step_idx
            if t % cfg.eval.every == 0 or t == cfg.steps:
                metrics |= evaluation()
                if "detector/synthetic_acc" in metrics:
                    metrics["gap/on_policy_minus_real"] = metrics["detector/synthetic_acc"] - metrics["real_dev/accuracy"]
            log.log(metrics, samples)
            if t % cfg.checkpoint_every == 0 or t == cfg.steps:
                trainer.save_checkpoint()
            print(
                f"step {t}: fool={metrics.get('fool_rate', float('nan')):.2f} "
                f"gate={metrics.get('gate/pass_rate', float('nan')):.2f} "
                f"det_acc={metrics.get('detector/synthetic_acc', float('nan')):.2f} "
                f"({metrics['time/step_seconds']:.0f}s)",
                flush=True,
            )
    finally:
        log.close()
        if sandbox:
            sandbox.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="configs/base.yaml")
    parser.add_argument("--arm", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("overrides", nargs="*", help="dotted key=value overrides, e.g. optim.lr=5e-6")
    args = parser.parse_args()
    run(load_config(args.base, args.arm, args.overrides), args.resume)


if __name__ == "__main__":
    main()
