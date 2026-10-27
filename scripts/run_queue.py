"""Run ablation arms x seeds one after another on the single GPU, resuming unfinished runs.

    uv run python scripts/run_queue.py --tier 1 --seeds 0 1 2
    uv run python scripts/run_queue.py --arms full a3_off --seeds 0 -- steps=100   # pilot
A run is skipped once its metrics.jsonl reaches the configured step count.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from advbug.config import load_config

TIERS = {
    1: ["full", "a1_frozen", "a1_uniform", "a2_off", "a3_off"],
    2: ["a5_k2", "a5_k4", "a5_k16", "a5u_k2", "a5u_k8", "a5u_k16"],  # A4 PPO is not written yet
    3: ["a6_partial", "a7_curriculum"],
}


def finished(run_dir: Path, steps: int) -> bool:
    metrics = run_dir / "metrics.jsonl"
    if not metrics.exists():
        return False
    lines = metrics.read_text().splitlines()
    return bool(lines) and json.loads(lines[-1]).get("step", 0) >= steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tier", type=int, choices=sorted(TIERS))
    parser.add_argument("--arms", nargs="*", default=[])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("overrides", nargs="*", help="passed to every run, after --")
    args = parser.parse_args()
    arms = args.arms or TIERS.get(args.tier, [])
    if not arms:
        parser.error("give --tier or --arms")

    for seed in args.seeds:  # seed-major: every arm gets seed 0 before any arm gets seed 1
        for arm in arms:
            arm_file = f"configs/arms/{arm}.yaml"
            overrides = [f"seed={seed}", *args.overrides]
            cfg = load_config("configs/base.yaml", arm_file, overrides)
            if finished(cfg.run_dir, cfg.steps):
                print(f"skip {cfg.run_dir} (done)")
                continue
            print(f"=== {cfg.run_dir}", flush=True)
            code = subprocess.call([sys.executable, "scripts/train.py", "--arm", arm_file, "--resume", *overrides])
            if code != 0:
                sys.exit(f"{cfg.run_dir} failed with exit code {code}")


if __name__ == "__main__":
    main()
