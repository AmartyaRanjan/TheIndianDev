"""Training-dynamics figures from runs/<arm>-s<seed>/metrics.jsonl (plan Phase 8).

    uv run python scripts/figures.py --arms full a1_frozen a1_uniform a2_off a3_off --out results/figures
One PNG per curve, one line per arm (mean over seeds, shaded min-max): injector class entropy, fool rate,
gate pass rate, real-dev accuracy, and the sim-to-real gap. Needs matplotlib (`uv pip install matplotlib`).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

CURVES = {
    "injector/class_entropy": "Injector class entropy (nats)",
    "fool_rate": "Fool rate",
    "gate/pass_rate": "Gate pass rate",
    "real_dev/accuracy": "Real-dev accuracy",
    "gap/on_policy_minus_real": "Sim-to-real gap (on-policy synthetic minus real-dev)",
    "detector/synthetic_acc": "On-policy synthetic accuracy",
}


def read_runs(runs: Path, arm: str) -> list[dict[int, dict[str, float]]]:
    out = []
    for path in sorted(runs.glob(f"{arm}-s*/metrics.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        out.append({int(r["step"]): r for r in rows if "step" in r})
    return out


def curve(seeds: list[dict[int, dict[str, float]]], key: str) -> tuple[list[int], list[float], list[float], list[float]]:
    by_step: dict[int, list[float]] = defaultdict(list)
    for run in seeds:
        for step, row in run.items():
            value = row.get(key)
            if isinstance(value, (int, float)) and value == value:  # skips NaN
                by_step[step].append(float(value))
    steps = sorted(by_step)
    return (
        steps,
        [sum(by_step[s]) / len(by_step[s]) for s in steps],
        [min(by_step[s]) for s in steps],
        [max(by_step[s]) for s in steps],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--out", default="results/figures")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = {arm: read_runs(runs, arm) for arm in args.arms}
    for arm, seeds in data.items():
        print(f"{arm}: {len(seeds)} seed(s)")
    for key, title in CURVES.items():
        fig, ax = plt.subplots(figsize=(6.4, 4))
        drawn = False
        for arm, seeds in data.items():
            steps, mean, lo, hi = curve(seeds, key)
            if steps:
                ax.plot(steps, mean, label=f"{arm} (n={len(seeds)})")
                ax.fill_between(steps, lo, hi, alpha=0.15)
                drawn = True
        if not drawn:
            plt.close(fig)
            print(f"skip {key}: no data")
            continue
        ax.set_xlabel("training step")
        ax.set_title(title)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = out / (key.replace("/", "_") + ".png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
