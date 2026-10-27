"""Supervised detector baseline ("why RL at all?"): fine-tune the detector adapter on gated AST-engine
mutants of the training corpus plus clean functions answered `no_bug`, with compute matched to an RL arm.

An RL arm sees steps x (train_mutants + train_clean) detector inputs (400 x 80 = 32,000 by default),
so --mutants 25600 --clean 6400 matches it. Score the result like any other detector:

    uv run python scripts/train_sft_detector.py
    uv run python scripts/eval.py --detector adapter --adapter runs/sft_detector/detector --split dev --name sft_detector
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from advbug.data.schema import CleanFunction, read_jsonl
from advbug.data.synthetic import uniform_mutants
from advbug.game.gate import Gate, GateConfig
from advbug.game.prompts import detector_messages
from advbug.rl.policy import LoraSettings, Policy
from advbug.rl.sft import sft
from advbug.sandbox import SandboxPool
from advbug.taxonomy import NO_BUG, TAXONOMY_FINGERPRINT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--corpus", default="data/corpus/train.jsonl")
    parser.add_argument("--mutants", type=int, default=25600)
    parser.add_argument("--clean", type=int, default=6400)
    parser.add_argument("--out", default="runs/sft_detector")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    corpus = list(read_jsonl(args.corpus, CleanFunction))
    texts: list[tuple[str, str]] = []  # (code shown to the detector, expected answer)
    classes: Counter[str] = Counter()
    with SandboxPool(workers=6) as sandbox:
        gate = Gate(GateConfig(), sandbox)
        while len(classes) == 0 or sum(classes.values()) < args.mutants:  # one gated mutant per function per pass
            fns = rng.sample(corpus, len(corpus))
            for fn, r in uniform_mutants(fns, gate, rng):
                if sum(classes.values()) >= args.mutants:
                    break
                texts.append((r.mutant, f"<line>{r.bug_lines[0]}</line><type>{r.claimed_class}</type>"))
                classes[r.claimed_class] += 1
    for fn in rng.sample(corpus, args.clean):
        texts.append((fn.code, f"<type>{NO_BUG}</type>"))
    print(f"{len(texts)} examples ({sum(classes.values())} mutants, {args.clean} clean)", flush=True)

    policy = Policy.from_pretrained(args.base_model, ["injector", "detector"], LoraSettings(args.lora_r, args.lora_alpha))
    end = policy.tokenizer.convert_tokens_to_ids("<|im_end|>")
    data = [
        (policy.prompt_ids(detector_messages(code)), policy.tokenizer(answer, add_special_tokens=False)["input_ids"] + [end])
        for code, answer in texts
    ]
    losses = sft(policy, "detector", data, lr=args.lr, epochs=args.epochs, seed=args.seed)
    path = policy.save_adapter("detector", Path(args.out))
    manifest = {
        "base_model": args.base_model,
        "taxonomy_fingerprint": TAXONOMY_FINGERPRINT,
        "examples": len(data),
        "mutants": sum(classes.values()),
        "clean": args.clean,
        "classes": dict(classes.most_common()),
        "lr": args.lr,
        "epochs": args.epochs,
        "loss_first": losses[:5],
        "loss_last": losses[-5:],
        "adapter": str(path),
    }
    (Path(args.out) / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
