"""Format warmup for the injector (plan Phase 3 rule: base-model format compliance < 80%).

The base model produced 0/256 gate-passing mutations, so it gives the injector no learning signal.
This fine-tunes only the injector adapter on AST-engine mutations of training-corpus functions,
written in the injector's output format and passed through the full gate. Every arm then starts
from this adapter (configs/base.yaml: injector.init_adapter); the detector is not warmed up.

    uv run python scripts/warmup_injector.py --functions 3000
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
from advbug.game.prompts import format_injector_output, injector_messages
from advbug.rl.policy import LoraSettings, Policy
from advbug.rl.sft import sft
from advbug.sandbox import SandboxPool
from advbug.taxonomy import TAXONOMY_FINGERPRINT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--corpus", default="data/corpus/train.jsonl")
    parser.add_argument("--functions", type=int, default=3000)
    parser.add_argument("--out", default="runs/warmup_injector")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    fns = rng.sample(list(read_jsonl(args.corpus, CleanFunction)), args.functions)
    with SandboxPool(workers=6) as sandbox:  # balanced classes, every example passes G2-G6
        pairs = uniform_mutants(fns, Gate(GateConfig(), sandbox), rng, balance=True)
    print(f"{len(pairs)} gated warmup examples from {len(fns)} functions", flush=True)

    policy = Policy.from_pretrained(args.base_model, ["injector", "detector"], LoraSettings(args.lora_r, args.lora_alpha))
    end = policy.tokenizer.convert_tokens_to_ids("<|im_end|>")
    data = []
    for fn, r in pairs:
        answer = format_injector_output(fn.code, r.mutant, r.claimed_class)
        if answer:
            prompt = policy.prompt_ids(injector_messages(fn.code))
            completion = policy.tokenizer(answer, add_special_tokens=False)["input_ids"] + [end]
            data.append((prompt, completion))
    losses = sft(policy, "injector", data, lr=args.lr, epochs=args.epochs, seed=args.seed)
    path = policy.save_adapter("injector", Path(args.out))
    manifest = {
        "base_model": args.base_model,
        "taxonomy_fingerprint": TAXONOMY_FINGERPRINT,
        "functions": len(fns),
        "examples": len(data),
        "classes": dict(Counter(r.claimed_class for _, r in pairs).most_common()),
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
