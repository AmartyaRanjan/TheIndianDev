"""Phase 3 check: can the base model act as an injector? Samples mutations and runs the full gate.

    uv run python scripts/probe_injector.py --functions 32 --samples 8 --gpu-memory 0.24 --eager
Prints the gate pass rate, the failure breakdown per check, the classes produced, and examples.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter

from transformers import AutoTokenizer

from advbug.data.schema import CleanFunction, read_jsonl
from advbug.game.gate import CHECKS, Gate, GateConfig
from advbug.game.prompts import injector_messages
from advbug.rl.policy import chat_prompt_ids
from advbug.rl.rollout import AdapterRef, VLLMGenerator
from advbug.sandbox import SandboxPool


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--corpus", default="data/corpus/val.jsonl")
    parser.add_argument("--functions", type=int, default=32)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--gpu-memory", type=float, default=0.24)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--adapter", help="injector LoRA adapter to probe instead of the base model")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    fns = random.Random(args.seed).sample(list(read_jsonl(args.corpus, CleanFunction)), args.functions)
    with SandboxPool(workers=4) as sandbox:  # spawned before CUDA initialises
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        generator = VLLMGenerator(
            args.base_model, max_lora_rank=32, gpu_memory_utilization=args.gpu_memory, max_loras=1, enforce_eager=args.eager
        )
        prompts = [chat_prompt_ids(tokenizer, injector_messages(fn.code)) for fn in fns]
        ref = AdapterRef("injector", 0, args.adapter) if args.adapter else None
        outs = generator.generate(prompts, ref, args.samples, 1.0, 128)
        pairs = [(fn, c.text) for fn, group in zip(fns, outs) for c in group]
        results = Gate(GateConfig(), sandbox).check(pairs)

    n = len(results)
    failed = Counter(r.failed_check for r in results if not r.ok)
    valid = [r for r in results if r.ok]
    print(f"{n} samples: gate pass rate {len(valid) / n:.3f}")
    print("failures: " + ", ".join(f"{c}={failed[c] / n:.3f}" for c in CHECKS))
    print("valid classes:", dict(Counter(r.claimed_class for r in valid).most_common()))
    groups_with_valid = sum(1 for i in range(len(fns)) if any(r.ok for r in results[i * args.samples : (i + 1) * args.samples]))
    print(f"functions with >=1 valid mutant: {groups_with_valid}/{len(fns)}")
    for (_, text), r in list(zip(pairs, results))[:3]:
        print(f"---- {r.failed_check or 'VALID'}:", repr(text[:300]))


if __name__ == "__main__":
    main()
