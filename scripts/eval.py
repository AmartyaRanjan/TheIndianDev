"""Score a detector on a split, with per-example records for paired significance tests.

    uv run python scripts/eval.py --detector linter --split dev
    uv run python scripts/eval.py --detector base --split dev [--few-shot]          # GPU
    uv run python scripts/eval.py --detector adapter --adapter runs/full-s0/adapters/detector/v400/detector --split test
    uv run python scripts/eval.py --detector claude --split dev [--batch] [--yes]   # paid API; dry run without --yes
    uv run python scripts/eval.py --detector gemini --split dev [--limit 30] [--yes]  # key in GEMINI_API_KEY or .env
Writes results/<name>/<split>.{predictions.jsonl,metrics.json}.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from advbug.data.schema import Example, read_jsonl
from advbug.eval.baselines import (
    AnswerCache,
    ClaudeSettings,
    GeminiSettings,
    claude_predictions,
    gemini_predictions,
    linter_predictions,
)
from advbug.eval.metrics import is_correct, line_distance, score
from advbug.game.prompts import DetectorPrediction, detector_messages, parse_detector
from advbug.taxonomy import NO_BUG

SPLITS = {"dev": "data/heldout/dev.jsonl", "test": "data/heldout/test.jsonl", "synthetic": "data/synthetic/val.jsonl"}


def open_weight_predictions(examples: list[Example], args: argparse.Namespace) -> list[DetectorPrediction | None]:
    from transformers import AutoTokenizer

    from advbug.rl.policy import chat_prompt_ids
    from advbug.rl.rollout import AdapterRef, VLLMGenerator

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    generator = VLLMGenerator(
        args.base_model,
        max_lora_rank=args.lora_rank,
        gpu_memory_utilization=args.gpu_memory,
        max_model_len=args.max_model_len,
        max_loras=1,
        enforce_eager=args.eager,
    )
    ref = AdapterRef("detector", 0, args.adapter) if args.detector == "adapter" else None
    prompts = [chat_prompt_ids(tokenizer, detector_messages(e.code, few_shot=args.few_shot)) for e in examples]
    outs = generator.generate(prompts, ref, 1, 0.0, 32)
    return [parse_detector(o[0].text) for o in outs]


def claude_cost_estimate(examples: list[Example]) -> str:
    """Rough dollars at claude-opus-5-5 list prices ($4/$20 per MTok); thinking dominates output."""
    input_tokens = sum(len(detector_messages(e.code)[1]["content"]) / 3.5 + 150 for e in examples)
    low, high = 300 * len(examples), 1500 * len(examples)  # thinking + answer tokens per request (assumed)
    dollars = lambda out: (input_tokens * 4 + out * 20) / 1e6  # noqa: E731
    return f"~{input_tokens / 1e6:.2f}M input tokens; ${dollars(low):.0f}-${dollars(high):.0f} (half that with --batch)"


def gemini_api_key() -> str:
    """GEMINI_API_KEY from the environment, else from the git-ignored .env file."""
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]
    env = Path(".env")
    for line in env.read_text().splitlines() if env.exists() else []:
        name, _, value = line.partition("=")
        if name.strip() == "GEMINI_API_KEY":
            return value.strip()
    raise SystemExit("GEMINI_API_KEY is not set (environment or .env)")


def api_token_report(cache: AnswerCache, examples: list[Example]) -> str:
    rows = [cache.rows[e.id] for e in examples if e.id in cache.rows]
    total = lambda key: sum(int(r.get(key) or 0) for r in rows)  # noqa: E731
    errors = sum(1 for r in rows if r.get("error"))
    served = sorted({str(r.get("served_by")) for r in rows if r.get("served_by")})
    return (
        f"{len(rows)} answers ({errors} errors) from {served}: input {total('input_tokens'):,} / "
        f"output {total('output_tokens'):,} / thinking {total('thinking_tokens'):,} tokens"
    )


def breakdowns(examples: list[Example], preds: list[DetectorPrediction | None]) -> dict[str, dict[str, dict[str, float]]]:
    groups: dict[str, dict[str, list[int]]] = {"source": defaultdict(list), "class": defaultdict(list), "location": defaultdict(list)}
    for i, e in enumerate(examples):
        groups["source"][e.source].append(i)
        groups["class"][e.bug_class].append(i)
        if e.location:
            groups["location"][e.location].append(i)
    return {
        name: {key: score([examples[i] for i in idx], [preds[i] for i in idx]) for key, idx in sorted(by.items())}
        for name, by in groups.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detector", required=True, choices=["linter", "base", "adapter", "claude", "gemini"])
    parser.add_argument("--split", default="dev", choices=sorted(SPLITS))
    parser.add_argument("--name", help="results/<name>/ (default derived from the detector)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--few-shot", action="store_true")
    parser.add_argument("--adapter", help="LoRA adapter directory for --detector adapter")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--gpu-memory", type=float, default=0.8, help="fraction of total GPU memory vLLM may take")
    parser.add_argument("--eager", action="store_true", help="skip CUDA graphs (for a crowded GPU)")
    parser.add_argument("--max-model-len", type=int, default=4096, help="few-shot prompts on long functions exceed 2048")
    parser.add_argument("--model", default=None, help="API model (defaults: claude-opus-5-5 / gemini-3.1-pro-preview)")
    parser.add_argument("--effort", default=ClaudeSettings.effort)
    parser.add_argument("--thinking-level", default=None, help="Gemini thinking level: low | medium | high")
    parser.add_argument("--batch", action="store_true", help="Claude via the Batches API (half price, no fallbacks)")
    parser.add_argument("--yes", action="store_true", help="actually spend on the Claude/Gemini API")
    args = parser.parse_args()

    examples = list(read_jsonl(SPLITS[args.split], Example))[: args.limit]
    args.model = args.model or {"claude": ClaudeSettings.model, "gemini": GeminiSettings.model}.get(args.detector)
    shot = "-fewshot" if args.few_shot else ""
    name = args.name or {
        "linter": "linter",
        "base": f"base{shot}",
        "adapter": f"adapter{shot}",
        "claude": f"claude-{args.model}-{args.effort}{shot}",
        "gemini": f"gemini-{args.model}{'-' + args.thinking_level if args.thinking_level else ''}{shot}",
    }[args.detector]
    out = Path("results") / name

    if args.detector == "linter":
        preds = linter_predictions(examples)
    elif args.detector == "claude":
        print(f"{len(examples)} requests to {args.model}: {claude_cost_estimate(examples)}")
        if not args.yes:
            print("dry run: re-run with --yes to send the requests")
            return
        settings = ClaudeSettings(model=args.model, effort=args.effort, few_shot=args.few_shot)
        preds = claude_predictions(examples, settings, AnswerCache(out / f"{args.split}.claude_cache.jsonl"), args.batch)
    elif args.detector == "gemini":
        input_tokens = sum(len(detector_messages(e.code, few_shot=args.few_shot)[-1]["content"]) / 3.5 + 150 for e in examples)
        print(f"{len(examples)} requests to {args.model}: ~{input_tokens / 1e6:.2f}M input tokens plus thinking/output")
        if not args.yes:
            print("dry run: re-run with --yes to send the requests")
            return
        cache = AnswerCache(out / f"{args.split}.gemini_cache.jsonl")
        settings = GeminiSettings(model=args.model, few_shot=args.few_shot, thinking_level=args.thinking_level)
        preds = gemini_predictions(examples, settings, cache, gemini_api_key())
        print(api_token_report(cache, examples))
    else:
        preds = open_weight_predictions(examples, args)

    out.mkdir(parents=True, exist_ok=True)
    with (out / f"{args.split}.predictions.jsonl").open("w") as f:
        for e, p in zip(examples, preds):
            row = {
                "id": e.id,
                "pair_id": e.pair_id,
                "source": e.source,
                "bug_class": e.bug_class,
                "location": e.location,
                "pred_class": p.bug_class if p else None,
                "pred_line": p.line if p else None,
                "correct": is_correct(e, p),
                "line_exact": e.bug_class != NO_BUG and line_distance(p.line if p else None, e.bug_lines) == 0,
            }
            f.write(json.dumps(row) + "\n")
    metrics = {"name": name, "split": args.split, "overall": score(examples, preds)} | breakdowns(examples, preds)
    (out / f"{args.split}.metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    o = metrics["overall"]
    print(
        f"{name} on {args.split}: accuracy={o['accuracy']:.3f} line_exact={o.get('line_exact', float('nan')):.3f} "
        f"type_acc={o.get('type_acc', float('nan')):.3f} FPR={o.get('false_positive_rate', float('nan')):.3f} "
        f"format_fail={o['format_failure']:.3f} macro_f1={o.get('macro_f1', float('nan')):.3f}"
    )


if __name__ == "__main__":
    main()
