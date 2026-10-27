"""Non-RL baselines (plan §4.8): static linter, and Claude zero/few-shot via the Anthropic API.

The open-weight base model (step 0) and trained adapters are scored by scripts/eval.py through
the same vLLM generator the training loop uses.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from advbug.data.schema import Example
from advbug.game.prompts import DetectorPrediction, detector_messages, parse_detector
from advbug.taxonomy import NO_BUG

LINTER_CLASS = "linter_warning"  # a localization-only prediction: never matches a taxonomy class


# --- static linter -----------------------------------------------------------------------------


def linter_predictions(examples: Sequence[Example]) -> list[DetectorPrediction]:
    """pyflakes: the first reported line is the predicted bug; no report means no_bug."""
    from pyflakes.api import check
    from pyflakes.reporter import Reporter

    class _Collect(Reporter):
        def __init__(self) -> None:
            self.lines: list[int] = []

        def flake(self, message) -> None:  # noqa: ANN001
            self.lines.append(message.lineno)

        def syntaxError(self, filename, msg, lineno, offset, text) -> None:  # noqa: ANN001, N802
            self.lines.append(lineno or 1)

        def unexpectedError(self, filename, msg) -> None:  # noqa: ANN001, N802
            pass

    preds = []
    for ex in examples:
        reporter = _Collect()
        check(ex.code, "snippet.py", reporter)
        preds.append(DetectorPrediction(LINTER_CLASS, min(reporter.lines)) if reporter.lines else DetectorPrediction(NO_BUG, None))
    return preds


# --- Claude ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ClaudeSettings:
    model: str = "claude-opus-5-5"
    effort: str = "medium"  # Claude Opus 5.5 always thinks; effort sets how much (its default is medium)
    max_tokens: int = 16000  # covers thinking plus the one-line answer
    few_shot: bool = False
    workers: int = 8


def _request(ex: Example, s: ClaudeSettings) -> dict[str, object]:
    messages = detector_messages(ex.code, few_shot=s.few_shot)
    return {
        "model": s.model,
        "max_tokens": s.max_tokens,
        "system": messages[0]["content"],
        "messages": messages[1:],
        "output_config": {"effort": s.effort},
    }


def _text(message) -> str:  # noqa: ANN001 - anthropic Message
    return "".join(b.text for b in message.content if b.type == "text")


class AnswerCache:
    """Per-example answers on disk so an interrupted or repeated run never pays twice."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.rows: dict[str, dict[str, object]] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                row = json.loads(line)
                self.rows[row["id"]] = row

    def add(self, row: dict[str, object]) -> None:
        self.rows[row["id"]] = row
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")


def claude_predictions(
    examples: Sequence[Example], settings: ClaudeSettings, cache: AnswerCache, use_batch: bool = False
) -> list[DetectorPrediction | None]:
    """Answers for every example (cached ones are not re-requested).

    Synchronous mode opts into server-side refusal fallbacks (`fallbacks: "default"`); a
    fallback-served answer is kept but its `served_by` model is recorded so the analysis can
    flag or exclude it. The Batches API (half price) does not accept `fallbacks`, so batch mode
    records refusals as format failures.
    """
    import anthropic

    client = anthropic.Anthropic()
    todo = [ex for ex in examples if ex.id not in cache.rows]
    if todo and use_batch:
        _run_batch(client, todo, settings, cache)
    elif todo:
        with ThreadPoolExecutor(settings.workers) as pool:
            for row in pool.map(lambda ex: _run_one(client, ex, settings), todo):
                cache.add(row)
    return [parse_detector(str(cache.rows[ex.id]["text"])) if ex.id in cache.rows else None for ex in examples]


def _run_one(client, ex: Example, s: ClaudeSettings) -> dict[str, object]:  # noqa: ANN001
    import anthropic

    try:
        response = client.beta.messages.create(
            **_request(ex, s), betas=["server-side-fallback-2026-07-01"], fallbacks="default"
        )
    except anthropic.BadRequestError as e:
        return {"id": ex.id, "text": "", "error": f"bad_request: {e.message}"[:300]}
    except anthropic.APIStatusError as e:  # retries for 429/5xx already exhausted by the SDK
        return {"id": ex.id, "text": "", "error": f"status {e.status_code}"}
    except anthropic.APIConnectionError:
        return {"id": ex.id, "text": "", "error": "connection"}
    refused = response.stop_reason == "refusal"
    return {
        "id": ex.id,
        "text": "" if refused else _text(response),
        "stop_reason": response.stop_reason,
        "served_by": response.model,
        "output_tokens": response.usage.output_tokens,
        "input_tokens": response.usage.input_tokens,
    }


def _run_batch(client, todo: Sequence[Example], s: ClaudeSettings, cache: AnswerCache) -> None:  # noqa: ANN001
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    ids = {f"ex-{i}": ex.id for i, ex in enumerate(todo)}  # custom_id must be [a-zA-Z0-9_-]{1,64}
    requests = [Request(custom_id=cid, params=MessageCreateParamsNonStreaming(**_request(ex, s))) for cid, ex in zip(ids, todo)]
    batch = client.messages.batches.create(requests=requests)
    print(f"submitted batch {batch.id} ({len(requests)} requests); polling ...", flush=True)
    while client.messages.batches.retrieve(batch.id).processing_status != "ended":
        time.sleep(60)
    for result in client.messages.batches.results(batch.id):  # arrives in any order: key by custom_id
        row: dict[str, object] = {"id": ids[result.custom_id], "text": ""}
        if result.result.type == "succeeded":
            msg = result.result.message
            refused = msg.stop_reason == "refusal"
            row |= {
                "text": "" if refused else _text(msg),
                "stop_reason": msg.stop_reason,
                "served_by": msg.model,
                "output_tokens": msg.usage.output_tokens,
                "input_tokens": msg.usage.input_tokens,
            }
        else:
            row["error"] = result.result.type
        cache.add(row)


# --- Gemini ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GeminiSettings:
    model: str = "gemini-3.1-pro-preview"
    max_output_tokens: int = 16384  # thinking tokens count against this limit
    thinking_level: str | None = None  # None = provider default; "low"/"medium"/"high" bound the thinking spend
    few_shot: bool = False
    workers: int = 4
    max_attempts: int = 6


def gemini_predictions(
    examples: Sequence[Example], settings: GeminiSettings, cache: AnswerCache, api_key: str
) -> list[DetectorPrediction | None]:
    """Answers from the Gemini API (AI Studio key); cached answers are not re-requested.

    Thinking is always on for these models and temperature is left at the provider default;
    the exact `model_version` that served each answer is recorded.
    """
    from google import genai

    client = genai.Client(api_key=api_key)
    todo = [ex for ex in examples if ex.id not in cache.rows]
    if todo:
        with ThreadPoolExecutor(settings.workers) as pool:
            for row in pool.map(lambda ex: _gemini_one(client, ex, settings), todo):
                cache.add(row)
    return [parse_detector(str(cache.rows[ex.id]["text"])) if ex.id in cache.rows else None for ex in examples]


def _gemini_one(client, ex: Example, s: GeminiSettings) -> dict[str, object]:  # noqa: ANN001 - genai.Client
    from google.genai import errors, types

    messages = detector_messages(ex.code, few_shot=s.few_shot)
    contents = [
        types.Content(role="model" if m["role"] == "assistant" else "user", parts=[types.Part(text=m["content"])])
        for m in messages[1:]
    ]
    thinking = types.ThinkingConfig(thinking_level=s.thinking_level.upper()) if s.thinking_level else None
    config = types.GenerateContentConfig(
        system_instruction=messages[0]["content"], max_output_tokens=s.max_output_tokens, thinking_config=thinking
    )
    for attempt in range(s.max_attempts):
        try:
            response = client.models.generate_content(model=s.model, contents=contents, config=config)
            break
        except errors.APIError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt + 1 < s.max_attempts:
                time.sleep(min(60.0, 5.0 * 2**attempt))  # rate limit or transient server error
                continue
            return {"id": ex.id, "text": "", "error": f"{e.code}: {str(e.message)[:200]}"}
    parts = response.candidates[0].content.parts if response.candidates and response.candidates[0].content else None
    text = "".join(p.text for p in parts or [] if p.text and not p.thought)
    usage = response.usage_metadata
    return {
        "id": ex.id,
        "text": text,
        "finish_reason": response.candidates[0].finish_reason.name if response.candidates else "NO_CANDIDATE",
        "served_by": response.model_version,
        "input_tokens": usage.prompt_token_count if usage else None,
        "output_tokens": usage.candidates_token_count if usage else None,
        "thinking_tokens": usage.thoughts_token_count if usage else None,
    }
