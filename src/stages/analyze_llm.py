"""Stage B.2 — LLM analysis: filler cuts + layout plan.

Two independent calls to the local mlx_lm.server (OpenAI-compatible):
  1. analyze_fillers(words) → FillerCutsResponse
  2. analyze_layout(words, duration) → LayoutPlanResponse

Both responses are validated against pydantic schemas; malformed
replies trigger up to LLM_MAX_RETRIES retries with a stricter prompt
suffix. Qwen3-style `<think>...</think>` blocks are stripped before
parsing (safe no-op for Llama which doesn't emit them).

Layout plan post-processing: if the model leaves gaps in coverage, we
fill them with the previous segment's layout (or default to "pip" for
a gap at the very start). Overlaps are resolved by truncating the
later segment.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field, ValidationError
from tenacity import retry, stop_after_attempt, wait_fixed

from src import config

log = logging.getLogger(__name__)


# --- schemas -----------------------------------------------------------

class FillerCut(BaseModel):
    start: float
    end: float
    reason: Literal["filler", "false_start", "repeat", "other"] = "filler"


class FillerCutsResponse(BaseModel):
    cuts: list[FillerCut] = Field(default_factory=list)


Layout = Literal["cam_full", "pip", "screen_full"]


class LayoutSegment(BaseModel):
    start: float
    end: float
    layout: Layout


class LayoutPlanResponse(BaseModel):
    segments: list[LayoutSegment]


# --- helpers ----------------------------------------------------------

THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove Qwen3-style <think>...</think> blocks before JSON parsing."""
    return THINK_TAG_RE.sub("", text).strip()


def _extract_json_body(text: str) -> str:
    """Find the outermost {...} block in model output.

    mlx_lm.server with response_format=json_object usually returns pure
    JSON, but some models wrap it in markdown fences or preamble. This
    helper extracts the JSON body robustly.
    """
    text = strip_thinking(text)
    # Try to find the first { and last matching }
    start = text.find("{")
    if start < 0:
        raise ValueError(f"no JSON object in model output: {text[:200]!r}")
    # Track brace depth to find matching close
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError(f"unbalanced braces in model output: {text[:200]!r}")


# --- LLM call wrapper --------------------------------------------------

def call_llm_json(
    system_prompt: str,
    user_payload: dict[str, Any],
    *,
    server_url: str | None = None,
    model: str | None = None,
    timeout_s: float | None = None,
    max_tokens: int = 8192,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """POST to /v1/chat/completions with JSON mode. Returns the parsed body.

    Does NOT validate the body against any schema — caller's responsibility.
    Raises on HTTP error, timeout, malformed JSON.
    """
    server_url = server_url or config.LLM_SERVER_URL
    model = model or config.LLM_MODEL
    timeout_s = timeout_s or config.LLM_TIMEOUT_SEC

    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": __import__("json").dumps(user_payload)},
        ],
    }
    r = httpx.post(
        f"{server_url.rstrip('/')}/chat/completions",
        json=payload,
        timeout=timeout_s,
    )
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    body = _extract_json_body(content)
    import json as _json
    return _json.loads(body)


# --- filler cuts -------------------------------------------------------

@retry(
    stop=stop_after_attempt(config.LLM_MAX_RETRIES),
    wait=wait_fixed(1),
    reraise=True,
)
def analyze_fillers(words: list[dict[str, Any]]) -> FillerCutsResponse:
    """LLM identifies filler + false-start + repeat ranges.

    Retries up to config.LLM_MAX_RETRIES on schema validation failure.
    """
    raw = call_llm_json(config.FILLER_PROMPT, {"words": words})
    try:
        return FillerCutsResponse(**raw)
    except ValidationError as e:
        log.warning("filler response failed schema: %s — retrying", e)
        raise


# --- layout plan -------------------------------------------------------

def _fill_coverage_gaps(
    segments: list[LayoutSegment],
    total_duration: float,
    default_layout: Layout = "pip",
) -> list[LayoutSegment]:
    """Post-process LLM layout output to guarantee the entire duration
    is covered with no gaps, no overlaps.

    - Gap at the start: insert default_layout from 0 to the first segment
    - Gaps between: extend previous segment to the next one's start
    - Overlaps: truncate the later segment's start to the earlier one's end
    - Gap at the end: extend the last segment to total_duration
    """
    if not segments:
        return [LayoutSegment(start=0.0, end=total_duration, layout=default_layout)]
    srt = sorted(segments, key=lambda s: s.start)
    fixed: list[LayoutSegment] = []

    # Leading gap
    if srt[0].start > 0.0:
        fixed.append(LayoutSegment(
            start=0.0, end=srt[0].start, layout=default_layout,
        ))
    fixed.append(srt[0])

    for nxt in srt[1:]:
        prev = fixed[-1]
        if nxt.start < prev.end:
            # Overlap — push nxt.start forward
            if nxt.end <= prev.end:
                # nxt fully inside prev: drop
                continue
            nxt = LayoutSegment(start=prev.end, end=nxt.end, layout=nxt.layout)
        elif nxt.start > prev.end:
            # Gap — extend prev
            fixed[-1] = LayoutSegment(
                start=prev.start, end=nxt.start, layout=prev.layout,
            )
        fixed.append(nxt)

    # Trailing gap
    last = fixed[-1]
    if last.end < total_duration:
        fixed[-1] = LayoutSegment(
            start=last.start, end=total_duration, layout=last.layout,
        )
    elif last.end > total_duration:
        fixed[-1] = LayoutSegment(
            start=last.start, end=total_duration, layout=last.layout,
        )
    return fixed


@retry(
    stop=stop_after_attempt(config.LLM_MAX_RETRIES),
    wait=wait_fixed(1),
    reraise=True,
)
def analyze_layout(
    words: list[dict[str, Any]],
    total_duration: float,
) -> LayoutPlanResponse:
    """LLM assigns cam_full / pip / screen_full to time ranges.

    Post-processes to ensure full coverage: fills gaps, truncates overlaps.
    """
    raw = call_llm_json(
        config.LAYOUT_PROMPT,
        {"words": words, "total_duration_s": total_duration},
    )
    try:
        parsed = LayoutPlanResponse(**raw)
    except ValidationError as e:
        log.warning("layout response failed schema: %s — retrying", e)
        raise
    # Fix up coverage before returning
    fixed = _fill_coverage_gaps(parsed.segments, total_duration)
    return LayoutPlanResponse(segments=fixed)


__all__ = [
    "FillerCut",
    "FillerCutsResponse",
    "Layout",
    "LayoutSegment",
    "LayoutPlanResponse",
    "strip_thinking",
    "call_llm_json",
    "analyze_fillers",
    "analyze_layout",
]
