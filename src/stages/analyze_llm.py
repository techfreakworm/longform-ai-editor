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

import json as _json
import logging
import re
import shutil
import subprocess
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field, ValidationError
from tenacity import retry, stop_after_attempt, wait_fixed

from src import config

log = logging.getLogger(__name__)


# --- schemas -----------------------------------------------------------

Confidence = Literal["high", "low"]


class FillerCut(BaseModel):
    start: float
    end: float
    reason: Literal["filler", "false_start", "repeat", "other"] = "filler"
    confidence: Confidence = "high"


class FillerCutsResponse(BaseModel):
    cuts: list[FillerCut] = Field(default_factory=list)


class DeadZoneCue(BaseModel):
    start: float
    end: float
    reason: str = ""
    confidence: Confidence = "high"


class DeadZoneCuesResponse(BaseModel):
    cues: list[DeadZoneCue] = Field(default_factory=list)


Layout = Literal["cam_full", "pip", "screen_full"]


class LayoutSegment(BaseModel):
    start: float
    end: float
    layout: Layout


class LayoutPlanResponse(BaseModel):
    segments: list[LayoutSegment]


ZoomStrength = Literal["soft", "normal", "strong"]


class ZoomHint(BaseModel):
    anchor_word_idx: int
    start: float
    end: float
    strength: ZoomStrength = "normal"
    reason: str = ""


class ZoomHintsResponse(BaseModel):
    hints: list[ZoomHint] = Field(default_factory=list)


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

_JSON_INSTRUCTION_SUFFIX = (
    "\n\nReturn ONLY a valid JSON object. No markdown fences, no preamble, no commentary."
)


def _have_claude_cli() -> bool:
    """True if the `claude` binary is on PATH."""
    return shutil.which("claude") is not None


def _maybe_prepend_sequential_thinking(system_prompt: str) -> str:
    """Prepend the sequential-thinking MCP instruction when enabled.

    Gated by config.USE_SEQUENTIAL_THINKING so tests and environments
    without the MCP server configured (check `.mcp.json` at repo root)
    can disable it cleanly.
    """
    if config.USE_SEQUENTIAL_THINKING:
        return config.SEQUENTIAL_THINKING_PREFIX + system_prompt
    return system_prompt


def _call_via_claude_cli(
    system_prompt: str,
    user_payload: dict[str, Any],
    *,
    model: str | None = None,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """Shell out to `claude -p` (Claude Code CLI non-interactive mode).

    Claude CLI has its own built-in agent system prompt; we can't replace
    it. We combine our system + user payload into one input that goes on
    stdin. The strict "Return ONLY a valid JSON object…" suffix tames
    Claude's tendency to add preamble.

    Uses the user's existing `claude login` session (Claude Max/Pro
    subscription) — no API key needed, calls count toward subscription
    quota rather than per-token billing.

    Passes `--effort <level>` (default "max") so Claude applies full
    reasoning budget to each call. Also passes `--mcp-config` so the
    non-interactive session can reach the sequential-thinking MCP
    declared in the project's .mcp.json — without this, `claude -p`
    only sees MCP servers from ~/.claude.json global config.
    """
    model = model or config.CLAUDE_MODEL
    timeout_s = timeout_s or config.LLM_TIMEOUT_SEC

    prompt_body = (
        _maybe_prepend_sequential_thinking(system_prompt).rstrip()
        + _JSON_INSTRUCTION_SUFFIX
        + "\n\nInput:\n"
        + _json.dumps(user_payload)
    )
    cmd = [
        "claude", "-p",
        "--model", model,
        "--effort", config.CLAUDE_EFFORT,
        "--output-format", "text",
        "--permission-mode", "bypassPermissions",
        "--no-session-persistence",
    ]
    if config.USE_SEQUENTIAL_THINKING and config.CLAUDE_MCP_CONFIG.exists():
        cmd.extend(["--mcp-config", str(config.CLAUDE_MCP_CONFIG)])
    log.debug("calling claude CLI: %s", cmd)
    r = subprocess.run(
        cmd,
        input=prompt_body,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"claude CLI exited {r.returncode}: {r.stderr[-400:] or r.stdout[-400:]}"
        )
    body = _extract_json_body(r.stdout)
    return _json.loads(body)


def _call_via_mlx_server(
    system_prompt: str,
    user_payload: dict[str, Any],
    *,
    server_url: str | None = None,
    model: str | None = None,
    timeout_s: float | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """POST to the local mlx_lm.server's /v1/chat/completions.

    Does NOT use `response_format={"type": "json_object"}` — mlx_lm.server's
    constrained-decoding path can hang on non-trivial prompts. Plain
    generation + strict prompt suffix is reliable on Llama 3.3 70B at
    temperature=0.
    """
    server_url = server_url or config.LLM_SERVER_URL
    model = model or config.LLM_MODEL
    timeout_s = timeout_s or config.LLM_TIMEOUT_SEC

    system_with_json = (
        _maybe_prepend_sequential_thinking(system_prompt).rstrip()
        + _JSON_INSTRUCTION_SUFFIX
    )
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_with_json},
            {"role": "user", "content": _json.dumps(user_payload)},
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
    return _json.loads(body)


def call_llm_json(
    system_prompt: str,
    user_payload: dict[str, Any],
    *,
    server_url: str | None = None,
    model: str | None = None,
    timeout_s: float | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Dispatcher: prefer Claude CLI if available, else local MLX.

    Behavior:
      * If `claude` is on PATH AND config.FORCE_LOCAL_LLM is False, try
        Claude CLI. On ANY failure (non-zero exit, malformed JSON, auth
        problem, timeout), log a warning and fall back to the local path.
      * Otherwise go straight to the local MLX path.

    This means a user with Claude Max/Pro gets a ~10% quality lift on
    layout decisions at zero marginal cost, and the pipeline still works
    perfectly without Claude — graceful degradation.
    """
    if not config.FORCE_LOCAL_LLM and _have_claude_cli():
        try:
            return _call_via_claude_cli(
                system_prompt, user_payload,
                model=None,         # use config.CLAUDE_MODEL
                timeout_s=timeout_s,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "Claude CLI call failed (%s); falling back to local MLX",
                exc,
            )
    return _call_via_mlx_server(
        system_prompt, user_payload,
        server_url=server_url,
        model=model,
        timeout_s=timeout_s,
        max_tokens=max_tokens,
        temperature=temperature,
    )


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
def analyze_dead_zone_cues(
    words: list[dict[str, Any]],
) -> DeadZoneCuesResponse:
    """LLM flags narrator-signaled skippable spans ("while installs", ...).

    Emits DeadZoneCue entries with a confidence flag. Low-confidence
    cues are intended to be verified downstream by a frame-inspection
    stage (`src/stages/verify_cuts.py`) before any content is cut.

    Retries up to config.LLM_MAX_RETRIES on schema validation failure.
    """
    raw = call_llm_json(config.DEAD_ZONE_CUES_PROMPT, {"words": words})
    try:
        return DeadZoneCuesResponse(**raw)
    except ValidationError as e:
        log.warning("dead_zone_cues response failed schema: %s — retrying", e)
        raise


@retry(
    stop=stop_after_attempt(config.LLM_MAX_RETRIES),
    wait=wait_fixed(1),
    reraise=True,
)
def analyze_zoom_hints(words: list[dict[str, Any]]) -> ZoomHintsResponse:
    """LLM identifies deictic zoom moments ("look at this", "notice", ...).

    The LLM emits windows in video timebase (words' own timestamps) along
    with a strength. Downstream: unify_segments merges these with
    cursor-driven zoom segments, using cursor position at hint start as
    the zoom centroid when cursor.csv is available, else screen center.

    Retries up to config.LLM_MAX_RETRIES on schema validation failure.
    """
    raw = call_llm_json(config.ZOOM_HINTS_PROMPT, {"words": words})
    try:
        return ZoomHintsResponse(**raw)
    except ValidationError as e:
        log.warning("zoom_hints response failed schema: %s — retrying", e)
        raise


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
    "Confidence",
    "FillerCut",
    "FillerCutsResponse",
    "DeadZoneCue",
    "DeadZoneCuesResponse",
    "Layout",
    "LayoutSegment",
    "LayoutPlanResponse",
    "ZoomStrength",
    "ZoomHint",
    "ZoomHintsResponse",
    "analyze_dead_zone_cues",
    "analyze_zoom_hints",
    "strip_thinking",
    "_maybe_prepend_sequential_thinking",
    "_have_claude_cli",
    "_call_via_claude_cli",
    "_call_via_mlx_server",
    "call_llm_json",
    "analyze_fillers",
    "analyze_layout",
]
