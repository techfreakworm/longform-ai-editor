"""Stage B.2 — LLM analysis: filler cuts + layout plan.

Two independent calls to the local MLX LLM server:
  1. filler_cuts — identify fillers, false starts, repeats
  2. layout_plan — segment transcript into cam_full / pip / screen_full

Both responses are validated against pydantic schemas; invalid responses
trigger up to LLM_MAX_RETRIES attempts with a stricter prompt suffix.

TODO(M2): implement. See IMPLEMENTATION_PLAN.md §M2.
"""
from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field


class FillerCut(BaseModel):
    start: float
    end: float
    reason: Literal["filler", "false_start", "repeat", "other"] = "filler"


class FillerCutsResponse(BaseModel):
    cuts: list[FillerCut] = Field(default_factory=list)


class LayoutSegment(BaseModel):
    start: float
    end: float
    layout: Literal["cam_full", "pip", "screen_full"]


class LayoutPlanResponse(BaseModel):
    segments: list[LayoutSegment]


class DeadZoneCue(BaseModel):
    start: float
    end: float
    reason: str = ""


class DeadZoneCuesResponse(BaseModel):
    cues: list[DeadZoneCue] = Field(default_factory=list)


THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Qwen3 sometimes emits <think>...</think> blocks; strip them before parsing."""
    return THINK_TAG_RE.sub("", text).strip()


def call_llm_json(prompt: str, user_payload: dict[str, Any]) -> dict[str, Any]:
    """POST to LLM server's /v1/chat/completions with response_format=json_object.

    TODO(M2): implement with httpx + tenacity retry.
    """
    raise NotImplementedError


def analyze_fillers(words: list[dict[str, Any]]) -> FillerCutsResponse:
    """TODO(M2): call LLM with FILLER_PROMPT, validate, retry on schema error."""
    raise NotImplementedError


def analyze_layout(words: list[dict[str, Any]], total_duration: float) -> LayoutPlanResponse:
    """TODO(M2): call LLM with LAYOUT_PROMPT, validate coverage (no gaps), fill holes with 'pip'."""
    raise NotImplementedError


def detect_dead_zone_cues(words: list[dict[str, Any]]) -> DeadZoneCuesResponse:
    """TODO(M3): LLM call for transcript-guided dead-zone cues (Stage C signal #4)."""
    raise NotImplementedError


__all__ = [
    "FillerCut",
    "FillerCutsResponse",
    "LayoutSegment",
    "LayoutPlanResponse",
    "DeadZoneCue",
    "DeadZoneCuesResponse",
    "strip_thinking",
    "call_llm_json",
    "analyze_fillers",
    "analyze_layout",
    "detect_dead_zone_cues",
]
