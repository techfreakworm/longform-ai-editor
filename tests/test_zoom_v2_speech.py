"""Tests for zoom v2 — speech-emphasis zooms (LLM hints → ZoomSegments)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.stages.analyze_llm import (
    ZoomHint,
    ZoomHintsResponse,
    analyze_zoom_hints,
)
from src.stages.cursor_zoom import (
    SPEECH_ZOOM_BY_STRENGTH,
    CursorEvent,
    ZoomSegment,
    _cursor_position_at,
    merge_zoom_segments,
    zoom_segments_from_hints,
)
from src.stages.unify_segments import load_zoom_hints


# --- schema -----------------------------------------------------------

def test_zoom_hint_schema_accepts_normal_payload() -> None:
    h = ZoomHint(
        anchor_word_idx=10, start=5.0, end=8.0,
        strength="normal", reason="look at this",
    )
    assert h.strength == "normal"


def test_zoom_hint_defaults_to_normal_strength() -> None:
    h = ZoomHint(anchor_word_idx=0, start=1.0, end=2.0)
    assert h.strength == "normal"


# --- cursor-position lookup ------------------------------------------

def test_cursor_position_at_returns_nearest_move() -> None:
    moves = [
        CursorEvent(t_s=1.0, x=0.3, y=0.3, is_click=False),
        CursorEvent(t_s=2.0, x=0.7, y=0.8, is_click=False),
    ]
    assert _cursor_position_at(moves, 1.9) == (0.7, 0.8)
    assert _cursor_position_at(moves, 1.1) == (0.3, 0.3)


def test_cursor_position_at_empty_moves_returns_none() -> None:
    assert _cursor_position_at([], 5.0) is None


def test_cursor_position_at_before_first_move_returns_none() -> None:
    moves = [CursorEvent(t_s=10.0, x=0.5, y=0.5, is_click=False)]
    # t well before first move and outside 0.5 s window → None
    assert _cursor_position_at(moves, 2.0) is None


# --- zoom_segments_from_hints ----------------------------------------

def test_hints_use_cursor_position_when_available() -> None:
    hints = [{"start": 5.0, "end": 7.5, "strength": "normal"}]
    moves = [CursorEvent(t_s=5.1, x=0.4, y=0.9, is_click=False)]
    out = zoom_segments_from_hints(hints, moves, duration_s=30.0)
    assert len(out) == 1
    assert out[0].cx == pytest.approx(0.4)
    assert out[0].cy == pytest.approx(0.9)
    assert out[0].zoom == SPEECH_ZOOM_BY_STRENGTH["normal"]


def test_hints_default_to_screen_center_without_cursor() -> None:
    hints = [{"start": 5.0, "end": 7.5, "strength": "strong"}]
    out = zoom_segments_from_hints(hints, [], duration_s=30.0)
    assert len(out) == 1
    assert (out[0].cx, out[0].cy) == (0.5, 0.5)
    assert out[0].zoom == SPEECH_ZOOM_BY_STRENGTH["strong"]


def test_hints_clamp_to_duration() -> None:
    hints = [{"start": 28.0, "end": 35.0}]
    out = zoom_segments_from_hints(hints, [], duration_s=30.0)
    assert len(out) == 1
    assert out[0].start == pytest.approx(28.0)
    assert out[0].end == pytest.approx(30.0)


def test_hints_reversed_or_zero_length_skipped() -> None:
    hints = [
        {"start": 5.0, "end": 4.0},   # reversed
        {"start": 10.0, "end": 10.0}, # zero length
    ]
    out = zoom_segments_from_hints(hints, [], duration_s=30.0)
    assert out == []


def test_hints_strength_maps_to_zoom_magnification() -> None:
    for s, expected in SPEECH_ZOOM_BY_STRENGTH.items():
        out = zoom_segments_from_hints(
            [{"start": 1.0, "end": 2.0, "strength": s}],
            [], duration_s=10.0,
        )
        assert out[0].zoom == expected


# --- merge_zoom_segments --------------------------------------------

def test_merge_keeps_non_overlapping_hints() -> None:
    a = [ZoomSegment(start=0, end=5, zoom=1.5, cx=0.3, cy=0.3)]
    b = [ZoomSegment(start=10, end=15, zoom=1.5, cx=0.5, cy=0.5)]
    out = merge_zoom_segments(a, b)
    assert len(out) == 2
    assert out[0].start == 0
    assert out[1].start == 10


def test_merge_drops_hint_whose_midpoint_overlaps_cursor_zoom() -> None:
    cursor = [ZoomSegment(start=5, end=10, zoom=1.5, cx=0.5, cy=0.5)]
    hint = [ZoomSegment(start=6, end=9, zoom=1.5, cx=0.5, cy=0.5)]
    out = merge_zoom_segments(cursor, hint)
    assert len(out) == 1
    assert out[0].start == 5


def test_merge_keeps_hint_adjacent_to_cursor_zoom() -> None:
    cursor = [ZoomSegment(start=5, end=10, zoom=1.5, cx=0.5, cy=0.5)]
    hint = [ZoomSegment(start=10, end=14, zoom=1.5, cx=0.5, cy=0.5)]
    out = merge_zoom_segments(cursor, hint)
    # Midpoint of hint is 12, outside [5, 10] → kept
    assert len(out) == 2


def test_merge_sorts_output_by_start() -> None:
    a = [ZoomSegment(start=20, end=25, zoom=1.5, cx=0.5, cy=0.5)]
    b = [ZoomSegment(start=5, end=8, zoom=1.5, cx=0.5, cy=0.5)]
    out = merge_zoom_segments(a, b)
    assert [z.start for z in out] == [5, 20]


# --- load_zoom_hints -------------------------------------------------

def test_load_zoom_hints_reads_json(tmp_path) -> None:
    p = tmp_path / "zoom_hints.json"
    p.write_text(json.dumps({
        "hints": [
            {"anchor_word_idx": 5, "start": 1.0, "end": 3.0, "strength": "normal"},
        ]
    }))
    out = load_zoom_hints(p)
    assert len(out) == 1
    assert out[0]["start"] == 1.0


def test_load_zoom_hints_skips_malformed_entries(tmp_path) -> None:
    p = tmp_path / "zoom_hints.json"
    p.write_text(json.dumps({
        "hints": [
            {"start": 1.0, "end": 3.0},
            {"no_start_or_end": 99},   # missing required fields
        ]
    }))
    out = load_zoom_hints(p)
    assert len(out) == 1


def test_load_zoom_hints_missing_file_returns_empty(tmp_path) -> None:
    assert load_zoom_hints(tmp_path / "nope.json") == []


def test_load_zoom_hints_corrupt_json_returns_empty(tmp_path) -> None:
    p = tmp_path / "zoom_hints.json"
    p.write_text("not json at all")
    assert load_zoom_hints(p) == []


# --- analyze_zoom_hints LLM wrapper ----------------------------------

def test_analyze_zoom_hints_parses_response() -> None:
    """The LLM's JSON payload parses into ZoomHintsResponse."""
    fake_llm_out = {
        "hints": [
            {"anchor_word_idx": 3, "start": 2.5, "end": 5.0,
             "strength": "normal", "reason": "look at this"}
        ]
    }
    with patch("src.stages.analyze_llm.call_llm_json", return_value=fake_llm_out):
        out = analyze_zoom_hints([{"word": "look", "start": 2.5, "end": 2.7}])
    assert len(out.hints) == 1
    assert out.hints[0].reason == "look at this"


def test_analyze_zoom_hints_accepts_empty_response() -> None:
    with patch("src.stages.analyze_llm.call_llm_json", return_value={"hints": []}):
        out = analyze_zoom_hints([])
    assert out.hints == []
