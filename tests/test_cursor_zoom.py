"""Tests for cursor_zoom — port of Cap's algorithm.

Covers each piece of the pipeline in isolation plus an end-to-end run on
the real cursor.csv fixture. Synthetic CSVs are generated in tmp_path as
needed — no external fixture files required for most tests.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.stages.cursor_zoom import (
    AUTO_ZOOM_AMOUNT,
    CLICK_GROUP_SPATIAL_THRESHOLD,
    CLICK_GROUP_TIME_THRESHOLD_SECS,
    CLICK_POST_PADDING,
    CLICK_PRE_PADDING,
    MAX_SEGMENT_DURATION,
    MERGE_GAP_THRESHOLD,
    MIN_SEGMENT_DURATION,
    MOVEMENT_POST_PADDING,
    MOVEMENT_PRE_PADDING,
    CursorEvent,
    ZoomSegment,
    click_intervals,
    compute_centroid,
    filter_short,
    generate_zoom_segments,
    group_clicks,
    merge_intervals,
    movement_intervals,
    parse_cursor_csv,
)

SCREEN_W = 1920
SCREEN_H = 1080


# --- synthetic CSV helper ------------------------------------------------

def _write_csv(path: Path, rows: list[dict]) -> Path:
    """rows: list of {t_s, x, y, event, [button], [down]}."""
    with path.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["t_s", "x", "y", "event", "button", "down"]
        )
        w.writeheader()
        for r in rows:
            w.writerow({
                "t_s": r["t_s"], "x": r["x"], "y": r["y"],
                "event": r["event"],
                "button": r.get("button", ""),
                "down": r.get("down", ""),
            })
    return path


# --- parse_cursor_csv ---------------------------------------------------

def test_parse_csv_normalizes_coords(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "c.csv", [
        {"t_s": 0.0, "x": 960, "y": 540, "event": "move"},
        {"t_s": 0.1, "x": 0, "y": 0, "event": "click", "button": "left", "down": "1"},
        {"t_s": 0.2, "x": 1920, "y": 1080, "event": "click", "button": "left", "down": "1"},
    ])
    clicks, moves = parse_cursor_csv(path, SCREEN_W, SCREEN_H)
    assert len(moves) == 1
    assert moves[0].x == pytest.approx(0.5)
    assert moves[0].y == pytest.approx(0.5)
    # Both clicks are at exactly the screen edges — within [0,1] inclusive.
    assert len(clicks) == 2


def test_parse_csv_drops_offscreen_clicks(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "c.csv", [
        {"t_s": 0.0, "x": -50, "y": 100, "event": "click", "button": "left", "down": "1"},
        {"t_s": 0.1, "x": 5000, "y": 100, "event": "click", "button": "left", "down": "1"},
        {"t_s": 0.2, "x": 960, "y": 540, "event": "click", "button": "left", "down": "1"},
    ])
    clicks, _ = parse_cursor_csv(path, SCREEN_W, SCREEN_H)
    assert len(clicks) == 1
    assert clicks[0].t_s == 0.2


def test_parse_csv_origin_offset(tmp_path: Path) -> None:
    """Cursor logged in macOS global space with origin at (1440, -1440)
    (external monitor above a 1440-wide macbook display) should normalize
    correctly when origin is passed."""
    path = _write_csv(tmp_path / "c.csv", [
        {"t_s": 0.0, "x": 2400, "y": -900, "event": "move"},
    ])
    _, moves = parse_cursor_csv(
        path, SCREEN_W, SCREEN_H, origin_x=1440, origin_y=-1440,
    )
    assert len(moves) == 1
    assert moves[0].x == pytest.approx((2400 - 1440) / 1920)
    assert moves[0].y == pytest.approx((-900 - -1440) / 1080)


def test_parse_csv_ignores_click_up(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "c.csv", [
        {"t_s": 0.0, "x": 100, "y": 100, "event": "click", "button": "left", "down": "1"},
        {"t_s": 0.01, "x": 100, "y": 100, "event": "click", "button": "left", "down": "0"},
    ])
    clicks, _ = parse_cursor_csv(path, SCREEN_W, SCREEN_H)
    assert len(clicks) == 1  # only the down edge


# --- group_clicks -------------------------------------------------------

def test_group_clicks_splits_by_time(tmp_path: Path) -> None:
    clicks = [
        CursorEvent(t_s=1.0, x=0.5, y=0.5, is_click=True),
        # same spot, but 3 s later — beyond CLICK_GROUP_TIME_THRESHOLD_SECS=2.5
        CursorEvent(t_s=4.0, x=0.5, y=0.5, is_click=True),
    ]
    groups = group_clicks(clicks, end_limit=100.0)
    assert len(groups) == 2


def test_group_clicks_splits_by_distance() -> None:
    clicks = [
        CursorEvent(t_s=1.0, x=0.1, y=0.1, is_click=True),
        # 1 s later but far away — beyond CLICK_GROUP_SPATIAL_THRESHOLD=0.15
        CursorEvent(t_s=2.0, x=0.9, y=0.9, is_click=True),
    ]
    groups = group_clicks(clicks, end_limit=100.0)
    assert len(groups) == 2


def test_group_clicks_joins_close_in_time_and_space() -> None:
    clicks = [
        CursorEvent(t_s=1.0, x=0.5, y=0.5, is_click=True),
        CursorEvent(t_s=1.8, x=0.52, y=0.49, is_click=True),  # Δt<2.5, dist<0.15
        CursorEvent(t_s=2.5, x=0.53, y=0.5, is_click=True),
    ]
    groups = group_clicks(clicks, end_limit=100.0)
    assert len(groups) == 1
    assert len(groups[0]) == 3


def test_group_clicks_respects_end_limit() -> None:
    clicks = [
        CursorEvent(t_s=1.0, x=0.5, y=0.5, is_click=True),
        CursorEvent(t_s=99.0, x=0.5, y=0.5, is_click=True),
    ]
    groups = group_clicks(clicks, end_limit=10.0)
    assert len(groups) == 1  # the second click is dropped


# --- click_intervals ----------------------------------------------------

def test_click_intervals_pads() -> None:
    groups = [[
        CursorEvent(t_s=5.0, x=0.5, y=0.5, is_click=True),
        CursorEvent(t_s=5.5, x=0.5, y=0.5, is_click=True),
    ]]
    ivs = click_intervals(groups, end_limit=100.0)
    assert len(ivs) == 1
    s, e = ivs[0]
    assert s == pytest.approx(5.0 - CLICK_PRE_PADDING)
    assert e == pytest.approx(5.5 + CLICK_POST_PADDING)


def test_click_intervals_clamps_to_zero() -> None:
    groups = [[CursorEvent(t_s=0.1, x=0.5, y=0.5, is_click=True)]]
    ivs = click_intervals(groups, end_limit=100.0)
    assert ivs[0][0] == 0.0  # padding would go negative; clamped


# --- movement_intervals -------------------------------------------------

def test_movement_emits_on_big_jump() -> None:
    """A single move with per-event distance > threshold emits an interval."""
    moves = [
        CursorEvent(t_s=0.0, x=0.1, y=0.1, is_click=False),
        # Per-step distance = 0.8 * sqrt(2) ≈ 1.13, far above threshold 0.02.
        CursorEvent(t_s=0.5, x=0.9, y=0.9, is_click=False),
    ]
    ivs = movement_intervals(moves, end_limit=100.0)
    assert len(ivs) == 1
    # Interval centered on the 0.5 s move with pre/post padding
    s, e = ivs[0]
    assert s == pytest.approx(0.5 - MOVEMENT_PRE_PADDING)
    assert e == pytest.approx(0.5 + MOVEMENT_POST_PADDING)


def test_movement_suppresses_tiny_jitter() -> None:
    """Sub-threshold per-step jitter that doesn't accumulate to the window
    threshold should emit NO intervals."""
    moves = [
        CursorEvent(t_s=t, x=0.5 + ((-1) ** i) * 0.001, y=0.5, is_click=False)
        for i, t in enumerate([0.0, 0.05, 0.10, 0.15, 0.20, 0.25])
    ]
    ivs = movement_intervals(moves, end_limit=100.0)
    assert ivs == []


def test_movement_shake_filter_suppresses_sustained_jitter() -> None:
    """Rapid zig-zag within SHAKE_FILTER_WINDOW_MS is filtered after startup.

    Cap's shake filter needs ≥ 4 events in the 150 ms window before it can
    count ≥ 2 reversals — so the first 3 events always leak through as
    startup. What matters: continuing jitter does NOT keep emitting. 20
    zigzag events should produce far fewer intervals than 20 straight-line
    events of the same per-step distance.
    """
    def gen(xs: list[float]) -> list[CursorEvent]:
        return [
            CursorEvent(t_s=i * 0.030, x=x, y=0.5, is_click=False)
            for i, x in enumerate(xs)
        ]

    n = 20
    # zigzag: alternates ±0.025 around center — per-step distance 0.05
    zigzag = gen([0.5 + ((-1) ** i) * 0.025 for i in range(n)])
    # straight: monotonic increase of 0.03 per step
    straight = gen([0.1 + i * 0.03 for i in range(n)])

    zig_ivs = movement_intervals(zigzag, end_limit=100.0)
    str_ivs = movement_intervals(straight, end_limit=100.0)

    # Shake filter must fire at some point — zigzag emits < 1/3 of straight.
    assert len(zig_ivs) < len(str_ivs) / 3, (len(zig_ivs), len(str_ivs))
    # And the absolute count stays in the startup-lag range (≤ 3).
    assert len(zig_ivs) <= 3


# --- merge + filter -----------------------------------------------------

def test_merge_intervals_joins_close_ranges() -> None:
    merged = merge_intervals([(0.0, 1.0), (1.5, 2.5), (5.0, 6.0)])
    # gap 0.5 ≤ MERGE_GAP_THRESHOLD (0.8) → merge first two
    assert merged == [(0.0, 2.5), (5.0, 6.0)]


def test_merge_intervals_separates_far_ranges() -> None:
    merged = merge_intervals([(0.0, 1.0), (2.0, 3.0)])
    # gap 1.0 > MERGE_GAP_THRESHOLD → keep separate
    assert merged == [(0.0, 1.0), (2.0, 3.0)]


def test_filter_short_drops_below_min() -> None:
    kept = filter_short([(0.0, 0.5), (1.0, 2.5)], min_duration=MIN_SEGMENT_DURATION)
    assert kept == [(1.0, 2.5)]


# --- centroid -----------------------------------------------------------

def test_compute_centroid_mean() -> None:
    moves = [
        CursorEvent(t_s=0.0, x=0.2, y=0.3, is_click=False),
        CursorEvent(t_s=0.5, x=0.4, y=0.7, is_click=False),
    ]
    cx, cy = compute_centroid(moves, 0.0, 1.0)
    assert cx == pytest.approx(0.3)
    assert cy == pytest.approx(0.5)


def test_compute_centroid_empty_fallback() -> None:
    cx, cy = compute_centroid([], 0.0, 1.0)
    assert (cx, cy) == (0.5, 0.5)


# --- end-to-end ---------------------------------------------------------

def test_generate_segments_clicks_only(tmp_path: Path) -> None:
    rows = [
        {"t_s": 5.0, "x": 960, "y": 540, "event": "click", "button": "left", "down": "1"},
        {"t_s": 5.3, "x": 980, "y": 540, "event": "click", "button": "left", "down": "1"},
    ]
    path = _write_csv(tmp_path / "c.csv", rows)
    segs = generate_zoom_segments(path, SCREEN_W, SCREEN_H, duration_s=30.0)
    assert len(segs) == 1
    s = segs[0]
    assert s.zoom == AUTO_ZOOM_AMOUNT
    assert s.start == pytest.approx(5.0 - CLICK_PRE_PADDING)
    assert s.end == pytest.approx(5.3 + CLICK_POST_PADDING)
    # Centroid fallback to (0.5, 0.5) since no moves in the interval
    assert s.cx == pytest.approx(0.5)
    assert s.cy == pytest.approx(0.5)


def test_generate_segments_caps_length(tmp_path: Path) -> None:
    """A sustained motion run exceeding MAX_SEGMENT_DURATION is clipped."""
    rows = []
    for i in range(60):
        rows.append({
            "t_s": i * 0.2,
            "x": 100 + i * 20,  # big per-event jumps → triggers movement_intervals
            "y": 500,
            "event": "move",
        })
    path = _write_csv(tmp_path / "c.csv", rows)
    segs = generate_zoom_segments(path, SCREEN_W, SCREEN_H, duration_s=30.0)
    assert len(segs) >= 1
    for s in segs:
        assert (s.end - s.start) <= MAX_SEGMENT_DURATION + 1e-6


def test_generate_segments_respects_stop_padding(tmp_path: Path) -> None:
    """Events in the last STOP_PADDING_SECONDS are dropped."""
    rows = [
        # event at t=9.8 with duration=10.0 → inside 0.5 s stop padding → dropped
        {"t_s": 9.8, "x": 100, "y": 500, "event": "click", "button": "left", "down": "1"},
    ]
    path = _write_csv(tmp_path / "c.csv", rows)
    segs = generate_zoom_segments(path, SCREEN_W, SCREEN_H, duration_s=10.0)
    assert segs == []


# --- Real fixture integration -----------------------------------------

@pytest.mark.slow
def test_generate_segments_on_real_cursor_csv(real_cursor: Path) -> None:
    """End-to-end against the real OBS-session cursor.csv.

    ext.mov is 2560×1440. We don't know the macOS origin for the external
    display on the user's machine, so we skip normalization sanity and just
    assert:
      - the function runs without crashing
      - returns a list of ZoomSegment
      - every segment respects duration & zoom invariants
    """
    # Assume external monitor at origin (0,0) for this test — ballpark is
    # fine, the actual per-creator origin is configured in M4 via .env.
    segs = generate_zoom_segments(
        real_cursor, screen_w=2560, screen_h=1440, duration_s=95.0,
    )
    assert isinstance(segs, list)
    for s in segs:
        assert isinstance(s, ZoomSegment)
        assert s.zoom == AUTO_ZOOM_AMOUNT
        assert s.end > s.start
        assert (s.end - s.start) >= MIN_SEGMENT_DURATION
        assert (s.end - s.start) <= MAX_SEGMENT_DURATION + 1e-6
        # Allow out-of-[0,1] centroids because we guessed the origin wrong;
        # just assert they're finite numbers.
        assert -10.0 <= s.cx <= 10.0
        assert -10.0 <= s.cy <= 10.0
