"""Tests for src/stages/cursor_idle.py."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.stages.cursor_idle import (
    IdleInterval,
    detect_cursor_idle_intervals,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = ["t_s", "event", "x", "y", "down"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            r.setdefault("x", 0.0)
            r.setdefault("y", 0.0)
            r.setdefault("down", "")
            writer.writerow(r)


def test_no_moves_means_entire_clip_is_idle(tmp_path) -> None:
    csv_path = tmp_path / "c.csv"
    _write_csv(csv_path, [])  # empty
    out = detect_cursor_idle_intervals(csv_path, duration_s=10.0, min_idle_sec=2.0)
    assert out == [IdleInterval(start=0.0, end=10.0)]


def test_leading_idle_emitted(tmp_path) -> None:
    """Moves start at t=3 → [0, 3] is a 3 s leading idle."""
    csv_path = tmp_path / "c.csv"
    _write_csv(csv_path, [
        {"t_s": 3.0, "event": "move"},
        {"t_s": 3.1, "event": "move"},
        {"t_s": 3.2, "event": "move"},
    ])
    out = detect_cursor_idle_intervals(csv_path, duration_s=5.0, min_idle_sec=2.0)
    assert len(out) == 1
    assert out[0] == IdleInterval(start=0.0, end=3.0)


def test_trailing_idle_emitted(tmp_path) -> None:
    csv_path = tmp_path / "c.csv"
    _write_csv(csv_path, [
        {"t_s": 0.5, "event": "move"},
        {"t_s": 1.0, "event": "move"},
        {"t_s": 1.5, "event": "move"},
    ])
    out = detect_cursor_idle_intervals(csv_path, duration_s=10.0, min_idle_sec=2.0)
    assert len(out) == 1
    assert out[0] == IdleInterval(start=1.5, end=10.0)


def test_middle_idle_between_active_bursts(tmp_path) -> None:
    """Two bursts of activity with a 4 s gap → one middle idle interval."""
    csv_path = tmp_path / "c.csv"
    _write_csv(csv_path, [
        {"t_s": 0.5, "event": "move"},
        {"t_s": 0.7, "event": "move"},
        # 4 s gap
        {"t_s": 4.7, "event": "move"},
        {"t_s": 5.0, "event": "move"},
    ])
    out = detect_cursor_idle_intervals(csv_path, duration_s=5.1, min_idle_sec=2.0)
    # Expected idles: [0, 0.5] (below threshold — skipped),
    #                 [0.7, 4.7] (4.0 s — kept)
    #                 [5.0, 5.1] (0.1 s — below threshold — skipped)
    assert len(out) == 1
    assert out[0].start == pytest.approx(0.7)
    assert out[0].end == pytest.approx(4.7)


def test_ignores_click_and_scroll_events(tmp_path) -> None:
    """Only 'move' events count — rapid clicks should not break idle run."""
    csv_path = tmp_path / "c.csv"
    _write_csv(csv_path, [
        {"t_s": 1.0, "event": "click", "down": "1"},
        {"t_s": 1.1, "event": "click", "down": "0"},
        {"t_s": 2.5, "event": "scroll"},
        # no moves → whole clip is idle
    ])
    out = detect_cursor_idle_intervals(csv_path, duration_s=10.0, min_idle_sec=2.0)
    assert len(out) == 1
    assert out[0] == IdleInterval(start=0.0, end=10.0)


def test_idle_below_threshold_not_emitted(tmp_path) -> None:
    """1.9 s gap with 2.0 s threshold → no interval."""
    csv_path = tmp_path / "c.csv"
    _write_csv(csv_path, [
        {"t_s": 1.0, "event": "move"},
        {"t_s": 2.9, "event": "move"},  # 1.9 s gap
        {"t_s": 3.5, "event": "move"},
    ])
    out = detect_cursor_idle_intervals(csv_path, duration_s=5.0, min_idle_sec=2.0)
    assert len(out) == 0


def test_moves_outside_duration_ignored(tmp_path) -> None:
    """A stray move with t > duration_s should not invalidate trailing idle."""
    csv_path = tmp_path / "c.csv"
    _write_csv(csv_path, [
        {"t_s": 0.5, "event": "move"},
        {"t_s": 999.0, "event": "move"},  # well past clip end
    ])
    out = detect_cursor_idle_intervals(csv_path, duration_s=10.0, min_idle_sec=2.0)
    # Only the 0.5 s move counts → trailing idle [0.5, 10.0]
    assert len(out) == 1
    assert out[0] == IdleInterval(start=0.5, end=10.0)


def test_malformed_rows_skipped(tmp_path) -> None:
    csv_path = tmp_path / "c.csv"
    with csv_path.open("w") as f:
        f.write("t_s,event\n")
        f.write("not_a_float,move\n")
        f.write("2.0,move\n")
    out = detect_cursor_idle_intervals(csv_path, duration_s=10.0, min_idle_sec=2.0)
    # Only t=2.0 s counts → leading idle [0, 2.0] is exactly 2.0 s (≥ threshold),
    # trailing idle [2.0, 10.0] is 8.0 s.
    assert len(out) == 2
    assert out[0] == IdleInterval(start=0.0, end=2.0)
    assert out[1] == IdleInterval(start=2.0, end=10.0)
