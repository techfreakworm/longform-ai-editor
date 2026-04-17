"""Tests for zoom-on-scroll / window-change detection.

Unit tests bypass ffmpeg by mocking `_sample_frames_gray`. The
clustering and event-detection math are testable without a real video.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.stages.scroll_zoom import (
    MAX_ZOOM_DURATION,
    MIN_ZOOM_DURATION,
    ChangeEvent,
    cluster_events_to_zooms,
    detect_changes_in_window,
    detect_scroll_zooms,
)


# --- cluster_events_to_zooms ----------------------------------------

def test_cluster_single_event_below_min_duration_dropped() -> None:
    """A single isolated event → span 0.2 + 0.6 = 0.8 s → kept (≥ 0.6 s)."""
    events = [ChangeEvent(t_s=5.0, cx_norm=0.5, cy_norm=0.5)]
    out = cluster_events_to_zooms(events, window_start_s=4.0, window_end_s=7.0,
                                  sample_rate_hz=2.0)
    assert len(out) == 1
    # Duration = end - start = (5.0 + 0.6) - (5.0 - 0.2) = 0.8
    assert (out[0].end - out[0].start) == pytest.approx(0.8)


def test_cluster_contiguous_events_become_one_zoom() -> None:
    """Three events at ~0.5 s spacing cluster into one zoom."""
    events = [
        ChangeEvent(t_s=5.0, cx_norm=0.3, cy_norm=0.5),
        ChangeEvent(t_s=5.5, cx_norm=0.35, cy_norm=0.5),
        ChangeEvent(t_s=6.0, cx_norm=0.4, cy_norm=0.5),
    ]
    out = cluster_events_to_zooms(events, 4.0, 10.0, sample_rate_hz=2.0)
    assert len(out) == 1
    # Centroid = mean
    assert out[0].cx == pytest.approx(0.35)


def test_cluster_split_by_gap() -> None:
    """Two events 3 s apart → two separate clusters."""
    events = [
        ChangeEvent(t_s=5.0, cx_norm=0.3, cy_norm=0.3),
        ChangeEvent(t_s=8.0, cx_norm=0.7, cy_norm=0.7),
    ]
    out = cluster_events_to_zooms(events, 0.0, 15.0, sample_rate_hz=2.0)
    assert len(out) == 2
    assert out[0].cx == pytest.approx(0.3)
    assert out[1].cx == pytest.approx(0.7)


def test_cluster_respects_window_bounds() -> None:
    """Pre/post padding should be clamped to the cursor-idle window."""
    events = [ChangeEvent(t_s=5.1, cx_norm=0.5, cy_norm=0.5)]
    # Use a wider window so clamping is observable without hitting MIN_ZOOM_DURATION.
    out = cluster_events_to_zooms(events, window_start_s=5.0,
                                  window_end_s=6.0, sample_rate_hz=2.0)
    assert len(out) == 1
    assert out[0].start >= 5.0
    assert out[0].end <= 6.0
    # Verify clamping on the left edge — requested start is 5.1 - 0.2 = 4.9
    # but window starts at 5.0 so it should clamp to 5.0.
    assert out[0].start == pytest.approx(5.0)


def test_cluster_caps_at_max_duration() -> None:
    """A very long change run is capped at MAX_ZOOM_DURATION."""
    events = [
        ChangeEvent(t_s=5.0 + 0.5 * i, cx_norm=0.5, cy_norm=0.5)
        for i in range(20)  # spans 10 s
    ]
    out = cluster_events_to_zooms(events, 4.0, 20.0, sample_rate_hz=2.0)
    assert len(out) == 1
    assert (out[0].end - out[0].start) <= MAX_ZOOM_DURATION + 1e-6


def test_cluster_empty_returns_empty() -> None:
    assert cluster_events_to_zooms([], 0.0, 10.0) == []


# --- detect_changes_in_window ---------------------------------------

def _make_blank_frame(w: int, h: int, fill: int = 0) -> np.ndarray:
    return np.full((h, w), fill, dtype=np.uint8)


def _make_frame_with_patch(
    w: int, h: int, bx: int, by: int, bw: int, bh: int,
    bg: int = 0, fg: int = 255,
) -> np.ndarray:
    f = _make_blank_frame(w, h, bg)
    f[by:by + bh, bx:bx + bw] = fg
    return f


def test_detect_changes_no_change_returns_empty(tmp_path) -> None:
    """Two identical frames → no change events."""
    f = _make_blank_frame(640, 360, fill=128)
    fake = ([f, f], [5.0, 5.5], (640, 360))
    with patch("src.stages.scroll_zoom._sample_frames_gray", return_value=fake):
        events = detect_changes_in_window(
            tmp_path / "s.mp4", 5.0, 6.0, diff_threshold=0.04,
        )
    assert events == []


def test_detect_changes_large_change_fires(tmp_path) -> None:
    """A big white patch appearing triggers a change with centroid at patch."""
    f0 = _make_blank_frame(640, 360, fill=0)
    f1 = _make_frame_with_patch(640, 360, bx=100, by=50, bw=200, bh=100)
    fake = ([f0, f1], [5.0, 5.5], (640, 360))
    with patch("src.stages.scroll_zoom._sample_frames_gray", return_value=fake):
        events = detect_changes_in_window(
            tmp_path / "s.mp4", 5.0, 6.0,
            diff_threshold=0.04, pixel_change_threshold=30,
        )
    assert len(events) == 1
    # Bbox: x 100..299, y 50..149 → bbox midpoint ≈ (199.5/640, 99.5/360).
    # Approximate equality — the half-pixel difference from numpy's
    # inclusive-min/max doesn't matter for zoom centering.
    assert events[0].cx_norm == pytest.approx(199.5 / 640, abs=5e-3)
    assert events[0].cy_norm == pytest.approx(99.5 / 360, abs=5e-3)


def test_detect_changes_below_threshold_ignored(tmp_path) -> None:
    """A tiny 10x10 patch on a 640x360 frame (~0.04% of pixels) is below 4%."""
    f0 = _make_blank_frame(640, 360, fill=0)
    f1 = _make_frame_with_patch(640, 360, bx=100, by=50, bw=10, bh=10)
    fake = ([f0, f1], [5.0, 5.5], (640, 360))
    with patch("src.stages.scroll_zoom._sample_frames_gray", return_value=fake):
        events = detect_changes_in_window(
            tmp_path / "s.mp4", 5.0, 6.0, diff_threshold=0.04,
        )
    assert events == []


def test_detect_changes_one_frame_returns_empty(tmp_path) -> None:
    """Can't diff a single frame."""
    fake = ([_make_blank_frame(640, 360)], [5.0], (640, 360))
    with patch("src.stages.scroll_zoom._sample_frames_gray", return_value=fake):
        events = detect_changes_in_window(tmp_path / "s.mp4", 5.0, 6.0)
    assert events == []


# --- detect_scroll_zooms end-to-end ----------------------------------

def test_detect_scroll_zooms_skips_windows_below_min(tmp_path) -> None:
    """A 1.0 s cursor-idle window is below MIN_WINDOW_SEC = 1.5 s."""
    # Should NOT call _sample_frames_gray at all.
    with patch("src.stages.scroll_zoom._sample_frames_gray",
               side_effect=AssertionError("should not be called")):
        out = detect_scroll_zooms(tmp_path / "s.mp4", [(5.0, 6.0)])
    assert out == []


def test_detect_scroll_zooms_handles_multiple_windows(tmp_path) -> None:
    """Two separate cursor-idle windows produce zooms only where changes occurred."""
    # Window 1 has a change; Window 2 is static.
    f_empty = _make_blank_frame(640, 360, fill=0)
    f_patch = _make_frame_with_patch(640, 360, bx=200, by=100, bw=200, bh=100)
    seq = iter([
        ([f_empty, f_patch, f_patch], [5.0, 5.5, 6.0], (640, 360)),  # changes once
        ([f_empty, f_empty, f_empty], [10.0, 10.5, 11.0], (640, 360)),  # no change
    ])
    with patch("src.stages.scroll_zoom._sample_frames_gray",
               side_effect=lambda *_a, **_kw: next(seq)):
        out = detect_scroll_zooms(
            tmp_path / "s.mp4",
            [(5.0, 7.0), (10.0, 12.0)],
        )
    assert len(out) == 1
    # Centroid for window-1 patch ≈ (0.469, 0.417)
    assert 0.4 < out[0].cx < 0.55
