"""Tests for element-aware zoom snapping (OCR target detection).

Unit tests mock the paddleocr reader so we don't need the ~300 MB wheel
installed. The snapping math is tested with synthetic ElementBox lists.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.stages.cursor_zoom import ZoomSegment
from src.stages.element_aware import (
    ElementBox,
    _cache_key,
    ocr_elements_at,
    snap_centroid_to_element,
    snap_zoom_segments,
)


# --- ElementBox -------------------------------------------------------

def test_element_box_centroid() -> None:
    e = ElementBox(x=100, y=200, w=40, h=20, text="Run")
    assert e.cx == 120
    assert e.cy == 210


# --- snap math --------------------------------------------------------

def test_snap_to_nearest_element_within_threshold(tmp_path) -> None:
    """Snap a cursor at the edge of a button to that button's center."""
    # Frame: 1920x1080. Cursor at (0.25, 0.5) = (480, 540).
    # Element at (470, 535) → (500, 555) → center (485, 545), ~7 px away.
    elements = [
        ElementBox(x=470, y=535, w=30, h=20, text="OK"),
        ElementBox(x=1000, y=100, w=50, h=20, text="Settings"),
    ]
    with patch("src.stages.element_aware.ocr_elements_at", return_value=elements):
        cx, cy = snap_centroid_to_element(
            0.25, 0.5, tmp_path / "screen.mp4", t_s=1.0,
            frame_w=1920, frame_h=1080, max_distance_px=150,
        )
    # Should snap to first element's center (485, 545) / (1920, 1080).
    assert cx == pytest.approx(485 / 1920)
    assert cy == pytest.approx(545 / 1080)


def test_snap_refuses_beyond_threshold(tmp_path) -> None:
    """If the nearest element is far away, return input unchanged."""
    elements = [ElementBox(x=1500, y=900, w=30, h=20, text="Done")]
    with patch("src.stages.element_aware.ocr_elements_at", return_value=elements):
        cx, cy = snap_centroid_to_element(
            0.2, 0.2,  # cursor far from element
            tmp_path / "screen.mp4", t_s=1.0,
            frame_w=1920, frame_h=1080, max_distance_px=150,
        )
    assert (cx, cy) == (0.2, 0.2)


def test_snap_with_no_elements_returns_input(tmp_path) -> None:
    """Empty OCR result → unchanged centroid."""
    with patch("src.stages.element_aware.ocr_elements_at", return_value=[]):
        cx, cy = snap_centroid_to_element(
            0.5, 0.5, tmp_path / "screen.mp4", t_s=1.0,
            frame_w=1920, frame_h=1080,
        )
    assert (cx, cy) == (0.5, 0.5)


def test_snap_zoom_segments_replaces_centroids(tmp_path) -> None:
    """snap_zoom_segments processes a list of ZoomSegments."""
    zooms = [
        ZoomSegment(start=1.0, end=3.0, zoom=1.5, cx=0.3, cy=0.3),
        ZoomSegment(start=5.0, end=7.0, zoom=1.5, cx=0.7, cy=0.7),
    ]
    elements_seq = [
        [ElementBox(x=576, y=324, w=10, h=10, text="A")],   # ≈ (0.3047, 0.3093)
        [],                                                  # no snap
    ]
    def fake_ocr(screen_path, t, cache_dir=None):
        return elements_seq.pop(0)

    with patch("src.stages.element_aware.ocr_elements_at", side_effect=fake_ocr):
        out = snap_zoom_segments(
            zooms, tmp_path / "screen.mp4",
            frame_w=1920, frame_h=1080,
        )
    assert len(out) == 2
    # First segment snapped
    assert out[0].cx != 0.3 or out[0].cy != 0.3
    # Second kept (no element)
    assert out[1].cx == 0.7 and out[1].cy == 0.7


# --- paddleocr graceful degradation ----------------------------------

def test_ocr_elements_at_without_paddleocr_returns_empty(tmp_path) -> None:
    """When paddleocr import fails → return []."""
    with patch("src.stages.element_aware._try_import_paddleocr", return_value=None):
        out = ocr_elements_at(tmp_path / "anyfile.mp4", 1.0)
    assert out == []


# --- cache key stability ---------------------------------------------

def test_cache_key_stable_for_same_file_and_second(tmp_path) -> None:
    f = tmp_path / "screen.mp4"
    f.write_bytes(b"fakevideodata")
    k1 = _cache_key(f, 5.1)
    k2 = _cache_key(f, 5.9)  # same integer second → same key
    k3 = _cache_key(f, 6.0)  # different second
    assert k1 == k2
    assert k1 != k3


def test_cache_key_different_file_different_key(tmp_path) -> None:
    f1 = tmp_path / "a.mp4"
    f1.write_bytes(b"a")
    f2 = tmp_path / "b.mp4"
    f2.write_bytes(b"different")
    assert _cache_key(f1, 1.0) != _cache_key(f2, 1.0)


def test_ocr_caches_to_disk(tmp_path) -> None:
    """Second call at same timestamp reads from cache, skipping OCR."""
    screen_path = tmp_path / "screen.mp4"
    screen_path.write_bytes(b"fakevideo")
    cache_dir = tmp_path / "cache"

    fake_reader = MagicMock()
    fake_reader.ocr.return_value = [[
        ([[10, 20], [40, 20], [40, 30], [10, 30]], ("Run", 0.9)),
    ]]

    with patch("src.stages.element_aware._try_import_paddleocr",
               return_value=fake_reader), \
         patch("src.stages.element_aware._extract_frame",
               side_effect=lambda sp, t, out: out.write_bytes(b"fakeframe") or out):
        out1 = ocr_elements_at(screen_path, 1.0, cache_dir=cache_dir)
        assert len(out1) == 1
        assert fake_reader.ocr.call_count == 1

        out2 = ocr_elements_at(screen_path, 1.0, cache_dir=cache_dir)
        assert len(out2) == 1
        # Second call should not hit the reader
        assert fake_reader.ocr.call_count == 1
