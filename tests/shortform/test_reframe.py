"""Tests for src/shortform/reframe.py.

Unit tests mock PySceneDetect / MediaPipe / cv2 / OneEuroFilter so we
don't need those heavy deps installed for the fast suite.
"""
from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import patch

import pytest

from src.shortform.reframe import (
    CropWindow,
    Scene,
    _cursor_centroid_per_scene,
    _scenes_or_single,
    _smooth_centroids,
    build_screen_crops,
    build_webcam_crops,
    detect_scenes,
)


# --- Scene / CropWindow dataclasses ---------------------------------

def test_scene_fields() -> None:
    s = Scene(start=1.0, end=5.0)
    assert s.end - s.start == 4.0


def test_crop_window_fields() -> None:
    c = CropWindow(start=0.0, end=10.0, cx=0.5, cy=0.5)
    assert (c.cx, c.cy) == (0.5, 0.5)


def test_scenes_or_single_empty_becomes_one() -> None:
    out = _scenes_or_single([], duration_s=30.0)
    assert len(out) == 1
    assert out[0] == Scene(start=0.0, end=30.0)


def test_scenes_or_single_pass_through() -> None:
    xs = [Scene(0.0, 5.0), Scene(5.0, 12.0)]
    assert _scenes_or_single(xs, duration_s=99.0) == xs


# --- detect_scenes fallback -----------------------------------------

def test_detect_scenes_returns_empty_when_scenedetect_missing(tmp_path) -> None:
    fake_path = tmp_path / "cam.mov"
    # Patch the import inside the function to raise
    with patch("src.shortform.reframe.Scene") as _unused:
        pass  # just importable

    # Simulate scenedetect import failure by patching the module import.
    with patch.dict("sys.modules", {"scenedetect": None}):
        out = detect_scenes(fake_path)
    assert out == []


# --- cursor centroid -------------------------------------------------

def _write_cursor_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = ["t_s", "event", "x", "y", "down"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            r.setdefault("x", 0.0)
            r.setdefault("y", 0.0)
            r.setdefault("down", "")
            writer.writerow(r)


def test_cursor_centroid_missing_csv_defaults_center(tmp_path) -> None:
    scenes = [Scene(0.0, 5.0), Scene(5.0, 10.0)]
    out = _cursor_centroid_per_scene(
        None, scenes,
        screen_w=1920, screen_h=1080,
        origin_x=0.0, origin_y=0.0, csv_to_video_offset_s=0.0,
    )
    assert out == [(0.5, 0.5), (0.5, 0.5)]


def test_cursor_centroid_computes_mean_per_scene(tmp_path) -> None:
    csv_path = tmp_path / "c.csv"
    _write_cursor_csv(csv_path, [
        # moves in scene 1 (0-5s) around (480, 270) → normalized (0.25, 0.25)
        {"t_s": 1.0, "event": "move", "x": 480, "y": 270},
        {"t_s": 2.0, "event": "move", "x": 480, "y": 270},
        # moves in scene 2 (5-10s) around (1440, 810) → (0.75, 0.75)
        {"t_s": 6.0, "event": "move", "x": 1440, "y": 810},
        {"t_s": 7.0, "event": "move", "x": 1440, "y": 810},
    ])
    scenes = [Scene(0.0, 5.0), Scene(5.0, 10.0)]
    out = _cursor_centroid_per_scene(
        csv_path, scenes,
        screen_w=1920, screen_h=1080,
        origin_x=0.0, origin_y=0.0, csv_to_video_offset_s=0.0,
    )
    assert len(out) == 2
    assert out[0] == pytest.approx((0.25, 0.25))
    assert out[1] == pytest.approx((0.75, 0.75))


def test_cursor_centroid_empty_scene_defaults_center(tmp_path) -> None:
    csv_path = tmp_path / "c.csv"
    _write_cursor_csv(csv_path, [
        {"t_s": 0.5, "event": "move", "x": 100, "y": 100},
    ])
    scenes = [Scene(10.0, 20.0)]  # scene has no moves
    out = _cursor_centroid_per_scene(
        csv_path, scenes,
        screen_w=1920, screen_h=1080,
        origin_x=0.0, origin_y=0.0, csv_to_video_offset_s=0.0,
    )
    assert out == [(0.5, 0.5)]


def test_cursor_centroid_applies_csv_offset(tmp_path) -> None:
    """A move at CSV t=2 s with offset=+3 s becomes video t=5 s."""
    csv_path = tmp_path / "c.csv"
    _write_cursor_csv(csv_path, [
        {"t_s": 2.0, "event": "move", "x": 960, "y": 540},
    ])
    scenes = [Scene(4.0, 7.0)]
    out = _cursor_centroid_per_scene(
        csv_path, scenes,
        screen_w=1920, screen_h=1080,
        origin_x=0.0, origin_y=0.0, csv_to_video_offset_s=3.0,
    )
    assert out == [pytest.approx((0.5, 0.5))]  # single event at exact center


# --- smoothing -------------------------------------------------------

def test_smooth_centroids_without_library_is_identity() -> None:
    """When OneEuroFilter isn't installed, centroids pass through."""
    centroids = [(0.3, 0.4), (0.6, 0.5)]
    scenes = [Scene(0.0, 5.0), Scene(5.0, 10.0)]
    with patch.dict("sys.modules", {"OneEuroFilter": None}):
        out = _smooth_centroids(centroids, scenes)
    assert out == centroids


def test_smooth_centroids_empty_returns_empty() -> None:
    assert _smooth_centroids([], []) == []


# --- build_webcam_crops: full stack mocked --------------------------

def test_build_webcam_crops_one_scene_fallback(tmp_path) -> None:
    """No scenedetect → single scene. No mediapipe → static offset."""
    cam = tmp_path / "cam.mov"
    cam.write_bytes(b"")
    with patch("src.shortform.reframe.detect_scenes", return_value=[]), \
         patch("src.shortform.reframe._face_centroid_per_scene_haar",
               return_value=[(0.4, 0.6)]), \
         patch("src.shortform.reframe._smooth_centroids",
               side_effect=lambda c, s: c):
        out = build_webcam_crops(cam, duration_s=30.0)
    assert len(out) == 1
    assert out[0].start == 0.0
    assert out[0].end == 30.0
    assert (out[0].cx, out[0].cy) == (0.4, 0.6)


def test_build_webcam_crops_multi_scene(tmp_path) -> None:
    cam = tmp_path / "cam.mov"
    cam.write_bytes(b"")
    scenes = [Scene(0.0, 10.0), Scene(10.0, 22.0)]
    with patch("src.shortform.reframe.detect_scenes", return_value=scenes), \
         patch("src.shortform.reframe._face_centroid_per_scene_haar",
               return_value=[(0.3, 0.5), (0.7, 0.5)]), \
         patch("src.shortform.reframe._smooth_centroids",
               side_effect=lambda c, s: c):
        out = build_webcam_crops(cam, duration_s=22.0)
    assert len(out) == 2
    assert out[0].cx == 0.3
    assert out[1].cx == 0.7


# --- build_screen_crops ---------------------------------------------

def test_build_screen_crops_without_cursor_centers(tmp_path) -> None:
    screen = tmp_path / "ext.mov"
    screen.write_bytes(b"")
    with patch("src.shortform.reframe.detect_scenes", return_value=[]), \
         patch("src.shortform.reframe._smooth_centroids",
               side_effect=lambda c, s: c):
        out = build_screen_crops(
            screen, cursor_csv=None, duration_s=30.0,
        )
    assert len(out) == 1
    assert (out[0].cx, out[0].cy) == (0.5, 0.5)


def test_build_screen_crops_uses_cursor(tmp_path) -> None:
    screen = tmp_path / "ext.mov"
    screen.write_bytes(b"")
    cursor = tmp_path / "c.csv"
    _write_cursor_csv(cursor, [
        {"t_s": 5.0, "event": "move", "x": 1440, "y": 810},  # (0.75, 0.75)
    ])
    with patch("src.shortform.reframe.detect_scenes",
               return_value=[Scene(0.0, 10.0)]), \
         patch("src.shortform.reframe._smooth_centroids",
               side_effect=lambda c, s: c):
        out = build_screen_crops(
            screen, cursor_csv=cursor, duration_s=10.0,
            screen_w=1920, screen_h=1080,
        )
    assert len(out) == 1
    assert out[0].cx == pytest.approx(0.75)
    assert out[0].cy == pytest.approx(0.75)
