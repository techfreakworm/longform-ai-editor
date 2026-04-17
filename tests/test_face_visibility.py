"""Tests for Stage B.3 — face visibility detection.

Unit tests mock ffmpeg pipe + PyObjC Vision so we don't need a real
webcam file or macOS. A @slow integration test runs the real stage on a
short fixture if both ffmpeg + PyObjC are available.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.stages.face_visibility import (
    FaceAbsence,
    detect_face_absent_intervals,
    run,
)


# ---------- helpers --------------------------------------------------

def _synthetic_visibility_sequence(pattern: list[bool]):
    """Return an iterable yielding dummy BMP bytes, one per boolean in `pattern`.

    Pair with a patched _frame_has_face that consumes the same list in order.
    """
    return [f"bmp{i}".encode() for i in range(len(pattern))]


# ---------- presence detection --------------------------------------

def test_all_frames_have_face_emits_no_absences(tmp_path) -> None:
    """If every sampled frame has a face, no absence intervals are emitted."""
    seq = [True, True, True, True]
    with patch("src.stages.face_visibility._try_import_vision", return_value=True), \
         patch("src.stages.face_visibility._ffmpeg_frame_iter",
               return_value=iter(_synthetic_visibility_sequence(seq))), \
         patch("src.stages.face_visibility._frame_has_face",
               side_effect=seq):
        out = detect_face_absent_intervals(tmp_path / "cam.mov")
    assert out == []


def test_short_absence_below_threshold_ignored(tmp_path) -> None:
    """A 0.5 s absence at 2 Hz (1 frame) is below the 2 s threshold → ignored."""
    # [present, absent, present] at 2 Hz -> absence 0.5s to 1.0s = 0.5 s
    seq = [True, False, True]
    with patch("src.stages.face_visibility._try_import_vision", return_value=True), \
         patch("src.stages.face_visibility._ffmpeg_frame_iter",
               return_value=iter(_synthetic_visibility_sequence(seq))), \
         patch("src.stages.face_visibility._frame_has_face",
               side_effect=seq):
        out = detect_face_absent_intervals(
            tmp_path / "cam.mov", sample_rate_hz=2.0, min_absence_sec=2.0,
        )
    assert out == []


def test_long_absence_emitted(tmp_path) -> None:
    """Seven absent frames at 2 Hz = 3.5 s run → exceeds 2 s threshold."""
    seq = [True, True] + [False] * 7 + [True, True]
    # Absence runs from idx=2 (t=1.0 s) to idx=9 (t=4.5 s). When idx=9
    # is "True", we close the run at t=4.5 s, so interval = [1.0, 4.5].
    with patch("src.stages.face_visibility._try_import_vision", return_value=True), \
         patch("src.stages.face_visibility._ffmpeg_frame_iter",
               return_value=iter(_synthetic_visibility_sequence(seq))), \
         patch("src.stages.face_visibility._frame_has_face",
               side_effect=seq):
        out = detect_face_absent_intervals(
            tmp_path / "cam.mov", sample_rate_hz=2.0, min_absence_sec=2.0,
        )
    assert len(out) == 1
    assert out[0].start == pytest.approx(1.0)
    assert out[0].end == pytest.approx(4.5)


def test_trailing_absence_to_eof_is_captured(tmp_path) -> None:
    """An absence that runs off the end of the clip is still emitted."""
    seq = [True, True] + [False] * 6  # ends absent, 6 * 0.5 = 3.0 s
    with patch("src.stages.face_visibility._try_import_vision", return_value=True), \
         patch("src.stages.face_visibility._ffmpeg_frame_iter",
               return_value=iter(_synthetic_visibility_sequence(seq))), \
         patch("src.stages.face_visibility._frame_has_face",
               side_effect=seq):
        out = detect_face_absent_intervals(
            tmp_path / "cam.mov", sample_rate_hz=2.0, min_absence_sec=2.0,
        )
    assert len(out) == 1
    assert out[0].start == pytest.approx(1.0)
    assert out[0].end == pytest.approx(4.0)  # 8 samples * 0.5 s


def test_multiple_absences_in_one_clip(tmp_path) -> None:
    """Two separate absences with a return-to-visible in between."""
    # at 2 Hz: T,T,F,F,F,F,F,T,T,F,F,F,F,F
    #          0 1 2 3 4 5 6 7 8 9 10 11 12 13
    # absence1: idx 2..7 → [1.0, 3.5]
    # absence2: idx 9..EOF → [4.5, 7.0]
    seq = [True, True, False, False, False, False, False,
           True, True,
           False, False, False, False, False]
    with patch("src.stages.face_visibility._try_import_vision", return_value=True), \
         patch("src.stages.face_visibility._ffmpeg_frame_iter",
               return_value=iter(_synthetic_visibility_sequence(seq))), \
         patch("src.stages.face_visibility._frame_has_face",
               side_effect=seq):
        out = detect_face_absent_intervals(
            tmp_path / "cam.mov", sample_rate_hz=2.0, min_absence_sec=2.0,
        )
    assert len(out) == 2
    assert out[0].start == pytest.approx(1.0)
    assert out[0].end == pytest.approx(3.5)
    assert out[1].start == pytest.approx(4.5)
    assert out[1].end == pytest.approx(7.0)


# ---------- fallback behavior ---------------------------------------

def test_returns_empty_when_vision_unavailable(tmp_path) -> None:
    """Non-macOS / missing PyObjC → no absences, no crash."""
    with patch("src.stages.face_visibility._try_import_vision", return_value=False):
        out = detect_face_absent_intervals(tmp_path / "cam.mov")
    assert out == []


# ---------- run() CLI entry -----------------------------------------

def test_run_writes_json_and_is_idempotent(tmp_path, monkeypatch) -> None:
    """run() writes face_absent.json; subsequent calls no-op when file exists."""
    from src import config
    monkeypatch.setattr(config, "WORK_DIR", tmp_path)

    with patch("src.stages.face_visibility._try_import_vision", return_value=True), \
         patch("src.stages.face_visibility._ffmpeg_frame_iter",
               return_value=iter(_synthetic_visibility_sequence([True, False, False, False, False, False, True]))), \
         patch("src.stages.face_visibility._frame_has_face",
               side_effect=[True, False, False, False, False, False, True]):
        out_path = run(tmp_path / "cam.mov")

    assert out_path == tmp_path / "face_absent.json"
    assert out_path.exists()
    import json as _json
    data = _json.loads(out_path.read_text())
    assert "absences" in data
    assert len(data["absences"]) == 1

    # Second call should skip (file exists) — we verify by NOT re-patching
    # _ffmpeg_frame_iter to raise if called.
    with patch("src.stages.face_visibility._ffmpeg_frame_iter",
               side_effect=AssertionError("should not re-run")):
        run(tmp_path / "cam.mov")
