"""Tests for Stage A — clap-cue sync.

Exercises real OBS-source-record fixtures from tests/fixtures/real/.
Ground truths (derived empirically from the 2026-04-17 01-49-37 recording):
  - Flash visible in ext.mov at OBS t ≈ 4.850s (±1 frame @ 60 fps = ±0.017s)
  - Cursor CSV has event=clap row at logger t ≈ 2.160s
  - Therefore csv_to_video_offset ≈ 4.850 − 2.160 = 2.690 s
  - No audible hand-clap → audio detector correctly rejects the keystroke
  - video_to_video offset = 0 (OBS-source-record files are already in sync)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from src.stages.sync_clap import (
    apply_offset,
    detect_clap_in_audio,
    detect_clap_in_csv,
    detect_flash_in_video,
    run,
)

# Ground truth from manual inspection of the 2026-04-17 clip.
GT_FLASH_S = 4.850
GT_CLAP_CSV_S = 2.160
GT_CSV_OFFSET_S = GT_FLASH_S - GT_CLAP_CSV_S  # 2.690
FPS = 60.0
ONE_FRAME_S = 1.0 / FPS


# ---------- pure detector tests ----------------------------------------

def test_detect_flash_in_ext(real_ext: Path) -> None:
    """Flash frame locates within ±1 frame of known ground truth."""
    t = detect_flash_in_video(real_ext)
    tol = ONE_FRAME_S + 1e-6
    assert abs(t - GT_FLASH_S) <= tol, (
        f"flash detected at {t:.4f}s, expected {GT_FLASH_S}s ± {tol:.4f}s"
    )


def test_detect_clap_in_csv(real_cursor: Path) -> None:
    """CSV clap row parses correctly."""
    t = detect_clap_in_csv(real_cursor)
    assert abs(t - GT_CLAP_CSV_S) <= 0.005, t


def test_detect_clap_in_csv_missing(tmp_path: Path) -> None:
    """Detector raises cleanly when CSV has no clap row."""
    csv_path = tmp_path / "no_clap.csv"
    csv_path.write_text("t_s,x,y,event,button,down\n0.5,100,200,move,,\n")
    with pytest.raises(RuntimeError, match="no event=clap"):
        detect_clap_in_csv(csv_path)


def test_detect_clap_audio_rejects_keystroke_near_flash(real_merged: Path) -> None:
    """No audible hand-clap — only a faint keystroke at ~t=4.68 s. When the
    detector searches within ±500 ms of the known flash time, the keystroke
    fails the "real hand clap" amplitude gates and the detector raises.
    """
    with pytest.raises(RuntimeError, match="no audible clap"):
        detect_clap_in_audio(real_merged, near_t=GT_FLASH_S, tolerance_s=0.5)


def test_detect_clap_audio_finds_speech_without_hint(real_merged: Path) -> None:
    """Without a `near_t` hint, the detector picks up the loudest onset in
    the whole window — which is a spoken word, not the clap. This confirms
    the run() orchestrator MUST pass near_t=flash_t, otherwise it would mis-
    identify dialogue as the clap.
    """
    t = detect_clap_in_audio(real_merged)
    # The loudest word onset in this clip is around 19 s.
    assert t > 10.0, (
        f"sanity check: without near_t the detector should find late speech, "
        f"not a short-timed clap. got t={t:.3f}s"
    )


# ---------- orchestrator tests -----------------------------------------

def test_run_with_csv_alignment(real_ext: Path, real_cam: Path, real_cursor: Path, tmp_path: Path) -> None:
    """End-to-end: write sync.json with csv_to_video_offset."""
    args = argparse.Namespace(
        screen=real_ext,
        webcam=real_cam,
        cursor=real_cursor,
        trim=False,
        manual_offset=None,
        work=tmp_path,
        verbose=0,
    )
    rc = run(args)
    assert rc == 0

    sync = json.loads((tmp_path / "sync.json").read_text())
    # Flash-based CSV alignment is the primary sync output
    assert sync["method"] in ("flash+csv", "flash+csv+audio")
    assert sync["flash_video_s"] is not None
    assert abs(sync["flash_video_s"] - GT_FLASH_S) <= ONE_FRAME_S + 1e-6
    assert sync["clap_csv_s"] is not None
    assert abs(sync["clap_csv_s"] - GT_CLAP_CSV_S) <= 0.005
    assert sync["csv_to_video_offset_s"] is not None
    assert abs(sync["csv_to_video_offset_s"] - GT_CSV_OFFSET_S) <= ONE_FRAME_S + 0.005
    # No audible clap → audio path rejected → video_to_video = 0
    assert sync["clap_audio_s"] is None
    assert sync["video_to_video_offset_s"] == 0.0
    assert 0.0 <= sync["confidence"] <= 1.0
    # Confidence should be high — the flash is ~160 luminance units
    assert sync["confidence"] > 0.6, sync["confidence"]

    # Without --trim, no synced mkvs are written
    assert sync["screen_synced"] is None
    assert sync["webcam_synced"] is None
    assert not (tmp_path / "screen_synced.mkv").exists()


def test_run_without_csv(real_ext: Path, real_cam: Path, tmp_path: Path) -> None:
    """No cursor CSV supplied: still detect the flash, skip CSV offset."""
    args = argparse.Namespace(
        screen=real_ext,
        webcam=real_cam,
        cursor=None,
        trim=False,
        manual_offset=None,
        work=tmp_path,
        verbose=0,
    )
    rc = run(args)
    assert rc == 0
    sync = json.loads((tmp_path / "sync.json").read_text())
    assert sync["method"] == "flash-only"
    assert sync["csv_to_video_offset_s"] is None
    assert sync["clap_csv_s"] is None


def test_run_manual_offset(real_ext: Path, real_cam: Path, tmp_path: Path) -> None:
    """--manual-offset bypasses detection entirely."""
    args = argparse.Namespace(
        screen=real_ext,
        webcam=real_cam,
        cursor=None,
        trim=False,
        manual_offset=0.25,
        work=tmp_path,
        verbose=0,
    )
    rc = run(args)
    assert rc == 0
    sync = json.loads((tmp_path / "sync.json").read_text())
    assert sync["method"] == "manual"
    assert sync["video_to_video_offset_s"] == 0.25
    assert sync["flash_video_s"] is None


@pytest.mark.slow
def test_run_with_trim(real_ext: Path, real_cam: Path, tmp_path: Path) -> None:
    """--trim writes screen_synced.mkv and webcam_synced.mkv."""
    args = argparse.Namespace(
        screen=real_ext,
        webcam=real_cam,
        cursor=None,
        trim=True,
        manual_offset=None,
        work=tmp_path,
        verbose=0,
    )
    rc = run(args)
    assert rc == 0
    assert (tmp_path / "screen_synced.mkv").exists()
    assert (tmp_path / "webcam_synced.mkv").exists()


def test_apply_offset_pass_through(real_ext: Path, real_cam: Path, tmp_path: Path) -> None:
    """offset=0 copies both inputs unchanged (up to stream-copy semantics)."""
    screen_out = tmp_path / "s.mkv"
    webcam_out = tmp_path / "w.mkv"
    apply_offset(real_ext, real_cam, 0.0, screen_out, webcam_out)
    assert screen_out.exists() and screen_out.stat().st_size > 0
    assert webcam_out.exists() and webcam_out.stat().st_size > 0
