"""Tests for src/shortform/render.py — per-layout ffmpeg filter graphs."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.shortform.reframe import CropWindow
from src.shortform.render import (
    SHORT_H,
    SHORT_W,
    VSTACK_HALF_H,
    ClipSpec,
    _face_crop_9_16,
    _safe_crop_center,
    _screen_crop_9_16,
    build_filter_complex,
    render_clip,
)


# --- helper crop filters --------------------------------------------

def test_safe_crop_center_clamps_extremes() -> None:
    assert _safe_crop_center(-0.5, 1.5) == (0.05, 0.95)
    assert _safe_crop_center(0.5, 0.5) == (0.5, 0.5)


def test_face_crop_produces_9_16_window() -> None:
    expr = _face_crop_9_16(0.4, 0.5)
    assert "crop=w='min(iw, ih*9/16)'" in expr
    assert "iw*0.4000" in expr


def test_screen_crop_without_zoom_has_no_scale() -> None:
    expr = _screen_crop_9_16(0.5, 0.5, zoom=1.0)
    assert "scale=iw*" not in expr
    assert "crop=" in expr


def test_screen_crop_with_zoom_prescales() -> None:
    expr = _screen_crop_9_16(0.5, 0.5, zoom=1.6)
    assert "scale=iw*1.600:ih*1.600" in expr
    assert "crop=" in expr


# --- per-layout filter graph ----------------------------------------

def _spec(layout, **kwargs):
    defaults = dict(
        start_s=0.0, end_s=30.0, layout=layout,
        webcam_crop=CropWindow(0.0, 30.0, 0.5, 0.5),
        screen_crop=CropWindow(0.0, 30.0, 0.5, 0.5),
    )
    defaults.update(kwargs)
    return ClipSpec(**defaults)


def test_cam_full_filter_has_single_output() -> None:
    graph = build_filter_complex(_spec("cam_full"))
    assert "[v]" in graph
    # Webcam is input 1 in our conventions
    assert "[1:v]" in graph
    assert "[0:v]" not in graph   # no screen needed


def test_screen_full_filter_uses_screen_input() -> None:
    graph = build_filter_complex(_spec("screen_full"))
    assert "[0:v]" in graph
    assert "[1:v]" not in graph


def test_split_vstack_uses_both_inputs_and_vstack() -> None:
    graph = build_filter_complex(_spec("split_vstack"))
    assert "[0:v]" in graph
    assert "[1:v]" in graph
    assert "vstack=inputs=2" in graph
    assert f"{SHORT_W}:{VSTACK_HALF_H}" in graph


def test_split_vstack_screen_zoom_default_prescales() -> None:
    """Screen portion pre-zoomed for phone legibility (user requirement)."""
    graph = build_filter_complex(_spec("split_vstack"))
    assert "scale=iw*1.600:ih*1.600" in graph


def test_split_vstack_screen_zoom_one_disables_prescale() -> None:
    spec = _spec("split_vstack", screen_zoom=1.0)
    graph = build_filter_complex(spec)
    # Only the final scale to SHORT_W×VSTACK_HALF_H should remain.
    assert "scale=iw*1.000:ih*1.000" not in graph


def test_pip_uses_circle_mask() -> None:
    graph = build_filter_complex(_spec("pip"))
    assert "alphamerge=shortest=1" in graph
    assert "movie=" in graph
    assert "overlay=" in graph


def test_pip_output_labelled_v() -> None:
    graph = build_filter_complex(_spec("pip"))
    assert graph.rstrip(";").endswith("[v]")


# --- captions burn-in ------------------------------------------------

def test_build_filter_appends_subtitle_filter(tmp_path) -> None:
    ass = tmp_path / "c.ass"
    ass.write_text("[Script Info]\n")
    spec = _spec("cam_full", captions_ass=ass)
    graph = build_filter_complex(spec)
    assert "subtitles=" in graph
    assert "[vbase]" in graph


def test_no_captions_means_no_subtitle_filter() -> None:
    graph = build_filter_complex(_spec("cam_full", captions_ass=None))
    assert "subtitles=" not in graph


# --- render_clip ffmpeg command ------------------------------------

def test_render_clip_builds_ffmpeg_cmd(tmp_path) -> None:
    out = tmp_path / "short.mp4"
    captured: dict[str, list[str]] = {}
    def fake_run(cmd, **_kw):
        captured["cmd"] = cmd
        class R:
            returncode = 0
            stderr = ""
        return R()

    with patch("src.shortform.render.subprocess.run", side_effect=fake_run):
        render_clip(
            _spec("cam_full"),
            tmp_path / "ext.mov",
            tmp_path / "cam.mov",
            tmp_path / "audio.mp4",
            out,
        )
    cmd = captured["cmd"]
    assert cmd[0] == "ffmpeg"
    assert "-filter_complex" in cmd
    assert "hevc_videotoolbox" in cmd
    # Three inputs
    assert cmd.count("-i") == 3
    assert str(out) == cmd[-1]


def test_render_clip_raises_on_ffmpeg_failure(tmp_path) -> None:
    class R:
        returncode = 1
        stderr = "bad filter"
    with patch("src.shortform.render.subprocess.run", return_value=R()):
        with pytest.raises(RuntimeError, match="bad filter"):
            render_clip(
                _spec("cam_full"),
                tmp_path / "ext.mov", tmp_path / "cam.mov", tmp_path / "a.mp4",
                tmp_path / "out.mp4",
            )


def test_render_clip_dry_run_does_not_invoke_ffmpeg(tmp_path) -> None:
    out = tmp_path / "dry.mp4"
    with patch("src.shortform.render.subprocess.run",
               side_effect=AssertionError("should not be called")):
        ret = render_clip(
            _spec("cam_full"),
            tmp_path / "ext.mov", tmp_path / "cam.mov", tmp_path / "a.mp4",
            out, dry_run=True,
        )
    assert ret == out
