"""Tests for Stage E — render filter-graph builder.

Pure-Python filter string generation is exhaustively unit-tested.
One integration test asks ffmpeg to PARSE the filter (without rendering)
to catch any ffmpeg-rejected syntax — that test is marked @slow so the
fast CI path skips it.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from src.stages.render import (
    RenderOptions,
    RenderSegment,
    atempo_chain,
    build_filter_complex,
    split_at_zoom_boundaries,
    zoom_crop_filter,
)
from src.stages.unify_segments import Segment, ZoomWindow


# --- helpers -----------------------------------------------------------

def S(start, end, layout="pip", speed=1.0, zooms=None):
    return Segment(start=start, end=end, speed=speed, layout=layout,
                   cursor_zooms=list(zooms or []))


def Z(start, end, cx=0.5, cy=0.5, zoom=1.5):
    return ZoomWindow(start=start, end=end, cx=cx, cy=cy, zoom=zoom)


# --- atempo_chain -----------------------------------------------------

def test_atempo_chain_identity() -> None:
    assert atempo_chain(1.0) == []


def test_atempo_chain_2x() -> None:
    assert atempo_chain(2.0) == ["atempo=2"]


def test_atempo_chain_4x_doubles() -> None:
    assert atempo_chain(4.0) == ["atempo=2", "atempo=2"]


def test_atempo_chain_8x_triples() -> None:
    assert atempo_chain(8.0) == ["atempo=2", "atempo=2", "atempo=2"]


def test_atempo_chain_non_power_of_two() -> None:
    # 3.0 = 2.0 * 1.5
    out = atempo_chain(3.0)
    assert out[0] == "atempo=2"
    assert out[1].startswith("atempo=1.5")


def test_atempo_chain_half_speed() -> None:
    # 0.5 is in the single-filter range, no chain needed
    out = atempo_chain(0.5)
    assert out == ["atempo=0.5"]


def test_atempo_chain_rejects_absurd_speed() -> None:
    with pytest.raises(ValueError):
        atempo_chain(32.0)
    with pytest.raises(ValueError):
        atempo_chain(0.1)


# --- zoom_crop_filter -------------------------------------------------

def test_zoom_crop_filter_center() -> None:
    z = Z(0, 1, cx=0.5, cy=0.5, zoom=1.5)
    out = zoom_crop_filter(z, 1920, 1080)
    # 1920/1.5 = 1280, 1080/1.5 = 720; centered at 320,180
    assert "crop=1280.000:720.000:320.000:180.000" == out


def test_zoom_crop_filter_edge_clamps() -> None:
    # Cursor at top-left corner (cx=0, cy=0) — crop_x would go negative, clamp to 0
    z = Z(0, 1, cx=0.0, cy=0.0, zoom=2.0)
    out = zoom_crop_filter(z, 1920, 1080)
    assert ":0.000:0.000" in out


def test_zoom_crop_filter_right_edge_clamps() -> None:
    # Cursor at right edge; crop_x should clamp so crop+w doesn't exceed source
    z = Z(0, 1, cx=1.0, cy=1.0, zoom=2.0)
    out = zoom_crop_filter(z, 1920, 1080)
    # Clamped crop_x = 1920 - 960 = 960; crop_y = 1080 - 540 = 540
    assert "crop=960.000:540.000:960.000:540.000" == out


# --- split_at_zoom_boundaries -----------------------------------------

def test_split_no_zoom_passthrough() -> None:
    segs = [S(0, 10, "pip")]
    out = split_at_zoom_boundaries(segs)
    assert len(out) == 1
    assert out[0].zoom is None
    assert (out[0].start, out[0].end) == (0, 10)


def test_split_single_zoom_three_pieces() -> None:
    segs = [S(0, 10, "pip", zooms=[Z(3, 6)])]
    out = split_at_zoom_boundaries(segs)
    assert len(out) == 3
    # pre-zoom
    assert out[0].zoom is None and (out[0].start, out[0].end) == (0, 3)
    # during zoom
    assert out[1].zoom is not None and (out[1].start, out[1].end) == (3, 6)
    # post-zoom
    assert out[2].zoom is None and (out[2].start, out[2].end) == (6, 10)


def test_split_zoom_at_start_two_pieces() -> None:
    segs = [S(0, 10, "pip", zooms=[Z(0, 4)])]
    out = split_at_zoom_boundaries(segs)
    assert len(out) == 2
    assert out[0].zoom is not None and (out[0].start, out[0].end) == (0, 4)
    assert out[1].zoom is None and (out[1].start, out[1].end) == (4, 10)


def test_split_zoom_at_end_two_pieces() -> None:
    segs = [S(0, 10, "pip", zooms=[Z(7, 10)])]
    out = split_at_zoom_boundaries(segs)
    assert len(out) == 2


def test_split_zoom_covers_all_one_piece() -> None:
    segs = [S(0, 10, "pip", zooms=[Z(0, 10)])]
    out = split_at_zoom_boundaries(segs)
    assert len(out) == 1
    assert out[0].zoom is not None


def test_split_two_zooms_five_pieces() -> None:
    segs = [S(0, 20, "pip", zooms=[Z(3, 5), Z(10, 12)])]
    out = split_at_zoom_boundaries(segs)
    # Expected: none(0-3), zoom(3-5), none(5-10), zoom(10-12), none(12-20)
    assert len(out) == 5
    assert [r.zoom is not None for r in out] == [False, True, False, True, False]


def test_split_preserves_speed_and_layout() -> None:
    segs = [S(0, 10, "screen_full", speed=4.0, zooms=[Z(3, 6)])]
    out = split_at_zoom_boundaries(segs)
    for r in out:
        assert r.speed == 4.0
        assert r.layout == "screen_full"


# --- build_filter_complex -----------------------------------------

def test_build_filter_empty_raises() -> None:
    with pytest.raises(ValueError, match="no render segments"):
        build_filter_complex([], RenderOptions())


def test_build_filter_single_cam_full_has_webcam_scale() -> None:
    rs = [RenderSegment(start=0, end=10, speed=1.0, layout="cam_full", zoom=None)]
    fc = build_filter_complex(rs, RenderOptions())
    # cam_full uses webcam as main output
    assert "[w0_trim]" in fc
    assert "scale=1920:1080" in fc
    # concat at end
    assert "concat=n=1:v=1:a=1[vout][aout]" in fc


def test_build_filter_single_pip_has_overlay() -> None:
    rs = [RenderSegment(start=0, end=10, speed=1.0, layout="pip", zoom=None)]
    fc = build_filter_complex(rs, RenderOptions())
    assert "overlay=W-w-32:H-h-32" in fc
    assert "[s0_bg]" in fc
    assert "[w0_pip]" in fc


def test_build_filter_screen_full_no_overlay() -> None:
    rs = [RenderSegment(start=0, end=10, speed=1.0, layout="screen_full", zoom=None)]
    fc = build_filter_complex(rs, RenderOptions())
    assert "overlay=" not in fc


def test_build_filter_speed_ramp_on_video_and_audio() -> None:
    rs = [RenderSegment(start=0, end=10, speed=4.0, layout="pip", zoom=None)]
    fc = build_filter_complex(rs, RenderOptions())
    # setpts divisor on video
    assert "setpts=(PTS-STARTPTS)/4" in fc
    # two-stage atempo on audio
    assert fc.count("atempo=2") >= 2


def test_build_filter_no_speed_change_omits_tempo() -> None:
    rs = [RenderSegment(start=0, end=10, speed=1.0, layout="pip", zoom=None)]
    fc = build_filter_complex(rs, RenderOptions())
    assert "atempo=" not in fc
    # setpts appears but with no divisor
    assert "setpts=(PTS-STARTPTS)[s0_trim]" in fc


def test_build_filter_zoom_adds_crop() -> None:
    rs = [RenderSegment(
        start=0, end=10, speed=1.0, layout="pip",
        zoom=Z(0, 10, cx=0.5, cy=0.5, zoom=1.5),
    )]
    opts = RenderOptions(screen_res=(1920, 1080))
    fc = build_filter_complex(rs, opts)
    assert "[s0_trim]crop=" in fc
    assert "[s0_crop]" in fc


def test_build_filter_zoom_skipped_when_layout_cam_full() -> None:
    """cam_full doesn't need the screen branch at all — a zoom on such a
    segment is meaningless and must not cause ffmpeg-rejected dangling
    outputs. We expect zero screen-branch emissions in the filter graph.
    """
    rs = [RenderSegment(
        start=0, end=10, speed=1.0, layout="cam_full",
        zoom=Z(0, 10, cx=0.5, cy=0.5, zoom=1.5),
    )]
    fc = build_filter_complex(rs, RenderOptions())
    # cam_full uses [w0_trim] for [v0]
    assert "[w0_trim]" in fc
    # NO screen-branch artifacts — the zoom crop would leave [s0_trim] dangling
    assert "[s0_trim]" not in fc
    assert "[s0_crop]" not in fc
    # [v0] appears exactly twice: once as output of cam_full scale, once in concat
    assert fc.count("[v0]") == 2


def test_build_filter_multiple_segments_concat_all() -> None:
    rs = [
        RenderSegment(start=0, end=10, speed=1.0, layout="cam_full", zoom=None),
        RenderSegment(start=10, end=30, speed=1.0, layout="pip", zoom=None),
        RenderSegment(start=30, end=50, speed=4.0, layout="pip", zoom=None),
    ]
    fc = build_filter_complex(rs, RenderOptions())
    # Every segment produces [vN] and [aN]
    for i in range(3):
        assert f"[v{i}]" in fc
        assert f"[a{i}]" in fc
    # Concat list
    assert "[v0][a0][v1][a1][v2][a2]concat=n=3:v=1:a=1[vout][aout]" in fc


def test_build_filter_trim_times_precise() -> None:
    rs = [RenderSegment(start=5.42, end=12.87, speed=1.0, layout="pip", zoom=None)]
    fc = build_filter_complex(rs, RenderOptions())
    assert "trim=start=5.4200:end=12.8700" in fc
    assert "atrim=start=5.4200:end=12.8700" in fc


def test_build_filter_custom_pip_geometry() -> None:
    rs = [RenderSegment(start=0, end=10, speed=1.0, layout="pip", zoom=None)]
    opts = RenderOptions(pip_w=640, pip_h=360, pip_margin=64)
    fc = build_filter_complex(rs, opts)
    assert "scale=640:360" in fc
    assert "overlay=W-w-64:H-h-64" in fc


# --- unknown-layout safety ------------------------------------------

def test_build_filter_unknown_layout_raises() -> None:
    # Bypass typing to test defensive branch
    rs = [RenderSegment(start=0, end=10, speed=1.0, layout="bogus", zoom=None)]  # type: ignore
    with pytest.raises(ValueError, match="unknown layout"):
        build_filter_complex(rs, RenderOptions())


# --- integration: ffmpeg actually parses our filter ----------------

@pytest.mark.slow
def test_filter_parses_with_ffmpeg(tmp_path: Path) -> None:
    """Write the filter to a script file and ask ffmpeg to validate it by
    requesting a 0-frame render of dummy inputs. Catches any syntax
    ffmpeg's parser rejects.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg not on PATH")

    rs = [
        RenderSegment(start=0, end=2, speed=1.0, layout="cam_full", zoom=None),
        RenderSegment(start=2, end=5, speed=1.0, layout="pip",
                      zoom=Z(2, 5, cx=0.5, cy=0.5, zoom=1.5)),
        RenderSegment(start=5, end=8, speed=4.0, layout="screen_full", zoom=None),
    ]
    fc = build_filter_complex(rs, RenderOptions())
    script = tmp_path / "fc.txt"
    script.write_text(fc)

    # Dummy inputs: small 10-s solid-color + sine-audio so trim works.
    # Use lavfi inputs so no real files needed.
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=blue:s=2560x1440:r=30:d=10",
        "-f", "lavfi", "-i", "color=c=red:s=1920x1080:r=30:d=10",
        "-f", "lavfi", "-i", "sine=frequency=220:duration=10",
        "-filter_complex_script", str(script),
        "-map", "[vout]", "-map", "[aout]",
        "-frames:v", "1",  # only render 1 frame — fast sanity check
        "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "aac",
        str(tmp_path / "out.mp4"),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, (
        f"ffmpeg rejected the filter graph:\n{r.stderr}\n\nscript:\n{fc}"
    )
    assert (tmp_path / "out.mp4").exists()
