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
    DEFAULT_EASE_DUR_S,
    RenderOptions,
    RenderSegment,
    atempo_chain,
    build_filter_complex,
    smooth_zoom_crop_filter,
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


# --- smooth_zoom_crop_filter ------------------------------------------

def test_smooth_zoom_returns_scale_crop_chain() -> None:
    z = Z(0, 5, cx=0.5, cy=0.5, zoom=1.5)
    out = smooth_zoom_crop_filter(z, 1920, 1080, local_duration=5.0)
    # Must be a scale-then-crop chain (crop alone can't do time-varying size).
    assert out.startswith("scale=")
    assert ",crop=" in out
    # Scale must be frame-evaluated for the zoom factor to change over time.
    assert "eval=frame" in out


def test_smooth_zoom_includes_hermite_cubic_terms() -> None:
    z = Z(0, 3, cx=0.5, cy=0.5, zoom=1.5)
    out = smooth_zoom_crop_filter(z, 1920, 1080, local_duration=3.0)
    # Cubic Hermite smoothstep: 3u^2 - 2u^3 for both ease-in and ease-out.
    assert "3*pow(clip(t/" in out
    assert "2*pow(clip(t/" in out
    # Ease-out half of the smoothstep uses (L - t).
    assert "(3-t)" in out


def test_smooth_zoom_includes_clip_for_edge_safety() -> None:
    """Crop position must never leave the scaled image bounds."""
    z = Z(0, 3, cx=0.5, cy=0.5, zoom=1.5)
    out = smooth_zoom_crop_filter(z, 1920, 1080, local_duration=3.0)
    # x/y are clamped to [0, scaled - out_dim] via clip().
    assert "clip(0.5*iw-1920/2,0,iw-1920)" in out
    assert "clip(0.5*ih-1080/2,0,ih-1080)" in out


def test_smooth_zoom_references_time_variable_t() -> None:
    z = Z(0, 4, cx=0.5, cy=0.5, zoom=1.5)
    out = smooth_zoom_crop_filter(z, 1920, 1080, local_duration=4.0)
    # The scale expression must use `t` (per-frame timestamp).
    assert "t/" in out


def test_smooth_zoom_crop_output_dims_match_input() -> None:
    """Crop must output (in_w × in_h) so downstream filters see a stable
    frame size. The math is: scale up by z(t), then crop back down to
    (in_w × in_h) centered on the cursor.
    """
    z = Z(0, 3, cx=0.5, cy=0.5, zoom=1.5)
    out = smooth_zoom_crop_filter(z, 1920, 1080, local_duration=3.0)
    assert "crop=w=1920:h=1080:" in out


def test_smooth_zoom_custom_ease_dur_appears_in_string() -> None:
    z = Z(0, 5, cx=0.5, cy=0.5, zoom=1.5)
    out_default = smooth_zoom_crop_filter(z, 1920, 1080, local_duration=5.0)
    out_fast = smooth_zoom_crop_filter(
        z, 1920, 1080, local_duration=5.0, ease_dur=0.15
    )
    # The two should produce different strings — the ease duration shows up
    # in the expression literally.
    assert out_default != out_fast
    assert "0.15" in out_fast
    assert "0.35" in out_default


def test_smooth_zoom_uses_local_duration_for_ease_out_term() -> None:
    """The ease-out half of the smoothstep references `(L - t)` where L
    is the local_duration — NOT the original segment length.
    """
    z = Z(0, 12, cx=0.5, cy=0.5, zoom=1.5)
    out = smooth_zoom_crop_filter(z, 1920, 1080, local_duration=2.5)
    # Literal 2.5 must appear in the (L-t) subexpression.
    assert "(2.5-t)" in out
    # Original end (12) should not appear (it'd mean we passed the wrong L).
    assert "(12-t)" not in out


def test_smooth_zoom_default_ease_dur_exported_constant() -> None:
    """The DEFAULT_EASE_DUR_S constant must be the value used when the
    caller doesn't pass an ease_dur — keeps the public API stable.
    """
    assert DEFAULT_EASE_DUR_S == 0.35
    z = Z(0, 5, cx=0.5, cy=0.5, zoom=1.5)
    out_default = smooth_zoom_crop_filter(z, 1920, 1080, local_duration=5.0)
    out_explicit = smooth_zoom_crop_filter(
        z, 1920, 1080, local_duration=5.0, ease_dur=DEFAULT_EASE_DUR_S,
    )
    assert out_default == out_explicit


def test_smooth_zoom_off_center_cursor_coords_in_expr() -> None:
    """Non-centered cx/cy values must appear in the crop offset math."""
    z = Z(0, 3, cx=0.25, cy=0.8, zoom=1.8)
    out = smooth_zoom_crop_filter(z, 2560, 1440, local_duration=3.0)
    assert "0.25*iw" in out
    assert "0.8*ih" in out
    # 1.8 is the zoom factor and must show up in the z(t) expression.
    assert "1.8" in out


def test_smooth_zoom_identity_when_target_zoom_is_one() -> None:
    """Documents the `zoom=1.0` edge case.

    With target_zoom=1, `z(t) = 1 + (1-1)*step = 1` — the scale is a no-op
    and crop is centered at 0,0 with full dimensions — effectively identity.
    This is correct but wasteful; higher-level code avoids emitting zooms
    of zoom=1 in the first place (cursor_zoom.py emits zoom=1.5).
    """
    z = Z(0, 3, cx=0.5, cy=0.5, zoom=1.0)
    out = smooth_zoom_crop_filter(z, 1920, 1080, local_duration=3.0)
    # Still a scale,crop chain (we don't special-case zoom=1).
    assert out.startswith("scale=")
    # The factor (Z-1) = 0 is still present, proving math is consistent.
    assert "(1-1)*" in out


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
    # The smooth zoom emits a scale,crop chain between [s0_trim] and [s0_crop].
    assert "[s0_trim]scale=" in fc
    assert ",crop=" in fc
    assert "[s0_crop]" in fc
    # And the expression must contain the Hermite + time-base markers.
    assert "3*pow(clip(t/" in fc
    assert "eval=frame" in fc


def test_build_filter_zoom_uses_local_duration_under_speed_ramp() -> None:
    """With speed=4x, a 10-s source zoom runs 2.5 s in local (post-setpts)
    time. The smooth zoom's ease curves should use that local duration,
    not the original 10 s — otherwise at high speed the ease-in/out would
    overshoot the visible segment duration.
    """
    rs = [RenderSegment(
        start=0, end=10, speed=4.0, layout="screen_full",
        zoom=Z(0, 10, cx=0.5, cy=0.5, zoom=1.5),
    )]
    fc = build_filter_complex(rs, RenderOptions())
    # 10/4 = 2.5 — this numeric literal must appear in the ease-out term.
    assert "(2.5-t)" in fc


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
    # pip_w/pip_h are only honored by the rect path; circle uses pip_diameter.
    opts = RenderOptions(pip_w=640, pip_h=360, pip_margin=64, pip_shape="rect")
    fc = build_filter_complex(rs, opts)
    assert "scale=640:360" in fc
    assert "overlay=W-w-64:H-h-64" in fc


# --- circle PIP -------------------------------------------------------

def test_render_options_default_pip_shape_is_circle() -> None:
    """Circle is the default shape — matches the user's recorded preference.
    If this breaks, most existing `pip` tests here are implicitly exercising
    the circle path and will need explicit `pip_shape="rect"` opts to
    keep testing the rectangle contract.
    """
    assert RenderOptions().pip_shape == "circle"


def test_build_filter_pip_circle_loads_mask_via_movie() -> None:
    rs = [RenderSegment(start=0, end=10, speed=1.0, layout="pip", zoom=None)]
    fc = build_filter_complex(rs, RenderOptions())
    # `movie=` is a SOURCE filter (no input label) — it reads the PNG from
    # disk inside filter_complex, no extra `-i` flag needed on the cmdline.
    assert "movie='" in fc
    assert "circle_mask.png" in fc
    # Mask must be scaled to the pip diameter and grayscaled for alphamerge.
    assert "format=gray" in fc
    # Single-frame PNG must be looped so alphamerge doesn't truncate the PIP
    # to one frame — loop=-1:size=1 buffers the frame and repeats forever.
    assert "loop=loop=-1:size=1" in fc


def test_build_filter_pip_circle_uses_alphamerge() -> None:
    rs = [RenderSegment(start=0, end=10, speed=1.0, layout="pip", zoom=None)]
    fc = build_filter_complex(rs, RenderOptions())
    assert "alphamerge" in fc
    # Webcam needs an alpha-capable pixel format BEFORE alphamerge so the
    # merged alpha survives through to overlay.
    assert "format=yuva420p" in fc


def test_build_filter_pip_circle_face_centered_crop() -> None:
    """Default cam_face_x/y = 0.5 → face-centered crop expression must
    reference 0.5 and min(iw,ih) in the webcam crop.
    """
    rs = [RenderSegment(start=0, end=10, speed=1.0, layout="pip", zoom=None)]
    fc = build_filter_complex(rs, RenderOptions())
    # Square side is min(iw, ih) so a 16:9 webcam gets cropped to a 1:1
    # square before being scaled into the circle — prevents squashed faces.
    assert "min(iw,ih)" in fc
    # Default cam_face_x = 0.5 → coefficient must appear in the x clip expr.
    assert "clip(0.5*iw-min(iw,ih)/2,0,iw-min(iw,ih))" in fc


def test_build_filter_pip_circle_custom_face_x_off_center() -> None:
    """Off-center speaker (cam_face_x=0.35) must shift the crop leftward —
    that 0.35 coefficient must show up literally in the clip expression.
    """
    rs = [RenderSegment(start=0, end=10, speed=1.0, layout="pip", zoom=None)]
    opts = RenderOptions(cam_face_x=0.35)
    fc = build_filter_complex(rs, opts)
    assert "clip(0.35*iw-min(iw,ih)/2,0,iw-min(iw,ih))" in fc


def test_build_filter_pip_circle_scales_to_diameter() -> None:
    rs = [RenderSegment(start=0, end=10, speed=1.0, layout="pip", zoom=None)]
    opts = RenderOptions(pip_diameter=420)
    fc = build_filter_complex(rs, opts)
    # Both the mask AND the cropped webcam are scaled to diameter×diameter
    # so they overlay pixel-perfectly in alphamerge.
    assert fc.count("scale=420:420") == 2


def test_build_filter_pip_circle_produces_w_pip_label() -> None:
    """The overlay line is shape-agnostic — both circle and rect paths
    must produce [w{i}_pip] so the overlay at the bottom works unchanged.
    """
    rs = [RenderSegment(start=0, end=10, speed=1.0, layout="pip", zoom=None)]
    fc = build_filter_complex(rs, RenderOptions())
    # alphamerge writes its result to [w0_pip] (same label as rect path).
    assert "[w0_sq][m0]alphamerge[w0_pip]" in fc


def test_build_filter_pip_rect_has_no_alphamerge_or_movie() -> None:
    """The rect (classic) path must NOT touch the mask or alpha plumbing."""
    rs = [RenderSegment(start=0, end=10, speed=1.0, layout="pip", zoom=None)]
    opts = RenderOptions(pip_shape="rect")
    fc = build_filter_complex(rs, opts)
    assert "alphamerge" not in fc
    assert "movie=" not in fc
    assert "format=gray" not in fc


def test_build_filter_pip_circle_overlay_unchanged() -> None:
    """Overlay positioning is shape-agnostic — same corner offset for both
    circle and rect. This guards against an accidental shape-dependent
    position change that would shift the PIP across renders.
    """
    rs = [RenderSegment(start=0, end=10, speed=1.0, layout="pip", zoom=None)]
    fc_circle = build_filter_complex(rs, RenderOptions(pip_shape="circle"))
    fc_rect = build_filter_complex(rs, RenderOptions(pip_shape="rect"))
    # Both graphs end with identical overlay coordinates on [v0].
    assert "overlay=W-w-32:H-h-32[v0]" in fc_circle
    assert "overlay=W-w-32:H-h-32[v0]" in fc_rect


def test_build_filter_pip_circle_unique_labels_per_segment() -> None:
    """Two PIP segments must get distinct mask/square labels so ffmpeg
    doesn't reject duplicate filter outputs.
    """
    rs = [
        RenderSegment(start=0, end=5, speed=1.0, layout="pip", zoom=None),
        RenderSegment(start=5, end=10, speed=1.0, layout="pip", zoom=None),
    ]
    fc = build_filter_complex(rs, RenderOptions())
    # Each segment gets its own mask load and square intermediate.
    assert "[m0]" in fc and "[m1]" in fc
    assert "[w0_sq]" in fc and "[w1_sq]" in fc
    # And its own alphamerge output.
    assert "[w0_pip]" in fc and "[w1_pip]" in fc


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


@pytest.mark.slow
def test_smooth_zoom_ease_visibly_changes_frames(tmp_path: Path) -> None:
    """Render a 4-s screen_full segment with a centered zoom over the whole
    range, then extract frames at t=0 (pre-ease, zoom≈1) and t=0.35/2
    (mid-ease, zoom≈midway). Assert the frames differ in SHA — the crop
    window slid / the scale factor changed, so the visible pixels should
    not be byte-identical.

    Uses `testsrc2` (ffmpeg's built-in gradient + timecode pattern) so any
    change in crop position or zoom level shows up in the frame content.
    """
    import hashlib
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg not on PATH")

    rs = [
        RenderSegment(
            start=0, end=4, speed=1.0, layout="screen_full",
            zoom=Z(0, 4, cx=0.3, cy=0.6, zoom=1.8),
        ),
    ]
    fc = build_filter_complex(rs, RenderOptions())
    script = tmp_path / "fc.txt"
    script.write_text(fc)

    # Use testsrc2 as the screen input — its timecode + gradient makes
    # crop-region changes obvious in the pixel data.
    out = tmp_path / "out.mp4"
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc2=s=2560x1440:r=30:d=6",
        "-f", "lavfi", "-i", "color=c=red:s=1920x1080:r=30:d=6",
        "-f", "lavfi", "-i", "sine=frequency=220:duration=6",
        "-filter_complex_script", str(script),
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "aac",
        str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f"ffmpeg failed:\n{r.stderr}"

    def _frame_hash(t: float) -> str:
        p = tmp_path / f"frame_{int(t*1000):04d}.png"
        r2 = subprocess.run(
            [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
             "-ss", f"{t:.3f}", "-i", str(out),
             "-frames:v", "1", str(p)],
            capture_output=True, text=True,
        )
        assert r2.returncode == 0, f"frame extract failed at t={t}: {r2.stderr}"
        return hashlib.sha1(p.read_bytes()).hexdigest()

    # t=0: pre-ease, zoom=1 → full frame visible.
    # t=0.2: inside ease-in (ease_dur=0.35) → zoom ~partial.
    # t=2.0: inside the hold plateau → full zoom.
    h_start = _frame_hash(0.0)
    h_mid_ease = _frame_hash(0.2)
    h_hold = _frame_hash(2.0)
    assert h_start != h_mid_ease, (
        "start and mid-ease frames are byte-identical — zoom didn't animate."
    )
    assert h_mid_ease != h_hold, (
        "mid-ease and hold frames are identical — ease curve never reached "
        "the full-zoom plateau."
    )
