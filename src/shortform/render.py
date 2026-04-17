"""Per-layout ffmpeg renderer for shortform clips.

Four layouts as specified in the plan doc:

  * `cam_full`      — webcam fills 1080×1920. 9:16 crop centered on face.
  * `screen_full`   — screen fills 1080×1920. 9:16 crop centered on
                      cursor/activity (legible UI on phone).
  * `split_vstack`  — screen on top 1080×960, webcam face-crop on bottom
                      1080×960. User refinement: the screen portion is
                      ZOOMED before being scaled down so code/UI is
                      legible at phone size, not just center-cropped.
  * `pip`           — screen main, webcam circle inset bottom-right.
                      Reuses the long-form circle mask.

This module builds ffmpeg filter strings; a higher-level pipeline
invokes ffmpeg subprocess per clip. The renderer accepts a clip window
(start_s, end_s), the pre-computed crop windows from `reframe.py`, and
an optional ASS caption path to burn in.
"""
from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src import config
from src.shortform.reframe import CropWindow

log = logging.getLogger(__name__)


Layout = Literal["cam_full", "screen_full", "split_vstack", "pip"]


# Default output geometry for all shortform layouts.
SHORT_W = 1080
SHORT_H = 1920
VSTACK_HALF_H = SHORT_H // 2  # 960 each


@dataclass
class ClipSpec:
    start_s: float
    end_s: float
    layout: Layout
    webcam_crop: CropWindow | None          # crop on the webcam input
    screen_crop: CropWindow | None          # crop on the screen input
    captions_ass: Path | None = None         # optional ASS file to burn
    # split_vstack screen-portion zoom factor applied BEFORE downscale.
    # > 1.0 zooms into cursor area for phone-size legibility.
    screen_zoom: float = 1.6


# ----- crop-center helpers ------------------------------------------

def _safe_crop_center(cx: float, cy: float) -> tuple[float, float]:
    """Clamp crop centers to [0.05, 0.95] so the crop window never
    exceeds the source frame edges."""
    return (min(max(cx, 0.05), 0.95), min(max(cy, 0.05), 0.95))


def _face_crop_9_16(cx: float, cy: float) -> str:
    """ffmpeg filter for a 9:16 face-centered crop on an input frame.

    Uses crop=W:H:x:y where W = floor(ih*9/16), H = ih. We expose the
    (cx, cy) as a normalized offset on the untouched input so the
    caller can feed any resolution.

    Returns a filter CHAIN without any input/output labels — caller
    wraps them.
    """
    cx_clamped, _cy = _safe_crop_center(cx, cy)
    # 9:16 is portrait → crop width is (ih * 9/16) on a 16:9 source.
    # x = cx * iw - ow/2, clamped into [0, iw - ow].
    return (
        f"crop=w='min(iw, ih*9/16)':h=ih:"
        f"x='max(0, min(iw-ow, iw*{cx_clamped:.4f} - ow/2))':y=0"
    )


def _screen_crop_9_16(cx: float, cy: float, zoom: float = 1.0) -> str:
    """ffmpeg filter for a 9:16 cursor-centered screen crop with optional
    zoom-before-crop so UI remains readable on a phone.

    A zoom > 1.0 pre-scales the screen, crops the 9:16 window around
    cursor, then downscales to SHORT_W×SHORT_H at render time.
    """
    cx_clamped, cy_clamped = _safe_crop_center(cx, cy)
    if zoom <= 1.001:
        return (
            f"crop=w='min(iw, ih*9/16)':h=ih:"
            f"x='max(0, min(iw-ow, iw*{cx_clamped:.4f} - ow/2))':y=0"
        )
    # Pre-scale by `zoom`, then crop centered on the (scaled) cursor.
    return (
        f"scale=iw*{zoom:.3f}:ih*{zoom:.3f},"
        f"crop=w='min(iw, ih*9/16)':h=ih:"
        f"x='max(0, min(iw-ow, iw*{cx_clamped:.4f} - ow/2))':"
        f"y='max(0, min(ih-oh, ih*{cy_clamped:.4f} - oh/2))'"
    )


def _circle_mask_filter(diameter: int = 320) -> str:
    """Filter chunk that loads the circle mask and produces a [mask] label."""
    mask_path = str(config.CIRCLE_MASK_PATH)
    return (
        f"movie={mask_path},scale={diameter}:{diameter},loop=loop=-1:size=1,"
        f"setpts=PTS-STARTPTS,format=gray[mask_src]"
    )


# ----- per-layout filter builders ------------------------------------

def _build_cam_full_filter(spec: ClipSpec) -> str:
    crop = spec.webcam_crop or CropWindow(
        start=spec.start_s, end=spec.end_s,
        cx=config.PIP_FACE_X, cy=config.PIP_FACE_Y,
    )
    crop_expr = _face_crop_9_16(crop.cx, crop.cy)
    chain = [
        f"[1:v]{crop_expr},scale={SHORT_W}:{SHORT_H},setsar=1[v]",
    ]
    return ";".join(chain)


def _build_screen_full_filter(spec: ClipSpec) -> str:
    crop = spec.screen_crop or CropWindow(
        start=spec.start_s, end=spec.end_s, cx=0.5, cy=0.5,
    )
    crop_expr = _screen_crop_9_16(crop.cx, crop.cy, zoom=1.0)
    chain = [
        f"[0:v]{crop_expr},scale={SHORT_W}:{SHORT_H},setsar=1[v]",
    ]
    return ";".join(chain)


def _build_split_vstack_filter(spec: ClipSpec) -> str:
    """Screen on top, webcam face-crop on bottom. Screen is pre-zoomed
    (via `screen_zoom`) for phone legibility — this is the user's
    explicit requirement.
    """
    scr_crop = spec.screen_crop or CropWindow(
        start=spec.start_s, end=spec.end_s, cx=0.5, cy=0.5,
    )
    cam_crop = spec.webcam_crop or CropWindow(
        start=spec.start_s, end=spec.end_s,
        cx=config.PIP_FACE_X, cy=config.PIP_FACE_Y,
    )
    screen_expr = _screen_crop_9_16(scr_crop.cx, scr_crop.cy,
                                     zoom=spec.screen_zoom)
    cam_expr = _face_crop_9_16(cam_crop.cx, cam_crop.cy)
    chain = [
        f"[0:v]{screen_expr},scale={SHORT_W}:{VSTACK_HALF_H},setsar=1[top]",
        f"[1:v]{cam_expr},scale={SHORT_W}:{VSTACK_HALF_H},setsar=1[bot]",
        "[top][bot]vstack=inputs=2[v]",
    ]
    return ";".join(chain)


def _build_pip_filter(spec: ClipSpec) -> str:
    """Screen fills the frame with a circular webcam inset bottom-right."""
    scr_crop = spec.screen_crop or CropWindow(
        start=spec.start_s, end=spec.end_s, cx=0.5, cy=0.5,
    )
    cam_crop = spec.webcam_crop or CropWindow(
        start=spec.start_s, end=spec.end_s,
        cx=config.PIP_FACE_X, cy=config.PIP_FACE_Y,
    )
    D = config.PIP_DIAMETER
    margin = 40
    pip_x = SHORT_W - D - margin
    pip_y = SHORT_H - D - margin
    chain = [
        f"[0:v]{_screen_crop_9_16(scr_crop.cx, scr_crop.cy, zoom=1.0)},"
        f"scale={SHORT_W}:{SHORT_H},setsar=1[bg]",
        # Webcam square crop + mask
        f"[1:v]{_face_crop_9_16(cam_crop.cx, cam_crop.cy)},"
        f"scale={D}:{D},format=yuva420p,setsar=1[cam_sq]",
        _circle_mask_filter(D),
        "[cam_sq][mask_src]alphamerge=shortest=1[cam_circle]",
        f"[bg][cam_circle]overlay=x={pip_x}:y={pip_y}[v]",
    ]
    return ";".join(chain)


LAYOUT_BUILDERS = {
    "cam_full": _build_cam_full_filter,
    "screen_full": _build_screen_full_filter,
    "split_vstack": _build_split_vstack_filter,
    "pip": _build_pip_filter,
}


# ----- top-level filter graph ---------------------------------------

def build_filter_complex(spec: ClipSpec) -> str:
    """Return the full -filter_complex string for this clip.

    Appends caption burn-in as a final pass when spec.captions_ass is
    set. The `[v]` label flowing out of the layout builder becomes
    `[v]subtitles=...[v]` so the mapping (`-map [v]`) stays stable.
    """
    graph = LAYOUT_BUILDERS[spec.layout](spec)
    if spec.captions_ass is not None:
        graph = (
            graph.rsplit("[v]", 1)[0]
            + "[vbase];"
            + f"[vbase]subtitles={spec.captions_ass}[v]"
        )
    return graph


# ----- ffmpeg invocation --------------------------------------------

def render_clip(
    spec: ClipSpec,
    screen_path: Path,
    webcam_path: Path,
    audio_path: Path,
    out_path: Path,
    *,
    bitrate: str = "8M",
    dry_run: bool = False,
) -> Path:
    """Run ffmpeg to produce `out_path` from this ClipSpec.

    Uses -ss/-to on both inputs for frame-precise trim. Hardware-encodes
    with hevc_videotoolbox (+ faststart). Audio: AAC 192k, 48 kHz.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    filter_complex = build_filter_complex(spec)

    cmd = [
        "ffmpeg", "-hide_banner", "-nostdin", "-y",
        "-ss", f"{spec.start_s:.3f}", "-to", f"{spec.end_s:.3f}",
        "-i", str(screen_path),
        "-ss", f"{spec.start_s:.3f}", "-to", f"{spec.end_s:.3f}",
        "-i", str(webcam_path),
        "-ss", f"{spec.start_s:.3f}", "-to", f"{spec.end_s:.3f}",
        "-i", str(audio_path),
        "-filter_complex", filter_complex,
        "-map", "[v]", "-map", "2:a",
        "-c:v", "hevc_videotoolbox", "-b:v", bitrate, "-tag:v", "hvc1",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-movflags", "+faststart",
        str(out_path),
    ]
    if dry_run:
        return out_path

    log.info("rendering clip [%.2f, %.2f] layout=%s → %s",
             spec.start_s, spec.end_s, spec.layout, out_path)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg shortform render failed:\n{r.stderr[-600:]}")
    return out_path


__all__ = [
    "Layout",
    "ClipSpec",
    "SHORT_W",
    "SHORT_H",
    "VSTACK_HALF_H",
    "build_filter_complex",
    "render_clip",
]
