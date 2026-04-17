"""Stage E — build one ffmpeg filter_complex pass and render final.mp4.

Takes the canonical segments.json from Stage D (unify_segments) and the
two raw source files, and produces a composited + trimmed + speed-ramped
+ layout-switched + optionally cursor-zoomed output.

Design:
  - `build_filter_complex(...)` is pure Python: segments → filter string.
    Fully unit-testable with no ffmpeg dependency.
  - `render(...)` writes the filter to a temp script file, invokes ffmpeg
    via -filter_complex_script (dodges shell-arg-length limits for long
    sessions), and maps vout/aout through hevc_videotoolbox to final.mp4.

Each segment in segments.json becomes ONE concat item. Segments with a
cursor_zoom list are pre-split at zoom boundaries so every render segment
has AT MOST one zoom — keeps the filter graph simple (constant crop per
segment, no per-frame crop expressions).

Input stream convention (3 inputs — screen, webcam-video, webcam-audio;
the last two may be the same file):
  [0:v]   screen video (silent or audio ignored)
  [1:v]   webcam video
  [2:a]   mic audio (the user's "merged.mp4" or webcam.mov or whatever)
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from src import config
from src.stages.unify_segments import Segment, ZoomWindow
from src.utils.ffmpeg_helpers import probe_resolution

log = logging.getLogger(__name__)

Layout = Literal["cam_full", "pip", "screen_full"]


# ---------- data model ------------------------------------------------

@dataclass
class RenderSegment:
    """One atomic piece of the final timeline.

    Has at most one cursor_zoom — parent Segment with multiple zooms is
    split into multiple RenderSegments (see split_at_zoom_boundaries).
    """
    start: float
    end: float
    speed: float
    layout: Layout
    zoom: ZoomWindow | None = None


@dataclass
class RenderOptions:
    output_w: int = 1920
    output_h: int = 1080
    output_fps: int = 30
    pip_w: int = 480
    pip_h: int = 270
    pip_margin: int = 32       # corner offset in pixels
    # Shape of the webcam PIP. "circle" punches the webcam through a
    # grayscale alpha mask (see circle_mask_path) before overlaying;
    # "rect" keeps the classic pip_w×pip_h rectangle.
    pip_shape: str = field(default_factory=lambda: config.PIP_SHAPE)
    pip_diameter: int = field(default_factory=lambda: config.PIP_DIAMETER)
    # Face position in the webcam frame (0.0–1.0). Used only by the
    # circle path — before scaling into the circle, the webcam is cropped
    # to a square of side=min(iw,ih) centered on (cam_face_x*iw, cam_face_y*ih).
    # Move toward 0.3 if you sit left of center, 0.7 if right of center.
    cam_face_x: float = field(default_factory=lambda: config.PIP_FACE_X)
    cam_face_y: float = field(default_factory=lambda: config.PIP_FACE_Y)
    circle_mask_path: Path = field(default_factory=lambda: config.CIRCLE_MASK_PATH)
    video_bitrate: str = "12M"
    audio_bitrate: str = "192k"
    audio_sample_rate: int = 48000
    pix_fmt: str = "yuv420p"
    screen_res: tuple[int, int] = (2560, 1440)
    webcam_res: tuple[int, int] = (1920, 1080)


# ---------- preprocess ------------------------------------------------

def split_at_zoom_boundaries(segments: list[Segment]) -> list[RenderSegment]:
    """Split each Segment at its cursor_zoom start/end times so every
    resulting RenderSegment has at most one zoom covering its entire range.

    A segment with no zooms produces exactly one RenderSegment. A segment
    with N zooms produces up to 2N+1 RenderSegments, alternating
    zoom / no-zoom / zoom / no-zoom / ...
    """
    out: list[RenderSegment] = []
    for seg in segments:
        if not seg.cursor_zooms:
            out.append(RenderSegment(
                start=seg.start, end=seg.end, speed=seg.speed,
                layout=seg.layout, zoom=None,
            ))
            continue
        # Sort zooms within segment and emit split pieces
        zooms = sorted(seg.cursor_zooms, key=lambda z: z.start)
        cur = seg.start
        for z in zooms:
            # Clamp zoom bounds defensively
            z_s = max(cur, z.start)
            z_e = min(seg.end, z.end)
            if z_s >= z_e:
                continue
            # Leading no-zoom gap
            if z_s > cur:
                out.append(RenderSegment(
                    start=cur, end=z_s, speed=seg.speed,
                    layout=seg.layout, zoom=None,
                ))
            # The zoomed piece
            out.append(RenderSegment(
                start=z_s, end=z_e, speed=seg.speed,
                layout=seg.layout, zoom=z,
            ))
            cur = z_e
        # Trailing no-zoom tail
        if cur < seg.end:
            out.append(RenderSegment(
                start=cur, end=seg.end, speed=seg.speed,
                layout=seg.layout, zoom=None,
            ))
    return out


# ---------- atempo chain for pitch-preserving speedup ----------------

def atempo_chain(speed: float, eps: float = 1e-6) -> list[str]:
    """Return a list of atempo filter args to achieve the given speed.

    ffmpeg's atempo is limited to [0.5, 2.0] per instance. Chain to reach
    higher factors: 4.0 → ["atempo=2","atempo=2"], 8.0 → three. Non-power-
    of-two factors use a trailing partial atempo (e.g. 3.0 → ["atempo=2",
    "atempo=1.5"]). Returns [] for speed ≈ 1.
    """
    if abs(speed - 1.0) < eps:
        return []
    if speed < 0.5 or speed > 16.0:
        raise ValueError(f"speed {speed} out of supported range [0.5, 16]")
    filters: list[str] = []
    remaining = speed
    while remaining > 2.0 + eps:
        filters.append("atempo=2")
        remaining /= 2.0
    # Final partial (always in [0.5, 2.0] here)
    if abs(remaining - 1.0) > eps:
        filters.append(f"atempo={remaining:g}")
    return filters


# ---------- zoom crop ------------------------------------------------

# Default ease-in/ease-out duration (seconds) for the Hermite smoothstep.
# 350 ms feels natural — not a snap, not a lazy slide. Can be overridden.
DEFAULT_EASE_DUR_S: float = 0.35


def zoom_crop_filter(zoom: ZoomWindow, in_w: int, in_h: int) -> str:
    """Build a static ffmpeg `crop=...` for the given zoom window.

    Crops a (in_w/zoom × in_h/zoom) region centered at (cx*in_w, cy*in_h),
    clamped to the source bounds. The caller scales the cropped result
    back up to the output resolution.

    Retained as a utility for tests and simple non-animated use cases.
    Production rendering uses `smooth_zoom_crop_filter` which adds ease
    in/out so the zoom doesn't snap visually.
    """
    crop_w = in_w / zoom.zoom
    crop_h = in_h / zoom.zoom
    crop_x = zoom.cx * in_w - crop_w / 2
    crop_y = zoom.cy * in_h - crop_h / 2
    crop_x = max(0.0, min(in_w - crop_w, crop_x))
    crop_y = max(0.0, min(in_h - crop_h, crop_y))
    return f"crop={crop_w:.3f}:{crop_h:.3f}:{crop_x:.3f}:{crop_y:.3f}"


def smooth_zoom_crop_filter(
    zoom: ZoomWindow,
    in_w: int,
    in_h: int,
    local_duration: float,
    ease_dur: float = DEFAULT_EASE_DUR_S,
) -> str:
    """Time-varying zoom filter with cubic-Hermite ease-in / hold / ease-out.

    Returns a `scale=...,crop=...` filter chain (as a single string) that
    smoothly zooms into the cursor over `ease_dur` seconds, holds at full
    zoom, then zooms back out over `ease_dur` seconds — all covering the
    range [0, local_duration] in local (post-trim, post-setpts) time.

    Why scale+crop instead of a single time-varying crop:
      ffmpeg's `crop` filter only re-evaluates `x`/`y` per frame — the
      output dimensions `w`/`h` are fixed at init time. So a crop whose
      *size* changes per frame is not possible with that filter alone.
      Workaround: upscale the input per frame by z(t) using `scale` with
      `eval=frame`, then center-crop a fixed (in_w × in_h) window from
      the scaled image, with x/y panning to keep the cursor framed.

    Math (`t` = local segment time, `D` = ease_dur, `L` = local_duration,
    `Z` = target_zoom, `CX`/`CY` = normalized cursor position):
        u_in  = clip(t/D, 0, 1)
        u_out = clip((L-t)/D, 0, 1)
        step  = min(3*u_in^2 - 2*u_in^3, 3*u_out^2 - 2*u_out^3)   # 0..1
        z(t)  = 1 + (Z-1) * step
        scaled_w = in_w * z(t);  scaled_h = in_h * z(t)
        crop_x = clip(CX*scaled_w - in_w/2, 0, scaled_w - in_w)
        crop_y = clip(CY*scaled_h - in_h/2, 0, scaled_h - in_h)

    The crop output is always (in_w × in_h), matching the scale's input
    — so downstream filters see a stable frame size.

    Escaping notes:
      - We emit this filter into a `-filter_complex_script` file, so
        shell-level escaping isn't needed.
      - Filter-graph colons and commas inside expressions MUST be single-
        quoted (`'...'`), otherwise ffmpeg splits on them. Hence every
        expression below is wrapped in single quotes.
      - Commas INSIDE an expression (e.g. `clip(a,b,c)`) stay unescaped
        because they're within a single-quoted string. No backslashes
        needed — those are only for quoting-style command-line usage.
    """
    z = zoom.zoom
    cx, cy = zoom.cx, zoom.cy
    d = ease_dur
    L = local_duration

    # The smoothstep expression. Computed symbolically so the string is
    # readable + auditable. `t` here is the ffmpeg per-frame timestamp
    # (seconds since the filter started processing frames — which, after
    # setpts=(PTS-STARTPTS)/speed, is local segment time).
    u_in = f"clip(t/{d:g},0,1)"
    u_out = f"clip(({L:g}-t)/{d:g},0,1)"
    hermite_in = f"(3*pow({u_in},2)-2*pow({u_in},3))"
    hermite_out = f"(3*pow({u_out},2)-2*pow({u_out},3))"
    step = f"min({hermite_in},{hermite_out})"
    z_expr = f"(1+({z:g}-1)*{step})"

    # scale output dimensions (time-varying).
    scale_w = f"{in_w}*{z_expr}"
    scale_h = f"{in_h}*{z_expr}"

    # crop window: fixed in_w × in_h from the scaled image, centered on
    # the cursor and edge-clamped. Inside crop, `iw`/`ih` are the scaled
    # input's current dimensions (i.e. in_w * z(t) / in_h * z(t)).
    crop_x = f"clip({cx:g}*iw-{in_w}/2,0,iw-{in_w})"
    crop_y = f"clip({cy:g}*ih-{in_h}/2,0,ih-{in_h})"

    return (
        f"scale=w='{scale_w}':h='{scale_h}':eval=frame:flags=bicubic,"
        f"crop=w={in_w}:h={in_h}:x='{crop_x}':y='{crop_y}'"
    )


# ---------- filter graph builder -------------------------------------

# Fill-frame scale helper: scales preserving aspect ratio, then center-crops
# any overhang. Produces pixel-exact WxH output with no letterboxing.
def _scale_fill(out_w: int, out_h: int) -> str:
    return (
        f"scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
        f"crop={out_w}:{out_h}"
    )


def _pip_circle_webcam_branch(i: int, opts: RenderOptions) -> list[str]:
    """Filter lines that turn [w{i}_trim] into a circular [w{i}_pip].

    Pipeline (three chains):
      1. `movie=<mask>` source → scale to PIP diameter → gray → [m{i}]
      2. [w{i}_trim] → face-centered square crop → scale to DxD → yuva420p → [w{i}_sq]
      3. [w{i}_sq] + [m{i}] → alphamerge → [w{i}_pip]

    Why face-centered crop FIRST, then scale to square:
      The webcam is typically 16:9. Scaling it directly to a square would
      squash the face. Cropping to a square of side=min(iw,ih) centered on
      the user's face preserves facial proportions; the speaker can sit
      off-center (cam_face_x != 0.5) without ending up with half a face in
      the circle.

    Escaping: expressions with commas are wrapped in single quotes so the
    filter-graph parser doesn't split on the inner commas. The mask path
    is single-quoted so colons/spaces in the filename don't break parsing.
    """
    D = opts.pip_diameter
    cx = opts.cam_face_x
    cy = opts.cam_face_y
    mask_path = str(opts.circle_mask_path)

    # Square side = shorter of the two input dimensions; ffmpeg resolves
    # `min(iw,ih)` at filter-init time, so this is a constant per segment.
    side = "min(iw,ih)"
    x_expr = f"clip({cx:g}*iw-{side}/2,0,iw-{side})"
    y_expr = f"clip({cy:g}*ih-{side}/2,0,ih-{side})"

    return [
        # `loop=-1:size=1` repeats the single-frame mask for the entire
        # segment — without it, alphamerge would truncate the PIP after
        # one frame. size=1 means "buffer 1 frame then loop it forever".
        (
            f"movie='{mask_path}',scale={D}:{D},format=gray,"
            f"loop=loop=-1:size=1[m{i}]"
        ),
        (
            f"[w{i}_trim]"
            f"crop=w='{side}':h='{side}':x='{x_expr}':y='{y_expr}',"
            f"scale={D}:{D},format=yuva420p,setsar=1"
            f"[w{i}_sq]"
        ),
        f"[w{i}_sq][m{i}]alphamerge[w{i}_pip]",
    ]


def _segment_branches(i: int, rs: RenderSegment, opts: RenderOptions) -> list[str]:
    """All filter lines needed to produce [v{i}] and [a{i}] for segment i.

    Only emits the input branches each layout actually needs — ffmpeg
    rejects unused filter outputs (they'd leave a `setpts:default`
    output unconnected), so cam_full skips the screen branch entirely
    and screen_full skips the webcam-video branch entirely. Audio is
    always needed.
    """
    lines: list[str] = []
    needs_screen_video = rs.layout in ("pip", "screen_full")
    needs_webcam_video = rs.layout in ("pip", "cam_full")

    pts_divisor = f"/{rs.speed:g}" if rs.speed != 1.0 else ""
    s_branch = ""

    # --- Screen video branch (trim + setpts + optional zoom crop) ----
    if needs_screen_video:
        lines.append(
            f"[0:v]trim=start={rs.start:.4f}:end={rs.end:.4f},"
            f"setpts=(PTS-STARTPTS){pts_divisor}[s{i}_trim]"
        )
        if rs.zoom is not None:
            # local_duration: time axis seen by filters AFTER the /speed
            # divisor in setpts=(PTS-STARTPTS)/speed. The smooth zoom's
            # ease curves run in this local timebase.
            local_duration = (rs.end - rs.start) / rs.speed
            lines.append(
                f"[s{i}_trim]"
                f"{smooth_zoom_crop_filter(rs.zoom, *opts.screen_res, local_duration)}"
                f"[s{i}_crop]"
            )
            s_branch = f"s{i}_crop"
        else:
            s_branch = f"s{i}_trim"

    # --- Webcam video branch -----------------------------------------
    if needs_webcam_video:
        lines.append(
            f"[1:v]trim=start={rs.start:.4f}:end={rs.end:.4f},"
            f"setpts=(PTS-STARTPTS){pts_divisor}[w{i}_trim]"
        )

    # --- Webcam audio branch (always needed) -------------------------
    atempo = atempo_chain(rs.speed)
    tempo_chain = ("," + ",".join(atempo)) if atempo else ""
    lines.append(
        f"[2:a]atrim=start={rs.start:.4f}:end={rs.end:.4f},"
        f"asetpts=PTS-STARTPTS{tempo_chain}[a{i}]"
    )

    # --- Layout composition ------------------------------------------
    if rs.layout == "cam_full":
        lines.append(
            f"[w{i}_trim]{_scale_fill(opts.output_w, opts.output_h)},"
            f"setsar=1[v{i}]"
        )
    elif rs.layout == "screen_full":
        lines.append(
            f"[{s_branch}]{_scale_fill(opts.output_w, opts.output_h)},"
            f"setsar=1[v{i}]"
        )
    elif rs.layout == "pip":
        lines.append(
            f"[{s_branch}]{_scale_fill(opts.output_w, opts.output_h)},"
            f"setsar=1[s{i}_bg]"
        )
        if opts.pip_shape == "circle":
            # Circle path: face-centered square crop + alphamerge with mask.
            # Emits [w{i}_pip] — the same label the rectangle path produces,
            # so the overlay line below is shape-agnostic.
            lines.extend(_pip_circle_webcam_branch(i, opts))
        else:
            lines.append(
                f"[w{i}_trim]{_scale_fill(opts.pip_w, opts.pip_h)},"
                f"setsar=1[w{i}_pip]"
            )
        lines.append(
            f"[s{i}_bg][w{i}_pip]"
            f"overlay=W-w-{opts.pip_margin}:H-h-{opts.pip_margin}"
            f"[v{i}]"
        )
    else:
        raise ValueError(f"unknown layout {rs.layout!r}")

    return lines


def build_filter_complex(
    render_segments: list[RenderSegment],
    options: RenderOptions,
) -> str:
    """Build the full `filter_complex` body for ffmpeg.

    Returns a newline-separated string of filter chains, terminating in
    `[vout]` and `[aout]` via a concat filter.
    """
    if not render_segments:
        raise ValueError("no render segments — nothing to render")

    lines: list[str] = []
    for i, rs in enumerate(render_segments):
        lines.extend(_segment_branches(i, rs, options))

    n = len(render_segments)
    concat_inputs = "".join(f"[v{i}][a{i}]" for i in range(n))
    lines.append(f"{concat_inputs}concat=n={n}:v=1:a=1[vout][aout]")

    return ";\n".join(lines)


# ---------- orchestration --------------------------------------------

def load_segments(segments_json: Path) -> list[Segment]:
    """Load segments.json into Segment dataclass instances."""
    doc = json.loads(segments_json.read_text())
    out: list[Segment] = []
    for s in doc["segments"]:
        zooms = [
            ZoomWindow(**z) for z in s.get("cursor_zooms", [])
        ]
        out.append(Segment(
            start=float(s["start"]),
            end=float(s["end"]),
            speed=float(s["speed"]),
            layout=s["layout"],
            cursor_zooms=zooms,
        ))
    return out


def render(
    screen: Path,
    webcam: Path,
    audio_source: Path,
    segments: list[Segment],
    output: Path,
    options: RenderOptions | None = None,
) -> None:
    """Build filter graph, write to temp script, invoke ffmpeg.

    Inputs (3 files; audio_source often == webcam or a separate merged file):
      -i screen     → [0]
      -i webcam     → [1]
      -i audio_src  → [2]
    """
    if options is None:
        options = RenderOptions()
        # Auto-detect source resolutions so the zoom crop math is correct.
        try:
            options.screen_res = probe_resolution(screen)
            options.webcam_res = probe_resolution(webcam)
        except Exception as e:
            log.warning("ffprobe failed; using default resolutions (%s)", e)

    rsegs = split_at_zoom_boundaries(segments)
    fc = build_filter_complex(rsegs, options)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, dir=output.parent
    ) as tf:
        tf.write(fc)
        script_path = Path(tf.name)

    try:
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
            "-i", str(screen),
            "-i", str(webcam),
            "-i", str(audio_source),
            "-filter_complex_script", str(script_path),
            "-map", "[vout]", "-map", "[aout]",
            "-c:v", "hevc_videotoolbox",
            "-b:v", options.video_bitrate,
            "-tag:v", "hvc1",
            "-pix_fmt", options.pix_fmt,
            "-c:a", "aac",
            "-b:a", options.audio_bitrate,
            "-ar", str(options.audio_sample_rate),
            "-movflags", "+faststart",
            str(output),
        ]
        log.info("ffmpeg cmd: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)
    finally:
        # Keep the script around on failure for debugging
        pass


def run(args: argparse.Namespace) -> int:
    work_dir = Path(args.work) if getattr(args, "work", None) else config.WORK_DIR
    screen: Path = args.screen
    webcam: Path = args.webcam
    audio: Path = getattr(args, "audio", None) or webcam
    segments_path: Path = args.segments
    output: Path = args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        segments = load_segments(segments_path)
        render(screen, webcam, audio, segments, output)
        print(f"[render] wrote {output}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[render] ERROR: {exc}", file=sys.stderr)
        return 1


__all__ = [
    "RenderSegment",
    "RenderOptions",
    "DEFAULT_EASE_DUR_S",
    "atempo_chain",
    "zoom_crop_filter",
    "smooth_zoom_crop_filter",
    "split_at_zoom_boundaries",
    "build_filter_complex",
    "load_segments",
    "render",
    "run",
]
