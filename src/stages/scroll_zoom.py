"""Zoom-on-scroll / window-change detection on the screen track.

Runs during cursor-idle windows only: if the screen content changes
significantly while the cursor is still, the presenter is probably
pointing the viewer at something new appearing on screen (scroll,
window switch, installer progress, test output, etc.). Emit a
ZoomSegment centered on the change region so the viewer can see it.

Algorithm:
  1. For each cursor-idle window ≥ MIN_WINDOW_SEC, sample frames at
     SAMPLE_RATE_HZ through ffmpeg.
  2. Compute per-pair absdiff with the prior frame, thresholded at
     ~30/255 per pixel (robust to video compression artifacts).
  3. If the fraction of changed pixels exceeds DIFF_THRESHOLD (default
     0.04 ≈ 4% of frame), record a change event with its bounding box.
  4. Cluster consecutive change events into zoom windows. Emit one
     ZoomSegment per cluster, centered on the bbox centroid.

This is deliberately cheaper than the OCR path — no per-frame OCR is
run and the whole pass processes a 30-min clip in well under 1 min on
M-class hardware. OpenCV is already a base dep so no new install.
"""
from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.stages.cursor_zoom import AUTO_ZOOM_AMOUNT, ZoomSegment

log = logging.getLogger(__name__)


# ----- tuning knobs --------------------------------------------------

MIN_WINDOW_SEC = 1.5            # skip idle windows shorter than this
SAMPLE_RATE_HZ = 2.0            # frame sample rate inside each window
PIXEL_CHANGE_THRESHOLD = 30     # per-pixel absdiff threshold (0-255)
MIN_ZOOM_DURATION = 0.6         # don't emit zooms shorter than this
MAX_ZOOM_DURATION = 4.0         # cap long change runs
PRE_PADDING_SEC = 0.2           # start zoom slightly before first change
POST_PADDING_SEC = 0.6          # hold zoom slightly past last change


# ----- types ---------------------------------------------------------

@dataclass
class ChangeEvent:
    t_s: float       # absolute video time of this sample
    cx_norm: float   # change-region centroid in [0,1]
    cy_norm: float


# ----- frame sampling ------------------------------------------------

def _sample_frames_gray(
    screen_path: Path, start_s: float, end_s: float,
    sample_rate_hz: float,
) -> tuple[list[np.ndarray], list[float], tuple[int, int]]:
    """Pipe grayscale raw frames via ffmpeg for [start_s, end_s).

    Returns (frames, timestamps, (w, h)). Frames are (h, w) uint8.
    """
    if end_s - start_s <= 0:
        return [], [], (0, 0)

    # Probe dimensions once per sample — ffmpeg reports them on stderr.
    probe_cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x",
        str(screen_path),
    ]
    r = subprocess.run(probe_cmd, capture_output=True, text=True)
    if r.returncode != 0 or "x" not in r.stdout.strip():
        raise RuntimeError(f"ffprobe dim failed: {r.stderr[-200:]}")
    w_str, h_str = r.stdout.strip().split("x")
    w, h = int(w_str), int(h_str)

    # Downscale to speed up the diff pass — 640-wide is plenty for
    # detecting change REGIONS (not text).
    scale_w = 640
    scale_h = int(h * scale_w / w)
    frame_bytes = scale_w * scale_h

    cmd = [
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error",
        "-ss", f"{start_s:.3f}",
        "-to", f"{end_s:.3f}",
        "-i", str(screen_path),
        "-vf", f"fps={sample_rate_hz},scale={scale_w}:{scale_h},format=gray",
        "-f", "rawvideo", "-pix_fmt", "gray",
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None

    frames: list[np.ndarray] = []
    times: list[float] = []
    idx = 0
    dt = 1.0 / sample_rate_hz
    while True:
        buf = proc.stdout.read(frame_bytes)
        if len(buf) < frame_bytes:
            break
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(scale_h, scale_w)
        frames.append(arr)
        times.append(start_s + idx * dt)
        idx += 1
    proc.stdout.close()
    proc.wait()
    if proc.returncode not in (0, None):
        err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        raise RuntimeError(f"scroll_zoom frame pipe failed: {err[-300:]}")

    return frames, times, (scale_w, scale_h)


# ----- core detection ------------------------------------------------

def detect_changes_in_window(
    screen_path: Path,
    start_s: float,
    end_s: float,
    *,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
    diff_threshold: float = 0.04,
    pixel_change_threshold: int = PIXEL_CHANGE_THRESHOLD,
) -> list[ChangeEvent]:
    """Return change events for a single cursor-idle window.

    Two consecutive frames are compared; if > diff_threshold fraction of
    pixels differ by ≥ pixel_change_threshold, a change event is emitted
    at the later frame's timestamp, with centroid = bbox midpoint of the
    changed pixels.
    """
    frames, times, (fw, fh) = _sample_frames_gray(
        screen_path, start_s, end_s, sample_rate_hz,
    )
    if len(frames) < 2:
        return []

    events: list[ChangeEvent] = []
    total_px = fw * fh
    for i in range(1, len(frames)):
        diff = np.abs(frames[i].astype(np.int16) - frames[i - 1].astype(np.int16))
        mask = diff >= pixel_change_threshold
        frac = mask.sum() / total_px
        if frac < diff_threshold:
            continue
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        cx_norm = (float(xs.min()) + float(xs.max())) / (2.0 * fw)
        cy_norm = (float(ys.min()) + float(ys.max())) / (2.0 * fh)
        events.append(ChangeEvent(t_s=times[i], cx_norm=cx_norm, cy_norm=cy_norm))
    return events


def cluster_events_to_zooms(
    events: list[ChangeEvent],
    window_start_s: float,
    window_end_s: float,
    *,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
) -> list[ZoomSegment]:
    """Group consecutive change events into zoom windows.

    Events within one sample interval (1/sample_rate_hz + small slack)
    of each other belong to the same cluster. Each cluster becomes one
    ZoomSegment with duration [first_event - pre, last_event + post]
    and centroid = mean of cluster centroids.
    """
    if not events:
        return []

    gap_limit = (1.5 / sample_rate_hz) + 0.05  # ~1.5 samples worth
    clusters: list[list[ChangeEvent]] = [[events[0]]]
    for ev in events[1:]:
        if ev.t_s - clusters[-1][-1].t_s <= gap_limit:
            clusters[-1].append(ev)
        else:
            clusters.append([ev])

    zooms: list[ZoomSegment] = []
    for c in clusters:
        start = max(window_start_s, c[0].t_s - PRE_PADDING_SEC)
        end = min(window_end_s, c[-1].t_s + POST_PADDING_SEC)
        if (end - start) < MIN_ZOOM_DURATION:
            continue
        if (end - start) > MAX_ZOOM_DURATION:
            end = start + MAX_ZOOM_DURATION
        cx = sum(e.cx_norm for e in c) / len(c)
        cy = sum(e.cy_norm for e in c) / len(c)
        zooms.append(ZoomSegment(
            start=start, end=end, zoom=AUTO_ZOOM_AMOUNT, cx=cx, cy=cy,
        ))
    return zooms


def detect_scroll_zooms(
    screen_path: Path,
    cursor_idle_intervals: list[tuple[float, float]],
    *,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
    diff_threshold: float = 0.04,
) -> list[ZoomSegment]:
    """End-to-end: cursor-idle intervals → zoom segments on content change.

    `cursor_idle_intervals` are already in VIDEO timebase (caller applies
    sync offset before passing in). Emitted ZoomSegments are in video
    timebase too.
    """
    all_zooms: list[ZoomSegment] = []
    for start_s, end_s in cursor_idle_intervals:
        if end_s - start_s < MIN_WINDOW_SEC:
            continue
        try:
            events = detect_changes_in_window(
                screen_path, start_s, end_s,
                sample_rate_hz=sample_rate_hz,
                diff_threshold=diff_threshold,
            )
        except RuntimeError as exc:
            log.warning("scroll_zoom skipping [%.2f, %.2f]: %s",
                        start_s, end_s, exc)
            continue
        all_zooms.extend(
            cluster_events_to_zooms(events, start_s, end_s,
                                    sample_rate_hz=sample_rate_hz)
        )
    return all_zooms


__all__ = [
    "ChangeEvent",
    "detect_changes_in_window",
    "cluster_events_to_zooms",
    "detect_scroll_zooms",
    "MIN_WINDOW_SEC",
    "SAMPLE_RATE_HZ",
    "PIXEL_CHANGE_THRESHOLD",
    "MIN_ZOOM_DURATION",
    "MAX_ZOOM_DURATION",
]
