"""Stage A — clap-cue sync.

Aligns the cursor.csv timebase to the video timebase using the flash cue
captured at record time (via cursor-tracker/screen_flash.py).

Secondary (optional): if the screen and webcam were recorded separately
(not via OBS source-record), trim them to a shared T=0 using an audible
hand-clap as a second alignment signal.

Typical OBS-source-record workflow:
  - screen recording, webcam recording, and the merged/main recording
    are all written by the same OBS session and share a wall-clock base.
  - cursor.csv is written by an independent process (cursor_logger.py)
    and does NOT share that base. Logger t=0 ≠ OBS t=0.
  - The clap hotkey triggers (a) a full-screen white flash visible in
    the screen recording and (b) a csv row {event=clap}.
  - Offset: csv_to_video_offset = t_flash_video − t_clap_csv

We detect:
  - t_flash_video: luminance-diff spike in the screen recording (strong,
    reliable — a full-screen white flash is unambiguous)
  - t_clap_csv: the event=clap row in the CSV
  - (optional) t_clap_audio: librosa onset in the audio track, used only
    when the user also physically hand-clapped. Useful to cross-check or
    to sync two separately-recorded files.

Output: work/sync.json with everything downstream needs.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import librosa
import numpy as np

from src import config
from src.utils.ffmpeg_helpers import run_ffmpeg


@dataclass
class SyncResult:
    # Core output consumed by downstream stages:
    csv_to_video_offset_s: float | None   # add to CSV t_s to get video t_s; None if no CSV
    video_to_video_offset_s: float         # trim amount between screen and webcam; 0 for OBS-sync

    # Detection telemetry:
    flash_video_s: float | None            # time of white flash in screen video
    clap_csv_s: float | None               # time of event=clap row in CSV
    clap_audio_s: float | None             # time of audio transient (if detected)

    method: str                            # "flash+csv" | "flash+csv+audio" | "manual" | "audio-only"
    confidence: float                      # 0–1; how sure we are about the primary offset

    # If you asked for trimmed outputs, the paths written:
    screen_synced: str | None = None
    webcam_synced: str | None = None


# ---------- detectors (pure) ------------------------------------------

def detect_flash_in_video(video_path: Path, search_window_s: float = 30.0) -> float:
    """Luminance-diff detection of the white-flash frame.

    Scans the first `search_window_s` seconds and returns the timestamp of
    the largest positive jump in per-frame mean intensity. A proper flash
    from screen_flash.py takes mean luminance from ~90 to ~250 — unmissable.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        cap.release()
        raise RuntimeError(f"invalid fps for {video_path}: {fps}")

    max_frames = int(round(search_window_s * fps))
    means: list[float] = []
    n = 0
    while n < max_frames:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        means.append(float(frame.mean()))
        n += 1
    cap.release()

    if len(means) < 10:
        raise RuntimeError(
            f"video too short or unreadable: {video_path} (got {len(means)} frames)"
        )

    means_arr = np.asarray(means, dtype=np.float64)
    diffs = np.diff(means_arr)
    flash_idx = int(np.argmax(diffs)) + 1  # +1: diff[i] = means[i+1] - means[i]
    # Sanity: require that the jump is actually meaningful. A real flash
    # lifts mean luminance by at least ~50 (BGR all at ~255 on a scene
    # averaging ~90). Below that we're probably just picking noise.
    if diffs[flash_idx - 1] < 25.0:
        raise RuntimeError(
            f"no flash detected (biggest luminance jump = "
            f"{diffs[flash_idx - 1]:.1f}, expected ≥25)"
        )
    return float(flash_idx / fps)


def detect_clap_in_audio(
    audio_path: Path,
    search_window_s: float = 30.0,
    near_t: float | None = None,
    tolerance_s: float = 0.5,
) -> float:
    """Loudest-onset detection via librosa, returning timestamp in audio file seconds.

    When `near_t` is given, restrict the search to [near_t - tolerance_s,
    near_t + tolerance_s]. This is important when the recording contains
    speech: a single loud word will beat a hand-clap on raw peak amplitude.
    For sync, we already know the approximate clap time (the flash time),
    so narrowing the window produces reliable results.

    Raises if no onset is detected OR the loudest peak fails the "real hand
    clap" sanity gates (min absolute amplitude ≥ 0.05, AND ≥ 25× ambient).
    """
    sr = 16000
    if near_t is not None:
        offset = max(0.0, near_t - tolerance_s)
        duration = 2.0 * tolerance_s
        time_base = offset
    else:
        offset = 0.0
        duration = search_window_s
        time_base = 0.0

    y, _ = librosa.load(
        str(audio_path), sr=sr, mono=True, offset=offset, duration=duration
    )
    if y.size == 0:
        raise RuntimeError(f"no audio decoded from {audio_path}")

    onset_times = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=True)
    if len(onset_times) == 0:
        raise RuntimeError(
            f"no onsets in audio window "
            f"[{offset:.2f}s, {offset + duration:.2f}s] of {audio_path}"
        )

    abs_y = np.abs(y)
    half_win = int(0.025 * sr)
    peaks = []
    for t in onset_times:
        idx = int(round(t * sr))
        lo, hi = max(0, idx - half_win), min(len(abs_y), idx + half_win)
        peaks.append(float(abs_y[lo:hi].max()) if hi > lo else 0.0)
    peaks_arr = np.asarray(peaks, dtype=np.float64)
    best = int(np.argmax(peaks_arr))

    # Sanity: real hand claps hit peak |amp| ≥ 0.05 (−26 dBFS) AND are 25×
    # louder than the ambient floor. Keystrokes, finger taps, and page
    # rustles generally fail at least one. Both must hold.
    noise_floor = float(np.median(np.abs(y))) + 1e-9
    peak = peaks_arr[best]
    min_peak = max(0.05, 25.0 * noise_floor)
    if peak < min_peak:
        raise RuntimeError(
            f"no audible clap detected (loudest onset peak {peak:.4f} "
            f"< threshold {min_peak:.4f}; noise floor {noise_floor:.5f})"
        )
    return float(time_base + onset_times[best])


def detect_clap_in_csv(cursor_csv: Path) -> float:
    """Read the cursor CSV, return the timestamp of the first event=clap row."""
    with cursor_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("event") == "clap":
                return float(row["t_s"])
    raise RuntimeError(f"no event=clap row in {cursor_csv}")


def _confidence_from_flash_magnitude(video_path: Path, flash_t: float) -> float:
    """Crude 0–1 confidence: how strong was the flash luminance jump?

    A full-frame white flash against a busy desktop jumps mean luminance
    by 150+ units (out of 255). We squash that to 0–1 via tanh(jump/80).
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = int(round(flash_t * fps))
    # Read 5 frames spanning the flash
    start = max(0, frame_idx - 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    means = []
    for _ in range(5):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        means.append(float(frame.mean()))
    cap.release()
    if len(means) < 2:
        return 0.0
    jump = max(means) - min(means)
    return float(np.tanh(max(0.0, jump) / 80.0))


# ---------- optional trimming ------------------------------------------

def apply_offset(
    screen_in: Path,
    webcam_in: Path,
    offset_s: float,
    screen_out: Path,
    webcam_out: Path,
) -> None:
    """Trim whichever of screen/webcam started earlier so they share T=0.

    offset_s > 0: webcam started AFTER screen — trim webcam head by offset_s.
    offset_s < 0: screen started AFTER webcam — trim screen head by |offset_s|.
    offset_s == 0: both pass through unchanged.

    Uses `-c copy` which snaps to the nearest earlier keyframe. For
    OBS output with GOP = 2 s this means ±2 s of accuracy loss on trim;
    downstream Stage E re-encodes anyway so precise offset comes from
    sync.json, not from trimmed file boundaries.
    """
    screen_out.parent.mkdir(parents=True, exist_ok=True)
    webcam_out.parent.mkdir(parents=True, exist_ok=True)

    if offset_s > 0:
        run_ffmpeg([
            "-ss", f"{offset_s:.6f}",
            "-i", str(webcam_in),
            "-c", "copy",
            str(webcam_out),
        ])
        run_ffmpeg(["-i", str(screen_in), "-c", "copy", str(screen_out)])
    elif offset_s < 0:
        run_ffmpeg([
            "-ss", f"{abs(offset_s):.6f}",
            "-i", str(screen_in),
            "-c", "copy",
            str(screen_out),
        ])
        run_ffmpeg(["-i", str(webcam_in), "-c", "copy", str(webcam_out)])
    else:
        run_ffmpeg(["-i", str(screen_in), "-c", "copy", str(screen_out)])
        run_ffmpeg(["-i", str(webcam_in), "-c", "copy", str(webcam_out)])


# ---------- orchestration ----------------------------------------------

def run(args: argparse.Namespace) -> int:
    """CLI entry for Stage A.

    Minimum workflow:
        lfe sync --screen ext.mov --webcam cam.mov --cursor cursor.csv
    This writes `work/sync.json` with csv_to_video_offset_s for downstream use.

    Pass --trim to also write screen_synced.mkv + webcam_synced.mkv trimmed
    to a shared T=0 (needed only when source files are NOT already in sync,
    e.g. recorded by separate devices).

    Pass --manual-offset N to bypass flash detection and use offset N.
    """
    screen: Path = args.screen
    webcam: Path = args.webcam
    cursor_csv: Path | None = getattr(args, "cursor", None)
    trim_videos: bool = bool(getattr(args, "trim", False))
    work_dir: Path = Path(args.work) if getattr(args, "work", None) else config.WORK_DIR
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Manual override path
        if getattr(args, "manual_offset", None) is not None:
            manual = float(args.manual_offset)
            result = SyncResult(
                csv_to_video_offset_s=None,
                video_to_video_offset_s=manual,
                flash_video_s=None,
                clap_csv_s=None,
                clap_audio_s=None,
                method="manual",
                confidence=1.0,
            )
        else:
            # Auto path: flash detection is the primary signal.
            flash_t = detect_flash_in_video(screen)

            # Optional CSV alignment
            clap_csv_t: float | None = None
            if cursor_csv is not None:
                try:
                    clap_csv_t = detect_clap_in_csv(cursor_csv)
                except RuntimeError as e:
                    print(f"[sync] WARN: {e}", file=sys.stderr)

            # Optional physical-clap audio detection (best-effort). Restrict
            # search to ±500 ms of the flash time — otherwise a single loud
            # word of later speech wins on peak amplitude.
            clap_audio_t: float | None = None
            try:
                clap_audio_t = detect_clap_in_audio(
                    webcam, near_t=flash_t, tolerance_s=0.5,
                )
            except RuntimeError:
                pass

            # Derive offsets
            csv_to_video = (flash_t - clap_csv_t) if clap_csv_t is not None else None
            video_to_video = (clap_audio_t - flash_t) if clap_audio_t is not None else 0.0

            # Method label
            if clap_audio_t is not None and clap_csv_t is not None:
                method = "flash+csv+audio"
            elif clap_csv_t is not None:
                method = "flash+csv"
            elif clap_audio_t is not None:
                method = "flash+audio"
            else:
                method = "flash-only"

            result = SyncResult(
                csv_to_video_offset_s=csv_to_video,
                video_to_video_offset_s=video_to_video,
                flash_video_s=flash_t,
                clap_csv_s=clap_csv_t,
                clap_audio_s=clap_audio_t,
                method=method,
                confidence=_confidence_from_flash_magnitude(screen, flash_t),
            )

        # Optional trimming (for non-OBS-synced source pairs)
        if trim_videos:
            screen_out = work_dir / "screen_synced.mkv"
            webcam_out = work_dir / "webcam_synced.mkv"
            apply_offset(screen, webcam, result.video_to_video_offset_s, screen_out, webcam_out)
            result.screen_synced = str(screen_out)
            result.webcam_synced = str(webcam_out)

        # Always write sync.json
        sync_path = work_dir / "sync.json"
        sync_path.write_text(json.dumps(asdict(result), indent=2))

        print(f"[sync] method={result.method} confidence={result.confidence:.3f}")
        if result.flash_video_s is not None:
            print(f"[sync] flash in screen at t={result.flash_video_s:.4f}s")
        if result.clap_csv_s is not None:
            print(f"[sync] csv clap at t={result.clap_csv_s:.4f}s → "
                  f"csv_to_video_offset = {result.csv_to_video_offset_s:+.4f}s")
        if result.clap_audio_s is not None:
            print(f"[sync] audio clap at t={result.clap_audio_s:.4f}s → "
                  f"video_to_video_offset = {result.video_to_video_offset_s:+.4f}s")
        print(f"[sync] wrote {sync_path}")
        if result.screen_synced:
            print(f"[sync] wrote {result.screen_synced}")
            print(f"[sync] wrote {result.webcam_synced}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[sync] ERROR: {exc}", file=sys.stderr)
        return 1


__all__ = [
    "SyncResult",
    "detect_flash_in_video",
    "detect_clap_in_audio",
    "detect_clap_in_csv",
    "apply_offset",
    "run",
]
