"""Stage A — clap-cue sync.

Align two independent video files (silent screen recording + webcam with mic
audio) to a shared T=0 using a clap cue that was captured during recording.

Signals:
  - Webcam audio: one loud onset (the physical clap) at t_webcam
  - Screen video: a full-screen white flash (from screen_flash.py triggered by
    cursor_logger.py's hotkey) at t_screen
  - offset_s = t_webcam - t_screen

If positive, webcam started after screen: trim offset_s off webcam's head.
If negative: trim |offset_s| off screen's head.

Precision: ±1 frame at 60 fps with a tight clap + flash.

TODO(M1): implement. See IMPLEMENTATION_PLAN.md §M1.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SyncResult:
    offset_s: float  # positive = trim webcam, negative = trim screen
    clap_webcam_s: float
    flash_screen_s: float
    method: str  # "auto" | "manual"
    confidence: float  # 0–1; auto detection quality


def detect_clap_in_audio(audio_path: Path, search_window_s: float = 30.0) -> float:
    """Librosa onset_detect on the first `search_window_s` of audio.

    Returns the timestamp of the highest-amplitude onset (the clap).

    TODO(M1): implement using librosa.onset.onset_detect + amplitude ranking
    over a 50 ms window.
    """
    raise NotImplementedError


def detect_flash_in_video(video_path: Path, search_window_s: float = 30.0) -> float:
    """Luminance-diff detection of the white-flash frame in the screen recording.

    Returns t_s of the flash frame (sub-frame precision via np.argmax on
    per-frame mean diffs).

    TODO(M1): implement using cv2.VideoCapture + numpy diff.
    """
    raise NotImplementedError


def apply_offset(
    screen_in: Path,
    webcam_in: Path,
    offset_s: float,
    screen_out: Path,
    webcam_out: Path,
) -> None:
    """ffmpeg -ss trim of whichever track started earlier, stream-copy.

    TODO(M1): implement using utils.ffmpeg_helpers.
    """
    raise NotImplementedError


def run(args: argparse.Namespace) -> int:
    """CLI entry for Stage A."""
    raise NotImplementedError(
        "sync_clap.run — see IMPLEMENTATION_PLAN.md §M1"
    )


__all__ = [
    "SyncResult",
    "detect_clap_in_audio",
    "detect_flash_in_video",
    "apply_offset",
    "run",
]
