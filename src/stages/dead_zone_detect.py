"""Stage C — dead-zone detection on screen recording.

Runs four independent detectors on screen + webcam, intersects their
outputs, and classifies remaining intervals by duration.

Detectors:
  1. ffmpeg freezedetect on screen (still > 2 s)
  2. ffmpeg silencedetect on webcam audio (quiet > 2 s)
  3. auto-editor motion:threshold=0.02 on screen (--export premiere)
  4. LLM transcript cues (Qwen3 prompt)

Intersection: keep ranges where ≥ 2 detectors agree AND duration > 2 s.

Classification:
  duration > 10 s          → action = "cut"
  3 s < duration ≤ 10 s    → action = "speed@8x"
  2 s < duration ≤ 3 s     → action = "speed@4x"

Outputs: `work/dead_zones.json` — [{start, end, action, detectors}, ...]

TODO(M3): implement. See IMPLEMENTATION_PLAN.md §M3.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


DeadZoneAction = Literal["cut", "speed@4x", "speed@8x"]


@dataclass
class DeadZone:
    start: float
    end: float
    action: DeadZoneAction
    detectors: list[str]  # which detectors fired: ["freeze", "silence", ...]


def run_freezedetect(screen_path: Path, db: float = -50.0, min_sec: float = 2.0) -> list[tuple[float, float]]:
    """ffmpeg -vf freezedetect, parse stderr for freeze_start/end.
    TODO(M3): implement via utils.log_parsers.
    """
    raise NotImplementedError


def run_silencedetect(webcam_path: Path, db: float = -30.0, min_sec: float = 2.0) -> list[tuple[float, float]]:
    """ffmpeg -af silencedetect, parse stderr.
    TODO(M3).
    """
    raise NotImplementedError


def run_motion_detect(screen_path: Path, threshold: float = 0.02) -> list[tuple[float, float]]:
    """auto-editor --edit motion --export premiere, parse FCP7 XML.
    TODO(M3).
    """
    raise NotImplementedError


def intersect_intervals(
    *detector_outputs: list[tuple[float, float]],
    min_agree: int = 2,
    min_duration: float = 2.0,
) -> list[tuple[float, float, list[int]]]:
    """Find time ranges where ≥ min_agree detectors are simultaneously "on".

    Returns (start, end, [detector_index, ...]) tuples.

    TODO(M3): implement via sweep-line over (start, +1) / (end, -1) events.
    """
    raise NotImplementedError


def classify(start: float, end: float) -> DeadZoneAction:
    duration = end - start
    if duration > 10.0:
        return "cut"
    if duration > 3.0:
        return "speed@8x"
    return "speed@4x"


def run(args: argparse.Namespace) -> int:
    raise NotImplementedError(
        "dead_zone_detect.run — see IMPLEMENTATION_PLAN.md §M3"
    )


__all__ = [
    "DeadZone",
    "DeadZoneAction",
    "run_freezedetect",
    "run_silencedetect",
    "run_motion_detect",
    "intersect_intervals",
    "classify",
    "run",
]
