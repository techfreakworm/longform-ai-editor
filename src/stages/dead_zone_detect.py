"""Stage C — dead-zone detection on the screen recording.

Runs independent detectors on the screen video and webcam audio, intersects
their outputs, and classifies remaining intervals by duration.

v1 detectors (shipped):
  1. ffmpeg freezedetect on screen  — still > min_sec
  2. ffmpeg silencedetect on audio  — quiet > min_sec

Future additions (TODO, see IMPLEMENTATION_PLAN.md §M3):
  3. auto-editor motion:threshold=X (FCP7 XML parse)
  4. LLM transcript cues ("while that installs", "let me skip")

Intersection rule: keep time ranges where ≥ `min_agree` detectors are
simultaneously "on" AND the resulting range duration ≥ `min_duration`.
With 2 detectors, that means both must fire — classic "silent narrator
while nothing moves on screen" signal, which is high-confidence dead-air.

Classification (per IMPLEMENTATION_PLAN.md §M3):
  duration > 10 s          → action = "cut"
  3 s < duration ≤ 10 s    → action = "speed@8x"
  2 s < duration ≤ 3 s     → action = "speed@4x"
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from src import config
from src.utils.log_parsers import parse_freezedetect, parse_silencedetect

log = logging.getLogger(__name__)


DeadZoneAction = Literal["cut", "speed@4x", "speed@8x"]


@dataclass
class DeadZone:
    start: float
    end: float
    action: DeadZoneAction
    detectors: list[str]  # which detectors fired for this range


# ---------- individual detectors --------------------------------------

def run_freezedetect(
    screen_path: Path,
    db: float = -50.0,
    min_sec: float = 2.0,
) -> list[tuple[float, float]]:
    """Run ffmpeg freezedetect on the screen track, parse stderr.

    The `n` parameter is in dB relative to full-scale (0 dB = no frame
    diff at all). -50 dB is more tolerant than ffmpeg's -60 dB default —
    avoids cursor-blink + menu-bar-clock false positives on a real
    desktop screencast.
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-nostdin",
        "-i", str(screen_path),
        "-vf", f"freezedetect=n={db}dB:d={min_sec}",
        "-map", "0:v:0",
        "-an",
        "-f", "null", "-",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg freezedetect failed: {r.stderr[-400:]}")
    return parse_freezedetect(r.stderr)


def run_silencedetect(
    audio_path: Path,
    db: float = -30.0,
    min_sec: float = 2.0,
) -> list[tuple[float, float]]:
    """Run ffmpeg silencedetect on the audio track.

    -30 dB noise floor matches auto-editor's default silence threshold
    used elsewhere in the pipeline. 2-second minimum avoids classifying
    natural speech pauses as dead zones.
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-nostdin",
        "-i", str(audio_path),
        "-af", f"silencedetect=noise={db}dB:d={min_sec}",
        "-vn",
        "-f", "null", "-",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg silencedetect failed: {r.stderr[-400:]}")
    return parse_silencedetect(r.stderr)


# ---------- intersection (sweep-line) --------------------------------

def intersect_intervals(
    *detector_outputs: list[tuple[float, float]],
    min_agree: int = 2,
    min_duration: float = 2.0,
    merge_gap: float = 0.5,
) -> list[tuple[float, float, list[int]]]:
    """Find time ranges where ≥ min_agree detectors are simultaneously on.

    Returns (start, end, [detector_indices_firing]) tuples.

    Uses sweep-line over interval boundary events. Per-detector open
    counts are tracked via a Counter so multiple abutting or overlapping
    intervals from the SAME detector are treated correctly — a detector
    is considered "on" while its count > 0.

    Tie ordering at event time: opens before closes, so two abutting
    intervals from the same detector bridge cleanly (count goes 1 → 2 → 1
    instead of 1 → 0 → 1, which would wrongly break coverage).

    Merges adjacent output ranges within merge_gap and drops ranges
    shorter than min_duration.
    """
    if not detector_outputs:
        return []

    events: list[tuple[float, int, int]] = []
    for idx, intervals in enumerate(detector_outputs):
        for s, e in intervals:
            if e > s:
                events.append((s, +1, idx))
                events.append((e, -1, idx))
    events.sort(key=lambda ev: (ev[0], -ev[1]))

    counts: Counter[int] = Counter()
    above = False
    range_start = 0.0
    range_detectors_union: set[int] = set()
    raw_ranges: list[tuple[float, float, list[int]]] = []

    for t, delta, idx in events:
        counts[idx] += delta
        active_detectors = {d for d, c in counts.items() if c > 0}
        is_above = len(active_detectors) >= min_agree

        if is_above and not above:
            range_start = t
            range_detectors_union = set(active_detectors)
            above = True
        elif above and not is_above:
            raw_ranges.append((range_start, t, sorted(range_detectors_union)))
            above = False
        if above:
            range_detectors_union |= active_detectors

    # Merge close ranges
    merged: list[tuple[float, float, list[int]]] = []
    for s, e, dets in raw_ranges:
        if merged and s - merged[-1][1] < merge_gap:
            prev_s, prev_e, prev_dets = merged[-1]
            merged[-1] = (
                prev_s,
                max(prev_e, e),
                sorted(set(prev_dets) | set(dets)),
            )
        else:
            merged.append((s, e, list(dets)))

    return [(s, e, dets) for (s, e, dets) in merged if (e - s) >= min_duration]


# ---------- classify ---------------------------------------------------

def classify(
    start: float,
    end: float,
    cut_min_sec: float = 10.0,
    speed_8x_min_sec: float = 3.0,
) -> DeadZoneAction:
    d = end - start
    if d > cut_min_sec:
        return "cut"
    if d > speed_8x_min_sec:
        return "speed@8x"
    return "speed@4x"


# ---------- CLI entry --------------------------------------------------

def run(args: argparse.Namespace) -> int:
    work_dir = Path(args.work) if getattr(args, "work", None) else config.WORK_DIR
    work_dir.mkdir(parents=True, exist_ok=True)

    screen: Path = args.screen
    webcam: Path = args.webcam
    audio: Path = getattr(args, "audio", None) or webcam

    freeze_db = float(getattr(args, "freeze_db", config.FREEZE_DB))
    freeze_min = float(getattr(args, "freeze_min", config.FREEZE_MIN_SEC))
    silence_db = float(getattr(args, "silence_db", config.SILENCE_DB))
    silence_min = float(getattr(args, "silence_min", config.SILENCE_MIN_SEC))
    cut_min = float(getattr(args, "cut_min", config.CUT_MIN_SEC))
    speed_8x_min = float(getattr(args, "speed_8x_min", config.SPEED_8X_MIN_SEC))

    try:
        print(f"[dead_zone] running freezedetect on {screen.name} ...")
        freezes = run_freezedetect(screen, db=freeze_db, min_sec=freeze_min)
        print(f"[dead_zone]   {len(freezes)} freeze intervals")

        print(f"[dead_zone] running silencedetect on {audio.name} ...")
        silences = run_silencedetect(audio, db=silence_db, min_sec=silence_min)
        print(f"[dead_zone]   {len(silences)} silence intervals")

        # Detector indices correspond to this order: 0=freeze, 1=silence.
        intersected = intersect_intervals(
            freezes, silences,
            min_agree=2,
            min_duration=max(freeze_min, silence_min),
        )
        detector_names = ["freezedetect", "silencedetect"]

        zones = [
            DeadZone(
                start=s, end=e,
                action=classify(s, e, cut_min, speed_8x_min),
                detectors=[detector_names[i] for i in dets],
            )
            for (s, e, dets) in intersected
        ]

        out_path = work_dir / "dead_zones.json"
        out_path.write_text(json.dumps(
            {"zones": [asdict(z) for z in zones]},
            indent=2,
        ))

        # Side artifact for triple-intersection hard-cut in unify_segments —
        # the raw silencedetect output without freeze intersection.
        silence_path = work_dir / "silence_intervals.json"
        silence_path.write_text(json.dumps(
            {
                "noise_db": silence_db,
                "min_sec": silence_min,
                "intervals": [{"start": s, "end": e} for s, e in silences],
            },
            indent=2,
        ))
        print(
            f"[dead_zone] {len(zones)} merged dead zones "
            f"(cut={sum(1 for z in zones if z.action == 'cut')}, "
            f"speed@8x={sum(1 for z in zones if z.action == 'speed@8x')}, "
            f"speed@4x={sum(1 for z in zones if z.action == 'speed@4x')})"
        )
        print(f"[dead_zone] wrote {out_path}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[dead_zone] ERROR: {exc}", file=sys.stderr)
        return 1


__all__ = [
    "DeadZone",
    "DeadZoneAction",
    "run_freezedetect",
    "run_silencedetect",
    "intersect_intervals",
    "classify",
    "run",
]
