"""Thin wrappers around subprocess + ffmpeg/ffprobe.

TODO: implement across milestones as needed by individual stages.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


def probe_duration(path: Path) -> float:
    """ffprobe -show_entries format=duration."""
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(r.stdout.strip())


def probe_resolution(path: Path) -> tuple[int, int]:
    """ffprobe stream width+height of first video stream."""
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "stream=width,height",
         "-of", "csv=s=,:p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    w, h = r.stdout.strip().split(",")
    return int(w), int(h)


def probe_fps(path: Path) -> float:
    """ffprobe stream r_frame_rate of first video stream."""
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    num, den = r.stdout.strip().split("/")
    return float(num) / float(den)


def run_ffmpeg(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    """Run ffmpeg with `-y -hide_banner -loglevel warning` prepended."""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning", *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


__all__ = ["probe_duration", "probe_resolution", "probe_fps", "run_ffmpeg"]
