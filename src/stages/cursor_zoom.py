"""Python port of CapSoftware/Cap's generate_zoom_segments_from_clicks_impl.

Input: cursor.csv from cursor-tracker + video dimensions + clip duration
Output: list of ZoomWindow (start, end, zoom, cx, cy) tuples

Algorithm verbatim from:
  https://github.com/CapSoftware/Cap/blob/main/apps/desktop/src-tauri/src/recording.rs
  (generate_zoom_segments_from_clicks_impl, lines 2125–2389)

All 14 constants preserved. License note: the algorithm is AGPL-3.0 in Cap's
source — porting to Python with attribution is permitted, but distributing
the resulting pipeline means the pipeline code is AGPL-3.0 too unless you
reimplement the algorithm from the documented design (the constants + the
prose description in research/06-filler-removal-and-autozoom.md §2.1).

TODO(M4): implement the clean-room port. See IMPLEMENTATION_PLAN.md §M4.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

# --- Constants (from Cap's recording.rs) -------------------------------
STOP_PADDING_SECONDS = 0.5
CLICK_GROUP_TIME_THRESHOLD_SECS = 2.5
CLICK_GROUP_SPATIAL_THRESHOLD = 0.15  # 15% of normalized screen
CLICK_PRE_PADDING = 0.4
CLICK_POST_PADDING = 1.8
MOVEMENT_PRE_PADDING = 0.3
MOVEMENT_POST_PADDING = 1.5
MERGE_GAP_THRESHOLD = 0.8
MIN_SEGMENT_DURATION = 1.0
MOVEMENT_WINDOW_SECONDS = 1.5
MOVEMENT_EVENT_DISTANCE_THRESHOLD = 0.02
MOVEMENT_WINDOW_DISTANCE_THRESHOLD = 0.08
AUTO_ZOOM_AMOUNT = 1.5
SHAKE_FILTER_THRESHOLD = 0.33
SHAKE_FILTER_WINDOW_MS = 150.0


@dataclass
class CursorEvent:
    t_s: float
    x: float  # normalized 0–1
    y: float  # normalized 0–1
    is_click: bool


@dataclass
class ZoomSegment:
    start: float
    end: float
    zoom: float
    cx: float  # normalized centroid 0–1
    cy: float


def parse_cursor_csv(csv_path: Path, screen_w: int, screen_h: int) -> tuple[list, list]:
    """Read cursor.csv, split into (clicks, moves), normalize coords."""
    raise NotImplementedError  # TODO(M4)


def group_clicks(clicks: list[CursorEvent], end_limit: float) -> list[list[CursorEvent]]:
    """Cluster clicks by temporal (2.5 s) AND spatial (15%) windows.
    TODO(M4).
    """
    raise NotImplementedError


def generate_move_intervals(moves: list[CursorEvent], end_limit: float) -> list[tuple[float, float]]:
    """Sweep cursor moves with rolling 1.5 s distance window + shake filter.
    TODO(M4).
    """
    raise NotImplementedError


def generate_zoom_segments(cursor_csv: Path, video_path: Path, duration_s: float) -> list[ZoomSegment]:
    """Top-level entry: CSV + video → list of 1.5× zoom segments.
    TODO(M4).
    """
    raise NotImplementedError


__all__ = [
    "CursorEvent",
    "ZoomSegment",
    "parse_cursor_csv",
    "group_clicks",
    "generate_move_intervals",
    "generate_zoom_segments",
]
