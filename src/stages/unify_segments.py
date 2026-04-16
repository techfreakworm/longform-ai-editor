"""Stage D — merge all decisions into the canonical segments.json.

Consumes:
  work/layout_plan.json    (the backbone — covers entire duration)
  work/filler_cuts.json    (intervals to remove)
  work/dead_zones.json     (intervals to cut or speed-ramp)
  cursor.csv (optional)    (for cursor_zoom annotations)

Emits:
  work/segments.json — [{in, out, speed, layout, cursor_zoom?}, ...]
  covering the entire final timeline with no gaps or overlaps.

Algorithm (in order — order matters!):
  1. Start with layout_plan as the backbone.
  2. Ripple-remove every filler_cuts entry.
  3. For each dead_zone:
       action=cut     → ripple-remove
       action=speed@N → keep, set speed=N on that portion
  4. Merge adjacent same-(layout,speed) segments.
  5. If cursor.csv present, annotate screen-visible segments with zoom windows
     from cursor_zoom.generate_zoom_segments().
  6. Validate: no gaps, no overlaps, every field present.

TODO(M4): implement. See IMPLEMENTATION_PLAN.md §M4.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


Layout = Literal["cam_full", "pip", "screen_full"]


@dataclass
class ZoomWindow:
    start: float  # relative to segment start
    end: float
    zoom: float
    cx: float  # 0–1 normalized
    cy: float


@dataclass
class Segment:
    in_: float
    out: float
    speed: float = 1.0
    layout: Layout = "pip"
    cursor_zoom: list[ZoomWindow] = field(default_factory=list)


def apply_cuts(timeline: list[Segment], cuts: list[tuple[float, float]]) -> list[Segment]:
    """Ripple-delete each (start, end) interval from the timeline.
    TODO(M4).
    """
    raise NotImplementedError


def apply_dead_zones(timeline: list[Segment], dead_zones) -> list[Segment]:
    """Apply dead-zone actions (cut or speed-ramp) to the timeline.
    TODO(M4).
    """
    raise NotImplementedError


def merge_adjacent(timeline: list[Segment]) -> list[Segment]:
    """Merge same-(layout,speed) neighbors.
    TODO(M4).
    """
    raise NotImplementedError


def annotate_cursor_zooms(timeline: list[Segment], cursor_csv: Path) -> list[Segment]:
    """Attach cursor-zoom windows to segments showing the screen.
    TODO(M4).
    """
    raise NotImplementedError


def validate(timeline: list[Segment], total_duration: float, tol: float = 1e-3) -> None:
    """Assert no gaps, no overlaps, all fields set.
    TODO(M4): raise ValueError with diagnostic on failure.
    """
    raise NotImplementedError


def run(args: argparse.Namespace) -> int:
    raise NotImplementedError(
        "unify_segments.run — see IMPLEMENTATION_PLAN.md §M4"
    )


__all__ = [
    "Layout",
    "ZoomWindow",
    "Segment",
    "apply_cuts",
    "apply_dead_zones",
    "merge_adjacent",
    "annotate_cursor_zooms",
    "validate",
    "run",
]
