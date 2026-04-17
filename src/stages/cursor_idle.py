"""Cursor-idle interval extraction from a cursor.csv log.

Emits time ranges where NO cursor move events occurred for at least
`min_idle_sec` seconds. A "move" here means any row with
`event == "move"`; clicks/scrolls are ignored because you can click
without moving.

Timebase: cursor-logger time (CSV t_s column, starts at logger t=0,
not video t=0). The caller (unify_segments) is responsible for
shifting these intervals into video timebase using
`csv_to_video_offset_s` from `work/sync.json`.

This is a lightweight complement to `cursor_zoom.py` — same CSV, but
we're asking the opposite question ("when was it boring?" vs.
"when was it active?").
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class IdleInterval:
    start: float  # seconds in CSV timebase
    end: float


def detect_cursor_idle_intervals(
    csv_path: Path,
    *,
    duration_s: float,
    min_idle_sec: float = 2.0,
) -> list[IdleInterval]:
    """Return intervals where the cursor did not move for ≥ min_idle_sec.

    Inclusive of leading and trailing idle: if the first move event is at
    t=3.4 s and min_idle_sec is 2.0, the interval [0.0, 3.4] is emitted
    (the user was idle from log start to first movement). Similarly for
    trailing idle up to `duration_s`.

    Args:
        csv_path: cursor.csv path (from cursor-tracker).
        duration_s: clip duration in CSV timebase, used to bound the
            trailing idle interval.
        min_idle_sec: minimum continuous idle duration to report.

    Returns:
        Sorted list of IdleInterval.
    """
    # Collect move-event timestamps only.
    move_times: list[float] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("event") != "move":
                continue
            try:
                t = float(row["t_s"])
            except (KeyError, ValueError):
                continue
            if 0.0 <= t <= duration_s:
                move_times.append(t)

    move_times.sort()

    intervals: list[IdleInterval] = []
    prev_t = 0.0
    for t in move_times:
        if (t - prev_t) >= min_idle_sec:
            intervals.append(IdleInterval(start=prev_t, end=t))
        prev_t = t

    # Trailing idle: from last move to end of clip.
    if (duration_s - prev_t) >= min_idle_sec:
        intervals.append(IdleInterval(start=prev_t, end=duration_s))

    return intervals


__all__ = [
    "IdleInterval",
    "detect_cursor_idle_intervals",
]
