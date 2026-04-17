"""Port of CapSoftware/Cap's `generate_zoom_segments_from_clicks_impl`.

Reads a cursor.csv from cursor-tracker and emits a list of ZoomSegments —
ranges of time where the screen should zoom in on the cursor, with a
centroid and zoom factor attached.

Timebase: the input CSV timestamps are in cursor-logger time (logger t=0,
not OBS t=0). Output ZoomSegment.start / end are in the SAME CSV timebase.
The caller (unify_segments) is responsible for adding
csv_to_video_offset_s from work/sync.json to shift into video timebase.

Coordinate system: pynput logs in macOS global screen space. On a multi-
monitor setup the recorded display has an origin offset and its own
dimensions. Caller provides screen_w, screen_h, origin_x, origin_y and
we transform to normalized [0,1] x [0,1] relative to THE recorded display.
Clicks outside [0,1] are dropped (they're on a different monitor).
Moves outside [0,1] are kept but their distance is not clamped — a burst
of off-screen motion is still real motion the algorithm should react to.

Algorithm constants are VERBATIM from Cap's recording.rs lines 2125–2389.
See research/06-filler-removal-and-autozoom.md §2.1 for the annotated
original. All 14 constants preserved; naming differs slightly to satisfy
PEP 8.
"""
from __future__ import annotations

import csv
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path


# --- Cap's constants (all 14) ------------------------------------------

STOP_PADDING_SECONDS = 0.5
CLICK_GROUP_TIME_THRESHOLD_SECS = 2.5
CLICK_GROUP_SPATIAL_THRESHOLD = 0.15
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

# Our own cap on segment length (not in Cap — safety for multi-minute idle
# loops). The user asked for ≤6 s zoom windows for a reasonable feel.
MAX_SEGMENT_DURATION = 6.0


# --- Types --------------------------------------------------------------

@dataclass
class CursorEvent:
    t_s: float
    x: float  # normalized [0, 1] if on-screen
    y: float  # normalized [0, 1] if on-screen
    is_click: bool  # True = down-edge click, False = move


@dataclass
class ZoomSegment:
    start: float       # seconds in CSV timebase
    end: float
    zoom: float        # scale factor (e.g. 1.5)
    cx: float          # normalized centroid x in [0, 1]
    cy: float          # normalized centroid y in [0, 1]


# Speech-emphasis zoom tuning. Strength maps to zoom magnification and
# factors into merge rules downstream.
SPEECH_ZOOM_BY_STRENGTH = {
    "soft": 1.25,
    "normal": 1.5,
    "strong": 1.7,
}


# --- Parse --------------------------------------------------------------

def parse_cursor_csv(
    csv_path: Path,
    screen_w: int,
    screen_h: int,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
) -> tuple[list[CursorEvent], list[CursorEvent]]:
    """Read cursor.csv, transform to the recorded-display frame, split
    into (clicks, moves).

    Only "down" clicks are retained (down=1); up-edges are ignored because
    the Cap algorithm operates on press events.

    Clicks at normalized coords outside [0,1] × [0,1] are dropped — they
    happened on a different display. Moves are kept regardless so the
    algorithm can detect inter-display cursor motion (rare, but real).
    """
    clicks: list[CursorEvent] = []
    moves: list[CursorEvent] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            event = row.get("event", "")
            try:
                t = float(row["t_s"])
                xg = float(row["x"])
                yg = float(row["y"])
            except (KeyError, ValueError):
                continue
            x = (xg - origin_x) / screen_w
            y = (yg - origin_y) / screen_h

            if event == "click" and row.get("down") == "1":
                if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                    clicks.append(CursorEvent(t_s=t, x=x, y=y, is_click=True))
            elif event == "move":
                moves.append(CursorEvent(t_s=t, x=x, y=y, is_click=False))

    return clicks, moves


# --- Click grouping -----------------------------------------------------

def group_clicks(
    clicks: list[CursorEvent], end_limit: float
) -> list[list[CursorEvent]]:
    """Cluster down-clicks by BOTH temporal (2.5 s) AND spatial (15%)
    proximity. A click joins an existing group if ANY member of that group
    is within both thresholds; otherwise starts a new group.
    """
    groups: list[list[CursorEvent]] = []
    for c in clicks:
        if c.t_s > end_limit:
            continue
        placed = False
        for g in groups:
            for o in g:
                time_close = abs(c.t_s - o.t_s) < CLICK_GROUP_TIME_THRESHOLD_SECS
                dist = math.hypot(c.x - o.x, c.y - o.y)
                if time_close and dist < CLICK_GROUP_SPATIAL_THRESHOLD:
                    g.append(c)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            groups.append([c])
    return groups


def click_intervals(
    groups: list[list[CursorEvent]], end_limit: float
) -> list[tuple[float, float]]:
    """Convert click groups to (start, end) intervals with padding."""
    out: list[tuple[float, float]] = []
    for g in groups:
        s = max(min(e.t_s for e in g) - CLICK_PRE_PADDING, 0.0)
        e = min(max(e.t_s for e in g) + CLICK_POST_PADDING, end_limit)
        if e > s:
            out.append((s, e))
    return out


# --- Movement sweep with shake filter ----------------------------------

def movement_intervals(
    moves: list[CursorEvent], end_limit: float
) -> list[tuple[float, float]]:
    """Sweep cursor moves, emit intervals around substantial motion.

    Maintains a rolling distance window of the last MOVEMENT_WINDOW_SECONDS
    seconds. A move event triggers an interval emit when EITHER:
      - the per-event step distance ≥ MOVEMENT_EVENT_DISTANCE_THRESHOLD, OR
      - the cumulative windowed distance ≥ MOVEMENT_WINDOW_DISTANCE_THRESHOLD

    Shake filter: inspect the last SHAKE_FILTER_WINDOW_MS of events; if
    there are ≥ 2 direction reversals AND the total traversed distance is
    small (< SHAKE_FILTER_THRESHOLD × 3), drop the event as jitter.
    """
    intervals: list[tuple[float, float]] = []
    # State
    last_pos: tuple[float, float] | None = None
    window: deque[tuple[float, float]] = deque()  # (t, per-step distance)
    win_dist = 0.0
    shake: deque[tuple[float, float, float]] = deque()  # (t_ms, x, y)

    moves_sorted = sorted(moves, key=lambda m: m.t_s)

    for m in moves_sorted:
        if m.t_s >= end_limit:
            break

        # Per-event distance
        if last_pos is None:
            d = 0.0
        else:
            d = math.hypot(m.x - last_pos[0], m.y - last_pos[1])
        last_pos = (m.x, m.y)
        if d <= 1e-9:
            continue

        # Shake filter
        shake.append((m.t_s * 1000.0, m.x, m.y))
        while shake and (m.t_s * 1000.0 - shake[0][0]) > SHAKE_FILTER_WINDOW_MS:
            shake.popleft()
        if len(shake) >= 3:
            pts = [(p[1], p[2]) for p in shake]
            # Reversal count: sign flip of successive segment dot products
            reversals = 0
            for i in range(1, len(pts) - 1):
                ax, ay = pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]
                bx, by = pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]
                if (ax * bx + ay * by) < 0:
                    reversals += 1
            total_dist = sum(
                math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
                for i in range(len(pts) - 1)
            )
            if reversals >= 2 and total_dist < SHAKE_FILTER_THRESHOLD * 3.0:
                continue

        # Rolling window
        window.append((m.t_s, d))
        win_dist += d
        while window and (m.t_s - window[0][0]) > MOVEMENT_WINDOW_SECONDS:
            win_dist -= window.popleft()[1]
        if win_dist < 0.0:
            win_dist = 0.0

        if d >= MOVEMENT_EVENT_DISTANCE_THRESHOLD or win_dist >= MOVEMENT_WINDOW_DISTANCE_THRESHOLD:
            s = max(m.t_s - MOVEMENT_PRE_PADDING, 0.0)
            e = min(m.t_s + MOVEMENT_POST_PADDING, end_limit)
            if e > s:
                intervals.append((s, e))

    return intervals


# --- Merge + filter -----------------------------------------------------

def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Sort and merge intervals that are within MERGE_GAP_THRESHOLD of each other."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged: list[tuple[float, float]] = []
    for s, e in intervals:
        if merged and s <= merged[-1][1] + MERGE_GAP_THRESHOLD:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def filter_short(
    intervals: list[tuple[float, float]], min_duration: float = MIN_SEGMENT_DURATION
) -> list[tuple[float, float]]:
    return [(s, e) for s, e in intervals if (e - s) >= min_duration]


# --- Centroid per segment ----------------------------------------------

def compute_centroid(
    moves: list[CursorEvent], s: float, e: float
) -> tuple[float, float]:
    """Mean (x, y) of moves within [s, e]. Falls back to screen center
    (0.5, 0.5) if no moves in that range."""
    in_seg = [m for m in moves if s <= m.t_s <= e]
    if not in_seg:
        return (0.5, 0.5)
    cx = sum(m.x for m in in_seg) / len(in_seg)
    cy = sum(m.y for m in in_seg) / len(in_seg)
    return (cx, cy)


# --- Top-level entry ----------------------------------------------------

def _cursor_position_at(
    moves: list[CursorEvent], t: float
) -> tuple[float, float] | None:
    """Return the cursor position closest to time `t` from the moves list.

    Returns None if moves is empty or `t` is before the first move
    timestamp minus 0.5 s (no reasonable estimate available).
    """
    if not moves:
        return None
    nearest = min(moves, key=lambda m: abs(m.t_s - t))
    if abs(nearest.t_s - t) > 0.5 and t < nearest.t_s:
        return None
    return (nearest.x, nearest.y)


def zoom_segments_from_hints(
    hints: list[dict],
    moves: list[CursorEvent],
    duration_s: float,
) -> list[ZoomSegment]:
    """Convert LLM ZoomHint entries into ZoomSegments in video timebase.

    Centroid resolution priority:
      1. Cursor position at `start` (via nearest move event, ±0.5 s) if
         cursor data is available.
      2. Screen center (0.5, 0.5) if no cursor data.

    Magnification comes from SPEECH_ZOOM_BY_STRENGTH (1.25 / 1.5 / 1.7).

    Args:
        hints: list of dicts {start, end, strength, ...}. Accepts dicts
            rather than a typed ZoomHint so callers don't need to import
            pydantic models.
        moves: cursor move events in CSV timebase. These should already
            be shifted into video timebase by the caller if a sync
            offset applies.
        duration_s: clip duration, used to clamp hint windows.
    """
    segs: list[ZoomSegment] = []
    for h in hints:
        start = max(0.0, float(h["start"]))
        end = min(duration_s, float(h["end"]))
        if end <= start:
            continue
        strength = h.get("strength", "normal")
        zoom = SPEECH_ZOOM_BY_STRENGTH.get(strength, AUTO_ZOOM_AMOUNT)
        pos = _cursor_position_at(moves, start)
        cx, cy = pos if pos is not None else (0.5, 0.5)
        segs.append(ZoomSegment(start=start, end=end, zoom=zoom, cx=cx, cy=cy))
    return segs


def merge_zoom_segments(
    a: list[ZoomSegment], b: list[ZoomSegment]
) -> list[ZoomSegment]:
    """Merge two zoom-segment lists, dropping b-entries that overlap a.

    Cursor-driven zooms (a) take precedence — they already have precise
    centroids and a validated duration. LLM-driven speech-emphasis zooms
    (b) fill gaps where the cursor didn't move.

    Overlap rule: if an LLM hint's midpoint falls inside any cursor zoom,
    the hint is dropped (cursor zoom already covers it). Otherwise the
    hint is kept as-is.
    """
    result = list(a)
    for h in b:
        mid = 0.5 * (h.start + h.end)
        overlaps = any(z.start <= mid <= z.end for z in a)
        if not overlaps:
            result.append(h)
    result.sort(key=lambda z: z.start)
    return result


def generate_zoom_segments(
    csv_path: Path,
    screen_w: int,
    screen_h: int,
    duration_s: float,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    max_segment_duration: float = MAX_SEGMENT_DURATION,
) -> list[ZoomSegment]:
    """End-to-end: cursor.csv → list of ZoomSegment.

    Caller provides the recorded display's dimensions (screen_w, screen_h)
    and its origin in macOS global screen space (origin_x, origin_y) —
    needed to normalize pynput's global coordinates onto [0,1].

    `duration_s` is the clip duration (from ffprobe) used to clamp the
    end_limit. We also drop events in the last STOP_PADDING_SECONDS so
    spurious hover-after-recording-stop doesn't emit phantom segments.

    Returns ZoomSegments in CSV timebase. The caller must shift them into
    video timebase using csv_to_video_offset_s from sync.json.
    """
    clicks, moves = parse_cursor_csv(
        csv_path, screen_w, screen_h, origin_x=origin_x, origin_y=origin_y
    )
    end_limit = max(0.0, duration_s - STOP_PADDING_SECONDS)

    groups = group_clicks(clicks, end_limit)
    ci = click_intervals(groups, end_limit)
    mi = movement_intervals(moves, end_limit)

    merged = merge_intervals(ci + mi)
    kept = filter_short(merged, MIN_SEGMENT_DURATION)

    segments: list[ZoomSegment] = []
    for s, e in kept:
        # Cap overly-long zooms to keep the feel reasonable.
        if (e - s) > max_segment_duration:
            e = s + max_segment_duration
        cx, cy = compute_centroid(moves, s, e)
        segments.append(
            ZoomSegment(start=s, end=e, zoom=AUTO_ZOOM_AMOUNT, cx=cx, cy=cy)
        )
    return segments


__all__ = [
    # constants — re-exported for tests and external tuning
    "STOP_PADDING_SECONDS",
    "CLICK_GROUP_TIME_THRESHOLD_SECS",
    "CLICK_GROUP_SPATIAL_THRESHOLD",
    "CLICK_PRE_PADDING",
    "CLICK_POST_PADDING",
    "MOVEMENT_PRE_PADDING",
    "MOVEMENT_POST_PADDING",
    "MERGE_GAP_THRESHOLD",
    "MIN_SEGMENT_DURATION",
    "MOVEMENT_WINDOW_SECONDS",
    "MOVEMENT_EVENT_DISTANCE_THRESHOLD",
    "MOVEMENT_WINDOW_DISTANCE_THRESHOLD",
    "AUTO_ZOOM_AMOUNT",
    "SHAKE_FILTER_THRESHOLD",
    "SHAKE_FILTER_WINDOW_MS",
    "MAX_SEGMENT_DURATION",
    # types
    "CursorEvent",
    "ZoomSegment",
    # api
    "parse_cursor_csv",
    "group_clicks",
    "click_intervals",
    "movement_intervals",
    "merge_intervals",
    "filter_short",
    "compute_centroid",
    "generate_zoom_segments",
    "SPEECH_ZOOM_BY_STRENGTH",
    "zoom_segments_from_hints",
    "merge_zoom_segments",
    "_cursor_position_at",
]
