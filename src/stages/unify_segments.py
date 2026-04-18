"""Stage D — merge all decisions into the canonical segments.json.

Consumes:
  work/layout_plan.json    (backbone — covers entire duration)
  work/filler_cuts.json    (intervals to remove entirely)
  work/dead_zones.json     (intervals to cut OR speed-ramp)
  cursor.csv  (optional)   (raw cursor log — zoom segments generated here)
  work/sync.json (optional) (carries csv_to_video_offset_s)

Emits:
  work/segments.json — the canonical edit list. Every entry has
    {"start", "end", "speed", "layout", "cursor_zooms"} where start / end
    are in ORIGINAL video (OBS) timebase. The render stage trims from
    the raw source at these times and concatenates.

Algorithm (order matters):
  1. Load layout_plan as the backbone (covers [0, source_duration]).
     Merge adjacent same-layout segments (cleans up LLM output).
  2. For every filler_cuts entry, SUBTRACT from the timeline. Splits
     the overlapping segment(s) into two — the gap is not rendered.
  3. For every dead_zone:
       action == "cut"      → subtract like a filler.
       action == "speed@Nx" → split at boundaries, mark inner pieces
                              with speed = N.
  4. If cursor.csv + sync.json provided, generate zoom segments (via
     cursor_zoom.generate_zoom_segments), shift from CSV time to video
     time using csv_to_video_offset_s, then attach to screen-visible
     segments (pip / screen_full) clipped to segment bounds.
  5. Validate: no overlapping segments, every field present, every
     segment's start < end, all segments within [0, source_duration].

Timestamp convention:
  - segment.start / segment.end: absolute original-video seconds
  - cursor_zooms[i].start / .end: absolute original-video seconds, and
    always satisfy segment.start ≤ zoom.start < zoom.end ≤ segment.end
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from src import config
from src.stages.cursor_idle import detect_cursor_idle_intervals
from src.stages.cursor_zoom import (
    CursorEvent,
    ZoomSegment,
    generate_zoom_segments,
    merge_zoom_segments,
    parse_cursor_csv,
    zoom_segments_from_hints,
)
from src.stages.dead_zone_detect import intersect_intervals
from src.stages.verify_cuts import VerifyInput
from src.stages.verify_cuts import run as verify_cuts_run

log = logging.getLogger(__name__)


Layout = Literal["cam_full", "pip", "screen_full"]
SCREEN_VISIBLE_LAYOUTS: frozenset[str] = frozenset({"pip", "screen_full"})

# How closely two floats must agree to be "equal" for timeline ops.
EPS = 1e-6


@dataclass
class DeadZoneCueEntry:
    start: float
    end: float
    reason: str
    confidence: Literal["high", "low"]


@dataclass
class ZoomWindow:
    start: float
    end: float
    zoom: float
    cx: float  # normalized [0,1]
    cy: float


@dataclass
class Segment:
    start: float
    end: float
    speed: float = 1.0
    layout: Layout = "pip"
    cursor_zooms: list[ZoomWindow] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "speed": self.speed,
            "layout": self.layout,
            "cursor_zooms": [asdict(z) for z in self.cursor_zooms],
        }


# ---------- loaders ---------------------------------------------------

def load_layout_plan(path: Path) -> list[Segment]:
    """Load layout_plan.json and pre-merge adjacent same-layout entries.

    Expected shape: {"segments": [{"start", "end", "layout"}, ...]}
    """
    data = json.loads(path.read_text())
    raw = data.get("segments", [])
    if not raw:
        raise ValueError(f"{path} has no segments")
    tl = [
        Segment(start=float(s["start"]), end=float(s["end"]), layout=s["layout"])
        for s in raw
    ]
    tl.sort(key=lambda s: s.start)
    return merge_adjacent(tl)


def load_cuts(path: Path, key: str = "cuts") -> list[tuple[float, float]]:
    """Load filler_cuts.json or similar. Returns list of (start, end)."""
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    entries = data.get(key, [])
    return [(float(e["start"]), float(e["end"])) for e in entries]


def load_dead_zones(path: Path) -> list[tuple[float, float, str]]:
    """Load dead_zones.json. Returns list of (start, end, action)."""
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    zones = data.get("zones", []) or data.get("dead_zones", [])
    return [(float(z["start"]), float(z["end"]), z["action"]) for z in zones]


def load_dead_zone_cues(path: Path) -> list[DeadZoneCueEntry]:
    """Load dead_zone_cues.json preserving the per-cue confidence.

    Missing file or malformed JSON → empty list (no cues). Entries
    without a `confidence` field default to "high" for backward
    compatibility with artifacts written before the verifier shipped.
    """
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return []
    cues: list[DeadZoneCueEntry] = []
    for c in data.get("cues", []) or []:
        if "start" not in c or "end" not in c:
            continue
        conf = c.get("confidence", "high")
        if conf not in ("high", "low"):
            conf = "high"
        cues.append(DeadZoneCueEntry(
            start=float(c["start"]),
            end=float(c["end"]),
            reason=str(c.get("reason", "")),
            confidence=conf,
        ))
    return cues


def load_zoom_hints(path: Path) -> list[dict]:
    """Load zoom_hints.json from analyze stage (speech-emphasis zooms).

    Returns list of dicts — caller passes them straight to
    `zoom_segments_from_hints`. Missing file or malformed entries degrade
    silently to an empty list.
    """
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return []
    hints = data.get("hints", []) or []
    return [h for h in hints if "start" in h and "end" in h]


def load_face_absent(path: Path) -> list[tuple[float, float]]:
    """Load face_absent.json from face_visibility stage.

    Returns list of (start, end) in CAM TIMEBASE. Since cam/ext/audio
    are synced to the same T=0 after Stage A, these are equivalent to
    video timebase as long as the webcam recording is the reference
    (which it is — sync.json.csv_to_video_offset_s is keyed to cam).
    """
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return [(float(a["start"]), float(a["end"])) for a in data.get("absences", [])]


def load_silence_intervals(path: Path) -> list[tuple[float, float]]:
    """Load the raw silencedetect output side-artifact from dead_zone_detect."""
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return [(float(x["start"]), float(x["end"])) for x in data.get("intervals", [])]


def _resolve_dead_zone_cues(
    cues: list[DeadZoneCueEntry],
    *,
    screen_path: Path | None,
    work_dir: Path,
) -> list[tuple[float, float]]:
    """Turn dead-zone cues into (start, end) cuts.

    High-confidence cues pass through. Low-confidence cues are:
      - routed through the frame-based verifier when VERIFY_UNCERTAIN_CUTS
        is set AND a screen recording is available,
      - dropped otherwise (they stay as hints, not cuts) so we don't
        delete content without evidence.
    Verifier outputs with decision == "reject" are also dropped; "accept"
    uses the original range; "trim" uses the narrower range.
    """
    if not cues:
        return []
    high = [c for c in cues if c.confidence == "high"]
    low = [c for c in cues if c.confidence == "low"]
    out: list[tuple[float, float]] = [(c.start, c.end) for c in high]

    if low and config.VERIFY_UNCERTAIN_CUTS and screen_path is not None and screen_path.exists():
        verify_inputs = [
            VerifyInput(
                start=c.start, end=c.end,
                reason=c.reason, kind="dead_zone",
            )
            for c in low
        ]
        results = verify_cuts_run(
            verify_inputs, screen_path=screen_path, work_dir=work_dir,
        )
        for r in results:
            if r.decision in ("accept", "trim"):
                out.append((r.start, r.end))
        print(
            f"[unify] verified {len(low)} low-conf cues: "
            f"{sum(1 for r in results if r.decision == 'accept')} accept, "
            f"{sum(1 for r in results if r.decision == 'trim')} trim, "
            f"{sum(1 for r in results if r.decision == 'reject')} reject"
        )
    return out


def compute_triple_intersection_cuts(
    face_absent: list[tuple[float, float]],
    cursor_idle: list[tuple[float, float]],
    narration_silent: list[tuple[float, float]],
    min_duration: float = 2.0,
) -> list[tuple[float, float]]:
    """All three signals must simultaneously indicate inactivity.

    Uses the sweep-line helper from dead_zone_detect with min_agree=3.
    Returned intervals become hard-cut ranges regardless of length
    (the render step never renders them).
    """
    ranges = intersect_intervals(
        face_absent, cursor_idle, narration_silent,
        min_agree=3,
        min_duration=min_duration,
    )
    return [(s, e) for (s, e, _dets) in ranges]


# ---------- core operations -------------------------------------------

def merge_adjacent(timeline: list[Segment]) -> list[Segment]:
    """Merge consecutive segments that:
      - share a boundary (prev.end ≈ next.start, i.e. contiguous), AND
      - share the same (layout, speed) pair.
    """
    if not timeline:
        return []
    out: list[Segment] = [
        Segment(
            start=timeline[0].start,
            end=timeline[0].end,
            speed=timeline[0].speed,
            layout=timeline[0].layout,
            cursor_zooms=list(timeline[0].cursor_zooms),
        )
    ]
    for seg in timeline[1:]:
        prev = out[-1]
        contiguous = abs(seg.start - prev.end) < EPS
        same_kind = seg.layout == prev.layout and abs(seg.speed - prev.speed) < EPS
        if contiguous and same_kind:
            prev.end = seg.end
            prev.cursor_zooms.extend(seg.cursor_zooms)
        else:
            out.append(Segment(
                start=seg.start, end=seg.end, speed=seg.speed,
                layout=seg.layout, cursor_zooms=list(seg.cursor_zooms),
            ))
    return out


def subtract_range(
    timeline: list[Segment], cut_start: float, cut_end: float
) -> list[Segment]:
    """Remove [cut_start, cut_end) from every overlapping segment.

    A segment fully inside the cut is deleted. A segment straddling either
    edge is shortened. A segment fully containing the cut is split into
    two sub-segments with matching (speed, layout, cursor_zooms cloned).
    """
    if cut_end <= cut_start:
        return timeline
    out: list[Segment] = []
    for seg in timeline:
        # No overlap
        if seg.end <= cut_start + EPS or seg.start >= cut_end - EPS:
            out.append(seg)
            continue
        # Fully swallowed by cut
        if cut_start - EPS <= seg.start and seg.end <= cut_end + EPS:
            continue
        # Left remainder
        if seg.start < cut_start - EPS:
            out.append(Segment(
                start=seg.start, end=cut_start, speed=seg.speed,
                layout=seg.layout,
                cursor_zooms=_clip_zooms(seg.cursor_zooms, seg.start, cut_start),
            ))
        # Right remainder
        if seg.end > cut_end + EPS:
            out.append(Segment(
                start=cut_end, end=seg.end, speed=seg.speed,
                layout=seg.layout,
                cursor_zooms=_clip_zooms(seg.cursor_zooms, cut_end, seg.end),
            ))
    return out


def apply_cuts(
    timeline: list[Segment], cuts: list[tuple[float, float]]
) -> list[Segment]:
    """Apply a list of (start, end) cuts in order."""
    tl = list(timeline)
    for s, e in cuts:
        tl = subtract_range(tl, s, e)
    return tl


def _parse_speed_action(action: str) -> float | None:
    """'speed@4x' → 4.0; 'speed@8x' → 8.0; 'cut' → None; other → None."""
    if not action.startswith("speed@"):
        return None
    tail = action[len("speed@"):]
    if tail.endswith("x"):
        tail = tail[:-1]
    try:
        return float(tail)
    except ValueError:
        return None


def apply_dead_zones(
    timeline: list[Segment], dead_zones: list[tuple[float, float, str]]
) -> list[Segment]:
    """For each dead zone: if action=='cut', subtract. Else speed-ramp."""
    tl = list(timeline)
    for start, end, action in dead_zones:
        if action == "cut":
            tl = subtract_range(tl, start, end)
        else:
            speed = _parse_speed_action(action)
            if speed is None or speed <= 0.0:
                log.warning("unknown dead_zone action %r, skipping", action)
                continue
            tl = _apply_speed_ramp(tl, start, end, speed)
    return tl


def _apply_speed_ramp(
    timeline: list[Segment], start: float, end: float, speed: float
) -> list[Segment]:
    """Split segments at [start, end); tag the interior pieces with the
    given speed. Outside pieces keep their existing speed.
    """
    if end <= start:
        return timeline
    out: list[Segment] = []
    for seg in timeline:
        if seg.end <= start + EPS or seg.start >= end - EPS:
            out.append(seg)
            continue
        # Left remainder
        if seg.start < start - EPS:
            out.append(Segment(
                start=seg.start, end=start, speed=seg.speed, layout=seg.layout,
                cursor_zooms=_clip_zooms(seg.cursor_zooms, seg.start, start),
            ))
        # Interior with new speed (intersect [start,end) and seg)
        i_start = max(seg.start, start)
        i_end = min(seg.end, end)
        if i_end > i_start + EPS:
            out.append(Segment(
                start=i_start, end=i_end, speed=speed, layout=seg.layout,
                cursor_zooms=_clip_zooms(seg.cursor_zooms, i_start, i_end),
            ))
        # Right remainder
        if seg.end > end + EPS:
            out.append(Segment(
                start=end, end=seg.end, speed=seg.speed, layout=seg.layout,
                cursor_zooms=_clip_zooms(seg.cursor_zooms, end, seg.end),
            ))
    return out


def _clip_zooms(
    zooms: list[ZoomWindow], lo: float, hi: float
) -> list[ZoomWindow]:
    """Keep only zooms overlapping [lo, hi); clip their bounds."""
    out: list[ZoomWindow] = []
    for z in zooms:
        if z.end <= lo + EPS or z.start >= hi - EPS:
            continue
        out.append(ZoomWindow(
            start=max(z.start, lo),
            end=min(z.end, hi),
            zoom=z.zoom, cx=z.cx, cy=z.cy,
        ))
    return out


def annotate_cursor_zooms(
    timeline: list[Segment], zoom_segments: list[ZoomSegment], csv_to_video_offset_s: float
) -> list[Segment]:
    """Attach zoom windows to screen-visible segments.

    Zoom segments come from cursor_zoom.py in CSV timebase; shift them to
    video time via csv_to_video_offset_s. Clip to each segment's bounds.
    Skip segments whose layout does not show the screen (cam_full).
    """
    # Shift once into video timebase
    shifted = [
        ZoomWindow(
            start=z.start + csv_to_video_offset_s,
            end=z.end + csv_to_video_offset_s,
            zoom=z.zoom, cx=z.cx, cy=z.cy,
        )
        for z in zoom_segments
    ]
    out: list[Segment] = []
    for seg in timeline:
        if seg.layout not in SCREEN_VISIBLE_LAYOUTS:
            # Screen not visible — no zoom makes sense.
            out.append(Segment(
                start=seg.start, end=seg.end, speed=seg.speed,
                layout=seg.layout, cursor_zooms=[],
            ))
            continue
        attached = _clip_zooms(shifted, seg.start, seg.end)
        out.append(Segment(
            start=seg.start, end=seg.end, speed=seg.speed,
            layout=seg.layout, cursor_zooms=attached,
        ))
    return out


# ---------- validation ------------------------------------------------

def validate(
    timeline: list[Segment], source_duration: float, tol: float = 1e-3
) -> None:
    """Raise ValueError with a clear diagnostic if the timeline is malformed.

    Checks:
      - every segment: start < end
      - every segment within [0, source_duration]
      - no overlaps between adjacent sorted segments
      - every field has a valid value
      - cursor_zooms inside their segment's bounds
    """
    if not timeline:
        raise ValueError("empty timeline")
    srt = sorted(timeline, key=lambda s: s.start)
    prev_end = -float("inf")
    for i, seg in enumerate(srt):
        if seg.end <= seg.start:
            raise ValueError(f"segment #{i} has end ≤ start: {seg}")
        if seg.start < -tol:
            raise ValueError(f"segment #{i} starts before 0: {seg}")
        if seg.end > source_duration + tol:
            raise ValueError(
                f"segment #{i} ends after source_duration "
                f"({seg.end} > {source_duration}): {seg}"
            )
        if seg.speed <= 0.0:
            raise ValueError(f"segment #{i} has non-positive speed: {seg}")
        if seg.layout not in ("cam_full", "pip", "screen_full"):
            raise ValueError(f"segment #{i} has unknown layout: {seg}")
        if seg.start + tol < prev_end:
            raise ValueError(
                f"segments overlap: #{i-1} ends {prev_end}, "
                f"#{i} starts {seg.start}"
            )
        prev_end = seg.end
        for j, z in enumerate(seg.cursor_zooms):
            if not (seg.start - tol <= z.start < z.end <= seg.end + tol):
                raise ValueError(
                    f"segment #{i} zoom #{j} outside segment bounds: "
                    f"seg=[{seg.start},{seg.end}], zoom=[{z.start},{z.end}]"
                )


# ---------- CLI entry -------------------------------------------------

def run(args: argparse.Namespace) -> int:
    work_dir: Path = Path(args.work) if getattr(args, "work", None) else config.WORK_DIR
    work_dir.mkdir(parents=True, exist_ok=True)

    layout_path = work_dir / "layout_plan.json"
    filler_path = work_dir / "filler_cuts.json"
    dead_path = work_dir / "dead_zones.json"
    dead_cues_path = work_dir / "dead_zone_cues.json"
    face_path = work_dir / "face_absent.json"
    silence_path = work_dir / "silence_intervals.json"
    sync_path = work_dir / "sync.json"
    segments_path = work_dir / "segments.json"

    try:
        timeline = load_layout_plan(layout_path)
        source_duration = timeline[-1].end

        filler_cuts = load_cuts(filler_path, key="cuts")
        cue_cuts = _resolve_dead_zone_cues(
            load_dead_zone_cues(dead_cues_path),
            screen_path=getattr(args, "screen", None),
            work_dir=work_dir,
        )
        timeline = apply_cuts(timeline, filler_cuts + cue_cuts)
        timeline = merge_adjacent(timeline)

        # Triple-intersection hard-cut: face absent ∩ cursor idle ∩ narration silent.
        # All three sources optional — the intersection is meaningful only when
        # all three are present. With any missing, falls through to just the
        # existing dead-zone pipeline.
        face_absent = load_face_absent(face_path)
        silent = load_silence_intervals(silence_path)
        cursor_csv: Path | None = getattr(args, "cursor", None)
        cursor_idle: list[tuple[float, float]] = []
        if cursor_csv is not None and cursor_csv.exists():
            idle_intervals = detect_cursor_idle_intervals(
                cursor_csv,
                duration_s=source_duration,
                min_idle_sec=config.CURSOR_IDLE_MIN_SEC,
            )
            # Shift cursor CSV timebase into video timebase if sync offset is known.
            offset = 0.0
            if sync_path.exists():
                sync_data = json.loads(sync_path.read_text())
                offset = float(sync_data.get("csv_to_video_offset_s") or 0.0)
            cursor_idle = [
                (iv.start + offset, iv.end + offset) for iv in idle_intervals
            ]
        if face_absent and cursor_idle and silent:
            triple_cuts = compute_triple_intersection_cuts(
                face_absent, cursor_idle, silent,
                min_duration=max(
                    config.FACE_ABSENT_MIN_SEC,
                    config.CURSOR_IDLE_MIN_SEC,
                    config.SILENCE_MIN_SEC,
                ),
            )
            if triple_cuts:
                print(
                    f"[unify] triple-intersection hard-cuts: {len(triple_cuts)} "
                    f"(face-absent ∩ cursor-idle ∩ narration-silent)"
                )
                timeline = apply_cuts(timeline, triple_cuts)
                timeline = merge_adjacent(timeline)

        dead_zones = load_dead_zones(dead_path)
        timeline = apply_dead_zones(timeline, dead_zones)
        timeline = merge_adjacent(timeline)

        # Optional cursor zoom + speech-emphasis zoom hints
        zoom_hints_path = work_dir / "zoom_hints.json"
        zoom_hints = load_zoom_hints(zoom_hints_path)

        if cursor_csv is not None and cursor_csv.exists() and sync_path.exists():
            sync_data = json.loads(sync_path.read_text())
            offset = sync_data.get("csv_to_video_offset_s")
            if offset is None:
                log.warning(
                    "sync.json has no csv_to_video_offset_s; "
                    "skipping cursor_zoom annotations"
                )
            else:
                # Screen dimensions + origin come from config or flags.
                screen_w = int(getattr(args, "screen_w", 2560))
                screen_h = int(getattr(args, "screen_h", 1440))
                origin_x = float(getattr(args, "origin_x", 0.0))
                origin_y = float(getattr(args, "origin_y", 0.0))
                cursor_zooms = generate_zoom_segments(
                    cursor_csv, screen_w=screen_w, screen_h=screen_h,
                    duration_s=source_duration,
                    origin_x=origin_x, origin_y=origin_y,
                )
                # Speech-emphasis zooms: shift cursor moves into video
                # time so the hint-centroid lookup works in the right
                # frame. Hints themselves are already in video time.
                _clicks, moves = parse_cursor_csv(
                    cursor_csv, screen_w=screen_w, screen_h=screen_h,
                    origin_x=origin_x, origin_y=origin_y,
                )
                moves_video = [
                    CursorEvent(t_s=m.t_s + offset, x=m.x, y=m.y, is_click=False)
                    for m in moves
                ]
                speech_zooms = zoom_segments_from_hints(
                    zoom_hints, moves_video, source_duration,
                )
                # Cursor zooms are in CSV timebase; shift them into
                # video time before merging.
                cursor_zooms_video = [
                    ZoomSegment(
                        start=z.start + offset, end=z.end + offset,
                        zoom=z.zoom, cx=z.cx, cy=z.cy,
                    )
                    for z in cursor_zooms
                ]
                merged_zooms = merge_zoom_segments(cursor_zooms_video, speech_zooms)

                screen_path: Path | None = getattr(args, "screen", None)

                # Optional scroll / window-change zoom: runs during
                # cursor-idle windows against the screen track.
                if (
                    config.USE_SCROLL_ZOOM
                    and screen_path is not None
                    and screen_path.exists()
                    and cursor_idle  # already shifted into video time above
                ):
                    from src.stages.scroll_zoom import detect_scroll_zooms
                    scroll_zooms = detect_scroll_zooms(
                        screen_path,
                        cursor_idle,
                        sample_rate_hz=config.SCROLL_ZOOM_SAMPLE_RATE_HZ,
                        diff_threshold=config.SCROLL_ZOOM_DIFF_THRESHOLD,
                    )
                    if scroll_zooms:
                        print(
                            f"[unify] scroll_zoom found {len(scroll_zooms)} "
                            f"content-change zooms"
                        )
                        merged_zooms = merge_zoom_segments(merged_zooms, scroll_zooms)

                # Optional element-aware snap (paddleocr). Gated by env
                # var so the expensive OCR pass is opt-in.
                if (
                    config.USE_ELEMENT_AWARE_ZOOM
                    and screen_path is not None
                    and screen_path.exists()
                ):
                    from src.stages.element_aware import snap_zoom_segments
                    merged_zooms = snap_zoom_segments(
                        merged_zooms, screen_path,
                        frame_w=config.VIDEO_RES_W,
                        frame_h=config.VIDEO_RES_H,
                        max_distance_px=config.ELEMENT_SNAP_MAX_PX,
                    )
                # Already in video timebase — pass offset=0.0 to annotate.
                timeline = annotate_cursor_zooms(timeline, merged_zooms, 0.0)
        elif zoom_hints:
            # No cursor.csv but we still have speech-emphasis hints —
            # centroids default to screen center.
            speech_zooms = zoom_segments_from_hints(zoom_hints, [], source_duration)
            timeline = annotate_cursor_zooms(timeline, speech_zooms, 0.0)

        validate(timeline, source_duration)

        doc = {
            "source_duration_s": source_duration,
            "segments": [s.to_dict() for s in timeline],
        }
        segments_path.write_text(json.dumps(doc, indent=2))

        total = sum(s.duration for s in timeline)
        cut_total = source_duration - total
        print(
            f"[unify] {len(timeline)} segments; "
            f"final duration {total:.2f}s "
            f"(source {source_duration:.2f}s, cut {cut_total:.2f}s)"
        )
        print(f"[unify] wrote {segments_path}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[unify] ERROR: {exc}", file=sys.stderr)
        return 1


__all__ = [
    "Layout",
    "SCREEN_VISIBLE_LAYOUTS",
    "DeadZoneCueEntry",
    "ZoomWindow",
    "Segment",
    "load_layout_plan",
    "load_cuts",
    "load_dead_zone_cues",
    "load_dead_zones",
    "load_face_absent",
    "load_silence_intervals",
    "load_zoom_hints",
    "compute_triple_intersection_cuts",
    "merge_adjacent",
    "subtract_range",
    "apply_cuts",
    "apply_dead_zones",
    "annotate_cursor_zooms",
    "validate",
    "run",
    "verify_cuts_run",
]
