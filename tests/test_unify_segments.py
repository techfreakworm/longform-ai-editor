"""Tests for Stage D — unify_segments (the decision brain).

Per IMPLEMENTATION_PLAN.md §M4, this module has the heaviest unit-test
coverage in the project because it's where the most subtle timeline
logic lives.

Organized by the four primitives + the orchestrator:
  - merge_adjacent
  - subtract_range
  - apply_dead_zones (and its speed-ramp helper)
  - annotate_cursor_zooms
  - validate
  - run() end-to-end with golden JSON output
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from src.stages.cursor_zoom import ZoomSegment
from src.stages.unify_segments import (
    EPS,
    SCREEN_VISIBLE_LAYOUTS,
    DeadZoneCueEntry,
    Segment,
    ZoomWindow,
    annotate_cursor_zooms,
    apply_cuts,
    apply_dead_zones,
    compute_triple_intersection_cuts,
    load_cuts,
    load_dead_zone_cues,
    load_dead_zones,
    load_face_absent,
    load_layout_plan,
    load_silence_intervals,
    merge_adjacent,
    run,
    subtract_range,
    validate,
)
from src.stages.verify_cuts import VerifyOutput


# --- helpers -----------------------------------------------------------

def S(start: float, end: float, layout: str = "pip", speed: float = 1.0,
      zooms: list[ZoomWindow] | None = None) -> Segment:
    """Shorthand Segment constructor."""
    return Segment(start=start, end=end, layout=layout, speed=speed,
                   cursor_zooms=list(zooms or []))


def Z(start: float, end: float, cx: float = 0.5, cy: float = 0.5,
      zoom: float = 1.5) -> ZoomWindow:
    return ZoomWindow(start=start, end=end, cx=cx, cy=cy, zoom=zoom)


# --- merge_adjacent ----------------------------------------------------

def test_merge_adjacent_joins_contiguous_same_kind() -> None:
    tl = [S(0, 10, "pip"), S(10, 20, "pip")]
    out = merge_adjacent(tl)
    assert len(out) == 1
    assert (out[0].start, out[0].end) == (0, 20)


def test_merge_adjacent_keeps_different_layouts() -> None:
    tl = [S(0, 10, "cam_full"), S(10, 20, "pip")]
    out = merge_adjacent(tl)
    assert len(out) == 2


def test_merge_adjacent_keeps_different_speeds() -> None:
    tl = [S(0, 10, "pip", speed=1.0), S(10, 20, "pip", speed=4.0)]
    out = merge_adjacent(tl)
    assert len(out) == 2


def test_merge_adjacent_does_not_merge_across_gap() -> None:
    tl = [S(0, 10, "pip"), S(15, 20, "pip")]  # gap 10→15
    out = merge_adjacent(tl)
    assert len(out) == 2


def test_merge_adjacent_empty() -> None:
    assert merge_adjacent([]) == []


def test_merge_adjacent_preserves_zooms_from_both() -> None:
    z1, z2 = Z(1, 2), Z(11, 12)
    tl = [S(0, 10, "pip", zooms=[z1]), S(10, 20, "pip", zooms=[z2])]
    out = merge_adjacent(tl)
    assert len(out) == 1
    assert len(out[0].cursor_zooms) == 2


# --- subtract_range ----------------------------------------------------

def test_subtract_range_no_overlap_passthrough() -> None:
    tl = [S(0, 10, "pip")]
    assert subtract_range(tl, 20, 30) == tl  # cut is entirely after


def test_subtract_range_cut_fully_contained_splits() -> None:
    tl = [S(0, 10, "pip")]
    out = subtract_range(tl, 3, 7)
    assert len(out) == 2
    assert (out[0].start, out[0].end) == (0, 3)
    assert (out[1].start, out[1].end) == (7, 10)


def test_subtract_range_cut_covers_segment_removes_it() -> None:
    tl = [S(5, 10, "pip")]
    assert subtract_range(tl, 0, 20) == []


def test_subtract_range_cut_on_left_edge() -> None:
    tl = [S(0, 10, "pip")]
    out = subtract_range(tl, 0, 3)
    assert len(out) == 1
    assert (out[0].start, out[0].end) == (3, 10)


def test_subtract_range_cut_on_right_edge() -> None:
    tl = [S(0, 10, "pip")]
    out = subtract_range(tl, 7, 10)
    assert len(out) == 1
    assert (out[0].start, out[0].end) == (0, 7)


def test_subtract_range_spans_two_segments() -> None:
    tl = [S(0, 10, "pip"), S(10, 20, "cam_full")]
    out = subtract_range(tl, 5, 15)
    # left seg → (0, 5); right seg → (15, 20); middle fully removed
    assert len(out) == 2
    assert (out[0].start, out[0].end, out[0].layout) == (0, 5, "pip")
    assert (out[1].start, out[1].end, out[1].layout) == (15, 20, "cam_full")


def test_subtract_range_clips_cursor_zooms_in_surviving_pieces() -> None:
    # Zoom from 2-8; cut 4-6 should leave zooms clipped in both halves.
    tl = [S(0, 10, "pip", zooms=[Z(2, 8)])]
    out = subtract_range(tl, 4, 6)
    # left (0,4) — zoom 2-4
    assert out[0].cursor_zooms[0].start == 2 and out[0].cursor_zooms[0].end == 4
    # right (6,10) — zoom 6-8
    assert out[1].cursor_zooms[0].start == 6 and out[1].cursor_zooms[0].end == 8


def test_subtract_range_degenerate_inputs_noop() -> None:
    tl = [S(0, 10, "pip")]
    # zero-length cut
    assert subtract_range(tl, 5, 5) == tl
    # reversed cut
    assert subtract_range(tl, 7, 3) == tl


# --- apply_cuts --------------------------------------------------------

def test_apply_cuts_composes_multiple() -> None:
    tl = [S(0, 20, "pip")]
    out = apply_cuts(tl, [(3, 5), (10, 12)])
    assert [(s.start, s.end) for s in out] == [(0, 3), (5, 10), (12, 20)]


# --- apply_dead_zones --------------------------------------------------

def test_dead_zone_cut_action_removes() -> None:
    tl = [S(0, 20, "pip")]
    out = apply_dead_zones(tl, [(5, 10, "cut")])
    assert [(s.start, s.end) for s in out] == [(0, 5), (10, 20)]


def test_dead_zone_speed_ramp_sets_speed_interior() -> None:
    tl = [S(0, 20, "pip")]
    out = apply_dead_zones(tl, [(5, 10, "speed@4x")])
    assert len(out) == 3
    assert out[0].speed == 1.0 and (out[0].start, out[0].end) == (0, 5)
    assert out[1].speed == 4.0 and (out[1].start, out[1].end) == (5, 10)
    assert out[2].speed == 1.0 and (out[2].start, out[2].end) == (10, 20)


def test_dead_zone_speed_8x() -> None:
    tl = [S(0, 20, "pip")]
    out = apply_dead_zones(tl, [(5, 15, "speed@8x")])
    # middle segment gets 8x
    middle = [s for s in out if s.speed == 8.0]
    assert len(middle) == 1
    assert (middle[0].start, middle[0].end) == (5, 15)


def test_dead_zone_speed_ramp_preserves_layout() -> None:
    tl = [S(0, 20, "screen_full")]
    out = apply_dead_zones(tl, [(5, 15, "speed@4x")])
    assert all(s.layout == "screen_full" for s in out)


def test_dead_zone_unknown_action_warns_and_skips() -> None:
    tl = [S(0, 20, "pip")]
    # Bogus action string should NOT corrupt the timeline
    out = apply_dead_zones(tl, [(5, 10, "do_something_weird")])
    assert out == tl


def test_dead_zone_speed_ramp_at_segment_boundary() -> None:
    """Ramp exactly aligned with segment boundaries should produce a clean split."""
    tl = [S(0, 10, "pip"), S(10, 20, "pip")]
    out = apply_dead_zones(tl, [(10, 20, "speed@4x")])
    # Should get: pip@1x 0-10, pip@4x 10-20
    assert len(out) == 2
    assert out[0].speed == 1.0 and (out[0].start, out[0].end) == (0, 10)
    assert out[1].speed == 4.0 and (out[1].start, out[1].end) == (10, 20)


# --- annotate_cursor_zooms --------------------------------------------

def test_cursor_zoom_skips_cam_full() -> None:
    tl = [S(0, 10, "cam_full")]
    zooms = [ZoomSegment(start=1, end=3, zoom=1.5, cx=0.5, cy=0.5)]
    out = annotate_cursor_zooms(tl, zooms, csv_to_video_offset_s=0.0)
    assert out[0].cursor_zooms == []


def test_cursor_zoom_attaches_to_pip() -> None:
    tl = [S(0, 10, "pip")]
    zooms = [ZoomSegment(start=1, end=3, zoom=1.5, cx=0.4, cy=0.6)]
    out = annotate_cursor_zooms(tl, zooms, csv_to_video_offset_s=0.0)
    assert len(out[0].cursor_zooms) == 1
    z = out[0].cursor_zooms[0]
    assert (z.start, z.end, z.cx, z.cy) == (1, 3, 0.4, 0.6)


def test_cursor_zoom_attaches_to_screen_full() -> None:
    tl = [S(0, 10, "screen_full")]
    zooms = [ZoomSegment(start=1, end=3, zoom=1.5, cx=0.5, cy=0.5)]
    out = annotate_cursor_zooms(tl, zooms, csv_to_video_offset_s=0.0)
    assert len(out[0].cursor_zooms) == 1


def test_cursor_zoom_shifts_by_csv_offset() -> None:
    # CSV zoom at 2-4; offset +10 → video zoom at 12-14
    tl = [S(0, 20, "pip")]
    zooms = [ZoomSegment(start=2, end=4, zoom=1.5, cx=0.5, cy=0.5)]
    out = annotate_cursor_zooms(tl, zooms, csv_to_video_offset_s=10.0)
    assert out[0].cursor_zooms[0].start == 12 and out[0].cursor_zooms[0].end == 14


def test_cursor_zoom_clipped_to_segment_bounds() -> None:
    # Zoom 0-20, segment 5-15 → zoom should clip to 5-15
    tl = [S(5, 15, "pip")]
    zooms = [ZoomSegment(start=0, end=20, zoom=1.5, cx=0.5, cy=0.5)]
    out = annotate_cursor_zooms(tl, zooms, csv_to_video_offset_s=0.0)
    assert out[0].cursor_zooms[0].start == 5
    assert out[0].cursor_zooms[0].end == 15


def test_cursor_zoom_dropped_if_outside_segment() -> None:
    tl = [S(0, 10, "pip")]
    zooms = [ZoomSegment(start=20, end=25, zoom=1.5, cx=0.5, cy=0.5)]
    out = annotate_cursor_zooms(tl, zooms, csv_to_video_offset_s=0.0)
    assert out[0].cursor_zooms == []


def test_screen_visible_layouts_set() -> None:
    assert SCREEN_VISIBLE_LAYOUTS == frozenset({"pip", "screen_full"})


# --- validate ---------------------------------------------------------

def test_validate_ok_on_clean_timeline() -> None:
    tl = [S(0, 10, "pip"), S(10, 20, "cam_full")]
    validate(tl, source_duration=20.0)


def test_validate_rejects_overlap() -> None:
    tl = [S(0, 10, "pip"), S(5, 15, "pip")]
    with pytest.raises(ValueError, match="overlap"):
        validate(tl, source_duration=20.0)


def test_validate_rejects_reversed_segment() -> None:
    tl = [Segment(start=10, end=5, layout="pip")]
    with pytest.raises(ValueError, match="end"):
        validate(tl, source_duration=20.0)


def test_validate_rejects_out_of_range() -> None:
    tl = [S(0, 25, "pip")]
    with pytest.raises(ValueError, match="source_duration"):
        validate(tl, source_duration=20.0)


def test_validate_rejects_unknown_layout() -> None:
    tl = [Segment(start=0, end=10, layout="nonsense")]
    with pytest.raises(ValueError, match="layout"):
        validate(tl, source_duration=20.0)


def test_validate_rejects_zero_speed() -> None:
    tl = [S(0, 10, "pip", speed=0.0)]
    with pytest.raises(ValueError, match="speed"):
        validate(tl, source_duration=20.0)


def test_validate_rejects_zoom_outside_segment() -> None:
    tl = [S(0, 10, "pip", zooms=[Z(15, 20)])]
    with pytest.raises(ValueError, match="outside segment"):
        validate(tl, source_duration=20.0)


def test_validate_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        validate([], source_duration=20.0)


# --- load_* helpers ---------------------------------------------------

def test_load_layout_plan_merges_adjacent(tmp_path: Path) -> None:
    (tmp_path / "layout_plan.json").write_text(json.dumps({
        "segments": [
            {"start": 0.0, "end": 5.0, "layout": "pip"},
            {"start": 5.0, "end": 10.0, "layout": "pip"},
            {"start": 10.0, "end": 15.0, "layout": "cam_full"},
        ],
    }))
    out = load_layout_plan(tmp_path / "layout_plan.json")
    assert len(out) == 2
    assert (out[0].start, out[0].end, out[0].layout) == (0, 10, "pip")


def test_load_cuts_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_cuts(tmp_path / "missing.json") == []


def test_load_cuts_parses(tmp_path: Path) -> None:
    (tmp_path / "filler.json").write_text(
        json.dumps({"cuts": [{"start": 1.1, "end": 2.2, "reason": "um"}]})
    )
    assert load_cuts(tmp_path / "filler.json") == [(1.1, 2.2)]


def test_load_dead_zones_accepts_either_key(tmp_path: Path) -> None:
    (tmp_path / "d1.json").write_text(
        json.dumps({"zones": [{"start": 1, "end": 2, "action": "cut"}]})
    )
    (tmp_path / "d2.json").write_text(
        json.dumps({"dead_zones": [{"start": 3, "end": 4, "action": "speed@4x"}]})
    )
    assert load_dead_zones(tmp_path / "d1.json") == [(1.0, 2.0, "cut")]
    assert load_dead_zones(tmp_path / "d2.json") == [(3.0, 4.0, "speed@4x")]


# --- end-to-end run() -------------------------------------------------

def _write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2))


def test_run_golden_filler_only(tmp_path: Path) -> None:
    """Typical scenario: LLM identifies fillers, no dead zones or cursor."""
    _write(tmp_path / "layout_plan.json", {
        "segments": [
            {"start": 0.0, "end": 30.0, "layout": "cam_full"},
            {"start": 30.0, "end": 120.0, "layout": "pip"},
            {"start": 120.0, "end": 150.0, "layout": "cam_full"},
        ],
    })
    _write(tmp_path / "filler_cuts.json", {
        "cuts": [
            {"start": 15.0, "end": 15.5, "reason": "um"},
            {"start": 50.0, "end": 50.3, "reason": "uh"},
        ],
    })
    args = argparse.Namespace(
        work=tmp_path, verbose=0, cursor=None,
        screen_w=2560, screen_h=1440, origin_x=0.0, origin_y=0.0,
    )
    rc = run(args)
    assert rc == 0

    doc = json.loads((tmp_path / "segments.json").read_text())
    assert doc["source_duration_s"] == 150.0
    segs = doc["segments"]
    # cam_full splits into (0,15) and (15.5,30); pip splits into (30,50) and (50.3,120);
    # cam_full (120,150). Total 5 segments.
    assert len(segs) == 5
    assert segs[0]["layout"] == "cam_full" and segs[0]["start"] == 0.0 and segs[0]["end"] == 15.0
    assert segs[1]["layout"] == "cam_full" and segs[1]["start"] == 15.5 and segs[1]["end"] == 30.0
    assert segs[2]["layout"] == "pip" and segs[2]["start"] == 30.0 and segs[2]["end"] == 50.0
    assert segs[3]["layout"] == "pip" and segs[3]["start"] == 50.3 and segs[3]["end"] == 120.0
    assert segs[4]["layout"] == "cam_full" and segs[4]["start"] == 120.0

    # All speeds default 1.0, no cursor zooms.
    for s in segs:
        assert s["speed"] == 1.0
        assert s["cursor_zooms"] == []


def test_run_golden_with_dead_zone_speed_ramp(tmp_path: Path) -> None:
    _write(tmp_path / "layout_plan.json", {
        "segments": [{"start": 0, "end": 60, "layout": "pip"}],
    })
    _write(tmp_path / "filler_cuts.json", {"cuts": []})
    _write(tmp_path / "dead_zones.json", {
        "zones": [{"start": 20, "end": 30, "action": "speed@4x"}],
    })

    args = argparse.Namespace(
        work=tmp_path, verbose=0, cursor=None,
        screen_w=2560, screen_h=1440, origin_x=0.0, origin_y=0.0,
    )
    rc = run(args)
    assert rc == 0
    segs = json.loads((tmp_path / "segments.json").read_text())["segments"]
    # (0,20) 1x, (20,30) 4x, (30,60) 1x
    assert len(segs) == 3
    assert segs[0]["speed"] == 1.0
    assert segs[1]["speed"] == 4.0
    assert segs[2]["speed"] == 1.0


def test_run_handles_missing_filler_and_deadzone(tmp_path: Path) -> None:
    """layout_plan.json is required; the other two inputs are optional."""
    _write(tmp_path / "layout_plan.json", {
        "segments": [{"start": 0, "end": 30, "layout": "pip"}],
    })
    args = argparse.Namespace(
        work=tmp_path, verbose=0, cursor=None,
        screen_w=2560, screen_h=1440, origin_x=0.0, origin_y=0.0,
    )
    rc = run(args)
    assert rc == 0
    segs = json.loads((tmp_path / "segments.json").read_text())["segments"]
    assert len(segs) == 1
    assert segs[0]["start"] == 0 and segs[0]["end"] == 30


def test_run_missing_layout_plan_fails(tmp_path: Path) -> None:
    args = argparse.Namespace(
        work=tmp_path, verbose=0, cursor=None,
        screen_w=2560, screen_h=1440, origin_x=0.0, origin_y=0.0,
    )
    rc = run(args)
    assert rc == 1  # loader raises, caught by run()'s except


def test_run_with_cursor_zoom(tmp_path: Path) -> None:
    """cursor.csv + sync.json → cursor_zooms attached to screen-visible segments."""
    _write(tmp_path / "layout_plan.json", {
        "segments": [
            {"start": 0, "end": 30, "layout": "cam_full"},
            {"start": 30, "end": 90, "layout": "pip"},
        ],
    })
    # sync offset is 10 — CSV t=X corresponds to video t=X+10.
    _write(tmp_path / "sync.json", {"csv_to_video_offset_s": 10.0})
    # cursor.csv with a click around CSV t=30 → video t=40 (falls inside pip segment).
    csv_path = tmp_path / "cursor.csv"
    csv_path.write_text(
        "t_s,x,y,event,button,down\n"
        # Some move events so centroid has data
        "28.0,1200,800,move,,\n"
        "29.0,1220,810,move,,\n"
        "30.0,1200,800,click,left,1\n"
        "30.1,1200,800,click,left,0\n"
        "30.5,1220,820,click,left,1\n"
        "30.6,1220,820,click,left,0\n"
        "31.0,1210,810,move,,\n"
    )

    args = argparse.Namespace(
        work=tmp_path, verbose=0, cursor=csv_path,
        screen_w=2560, screen_h=1440, origin_x=0.0, origin_y=0.0,
    )
    rc = run(args)
    assert rc == 0
    doc = json.loads((tmp_path / "segments.json").read_text())
    segs = doc["segments"]
    # cam_full gets no zooms; pip should have at least one.
    cam = next(s for s in segs if s["layout"] == "cam_full")
    pip = next(s for s in segs if s["layout"] == "pip")
    assert cam["cursor_zooms"] == []
    # The click burst at CSV t=30 → video t≈40. Should produce a zoom segment
    # inside pip (30–90).
    assert len(pip["cursor_zooms"]) >= 1
    for z in pip["cursor_zooms"]:
        assert 30 <= z["start"] < z["end"] <= 90


def test_run_produces_valid_output(tmp_path: Path) -> None:
    """Regardless of inputs, run() must produce a timeline that passes
    validate(). Combination of filler + dead-zone + layouts."""
    _write(tmp_path / "layout_plan.json", {
        "segments": [
            {"start": 0, "end": 20, "layout": "cam_full"},
            {"start": 20, "end": 100, "layout": "pip"},
            {"start": 100, "end": 130, "layout": "screen_full"},
            {"start": 130, "end": 150, "layout": "cam_full"},
        ],
    })
    _write(tmp_path / "filler_cuts.json", {
        "cuts": [
            {"start": 10, "end": 10.5, "reason": "um"},
            {"start": 25, "end": 25.4, "reason": "false_start"},
        ],
    })
    _write(tmp_path / "dead_zones.json", {
        "zones": [
            {"start": 40, "end": 55, "action": "speed@8x"},
            {"start": 105, "end": 110, "action": "cut"},
        ],
    })
    args = argparse.Namespace(
        work=tmp_path, verbose=0, cursor=None,
        screen_w=2560, screen_h=1440, origin_x=0.0, origin_y=0.0,
    )
    rc = run(args)
    assert rc == 0
    doc = json.loads((tmp_path / "segments.json").read_text())
    # Reconstruct Segment objects and re-validate via the library function.
    segs = [
        Segment(
            start=s["start"], end=s["end"], speed=s["speed"], layout=s["layout"],
            cursor_zooms=[ZoomWindow(**z) for z in s["cursor_zooms"]],
        )
        for s in doc["segments"]
    ]
    # Source duration from output
    validate(segs, source_duration=doc["source_duration_s"])

    # Spot-check: the speed@8x segment must be present
    ramp_segs = [s for s in segs if abs(s.speed - 8.0) < EPS]
    assert len(ramp_segs) == 1
    assert (ramp_segs[0].start, ramp_segs[0].end) == (40, 55)
    # The cut at (105, 110) must leave a gap in the screen_full range
    sf_ranges = [(s.start, s.end) for s in segs if s.layout == "screen_full"]
    assert sf_ranges == [(100, 105), (110, 130)]


# --- load_dead_zone_cues ----------------------------------------------

def test_load_dead_zone_cues_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_dead_zone_cues(tmp_path / "nope.json") == []


def test_load_dead_zone_cues_preserves_confidence(tmp_path: Path) -> None:
    path = tmp_path / "dead_zone_cues.json"
    _write(path, {"cues": [
        {"start": 10.0, "end": 120.0, "reason": "installs",
         "confidence": "low"},
        {"start": 200.0, "end": 205.0, "reason": "skip",
         "confidence": "high"},
    ]})
    cues = load_dead_zone_cues(path)
    assert len(cues) == 2
    assert cues[0].confidence == "low"
    assert cues[0].reason == "installs"
    assert cues[1].confidence == "high"


def test_load_dead_zone_cues_defaults_confidence_high_for_backward_compat(
    tmp_path: Path,
) -> None:
    """Old artifacts without the field must still load."""
    path = tmp_path / "dead_zone_cues.json"
    _write(path, {"cues": [{"start": 10.0, "end": 20.0, "reason": "old"}]})
    cues = load_dead_zone_cues(path)
    assert cues[0].confidence == "high"


def test_load_dead_zone_cues_malformed_json_returns_empty(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("not json at all")
    assert load_dead_zone_cues(path) == []


# --- run() with dead-zone cues ---------------------------------------

def test_run_applies_high_confidence_dead_zone_cues(tmp_path: Path) -> None:
    """High-confidence cues always apply, regardless of VERIFY flag."""
    _write(tmp_path / "layout_plan.json", {
        "segments": [{"start": 0, "end": 300, "layout": "pip"}],
    })
    _write(tmp_path / "dead_zone_cues.json", {"cues": [
        {"start": 100.0, "end": 180.0, "reason": "installs",
         "confidence": "high"},
    ]})
    args = argparse.Namespace(
        work=tmp_path, verbose=0, cursor=None,
        screen_w=2560, screen_h=1440, origin_x=0.0, origin_y=0.0,
    )
    rc = run(args)
    assert rc == 0
    segs = json.loads((tmp_path / "segments.json").read_text())["segments"]
    # Original 0–300 pip split by the cue cut into 0–100 + 180–300.
    ranges = [(s["start"], s["end"]) for s in segs]
    assert ranges == [(0.0, 100.0), (180.0, 300.0)]


def test_run_skips_low_confidence_cues_when_verify_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With VERIFY_UNCERTAIN_CUTS off, low-confidence cues are ignored —
    they do NOT cut, because we haven't verified them."""
    monkeypatch.setattr("src.config.VERIFY_UNCERTAIN_CUTS", False)
    _write(tmp_path / "layout_plan.json", {
        "segments": [{"start": 0, "end": 300, "layout": "pip"}],
    })
    _write(tmp_path / "dead_zone_cues.json", {"cues": [
        {"start": 100.0, "end": 180.0, "reason": "maybe",
         "confidence": "low"},
    ]})
    args = argparse.Namespace(
        work=tmp_path, verbose=0, cursor=None,
        screen_w=2560, screen_h=1440, origin_x=0.0, origin_y=0.0,
    )
    rc = run(args)
    assert rc == 0
    segs = json.loads((tmp_path / "segments.json").read_text())["segments"]
    # No cut applied — single segment 0–300.
    assert len(segs) == 1
    assert (segs[0]["start"], segs[0]["end"]) == (0.0, 300.0)


def test_run_with_verifier_accept_applies_cut(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.config.VERIFY_UNCERTAIN_CUTS", True)
    _write(tmp_path / "layout_plan.json", {
        "segments": [{"start": 0, "end": 300, "layout": "pip"}],
    })
    _write(tmp_path / "dead_zone_cues.json", {"cues": [
        {"start": 100.0, "end": 180.0, "reason": "install",
         "confidence": "low"},
    ]})
    screen_path = tmp_path / "screen.mp4"
    screen_path.write_bytes(b"x")

    def fake_verify(candidates: list, *, screen_path: Path, work_dir: Path,
                    **_kw: object) -> list[VerifyOutput]:
        return [
            VerifyOutput(
                start=c.start, end=c.end, decision="accept",
                rationale="frames identical",
                original_start=c.start, original_end=c.end,
                frame_timestamps=(c.start, c.end), recursion_depth=0,
            ) for c in candidates
        ]

    monkeypatch.setattr("src.stages.unify_segments.verify_cuts_run", fake_verify)

    args = argparse.Namespace(
        work=tmp_path, verbose=0, cursor=None, screen=screen_path,
        screen_w=2560, screen_h=1440, origin_x=0.0, origin_y=0.0,
    )
    rc = run(args)
    assert rc == 0
    segs = json.loads((tmp_path / "segments.json").read_text())["segments"]
    ranges = [(s["start"], s["end"]) for s in segs]
    assert ranges == [(0.0, 100.0), (180.0, 300.0)]


def test_run_with_verifier_reject_keeps_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.config.VERIFY_UNCERTAIN_CUTS", True)
    _write(tmp_path / "layout_plan.json", {
        "segments": [{"start": 0, "end": 300, "layout": "pip"}],
    })
    _write(tmp_path / "dead_zone_cues.json", {"cues": [
        {"start": 100.0, "end": 180.0, "reason": "fuzzy",
         "confidence": "low"},
    ]})
    screen_path = tmp_path / "screen.mp4"
    screen_path.write_bytes(b"x")

    def fake_verify(candidates: list, **_kw: object) -> list[VerifyOutput]:
        return [
            VerifyOutput(
                start=c.start, end=c.end, decision="reject",
                rationale="code appears in frame 3",
                original_start=c.start, original_end=c.end,
                frame_timestamps=(c.start, c.end), recursion_depth=0,
            ) for c in candidates
        ]

    monkeypatch.setattr("src.stages.unify_segments.verify_cuts_run", fake_verify)

    args = argparse.Namespace(
        work=tmp_path, verbose=0, cursor=None, screen=screen_path,
        screen_w=2560, screen_h=1440, origin_x=0.0, origin_y=0.0,
    )
    rc = run(args)
    assert rc == 0
    segs = json.loads((tmp_path / "segments.json").read_text())["segments"]
    # Cut rejected → single segment 0–300 survives.
    assert len(segs) == 1
    assert (segs[0]["start"], segs[0]["end"]) == (0.0, 300.0)


def test_run_with_verifier_trim_applies_narrower_cut(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.config.VERIFY_UNCERTAIN_CUTS", True)
    _write(tmp_path / "layout_plan.json", {
        "segments": [{"start": 0, "end": 300, "layout": "pip"}],
    })
    _write(tmp_path / "dead_zone_cues.json", {"cues": [
        {"start": 100.0, "end": 180.0, "reason": "partial",
         "confidence": "low"},
    ]})
    screen_path = tmp_path / "screen.mp4"
    screen_path.write_bytes(b"x")

    def fake_verify(candidates: list, **_kw: object) -> list[VerifyOutput]:
        # Trim tail — narrower cut is [100, 140], not the full [100, 180].
        out = []
        for c in candidates:
            out.append(VerifyOutput(
                start=c.start, end=140.0, decision="trim",
                rationale="activity begins at t=140",
                original_start=c.start, original_end=c.end,
                frame_timestamps=(c.start, c.end), recursion_depth=0,
            ))
        return out

    monkeypatch.setattr("src.stages.unify_segments.verify_cuts_run", fake_verify)

    args = argparse.Namespace(
        work=tmp_path, verbose=0, cursor=None, screen=screen_path,
        screen_w=2560, screen_h=1440, origin_x=0.0, origin_y=0.0,
    )
    rc = run(args)
    assert rc == 0
    segs = json.loads((tmp_path / "segments.json").read_text())["segments"]
    ranges = [(s["start"], s["end"]) for s in segs]
    # Narrower cut [100, 140] — rest 140..300 preserved.
    assert ranges == [(0.0, 100.0), (140.0, 300.0)]


def test_run_verifier_skipped_without_screen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even with VERIFY_UNCERTAIN_CUTS=1, if no screen is available the
    low-confidence cues fall back to the "skip" behavior — the verifier
    has no frames to examine."""
    monkeypatch.setattr("src.config.VERIFY_UNCERTAIN_CUTS", True)
    _write(tmp_path / "layout_plan.json", {
        "segments": [{"start": 0, "end": 300, "layout": "pip"}],
    })
    _write(tmp_path / "dead_zone_cues.json", {"cues": [
        {"start": 100.0, "end": 180.0, "reason": "maybe",
         "confidence": "low"},
    ]})

    calls: list[int] = []

    def should_not_run(*_a: object, **_kw: object) -> list:
        calls.append(1)
        return []

    monkeypatch.setattr("src.stages.unify_segments.verify_cuts_run",
                        should_not_run)

    args = argparse.Namespace(
        work=tmp_path, verbose=0, cursor=None,  # no screen= attribute
        screen_w=2560, screen_h=1440, origin_x=0.0, origin_y=0.0,
    )
    rc = run(args)
    assert rc == 0
    assert calls == []  # verifier never invoked
    segs = json.loads((tmp_path / "segments.json").read_text())["segments"]
    assert len(segs) == 1  # no cut applied


# --- triple-intersection hard-cut ------------------------------------

def test_triple_intersection_requires_all_three_sources() -> None:
    """All three signals overlap on [5, 10] — that range is returned."""
    out = compute_triple_intersection_cuts(
        face_absent=[(0, 12)],
        cursor_idle=[(5, 20)],
        narration_silent=[(3, 10)],
    )
    assert out == [(5.0, 10.0)]


def test_triple_intersection_two_signals_is_empty() -> None:
    """Only two signals active → no triple-intersection output."""
    out = compute_triple_intersection_cuts(
        face_absent=[(0, 10)],
        cursor_idle=[(0, 10)],
        narration_silent=[],  # third source empty
    )
    assert out == []


def test_triple_intersection_respects_min_duration() -> None:
    """A 1.5 s triple overlap should be dropped under a 2 s threshold."""
    out = compute_triple_intersection_cuts(
        face_absent=[(5, 6.5)],
        cursor_idle=[(5, 6.5)],
        narration_silent=[(5, 6.5)],
        min_duration=2.0,
    )
    assert out == []


def test_triple_intersection_merges_disjoint_pieces() -> None:
    """Two separate triple-overlap ranges both get emitted."""
    out = compute_triple_intersection_cuts(
        face_absent=[(0, 5), (10, 15)],
        cursor_idle=[(0, 5), (10, 15)],
        narration_silent=[(0, 5), (10, 15)],
        min_duration=2.0,
    )
    assert out == [(0.0, 5.0), (10.0, 15.0)]


def test_load_face_absent_reads_json(tmp_path) -> None:
    p = tmp_path / "face_absent.json"
    p.write_text(json.dumps({
        "absences": [
            {"start": 1.0, "end": 3.0},
            {"start": 10.0, "end": 12.5},
        ]
    }))
    assert load_face_absent(p) == [(1.0, 3.0), (10.0, 12.5)]


def test_load_face_absent_missing_returns_empty(tmp_path) -> None:
    assert load_face_absent(tmp_path / "nope.json") == []


def test_load_silence_intervals_reads_json(tmp_path) -> None:
    p = tmp_path / "silence_intervals.json"
    p.write_text(json.dumps({
        "intervals": [{"start": 2.0, "end": 4.0}]
    }))
    assert load_silence_intervals(p) == [(2.0, 4.0)]


def test_load_silence_intervals_missing_returns_empty(tmp_path) -> None:
    assert load_silence_intervals(tmp_path / "nope.json") == []

