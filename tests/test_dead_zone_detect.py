"""Tests for Stage C — dead-zone detection.

Unit-tests the pure pieces (intersect_intervals, classify) with hand-
crafted inputs. The ffmpeg-invoking detectors run once as an integration
test against the real fixture set (marked @slow).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from src.stages.dead_zone_detect import (
    classify,
    intersect_intervals,
    run,
    run_freezedetect,
    run_silencedetect,
)


# --- intersect_intervals ----------------------------------------------

def test_intersect_requires_agreement() -> None:
    """Two detectors: both must fire at the same time for the range to survive."""
    freeze = [(0.0, 10.0)]
    silence = [(5.0, 15.0)]
    out = intersect_intervals(freeze, silence, min_agree=2, min_duration=0.0)
    assert [(s, e) for (s, e, _) in out] == [(5.0, 10.0)]


def test_intersect_no_overlap_returns_empty() -> None:
    freeze = [(0.0, 5.0)]
    silence = [(10.0, 15.0)]
    out = intersect_intervals(freeze, silence, min_agree=2, min_duration=0.0)
    assert out == []


def test_intersect_single_detector_does_not_fire_alone() -> None:
    """min_agree=2 with only one detector input should return []."""
    freeze = [(0.0, 10.0)]
    out = intersect_intervals(freeze, [], min_agree=2, min_duration=0.0)
    assert out == []


def test_intersect_reports_firing_detectors() -> None:
    """The 3rd element of each tuple lists which detector indices fired."""
    out = intersect_intervals(
        [(0, 10)], [(5, 15)], min_agree=2, min_duration=0.0,
    )
    _, _, dets = out[0]
    assert dets == [0, 1]


def test_intersect_min_duration_drops_short_ranges() -> None:
    """Overlap shorter than min_duration is filtered out."""
    freeze = [(0, 3)]
    silence = [(2, 10)]
    # Overlap is (2, 3) = 1 s — below the 2 s default
    out = intersect_intervals(freeze, silence, min_agree=2, min_duration=2.0)
    assert out == []


def test_intersect_merges_within_gap() -> None:
    """Two close overlaps separated by a tiny gap should merge."""
    freeze = [(0, 5), (5.1, 10)]
    silence = [(0, 10)]
    out = intersect_intervals(
        freeze, silence, min_agree=2, min_duration=0.0, merge_gap=0.5,
    )
    # Gap between the two freeze-only-closed-then-reopened ranges is 0.1 s
    # which is less than merge_gap=0.5 → merge into [0, 10]
    assert len(out) == 1
    assert out[0][0] == 0 and out[0][1] == 10


def test_intersect_does_not_merge_beyond_gap() -> None:
    freeze = [(0, 5), (7, 12)]
    silence = [(0, 12)]
    out = intersect_intervals(
        freeze, silence, min_agree=2, min_duration=0.0, merge_gap=0.5,
    )
    # Gap is 2 s, much larger than merge_gap — keep separate
    assert len(out) == 2


def test_intersect_three_detectors_min_agree_2() -> None:
    """With 3 detectors, min_agree=2 means any two of them at a time."""
    a = [(0, 5)]
    b = [(3, 8)]
    c = [(7, 10)]
    out = intersect_intervals(a, b, c, min_agree=2, min_duration=0.0)
    # (3, 5): a+b; (7, 8): b+c.
    ranges = [(s, e) for (s, e, _) in out]
    assert (3.0, 5.0) in ranges
    assert (7.0, 8.0) in ranges


def test_intersect_empty_inputs() -> None:
    assert intersect_intervals() == []
    assert intersect_intervals([]) == []
    assert intersect_intervals([], []) == []


def test_intersect_abutting_intervals_bridge() -> None:
    """Abutting intervals from different detectors at the same instant
    should not cause false "coverage drop" at the boundary — opens
    should be processed before closes when times tie.
    """
    freeze = [(0, 5), (5, 10)]  # two abutting
    silence = [(0, 10)]
    out = intersect_intervals(freeze, silence, min_agree=2, min_duration=0.0)
    # Should emit one contiguous [0, 10]
    assert len(out) == 1
    assert out[0][0] == 0 and out[0][1] == 10


# --- classify ---------------------------------------------------------

def test_classify_speed_4x() -> None:
    assert classify(0, 2.5) == "speed@4x"  # 2.5 s
    assert classify(0, 3.0) == "speed@4x"  # exactly 3 s still 4x (not > 3)


def test_classify_speed_8x() -> None:
    assert classify(0, 3.1) == "speed@8x"
    assert classify(0, 10.0) == "speed@8x"


def test_classify_cut() -> None:
    assert classify(0, 10.1) == "cut"
    assert classify(0, 100.0) == "cut"


def test_classify_custom_thresholds() -> None:
    assert classify(0, 5.0, cut_min_sec=4.0, speed_8x_min_sec=2.0) == "cut"
    assert classify(0, 3.0, cut_min_sec=4.0, speed_8x_min_sec=2.0) == "speed@8x"


# --- CLI run (with mocked detectors) ---------------------------------

def test_run_writes_dead_zones_json(tmp_path, monkeypatch) -> None:
    """End-to-end run() with mocked ffmpeg detectors. Verifies JSON format
    and classification without touching the filesystem for video input."""
    from src.stages import dead_zone_detect

    monkeypatch.setattr(
        dead_zone_detect, "run_freezedetect",
        lambda *_a, **_k: [(0, 5), (10, 25)],
    )
    monkeypatch.setattr(
        dead_zone_detect, "run_silencedetect",
        lambda *_a, **_k: [(0, 30)],
    )

    args = argparse.Namespace(
        screen=tmp_path / "dummy_screen.mov",
        webcam=tmp_path / "dummy_webcam.mov",
        audio=None, words=None,
        work=tmp_path, verbose=0,
        freeze_db=-50, freeze_min=2.0,
        silence_db=-30, silence_min=2.0,
        cut_min=10.0, speed_8x_min=3.0,
    )
    rc = run(args)
    assert rc == 0

    doc = json.loads((tmp_path / "dead_zones.json").read_text())
    zones = doc["zones"]
    assert len(zones) == 2
    # (0, 5) duration 5 → speed@8x; detectors both 0 and 1
    assert zones[0]["start"] == 0 and zones[0]["end"] == 5
    assert zones[0]["action"] == "speed@8x"
    assert set(zones[0]["detectors"]) == {"freezedetect", "silencedetect"}
    # (10, 25) duration 15 → cut
    assert zones[1]["start"] == 10 and zones[1]["end"] == 25
    assert zones[1]["action"] == "cut"


def test_run_no_overlaps_produces_empty_zones(tmp_path, monkeypatch) -> None:
    from src.stages import dead_zone_detect
    monkeypatch.setattr(dead_zone_detect, "run_freezedetect",
                        lambda *_a, **_k: [(0, 5)])
    monkeypatch.setattr(dead_zone_detect, "run_silencedetect",
                        lambda *_a, **_k: [(20, 25)])
    args = argparse.Namespace(
        screen=tmp_path / "a.mov", webcam=tmp_path / "b.mov",
        audio=None, words=None,
        work=tmp_path, verbose=0,
        freeze_db=-50, freeze_min=2.0,
        silence_db=-30, silence_min=2.0,
        cut_min=10.0, speed_8x_min=3.0,
    )
    rc = run(args)
    assert rc == 0
    doc = json.loads((tmp_path / "dead_zones.json").read_text())
    assert doc["zones"] == []


# --- real ffmpeg integration against real fixtures -------------------

@pytest.mark.slow
def test_freezedetect_real_fixture(real_ext: Path) -> None:
    """ffmpeg freezedetect on the real ext.mov must return at least one
    interval — the session had several install-wait moments that were
    visually static."""
    intervals = run_freezedetect(real_ext, db=-50.0, min_sec=2.0)
    assert len(intervals) >= 1
    for s, e in intervals:
        assert e > s
        assert s >= 0


@pytest.mark.slow
def test_silencedetect_real_fixture(real_merged: Path) -> None:
    """Real merged.mp4 has several silent stretches (before/after speaking)."""
    intervals = run_silencedetect(real_merged, db=-30.0, min_sec=2.0)
    assert len(intervals) >= 1
    for s, e in intervals:
        assert e > s
        assert s >= 0


@pytest.mark.slow
def test_run_real_fixture_produces_valid_dead_zones(
    real_ext: Path, real_cam: Path, real_merged: Path, tmp_path: Path
) -> None:
    """End-to-end on real fixtures — verify dead_zones.json is well-formed
    and that every emitted zone passes duration + action invariants.
    """
    args = argparse.Namespace(
        screen=real_ext,
        webcam=real_cam,
        audio=real_merged,  # user's authoritative audio source
        words=None,
        work=tmp_path, verbose=0,
        freeze_db=-50, freeze_min=2.0,
        silence_db=-30, silence_min=2.0,
        cut_min=10.0, speed_8x_min=3.0,
    )
    rc = run(args)
    assert rc == 0
    doc = json.loads((tmp_path / "dead_zones.json").read_text())
    for z in doc["zones"]:
        assert z["end"] > z["start"]
        d = z["end"] - z["start"]
        assert d >= 2.0
        # Action matches classification rules
        if d > 10.0:
            assert z["action"] == "cut"
        elif d > 3.0:
            assert z["action"] == "speed@8x"
        else:
            assert z["action"] == "speed@4x"
        assert set(z["detectors"]).issubset({"freezedetect", "silencedetect"})
