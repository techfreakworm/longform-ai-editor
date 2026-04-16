"""Smoke tests for src.utils.timecodes — a small, pure-Python module that
is actually implemented today (not a TODO). Serves as the first working test
so CI doesn't collapse on day one.
"""
from __future__ import annotations

from src.utils.timecodes import frame_to_sec, overlap, sec_to_frame, subtract_interval


def test_sec_to_frame_round_nearest():
    assert sec_to_frame(0.0, 30.0) == 0
    assert sec_to_frame(1.0, 30.0) == 30
    assert sec_to_frame(0.5, 30.0) == 15
    assert sec_to_frame(0.516, 30.0) == 15


def test_frame_to_sec_inverse():
    for fps in (24.0, 30.0, 60.0):
        for f in (0, 1, 60, 1800):
            sec = frame_to_sec(f, fps)
            assert sec_to_frame(sec, fps) == f


def test_overlap_basic():
    assert overlap((0, 10), (5, 15)) == 5.0
    assert overlap((0, 10), (10, 20)) == 0.0
    assert overlap((0, 10), (20, 30)) == 0.0
    assert overlap((0, 10), (3, 7)) == 4.0


def test_subtract_interval_middle():
    assert subtract_interval((0, 10), (3, 7)) == [(0, 3), (7, 10)]


def test_subtract_interval_left_edge():
    assert subtract_interval((0, 10), (0, 3)) == [(3, 10)]


def test_subtract_interval_right_edge():
    assert subtract_interval((0, 10), (7, 10)) == [(0, 7)]


def test_subtract_interval_disjoint():
    assert subtract_interval((0, 10), (20, 30)) == [(0, 10)]


def test_subtract_interval_covers_all():
    assert subtract_interval((3, 7), (0, 10)) == []
