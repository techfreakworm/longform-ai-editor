"""Tests for frame-based verification of low-confidence cut decisions.

The verifier samples N frames uniformly across a proposed cut span,
asks a multimodal Claude whether the span is really removable, and
returns accept / reject / trim.

All tests mock the Claude call via dependency injection so the suite
stays fast and offline. Real-Claude behavior is covered by a single
@pytest.mark.slow integration test (skipped when `claude` is absent).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from src.stages.verify_cuts import (
    VerifyInput,
    VerifyOutput,
    extract_frames,
    run,
    sample_frame_timestamps,
    verify_one,
)


# --- sample_frame_timestamps ------------------------------------------

def test_sample_frame_timestamps_long_interval_returns_n_uniform() -> None:
    ts = sample_frame_timestamps(0.0, 70.0, n=8, min_gap=2.0)
    assert len(ts) == 8
    # uniformly spaced including both endpoints
    assert ts[0] == pytest.approx(0.0)
    assert ts[-1] == pytest.approx(70.0)
    diffs = [b - a for a, b in zip(ts, ts[1:])]
    assert all(abs(d - diffs[0]) < 1e-6 for d in diffs)


def test_sample_frame_timestamps_short_interval_falls_back_to_min_gap() -> None:
    """A 5 s span with min_gap=2 should yield 3 samples at [0,2.5,5] —
    can't fit 8 × 2 s in 5 s, so we drop n to fit within the gap budget."""
    ts = sample_frame_timestamps(0.0, 5.0, n=8, min_gap=2.0)
    assert len(ts) == 3
    assert ts[0] == pytest.approx(0.0)
    assert ts[-1] == pytest.approx(5.0)


def test_sample_frame_timestamps_tiny_interval_returns_endpoints() -> None:
    ts = sample_frame_timestamps(10.0, 10.5, n=8, min_gap=2.0)
    assert ts == [10.0, 10.5]


def test_sample_frame_timestamps_zero_length_returns_single() -> None:
    ts = sample_frame_timestamps(10.0, 10.0, n=8, min_gap=2.0)
    assert ts == [10.0]


def test_sample_frame_timestamps_negative_range_raises() -> None:
    with pytest.raises(ValueError):
        sample_frame_timestamps(10.0, 5.0, n=8, min_gap=2.0)


# --- extract_frames ---------------------------------------------------

def test_extract_frames_deterministic_filenames(tmp_path: Path) -> None:
    """Same (screen, timestamp) pair must hash to the same output path so
    repeat calls hit the cache."""
    # Need a real file so stat().st_mtime is stable.
    fake_src = tmp_path / "screen.mp4"
    fake_src.write_bytes(b"ignored content for hash")

    frames = [(0.5, tmp_path / "frame_0.jpg"), (10.0, tmp_path / "frame_1.jpg")]

    def fake_runner(cmd: list[str], **_kw: Any) -> None:
        # Simulate ffmpeg writing the output file.
        out = Path(cmd[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"jpg")

    with patch("src.stages.verify_cuts._run_ffmpeg", side_effect=fake_runner):
        out1 = extract_frames(fake_src, [0.5, 10.0], tmp_path)
        out2 = extract_frames(fake_src, [0.5, 10.0], tmp_path)

    assert [t for t, _ in out1] == [0.5, 10.0]
    assert [p for _, p in out1] == [p for _, p in out2]
    assert all(p.exists() for _, p in out1)


def test_extract_frames_skips_ffmpeg_on_cache_hit(tmp_path: Path) -> None:
    fake_src = tmp_path / "screen.mp4"
    fake_src.write_bytes(b"content")

    call_count = 0

    def counting_runner(cmd: list[str], **_kw: Any) -> None:
        nonlocal call_count
        call_count += 1
        out = Path(cmd[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"jpg")

    with patch("src.stages.verify_cuts._run_ffmpeg", side_effect=counting_runner):
        extract_frames(fake_src, [1.0], tmp_path)
        first_calls = call_count
        extract_frames(fake_src, [1.0], tmp_path)
        second_calls = call_count

    assert first_calls == 1
    assert second_calls == 1  # cache hit — no new ffmpeg invocation


# --- verify_one -------------------------------------------------------

def _fake_verifier(response: dict) -> Any:
    def _call(_inp: VerifyInput, _frames: list[tuple[float, Path]],
              **_kw: Any) -> dict:
        return response
    return _call


def test_verify_one_accept(tmp_path: Path) -> None:
    inp = VerifyInput(start=100.0, end=160.0,
                      reason="while installs", kind="dead_zone")
    with patch("src.stages.verify_cuts.extract_frames",
               return_value=[(100.0, tmp_path / "f.jpg"),
                             (160.0, tmp_path / "g.jpg")]):
        out = verify_one(
            inp, screen_path=tmp_path / "screen.mp4", work_dir=tmp_path,
            _call_verifier=_fake_verifier({
                "decision": "accept",
                "start": 100.0, "end": 160.0,
                "rationale": "frames visually identical",
            }),
        )
    assert out.decision == "accept"
    assert (out.start, out.end) == (100.0, 160.0)
    assert out.original_start == 100.0
    assert out.original_end == 160.0


def test_verify_one_reject(tmp_path: Path) -> None:
    inp = VerifyInput(start=100.0, end=160.0,
                      reason="while installs", kind="dead_zone")
    with patch("src.stages.verify_cuts.extract_frames",
               return_value=[(100.0, tmp_path / "f.jpg")]):
        out = verify_one(
            inp, screen_path=tmp_path / "screen.mp4", work_dir=tmp_path,
            _call_verifier=_fake_verifier({
                "decision": "reject",
                "start": 100.0, "end": 160.0,
                "rationale": "code editor opens halfway through",
            }),
        )
    assert out.decision == "reject"


def test_verify_one_trim_tail_edge_aligned(tmp_path: Path) -> None:
    inp = VerifyInput(start=100.0, end=160.0,
                      reason="while installs", kind="dead_zone")
    with patch("src.stages.verify_cuts.extract_frames",
               return_value=[(100.0, tmp_path / "f.jpg")]):
        out = verify_one(
            inp, screen_path=tmp_path / "screen.mp4", work_dir=tmp_path,
            max_recursion=0,  # don't recurse — test single-pass behavior
            _call_verifier=_fake_verifier({
                "decision": "trim",
                "start": 100.0, "end": 130.0,
                "rationale": "meaningful output appears at t=140",
            }),
        )
    assert out.decision == "trim"
    assert (out.start, out.end) == (100.0, 130.0)


def test_verify_one_trim_head_edge_aligned(tmp_path: Path) -> None:
    inp = VerifyInput(start=100.0, end=160.0,
                      reason="while installs", kind="dead_zone")
    with patch("src.stages.verify_cuts.extract_frames",
               return_value=[(100.0, tmp_path / "f.jpg")]):
        out = verify_one(
            inp, screen_path=tmp_path / "screen.mp4", work_dir=tmp_path,
            max_recursion=0,
            _call_verifier=_fake_verifier({
                "decision": "trim",
                "start": 130.0, "end": 160.0,
                "rationale": "activity at the start fades by t=130",
            }),
        )
    assert out.decision == "trim"
    assert (out.start, out.end) == (130.0, 160.0)


def test_verify_one_trim_middle_becomes_reject(tmp_path: Path) -> None:
    """Claude returns a trim that excludes the middle (i.e. a new range
    that doesn't touch either original edge). Per spec we reject + log
    rather than try to emit two cuts."""
    inp = VerifyInput(start=100.0, end=160.0,
                      reason="let me skip", kind="dead_zone")
    with patch("src.stages.verify_cuts.extract_frames",
               return_value=[(100.0, tmp_path / "f.jpg")]):
        out = verify_one(
            inp, screen_path=tmp_path / "screen.mp4", work_dir=tmp_path,
            max_recursion=0,
            _call_verifier=_fake_verifier({
                "decision": "trim",
                "start": 120.0, "end": 140.0,  # middle slice of [100,160]
                "rationale": "noisy",
            }),
        )
    assert out.decision == "reject"
    assert "middle" in out.rationale.lower()
    # Original bounds preserved for audit.
    assert (out.original_start, out.original_end) == (100.0, 160.0)


def test_verify_one_trim_range_outside_bounds_becomes_reject(tmp_path: Path) -> None:
    inp = VerifyInput(start=100.0, end=160.0,
                      reason="fuzzy", kind="dead_zone")
    with patch("src.stages.verify_cuts.extract_frames",
               return_value=[(100.0, tmp_path / "f.jpg")]):
        out = verify_one(
            inp, screen_path=tmp_path / "screen.mp4", work_dir=tmp_path,
            max_recursion=0,
            _call_verifier=_fake_verifier({
                "decision": "trim",
                "start": 90.0, "end": 200.0,   # wider than original
                "rationale": "confused",
            }),
        )
    assert out.decision == "reject"


def test_verify_one_recursion_bounded(tmp_path: Path) -> None:
    """Claude keeps saying "trim" on progressively narrower ranges — the
    recursion must stop at max_recursion and return the latest trim."""
    inp = VerifyInput(start=0.0, end=100.0, reason="x", kind="dead_zone")
    calls: list[tuple[float, float]] = []

    def shrinking(_inp: VerifyInput, _frames: list[tuple[float, Path]],
                  **_kw: Any) -> dict:
        calls.append((_inp.start, _inp.end))
        new_end = _inp.end * 0.5
        return {
            "decision": "trim",
            "start": _inp.start, "end": new_end,
            "rationale": "keep shrinking",
        }

    with patch("src.stages.verify_cuts.extract_frames",
               return_value=[(0.0, tmp_path / "f.jpg")]):
        out = verify_one(
            inp, screen_path=tmp_path / "screen.mp4", work_dir=tmp_path,
            max_recursion=2,
            _call_verifier=shrinking,
        )
    # 1 initial + 2 recursive = 3 calls max
    assert len(calls) == 3
    assert out.decision == "trim"
    # final range is 100 * 0.5^3 on end
    assert out.end == pytest.approx(12.5)
    assert out.recursion_depth == 2


def test_verify_one_accept_short_circuits_recursion(tmp_path: Path) -> None:
    """Accept on the first pass → no recursion regardless of max_recursion."""
    inp = VerifyInput(start=0.0, end=60.0, reason="x", kind="dead_zone")
    calls: list[int] = []

    def count_calls(*_a: Any, **_k: Any) -> dict:
        calls.append(1)
        return {"decision": "accept",
                "start": 0.0, "end": 60.0,
                "rationale": "ok"}

    with patch("src.stages.verify_cuts.extract_frames",
               return_value=[(0.0, tmp_path / "f.jpg")]):
        out = verify_one(
            inp, screen_path=tmp_path / "screen.mp4", work_dir=tmp_path,
            max_recursion=5,
            _call_verifier=count_calls,
        )
    assert len(calls) == 1
    assert out.decision == "accept"


# --- run (batch) + audit log ------------------------------------------

def test_run_writes_audit_log(tmp_path: Path) -> None:
    screen = tmp_path / "screen.mp4"
    screen.write_bytes(b"x")
    inputs = [
        VerifyInput(start=10.0, end=70.0, reason="install", kind="dead_zone"),
        VerifyInput(start=200.0, end=260.0, reason="skip", kind="dead_zone"),
    ]

    responses = iter([
        {"decision": "accept", "start": 10.0, "end": 70.0, "rationale": "a"},
        {"decision": "reject", "start": 200.0, "end": 260.0, "rationale": "b"},
    ])

    def stubbed(*_a: Any, **_k: Any) -> dict:
        return next(responses)

    with patch("src.stages.verify_cuts.extract_frames",
               return_value=[(10.0, tmp_path / "f.jpg")]):
        outs = run(inputs, screen_path=screen, work_dir=tmp_path,
                   _call_verifier=stubbed)

    assert len(outs) == 2
    assert outs[0].decision == "accept"
    assert outs[1].decision == "reject"

    log_path = tmp_path / "verify_log.json"
    assert log_path.exists()
    log = json.loads(log_path.read_text())
    assert len(log["entries"]) == 2
    assert log["entries"][0]["decision"] == "accept"
    assert log["entries"][1]["decision"] == "reject"
    # Audit fields present
    assert "original_start" in log["entries"][0]
    assert "frame_timestamps" in log["entries"][0]
    assert "rationale" in log["entries"][0]


def test_run_empty_candidates(tmp_path: Path) -> None:
    screen = tmp_path / "screen.mp4"
    screen.write_bytes(b"x")
    outs = run([], screen_path=screen, work_dir=tmp_path)
    assert outs == []
    # No audit log is created for an empty batch.
    assert not (tmp_path / "verify_log.json").exists()
