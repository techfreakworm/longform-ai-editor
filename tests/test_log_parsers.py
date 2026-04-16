"""Tests for src.utils.log_parsers.

Stderr fixture strings are taken from real ffmpeg runs on an M5 Max —
the exact format the detector stage will encounter in practice.
"""
from __future__ import annotations

from src.utils.log_parsers import parse_freezedetect, parse_silencedetect


# Real stderr snippet from:
# ffmpeg -i ext.mov -vf "freezedetect=n=-50dB:d=2" -map 0:v:0 -f null -
FREEZE_STDERR_REAL = """
frame=  261 fps=0.0 q=-0.0 size=N/A time=00:00:04.35 bitrate=N/A speed=8.61x    [freezedetect @ 0x72ac07e80] lavfi.freezedetect.freeze_start: 5.883333
frame=  492 fps=487 q=-0.0 size=N/A time=00:00:08.20 bitrate=N/A speed=8.12x    [freezedetect @ 0x72ac07e80] lavfi.freezedetect.freeze_duration: 7.816667
[freezedetect @ 0x72ac07e80] lavfi.freezedetect.freeze_end: 13.7
frame=  917 fps=454 q=-0.0 size=N/A time=00:00:15.28 bitrate=N/A speed=7.56x    [freezedetect @ 0x72ac07e80] lavfi.freezedetect.freeze_start: 13.7
[freezedetect @ 0x72ac07e80] lavfi.freezedetect.freeze_duration: 3.0
[freezedetect @ 0x72ac07e80] lavfi.freezedetect.freeze_end: 16.7
"""

# Real stderr snippet from:
# ffmpeg -i merged.mp4 -af "silencedetect=noise=-30dB:d=2" -f null -
SILENCE_STDERR_REAL = """
[silencedetect @ 0x8eac57d20] silence_start: 0
frame=  412 fps=268 q=-0.0 speed=4.08x [silencedetect @ 0x8eac57d20] silence_end: 7.185896 | silence_duration: 7.185896
frame=  965 fps=211 q=-0.0 speed=3.39x [silencedetect @ 0x8eac57d20] silence_start: 13.864979
[silencedetect @ 0x8eac57d20] silence_end: 17.288896 | silence_duration: 3.423917
"""


def test_parse_freezedetect_pairs_real_output() -> None:
    intervals = parse_freezedetect(FREEZE_STDERR_REAL)
    assert len(intervals) == 2
    assert intervals[0] == (5.883333, 13.7)
    assert intervals[1] == (13.7, 16.7)


def test_parse_silencedetect_pairs_real_output() -> None:
    intervals = parse_silencedetect(SILENCE_STDERR_REAL)
    assert len(intervals) == 2
    assert intervals[0] == (0.0, 7.185896)
    assert intervals[1] == (13.864979, 17.288896)


def test_parse_freezedetect_drops_unclosed_start() -> None:
    """A freeze that's still open when ffmpeg stops has no matching end —
    we can't know the real end time, so drop it.
    """
    stderr = (
        "[freezedetect] lavfi.freezedetect.freeze_start: 5.0\n"
        "[freezedetect] lavfi.freezedetect.freeze_duration: 3.0\n"
        "[freezedetect] lavfi.freezedetect.freeze_end: 8.0\n"
        "[freezedetect] lavfi.freezedetect.freeze_start: 20.0\n"
        # no freeze_end — file ended
    )
    intervals = parse_freezedetect(stderr)
    assert intervals == [(5.0, 8.0)]


def test_parse_freezedetect_empty_stderr_returns_empty() -> None:
    assert parse_freezedetect("") == []
    assert parse_freezedetect("some unrelated ffmpeg noise") == []


def test_parse_silencedetect_drops_unclosed_start() -> None:
    stderr = (
        "[silencedetect] silence_start: 10.0\n"
        "[silencedetect] silence_end: 15.0 | silence_duration: 5.0\n"
        "[silencedetect] silence_start: 25.0\n"
    )
    intervals = parse_silencedetect(stderr)
    assert intervals == [(10.0, 15.0)]


def test_parse_silencedetect_integer_start_time() -> None:
    """ffmpeg sometimes emits `silence_start: 0` (int) at t=0."""
    stderr = "[silencedetect] silence_start: 0\n[silencedetect] silence_end: 2.5 | ...\n"
    assert parse_silencedetect(stderr) == [(0.0, 2.5)]


def test_parse_rejects_reversed_intervals() -> None:
    """end < start should be filtered (defensive; shouldn't happen from ffmpeg)."""
    stderr = (
        "freeze_start: 10.0\n"
        "freeze_end: 5.0\n"
    )
    assert parse_freezedetect(stderr) == []


def test_parse_equals_or_colon_delimiter() -> None:
    """The regex tolerates either `:` or `=` between key and value, so
    metadata-frame format also parses."""
    stderr_eq = "freeze_start=10.0\nfreeze_end=15.0\n"
    assert parse_freezedetect(stderr_eq) == [(10.0, 15.0)]
