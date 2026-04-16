"""Parsers for ffmpeg filter log output — freezedetect, silencedetect.

ffmpeg emits these as stderr log lines (not machine-readable metadata
frames by default). Real format from `ffmpeg -vf freezedetect=... -f null -`:

  [freezedetect @ 0x...] lavfi.freezedetect.freeze_start: 5.883333
  [freezedetect @ 0x...] lavfi.freezedetect.freeze_duration: 7.816667
  [freezedetect @ 0x...] lavfi.freezedetect.freeze_end: 13.7

And from `ffmpeg -af silencedetect=... -f null -`:

  [silencedetect @ 0x...] silence_start: 0
  [silencedetect @ 0x...] silence_end: 7.185896 | silence_duration: 7.185896

We don't use `silence_duration` — it's derivable from (end - start).

The lines are interleaved with ffmpeg's per-frame progress (`frame=...`)
on the same stderr. Regex-based extraction works cleanly across that.
"""
from __future__ import annotations

import re

# [:=] tolerates either a colon (ffmpeg default) or equals sign (older
# `-metadata` output). Float regex is permissive; real values always look
# like ints or decimals without exponent/sign.
_FLOAT = r"[\d.]+"

_FREEZE_START_RE = re.compile(rf"freeze_start[:=]\s*({_FLOAT})")
_FREEZE_END_RE = re.compile(rf"freeze_end[:=]\s*({_FLOAT})")
_SILENCE_START_RE = re.compile(rf"silence_start[:=]\s*({_FLOAT})")
_SILENCE_END_RE = re.compile(rf"silence_end[:=]\s*({_FLOAT})")


def _pair_starts_and_ends(
    stderr: str,
    start_re: re.Pattern[str],
    end_re: re.Pattern[str],
) -> list[tuple[float, float]]:
    """Extract ordered starts and ends, pair them, keep only well-formed intervals."""
    starts = [float(m.group(1)) for m in start_re.finditer(stderr)]
    ends = [float(m.group(1)) for m in end_re.finditer(stderr)]
    intervals: list[tuple[float, float]] = []
    for i, s in enumerate(starts):
        if i >= len(ends):
            # Unclosed start: detector was still inside the region when the
            # input ran out. Drop — we don't know the real end.
            break
        e = ends[i]
        if e > s:
            intervals.append((s, e))
    return intervals


def parse_freezedetect(stderr: str) -> list[tuple[float, float]]:
    """Pair up freeze_start / freeze_end lines into (start, end) intervals."""
    return _pair_starts_and_ends(stderr, _FREEZE_START_RE, _FREEZE_END_RE)


def parse_silencedetect(stderr: str) -> list[tuple[float, float]]:
    """Pair up silence_start / silence_end lines into (start, end) intervals."""
    return _pair_starts_and_ends(stderr, _SILENCE_START_RE, _SILENCE_END_RE)


__all__ = ["parse_freezedetect", "parse_silencedetect"]
