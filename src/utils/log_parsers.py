"""Parsers for ffmpeg filter log output — freezedetect, silencedetect, scdet.

ffmpeg emits these as stderr log lines like:
  [freezedetect @ 0x...] lavfi.freeze_start: 12.345
  [freezedetect @ 0x...] lavfi.freeze_duration: 3.210
  [freezedetect @ 0x...] lavfi.freeze_end: 15.555

TODO(M3): implement. Unit-test with fixture log strings.
"""
from __future__ import annotations

import re


FREEZE_START_RE = re.compile(r"freeze_start:\s*([\d.]+)")
FREEZE_END_RE = re.compile(r"freeze_end:\s*([\d.]+)")
SILENCE_START_RE = re.compile(r"silence_start:\s*([\d.]+)")
SILENCE_END_RE = re.compile(r"silence_end:\s*([\d.]+)")


def parse_freezedetect(stderr: str) -> list[tuple[float, float]]:
    """Pair up freeze_start/freeze_end lines into intervals.
    TODO(M3).
    """
    raise NotImplementedError


def parse_silencedetect(stderr: str) -> list[tuple[float, float]]:
    """Pair up silence_start/silence_end lines into intervals.
    TODO(M3).
    """
    raise NotImplementedError


__all__ = ["parse_freezedetect", "parse_silencedetect"]
