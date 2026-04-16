"""Second ↔ frame conversion helpers and small time-range utilities.

Keeping this in one place avoids scattered rounding bugs across stages.
"""
from __future__ import annotations


def sec_to_frame(t: float, fps: float) -> int:
    """Round-nearest seconds → frame index."""
    return int(round(t * fps))


def frame_to_sec(n: int, fps: float) -> float:
    return n / fps


def overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Duration of overlap between two (start, end) intervals, 0 if none."""
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    return max(0.0, hi - lo)


def subtract_interval(
    base: tuple[float, float],
    cut: tuple[float, float],
) -> list[tuple[float, float]]:
    """Return `base` minus `cut`, as a list of 0–2 intervals."""
    bs, be = base
    cs, ce = cut
    if ce <= bs or cs >= be:
        return [base]
    out = []
    if cs > bs:
        out.append((bs, cs))
    if ce < be:
        out.append((ce, be))
    return out


__all__ = ["sec_to_frame", "frame_to_sec", "overlap", "subtract_interval"]
