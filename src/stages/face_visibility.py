"""Stage B.3 — face visibility detection on the webcam track.

Samples `cam.mov` at FACE_SAMPLE_RATE_HZ (default 2 Hz) through ffmpeg
into an in-memory pipe, runs Apple Vision `VNDetectFaceRectanglesRequest`
on each frame via PyObjC, and emits time ranges where **no face** was
detected continuously for ≥ FACE_ABSENT_MIN_SEC.

Output is consumed by `unify_segments` which intersects it with
cursor-idle and narration-silent intervals to produce triple-confirmed
hard-cut ranges.

The sample-rate / threshold tradeoff:
  * 2 Hz × 2 s threshold ≈ 4 samples minimum per absence → ample margin
    against single-frame false negatives (hand in front, speaker looks
    down, Vision misses a profile view).
  * Lower rates save ffmpeg seek cost but can miss short look-offs.
  * Higher rates pay Apple Vision cost on every frame for no gain —
    the minimum absence window is the real resolution limit.

macOS-only. On other platforms the stage degrades to "all visible"
so downstream stages behave as if the feature were off.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from src import config

log = logging.getLogger(__name__)


@dataclass
class FaceAbsence:
    start: float
    end: float


# ---------- PyObjC lazy import --------------------------------------

_vision_imports_attempted = False
_vision_available = False


def _try_import_vision() -> bool:
    """Import PyObjC Vision / Quartz lazily.

    Kept lazy so the module stays importable on non-macOS (tests, CI)
    and so import cost isn't paid by pipelines that never call the
    face-visibility stage.
    """
    global _vision_imports_attempted, _vision_available
    if _vision_imports_attempted:
        return _vision_available
    _vision_imports_attempted = True

    if sys.platform != "darwin":
        log.warning(
            "face_visibility: platform %s is not macOS — falling back to 'always visible'",
            sys.platform,
        )
        return False

    try:
        import Quartz  # noqa: F401  (CoreGraphics / CGImage)
        import Vision  # noqa: F401
        from Foundation import NSData  # noqa: F401
    except ImportError as exc:
        log.warning(
            "face_visibility: PyObjC Vision/Quartz not installed (%s) — "
            "run `pip install pyobjc-framework-Vision pyobjc-framework-Quartz` "
            "to enable face-absence detection",
            exc,
        )
        return False

    _vision_available = True
    return True


# ---------- frame pipe from ffmpeg ----------------------------------

def _ffmpeg_frame_iter(cam_path: Path, sample_rate_hz: float):
    """Yield raw BMP-encoded frames from cam_path at `sample_rate_hz`.

    BMP is used because it is the simplest container Apple Vision can
    decode without extra tooling — one image = one ffmpeg output.
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error",
        "-i", str(cam_path),
        "-vf", f"fps={sample_rate_hz}",
        "-f", "image2pipe",
        "-vcodec", "bmp",
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None

    buf = b""
    try:
        while True:
            chunk = proc.stdout.read(1 << 16)
            if not chunk:
                break
            buf += chunk
            while True:
                if len(buf) < 6 or buf[0:2] != b"BM":
                    break
                size = int.from_bytes(buf[2:6], "little")
                if size == 0 or len(buf) < size:
                    break
                yield buf[:size]
                buf = buf[size:]
    finally:
        proc.stdout.close()
        proc.wait()
    if proc.returncode not in (0, None):
        err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        raise RuntimeError(f"ffmpeg frame pipe failed: {err[-400:]}")


# ---------- face detector -------------------------------------------

def _frame_has_face(bmp_bytes: bytes) -> bool:
    """Return True if Apple Vision detects at least one face in the BMP."""
    from Foundation import NSData
    from Quartz import (
        CGImageSourceCreateImageAtIndex,
        CGImageSourceCreateWithData,
    )
    from Vision import (
        VNDetectFaceRectanglesRequest,
        VNImageRequestHandler,
    )

    ns_data = NSData.dataWithBytes_length_(bmp_bytes, len(bmp_bytes))
    source = CGImageSourceCreateWithData(ns_data, None)
    if source is None:
        return False
    cg_image = CGImageSourceCreateImageAtIndex(source, 0, None)
    if cg_image is None:
        return False

    req = VNDetectFaceRectanglesRequest.alloc().init()
    handler = VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, None)
    success, _err = handler.performRequests_error_([req], None)
    if not success:
        return False
    results = req.results() or []
    return len(results) > 0


# ---------- main entry ----------------------------------------------

def detect_face_absent_intervals(
    cam_path: Path,
    *,
    sample_rate_hz: float | None = None,
    min_absence_sec: float | None = None,
) -> list[FaceAbsence]:
    """Return intervals where the webcam has no detectable face for
    at least `min_absence_sec` seconds.

    Samples frames via ffmpeg at `sample_rate_hz`. Each sample's
    timestamp is inferred from its index (index / sample_rate_hz) — exact
    enough for the ≥ 2 s threshold this feature cares about.

    When Apple Vision is unavailable (non-macOS or missing PyObjC), logs
    a warning and returns an empty list so downstream stages behave as
    if the feature were off.
    """
    sample_rate_hz = sample_rate_hz or config.FACE_SAMPLE_RATE_HZ
    min_absence_sec = min_absence_sec or config.FACE_ABSENT_MIN_SEC

    if not _try_import_vision():
        return []

    dt = 1.0 / sample_rate_hz
    absences: list[FaceAbsence] = []
    run_start: float | None = None

    for idx, bmp in enumerate(_ffmpeg_frame_iter(cam_path, sample_rate_hz)):
        t = idx * dt
        has_face = _frame_has_face(bmp)
        if has_face:
            if run_start is not None and (t - run_start) >= min_absence_sec:
                absences.append(FaceAbsence(start=run_start, end=t))
            run_start = None
        else:
            if run_start is None:
                run_start = t

    # Trailing run that lasted to EOF.
    if run_start is not None:
        final_t = (idx + 1) * dt  # type: ignore[name-defined]
        if (final_t - run_start) >= min_absence_sec:
            absences.append(FaceAbsence(start=run_start, end=final_t))

    return absences


def run(cam_path: Path, out_path: Path | None = None) -> Path:
    """CLI entry: detect → write work/face_absent.json.

    Idempotent: caller deletes the output to force re-run.
    """
    out_path = out_path or (config.WORK_DIR / "face_absent.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        log.info("face_visibility: %s exists, skipping", out_path)
        return out_path

    log.info("face_visibility: scanning %s", cam_path)
    absences = detect_face_absent_intervals(cam_path)
    payload = {
        "cam_path": str(cam_path),
        "sample_rate_hz": config.FACE_SAMPLE_RATE_HZ,
        "min_absence_sec": config.FACE_ABSENT_MIN_SEC,
        "absences": [asdict(a) for a in absences],
    }
    out_path.write_text(json.dumps(payload, indent=2))
    log.info("face_visibility: found %d absence interval(s) → %s", len(absences), out_path)
    return out_path


__all__ = [
    "FaceAbsence",
    "detect_face_absent_intervals",
    "run",
]
