"""Element-aware zoom target snapping via OCR.

Given a raw cursor-centroid (cx, cy) + a screen video at time `t`, run
OCR on the nearest frame and snap the centroid to the midpoint of the
closest text element's bounding box — but only if it's within a small
pixel threshold. Otherwise leave the centroid alone.

Why: cursor_zoom picks its centroid from mean cursor position, which
can land mid-whitespace between two buttons, mid-scrollbar, or on the
edge of a text block. OCR lets us reach for the SEMANTIC target the
presenter was pointing at — the button, the line of code, the field —
instead of the raw mouse coordinate.

Trade-offs this respects:
  - OCR is expensive (~300 ms per frame on CPU). We cache per
    (screen_path mtime + size, timestamp_rounded_to_1s) so re-runs are
    near-free.
  - paddleocr is an optional dep (`pip install '.[zoom-ocr]'`). Without
    it, `snap_centroid_to_element` is a no-op: returns its inputs
    unchanged. Callers need not branch.
  - Snap threshold is intentionally tight (150 px default, ~8% of a
    1920-wide frame). Prefer a raw cursor centroid over a far-away
    "semantic match" that would teleport the zoom elsewhere.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


_paddle_imports_attempted = False
_paddle_available = False
_paddle_reader = None  # type: ignore[assignment]


def _try_import_paddleocr():
    """Lazy singleton import of paddleocr's PaddleOCR reader."""
    global _paddle_imports_attempted, _paddle_available, _paddle_reader
    if _paddle_imports_attempted:
        return _paddle_reader
    _paddle_imports_attempted = True
    try:
        from paddleocr import PaddleOCR

        _paddle_reader = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
        _paddle_available = True
    except Exception as exc:  # noqa: BLE001 — third-party can throw many ways
        log.warning(
            "element_aware: paddleocr unavailable (%s) — "
            "falling back to raw cursor centroid. "
            "Install with `pip install '.[zoom-ocr]'`.",
            exc,
        )
        _paddle_reader = None
    return _paddle_reader


# ---------- bounding-box type ---------------------------------------

@dataclass
class ElementBox:
    x: float   # top-left in pixel coords
    y: float
    w: float
    h: float
    text: str

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0


# ---------- frame extraction + OCR ----------------------------------

def _extract_frame(
    screen_path: Path, t_s: float, out_path: Path
) -> Path:
    """Extract a single frame at time t_s to out_path (PNG)."""
    cmd = [
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
        "-ss", f"{t_s:.3f}",
        "-i", str(screen_path),
        "-frames:v", "1",
        "-f", "image2",
        str(out_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extract failed: {r.stderr[-300:]}")
    return out_path


def _cache_key(screen_path: Path, t_s: float) -> str:
    """Stable per-(file identity, second) cache key."""
    try:
        st = screen_path.stat()
        stamp = f"{st.st_size}:{int(st.st_mtime)}"
    except FileNotFoundError:
        stamp = "no-stat"
    rounded = int(t_s)  # 1 s resolution — OCR is frame-static
    h = hashlib.md5(f"{screen_path}|{stamp}|{rounded}".encode()).hexdigest()
    return h[:16]


def ocr_elements_at(
    screen_path: Path,
    t_s: float,
    *,
    cache_dir: Path | None = None,
) -> list[ElementBox]:
    """Return all OCR'd text boxes on the screen at time t_s.

    Cached on disk under `cache_dir` (default `<screen parent>/.ocr_cache`)
    so repeat queries at the same (second, file) are free. Returns []
    gracefully when paddleocr isn't installed.
    """
    reader = _try_import_paddleocr()
    if reader is None:
        return []

    cache_dir = cache_dir or (screen_path.parent / ".ocr_cache")
    cache_dir.mkdir(exist_ok=True)
    key = _cache_key(screen_path, t_s)
    cache_path = cache_dir / f"{key}.json"
    if cache_path.exists():
        data = json.loads(cache_path.read_text())
        return [ElementBox(**e) for e in data]

    frame_path = cache_dir / f"{key}.png"
    _extract_frame(screen_path, t_s, frame_path)

    raw = reader.ocr(str(frame_path), cls=False)
    elements: list[ElementBox] = []
    # paddleocr 2.x returns a list per image; each item is (bbox, (text, conf)).
    if raw and isinstance(raw, list) and raw[0] is not None:
        for det in raw[0]:
            try:
                bbox = det[0]
                text = det[1][0]
            except (IndexError, TypeError):
                continue
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            x = float(min(xs))
            y = float(min(ys))
            w = float(max(xs) - x)
            h = float(max(ys) - y)
            elements.append(ElementBox(x=x, y=y, w=w, h=h, text=str(text)))

    cache_path.write_text(json.dumps(
        [e.__dict__ for e in elements], indent=2,
    ))
    # Remove the .png to save disk once results are cached.
    try:
        frame_path.unlink()
    except FileNotFoundError:
        pass
    return elements


# ---------- snap API --------------------------------------------------

def snap_centroid_to_element(
    cx_norm: float,
    cy_norm: float,
    screen_path: Path,
    t_s: float,
    *,
    frame_w: int,
    frame_h: int,
    max_distance_px: float = 150.0,
) -> tuple[float, float]:
    """Snap a normalized centroid to the nearest OCR'd element's center.

    Args:
        cx_norm, cy_norm: zoom centroid in [0,1] normalized coords.
        screen_path: screen video path.
        t_s: time in video seconds.
        frame_w, frame_h: pixel dimensions of the screen frame.
        max_distance_px: snap only if the nearest element's center is
            within this many pixels of (cx_norm, cy_norm) projected to
            pixel space. Prevents teleporting the zoom to a distant
            unrelated element.

    Returns:
        (cx_norm, cy_norm) — either snapped or unchanged.
    """
    elements = ocr_elements_at(screen_path, t_s)
    if not elements:
        return cx_norm, cy_norm

    cx_px = cx_norm * frame_w
    cy_px = cy_norm * frame_h

    best: ElementBox | None = None
    best_d = float("inf")
    for e in elements:
        d = math.hypot(e.cx - cx_px, e.cy - cy_px)
        if d < best_d:
            best = e
            best_d = d

    if best is None or best_d > max_distance_px:
        return cx_norm, cy_norm

    return best.cx / frame_w, best.cy / frame_h


def snap_zoom_segments(
    zooms: list,  # list[ZoomSegment] — avoid circular import
    screen_path: Path,
    *,
    frame_w: int,
    frame_h: int,
    max_distance_px: float = 150.0,
) -> list:
    """Apply snap_centroid_to_element to a list of ZoomSegments in-place.

    Kept as a convenience wrapper so unify_segments can run one call
    rather than iterating. Importing ZoomSegment here would cycle; the
    caller provides the list and we mutate each entry's (cx, cy).
    """
    # Only run OCR once per unique (rounded) t — segments often cluster.
    from src.stages.cursor_zoom import ZoomSegment  # local to avoid cycle

    out: list[ZoomSegment] = []
    for z in zooms:
        new_cx, new_cy = snap_centroid_to_element(
            z.cx, z.cy, screen_path, z.start,
            frame_w=frame_w, frame_h=frame_h,
            max_distance_px=max_distance_px,
        )
        out.append(ZoomSegment(
            start=z.start, end=z.end, zoom=z.zoom,
            cx=new_cx, cy=new_cy,
        ))
    return out


__all__ = [
    "ElementBox",
    "ocr_elements_at",
    "snap_centroid_to_element",
    "snap_zoom_segments",
]
