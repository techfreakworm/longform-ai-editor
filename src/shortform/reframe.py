"""Face-tracked + cursor-aware reframing for 9:16 portrait output.

Produces per-scene crop windows for both tracks:

  * **Webcam track** (`cam_full`, `split_vstack` bottom, `pip` inset) —
    crop centered on the presenter's face using MediaPipe Face
    Landmarker. Falls back to a static offset from
    `config.PIP_FACE_X / PIP_FACE_Y` when MediaPipe isn't installed.

  * **Screen track** (`screen_full`, `split_vstack` top, `pip` main) —
    crop centered on cursor activity so code / UI remains legible on a
    phone. Uses the cursor CSV (same file long-form consumes). Falls
    back to screen-center when no CSV available.

Both tracks are smoothed across time with a 1€ Filter (Casiez, Roussel,
Vogel CHI 2012) so pans feel natural rather than jittery.

Scene boundaries come from PySceneDetect's AdaptiveDetector — a
re-centering event is more acceptable at a scene cut than mid-shot, so
clamping updates to scene boundaries is the default.

Heavy deps are imported lazily and **optional** (install via
`pip install '.[shortform]'`). The module's core data types and
bootstrap-friendly paths (static-offset fallback) always work.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from src import config

log = logging.getLogger(__name__)


# ----- data types ----------------------------------------------------

@dataclass
class Scene:
    start: float  # seconds
    end: float


@dataclass
class CropWindow:
    """A crop rectangle to apply over [start, end].

    (cx, cy) are normalized [0, 1] in the SOURCE frame's space. The
    renderer picks the aspect ratio (9:16 for cam_full/screen_full,
    9:8 for vstack halves, etc.) and translates to pixel coords at
    render time.
    """
    start: float
    end: float
    cx: float
    cy: float


# ----- scene detection ----------------------------------------------

def detect_scenes(
    video_path: Path, min_scene_len_frames: int = 15,
) -> list[Scene]:
    """Run PySceneDetect's AdaptiveDetector over `video_path`.

    Falls back to one scene [0, duration] if scenedetect isn't installed
    or detection fails.
    """
    try:
        from scenedetect import AdaptiveDetector, detect

        scenes_raw = detect(
            str(video_path),
            AdaptiveDetector(
                adaptive_threshold=3.0,
                min_scene_len=min_scene_len_frames,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("detect_scenes: scenedetect unavailable/failed (%s) — "
                    "treating clip as single scene", exc)
        return []

    scenes: list[Scene] = []
    for s, e in scenes_raw:
        scenes.append(Scene(start=s.get_seconds(), end=e.get_seconds()))
    return scenes


def _scenes_or_single(scenes: list[Scene], duration_s: float) -> list[Scene]:
    """Coerce empty scene list to one big scene."""
    if not scenes:
        return [Scene(start=0.0, end=duration_s)]
    return scenes


# ----- face tracking -------------------------------------------------

def _detect_faces_in_frame(frame_bgr, haar_cascade) -> list[tuple[float, float, float, float]]:
    """Return list of (x, y, w, h) relative [0,1] face boxes in a BGR frame."""
    import cv2

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    rects = haar_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60),
    )
    return [
        (x / w, y / h, rw / w, rh / h)
        for (x, y, rw, rh) in rects
    ]


def _face_centroid_per_scene_haar(
    webcam_path: Path, scenes: list[Scene],
) -> list[tuple[float, float]]:
    """Per-scene MEDIAN face centroid via OpenCV Haar cascade.

    OpenCV is already a base dep — no extra install needed. Haar is less
    accurate than MediaPipe Face Mesh but plenty for PIP re-centering
    on a solo creator's frontal webcam. Returns (config.PIP_FACE_X,
    config.PIP_FACE_Y) for scenes where no face is detected.

    MediaPipe replaced its `solutions.face_detection` API with the
    Tasks API in 0.10.x, which requires downloading a .task file.
    Using Haar keeps us zero-config.
    """
    try:
        import cv2
    except Exception as exc:  # noqa: BLE001
        log.warning("opencv not installed (%s) — static face offset", exc)
        return [(config.PIP_FACE_X, config.PIP_FACE_Y) for _ in scenes]

    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        haar = cv2.CascadeClassifier(cascade_path)
        if haar.empty():
            raise RuntimeError(f"failed to load Haar cascade at {cascade_path}")
    except Exception as exc:  # noqa: BLE001
        log.warning("Haar cascade unavailable (%s) — static face offset", exc)
        return [(config.PIP_FACE_X, config.PIP_FACE_Y) for _ in scenes]

    cap = cv2.VideoCapture(str(webcam_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    centroids: list[tuple[float, float]] = []
    try:
        for scene in scenes:
            samples_t = [
                scene.start + (scene.end - scene.start) * frac
                for frac in (0.2, 0.4, 0.6, 0.8)
            ]
            xs: list[float] = []
            ys: list[float] = []
            for t in samples_t:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
                ok, frame = cap.read()
                if not ok:
                    continue
                boxes = _detect_faces_in_frame(frame, haar)
                if not boxes:
                    continue
                # pick largest by area
                x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
                xs.append(x + w / 2.0)
                ys.append(y + h / 2.0)
            if xs:
                xs.sort(); ys.sort()
                mid = len(xs) // 2
                centroids.append((xs[mid], ys[mid]))
            else:
                centroids.append((config.PIP_FACE_X, config.PIP_FACE_Y))
    finally:
        cap.release()

    return centroids


# Legacy name preserved for test back-compat; points at the Haar impl.
_face_centroid_per_scene_mediapipe = _face_centroid_per_scene_haar


# ----- cursor centroid on screen track ------------------------------

def _cursor_centroid_per_scene(
    cursor_csv: Path | None,
    scenes: list[Scene],
    *,
    screen_w: int,
    screen_h: int,
    origin_x: float,
    origin_y: float,
    csv_to_video_offset_s: float,
) -> list[tuple[float, float]]:
    """Mean cursor position (cx, cy ∈ [0,1]) per scene.

    Returns (0.5, 0.5) for scenes with no cursor data or when the CSV
    isn't available — safe fallback to screen-center crop.
    """
    if cursor_csv is None or not cursor_csv.exists():
        return [(0.5, 0.5) for _ in scenes]

    from src.stages.cursor_zoom import parse_cursor_csv

    _clicks, moves = parse_cursor_csv(
        cursor_csv,
        screen_w=screen_w, screen_h=screen_h,
        origin_x=origin_x, origin_y=origin_y,
    )
    out: list[tuple[float, float]] = []
    for scene in scenes:
        # Cursor CSV is in logger timebase → shift to video time.
        moves_in_scene = [
            m for m in moves
            if scene.start <= (m.t_s + csv_to_video_offset_s) <= scene.end
        ]
        if not moves_in_scene:
            out.append((0.5, 0.5))
            continue
        cx = sum(m.x for m in moves_in_scene) / len(moves_in_scene)
        cy = sum(m.y for m in moves_in_scene) / len(moves_in_scene)
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        out.append((cx, cy))
    return out


# ----- One-Euro smoothing -------------------------------------------

def _smooth_centroids(
    centroids: list[tuple[float, float]],
    scenes: list[Scene],
    *,
    mincutoff: float = 0.8,
    beta: float = 0.05,
) -> list[tuple[float, float]]:
    """Apply OneEuroFilter across scene-boundary centroids.

    A single filter pass across scene centers — scenes are small enough
    (seconds–minutes) that we treat each scene midpoint as one filter
    input. Output replaces each scene's centroid with the smoothed one.

    Falls back to identity if OneEuroFilter isn't installed.
    """
    try:
        from OneEuroFilter import OneEuroFilter
    except Exception as exc:  # noqa: BLE001
        log.warning("OneEuroFilter unavailable (%s) — no smoothing", exc)
        return list(centroids)

    if not centroids:
        return []

    freq = 1.0  # one "sample" per scene
    fx = OneEuroFilter(freq=freq, mincutoff=mincutoff, beta=beta, dcutoff=1.0)
    fy = OneEuroFilter(freq=freq, mincutoff=mincutoff, beta=beta, dcutoff=1.0)

    out: list[tuple[float, float]] = []
    for i, (cx, cy) in enumerate(centroids):
        t = (scenes[i].start + scenes[i].end) / 2.0
        out.append((fx(cx, t), fy(cy, t)))
    return out


# ----- public API ----------------------------------------------------

def build_webcam_crops(
    webcam_path: Path, duration_s: float,
) -> list[CropWindow]:
    """Face-tracked crop windows for the webcam track."""
    scenes = _scenes_or_single(detect_scenes(webcam_path), duration_s)
    centroids = _face_centroid_per_scene_haar(webcam_path, scenes)
    smoothed = _smooth_centroids(centroids, scenes)
    return [
        CropWindow(start=s.start, end=s.end, cx=cx, cy=cy)
        for s, (cx, cy) in zip(scenes, smoothed, strict=True)
    ]


def build_screen_crops(
    screen_path: Path,
    cursor_csv: Path | None,
    duration_s: float,
    *,
    screen_w: int = 2560,
    screen_h: int = 1440,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    csv_to_video_offset_s: float = 0.0,
) -> list[CropWindow]:
    """Cursor-aware crop windows for the screen track. Falls back to
    screen-center when the cursor CSV isn't available."""
    scenes = _scenes_or_single(detect_scenes(screen_path), duration_s)
    centroids = _cursor_centroid_per_scene(
        cursor_csv, scenes,
        screen_w=screen_w, screen_h=screen_h,
        origin_x=origin_x, origin_y=origin_y,
        csv_to_video_offset_s=csv_to_video_offset_s,
    )
    smoothed = _smooth_centroids(centroids, scenes)
    return [
        CropWindow(start=s.start, end=s.end, cx=cx, cy=cy)
        for s, (cx, cy) in zip(scenes, smoothed, strict=True)
    ]


__all__ = [
    "Scene",
    "CropWindow",
    "detect_scenes",
    "build_webcam_crops",
    "build_screen_crops",
]
