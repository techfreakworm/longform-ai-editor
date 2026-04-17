# Future improvements

Living document. Add a dated entry when a new idea has concrete signal
behind it; promote to a GitHub issue when someone commits to implementing
it. The goal is to keep the backlog in one place without derailing the
current feature set.

---

## 1. Face-aware PIP positioning — auto-detect `cam_face_x/y`

**Current state (2026-04-17):** `config.PIP_FACE_X` / `PIP_FACE_Y` are
**static user offsets** (default `0.5, 0.5`). Before scaling into the
circle, `render.py` crops the webcam to a square of side `min(iw, ih)`
centered at `(cam_face_x * iw, cam_face_y * ih)`. Works perfectly when
the speaker sits in a fixed spot; requires manual retune if seating
position changes between recordings.

**Upgrade path:** one-pass face detection over `cam.mov`, take the
median face center across sampled frames, persist as metadata alongside
the render inputs so downstream renders reuse it without re-detecting.

### Option matrix

| Option | Runtime | Accuracy | macOS ARM native | Notes |
|--------|--------:|---------:|:----------------:|-------|
| **Apple Vision `VNDetectFaceRectanglesRequest`** | ~2 ms/frame on M5 | Excellent for well-lit frontal face | ✅ (CoreImage-backed) | PyObjC binding already in use for `screen_flash.py`; zero extra install. Preferred. |
| **mediapipe Face Detection** | ~4 ms/frame | Excellent | ⚠️ x86_64 wheels only on PyPI as of 2026; must install from `mediapipe-silicon` fork or build from source | Cross-platform if ever needed. |
| **InsightFace (buffalo_sc / scrfd)** | ~8 ms/frame on CPU | State-of-the-art (handles profile, occlusion) | ✅ (onnxruntime-arm64) | Overkill for solo creator with frontal camera. Good for group shots. |
| **OpenCV Haar cascade (`haarcascade_frontalface_default.xml`)** | ~1 ms/frame | Fair — misses under poor lighting | ✅ (already in requirements) | Cheapest. Baseline-tier accuracy. |
| **OpenCV DNN (Caffe / YuNet)** | ~3 ms/frame | Very good | ✅ | Middle ground: better than Haar, simpler than mediapipe. YuNet 2023 update is compact + fast. |

### One-pass median vs per-frame tracking

**One-pass median** (recommended first step):
- Sample every Nth frame (e.g. 1 frame/sec = cheap).
- Run detector; collect face-center coordinates.
- Reject outliers (detector false positives); take median of the rest.
- Write to `work/face_center.json`: `{"x": 0.42, "y": 0.48}`.
- `render.py` reads that file; falls back to config defaults if missing.

**Why median, not mean:** median is robust to detector misses and
occlusion frames (hand in front of face, speaker looks down). A single
bad frame shouldn't drag the PIP off-center for the whole render.

**Per-frame tracking** (only if needed later):
- Keyframed crop expression driven by smoothed per-frame face center.
- Requires generating an ffmpeg sendcmd / zoompan-style expression or
  pre-rendering the cropped webcam track.
- Useful if speaker physically moves during the clip (e.g. standing
  tutorials, walking demos). Not needed for seated tutorials.

### Proposed implementation sketch

```python
# src/stages/detect_face.py  (new stage, runs between sync and analyze)
def detect_face_center(cam_path: Path, sample_every: float = 1.0) -> tuple[float, float]:
    """Return median face center in [0,1] coords using Apple Vision.
    Caches result to work/face_center.json (content-hashed by cam_path
    mtime + size) so subsequent renders are instant."""
    # Apple Vision path (PyObjC):
    #   1. ffmpeg -i cam_path -vf fps=1/sample_every -f image2pipe -vcodec bmp pipe:1
    #   2. For each BMP: CIImage → VNImageRequestHandler → VNDetectFaceRectanglesRequest
    #   3. Take boundingBox.midX, midY; collect; return median.
    ...
```

Then wire into `RenderOptions.cam_face_x/y` via a `--auto-face` CLI
flag on `render` that reads from `work/face_center.json` when present.

### Open question

Should we do face tracking BEFORE the unify stage so that cursor-zoom
logic can be aware of the face? Probably no — face and cursor serve
different layouts (cam vs screen-full) and don't cross-reference today.

---

## 2. Feathered circle edge (soft mask)

**Current state:** `assets/circle_mask.png` is a hard-edge circle. The
1–2 px anti-aliasing from the source JPG is all we get.

**Upgrade:** generate a version of the mask with a 6–10 px Gaussian
feather at the circle edge. Softer transition reads as less clinical,
matches the Descript / Loom / Riverside PIP aesthetic.

**Implementation:** one-shot PIL script in `scripts/make_circle_mask.py`
that takes a diameter and feather radius, writes to `assets/`. Keep
the existing hard-edge mask as a default; add
`config.PIP_EDGE_FEATHER = 0` (off by default) that picks a feathered
variant when > 0.

---

## 3. Circle PIP border / drop-shadow

**Current state:** circle is drawn edge-to-edge with no visual
separator from the screen background.

**Upgrade:** optional 2–4 px border (configurable color) + subtle drop
shadow so the PIP "lifts off" the screen. Common patterns:
- Border: pre-bake into the mask PNG (slight ring on inner edge).
- Shadow: second `overlay` pass with a blurred black disc offset by a
  few pixels — computationally cheap.

Defer until there's a visual reason — for now the clean circle is enough.

---

## 4. Multi-speaker PIP (future)

Not on the roadmap but worth noting so we don't architect ourselves out
of it: if a second speaker ever joins, we'd need:
- Two face-center detections (one per speaker's webcam).
- Two circle PIPs, different corners.
- Possibly a "host fills, guest insets" layout.

The current `RenderOptions.pip_shape` / `cam_face_x/y` model would need
to become per-webcam. Log as a design note; no action.

---

## 5. Face-aware cursor-zoom avoidance

When the cursor-zoom overlaps with the PIP circle's corner, the cursor
may end up visually behind the PIP. Low priority because:
- PIP is ~320 px in a 1920×1080 frame → 3% of area.
- Cursor usually centers on the active UI element, which is rarely in
  the PIP's corner.

If this bites, the fix is to penalize zooms whose final scaled center
falls within a margin of the PIP rectangle — add to `cursor_zoom.py`.
