# Future improvements

Living document. Add a dated entry when a new idea has concrete signal
behind it; promote to a GitHub issue when someone commits to implementing
it. The goal is to keep the backlog in one place without derailing the
current feature set.

---

## 0. Cursor-less clip quality (user-reported 2026-04-18)

**Current state:** when a source recording has no `cursor.csv`, several
high-value features fall back to neutral behavior:

- **Triple-intersection hard-cut** is dormant (requires face-absent ∩
  cursor-idle ∩ narration-silent; cursor-idle source is empty).
- **Speech-emphasis zooms** still fire but cx/cy defaults to screen
  center (0.5, 0.5) — they zoom into the middle of whatever's on screen
  rather than the actual UI element being discussed.
- **Element-aware OCR snap** still works when enabled, but has nothing
  to snap FROM (the input centroid is already the screen center).
- **Zoom-on-scroll** is dormant (requires cursor-idle windows).
- **Shortform screen-crop** defaults to center for all scenes — split_vstack
  ends up center-cropping screen content that was likely framed with the
  action at a specific part of the 16:9 source.

The output is **visibly worse** on cursor-less inputs. The 2026-04-17
stitched clip run confirmed this: 3 shortform clips all picked
`split_vstack` layout but the screen top-half was center-cropped, missing
the actual UI elements.

**Upgrade paths (pick one or combine):**

1. **On-screen cursor tracking from the recording itself.** Run a
   lightweight cursor detector (OpenCV template-matching against the macOS
   arrow cursor sprite; or CoreGraphics `CGEventSourceGetEventMask` on a
   captured screen stream) to reconstruct cursor.csv post-hoc. Medium
   effort, high payoff.
2. **Activity-centered screen crop as cursor fallback.** Re-use
   `src/stages/scroll_zoom.py`'s frame-diff logic but on the whole video
   (not just cursor-idle windows) to find the "most-active" region per
   scene. Centroid of recent activity becomes the shortform screen crop.
   Low effort, medium payoff.
3. **OCR-driven screen crop.** Run `src/stages/element_aware.py`
   periodically across the clip; the bounding box of the most prominent
   non-chrome text block becomes the crop centroid. Works well for
   code walkthroughs. Medium effort, high payoff for tutorial content
   specifically.
4. **LLM-driven screen-crop hints.** Add a prompt similar to zoom-hints
   that asks the LLM to emit "on-screen-region-of-interest" per
   sentence ("top-left terminal", "bottom-right button"), resolved
   against a 3×3 grid. Cheap LLM call; degrades gracefully.

Recommendation: ship **(2) activity-centered screen crop** first (shares
almost all code with scroll_zoom), then layer **(3) OCR** on top for
scenes where (2) yields a low-confidence centroid.

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

---

## 6. Shortform: music bed + sidechain ducking

**Deferred from v1 (2026-04-17).** Plan docs at
`/Users/techfreakworm/Projects/editing/plans/short-form-pipeline.md`
§7b.2 already sketch the filter graph. Add a `--music PATH` flag to the
shortform CLI, duck under narration via `sidechaincompress=threshold=0.05:
ratio=8:attack=5:release=300`, mix after loudnorm. +0.5 day work.

---

## 7. Shortform: Submagic-style animated caption templates

**Deferred from v1 (2026-04-17).** Current shortform uses the pure-Python
ASS karaoke writer (or stable-ts when installed) with fixed styling.
Plan mentions rendering a transparent overlay via Remotion (Node) for
animations beyond karaoke fills. Out of scope until we have real demand
for per-template styling.

Reference: AgriciDaniel/claude-shorts Remotion components, Aegisub
override tags (`\fad`, `\t(0,200,\fscx120\fscy120)`).

---

## 8. Shortform: multi-speaker active-speaker detection

**Deferred.** Mouth-aspect-ratio detection via MediaPipe Face Landmarker
was in the plan but reframe.py ended up using OpenCV Haar cascade
(mediapipe 0.10.x removed the `solutions` API). For solo creator content
Haar is enough. For broadcast-grade ASD on multi-guest podcasts, add
TalkNet-ASD (https://github.com/TaoRuijie/TalkNet-ASD) — significantly
more compute + training data.

---

## 9. Shortform: pre-filter candidate cost

**Current state (2026-04-18):** first real-world run scored 17 candidates
via Claude CLI at `--effort max` = ~14 min. The pipeline pre-filters
by duration (0.5× sweet_min to 2.5× sweet_max) in `pipeline.py`, but
cost still scales linearly with candidate count.

**Upgrade paths:**

1. **Heuristic prefilter stage** — drop candidates with audio-energy peak
   below a threshold, or with word count < 50 (too little substance) /
   > 250 (rambling). Runs in milliseconds; could cut candidate count
   ~50%.
2. **Two-pass LLM scoring** — first a cheap `sonnet` pass over all
   candidates to get a rough rank, then `opus --effort max` on the
   top-K. Halves the effort-max cost.
3. **Batch scoring** — send 5–10 candidates per Claude call instead of
   one. Saves prompt + thinking overhead. Requires prompt tuning to
   keep per-candidate accuracy.

Recommendation: ship (1) first (trivial), then (2) if scoring latency
still matters on real hardware.

---

## 10. Cosmetic: misleading "model=..." log line

When Claude CLI is the active backend, Stage B logs
`model=mlx-community/Llama-3.3-70B-Instruct-4bit` because it prints
`config.LLM_MODEL` regardless of dispatcher. Fixed in a follow-up — now
labels `backend=claude (opus, effort=max)` vs `backend=mlx (...)`. See
`src/stages/transcribe.py::run_analyze`.

---

## 11. Bash/subprocess timeouts killing long runs

**Current state:** running the pipeline via a background-task mechanism
with a 10-min kill timer silently terminates Python mid-run. The
workaround is `nohup python -m src.cli ... & disown`. For interactive
shells + CI this isn't an issue; documented as a gotcha in
`CLAUDE.md §6.12`.

No code change needed — the pipeline itself handles everything correctly
when given sufficient wall-clock.
