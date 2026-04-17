# long-form-editor

**100%-local dual-track long-form + shortform YouTube editor for Apple Silicon.**
No cloud credits, no per-render fees, no SaaS. Runs on your machine end-to-end.

```
OBS session (ext.mov + cam.mov + merged.mp4 [+ cursor.csv])
                         ↓
   ┌──────────────────────────────────────────────────────────────┐
   │ sync → transcribe → LLM (filler + layout + zoom hints)       │
   │      → dead-zone detect → face-visibility → unify            │
   │      → render (chunked) → polish (loudnorm)                  │
   └──────────────────────────────────────────────────────────────┘
                         ↓
                     final.mp4   (1920×1080, HEVC, EBU R128)
                         │
                         │ optional
                         ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ shortform: transcribe → topical segment → LLM score + rank   │
   │          → reframe (face + cursor) → karaoke captions        │
   │          → render (4 layouts, 9:16 portrait)                 │
   └──────────────────────────────────────────────────────────────┘
                         ↓
                     N × 1080×1920 HEVC shorts
```

---

## What it actually does (real measurements)

**Short clip** (95.75 s test session, M5 Max): polished **44.8 s** 1920×1080 HEVC
in ~8 s of render wall-clock. Pipeline now also invokes Claude CLI with
`--effort max` + sequential-thinking MCP — 3 LLM calls (filler / layout /
zoom hints) take ~60 s total on a clip this small.

**Long clip** (34.7 min stitched 3-session recording, M5 Max): polished
**27:59** 1920×1080 HEVC, **29 min wall-clock total**. 3664 words transcribed
(~60 s via mlx-whisper on GPU), Claude CLI LLM phase with MCP `~15 min`
(6 filler cuts + 14 layout segments + **9 zoom hints**), face-visibility
detected 51 absence intervals, 84 dead zones → 179 segments rendered in
7 chunks of ≤30 each + concat-demuxer stitch.

**Shortform run** (same 34.7-min stitched clip → 3 portrait shorts): total
**~50 min** first run, 2–3 min on re-runs (ranked-cache). Scoring 17
candidates via Claude CLI `--effort max` is the long leg at ~14 min;
face-crop + 3 renders ~2 min combined. Output: 3 × **1080×1920 HEVC**
(23–34 s each) with karaoke captions burned in.

| Stage | Role | Status | Tests |
|---|---|---|---|
| M0 · scaffold | install scripts, venv, CLI | ✅ | — |
| M0b · cursor tracker | record-time cursor/click/clap log | ✅ | — |
| M1 · Stage A sync | clap-cue alignment, CSV ↔ video time | ✅ | 18 |
| cursor_zoom (Cap port) | AGPL algorithm port for auto-zoom | ✅ | 22 |
| M2 · Stage B transcribe + LLM | mlx-whisper + Claude CLI (effort=max + MCP) | ✅ | 38 |
| zoom v2 — speech-emphasis | LLM-driven "look at this" zoom markers | ✅ | 19 |
| zoom v2 — element-aware OCR | paddleocr snap to UI elements (opt-in) | ✅ | 9 |
| zoom v2 — zoom-on-scroll | frame-diff zoom during cursor idle (opt-in) | ✅ | 12 |
| M3 · Stage C dead-zone detect | freeze ∩ silence + silence side-artifact | ✅ | 32 |
| Stage C.2 — face visibility | Apple Vision face absence detection | ✅ | 7 |
| cursor-idle intervals | CSV → no-move windows for triple-intersect | ✅ | 8 |
| M4 · Stage D unify | merge + triple-intersection hard-cut | ✅ | 53 |
| M5 · Stage E render | chunked filter_complex + circle PIP | ✅ | 56 |
| M6 · Stage F polish | EBU R128 loudnorm + optional denoise | ✅ | 10 |
| **Shortform v1** | portrait 9:16 pipeline (4 layouts) | ✅ | 88 |

**369 fast tests + 11 slow integration tests** (real ffmpeg / LLM / fixtures).

---

## Quick start

```bash
git clone git@github.com:techfreakworm/longform-ai-editor.git ~/Projects/long-form-editor
cd ~/Projects/long-form-editor

# 1. install system deps + Python deps + hf login (~1 min on warm macOS)
./scripts/install.sh

# 2. download models (Whisper + Llama 3.3 70B, ~41 GB, one time)
./scripts/download_models.sh

# 3. set up cursor tracker (separate venv, needs Accessibility perms once)
./scripts/install_cursor_tracker.sh
```

After the `install_cursor_tracker.sh` run, open **System Settings → Privacy & Security**
and grant your terminal **Accessibility** + **Input Monitoring**, then fully
quit (`Cmd+Q`) and reopen that terminal.

---

## Record a session

Two terminals running in parallel:

**Terminal 1 — OBS:**

1. Install the [`obs-source-record`](https://github.com/exeldro/obs-source-record/releases)
   plugin (one-time, download the macOS arm64 `.pkg`).
2. Attach a Source Record filter to your Display Capture (writes `ext.mov`) and
   your Video Capture Device (writes `cam.mov` with mic).
3. Start OBS's normal recording too (writes `merged.mp4` — we only use its audio).
4. Press Record.

**Terminal 2 — cursor tracker:**

```bash
cd ~/Projects/long-form-editor
./cursor-tracker/record.sh ~/sessions/episode_01.csv
```

**Within 2-3 seconds** of starting both, press **`Ctrl+Option+Cmd+K`** and
**clap your hands audibly**. Both monitors flash briefly (so the flash is
captured regardless of which display OBS is recording) and a `clap` row is
written to the CSV. This three-way marker (visible flash + audible clap + CSV
row) is what lets the sync stage align everything later.

Record your tutorial. When done:
- `Ctrl+C` the cursor tracker → CSV saved
- Stop OBS → `.mov`/`.mp4` files written

---

## Run the pipeline

### LLM backend — Claude CLI (preferred) or local MLX

All long-form + shortform LLM calls (filler cuts, layout plan, zoom hints,
shortform scoring) auto-dispatch based on what's available:

| Condition | Backend |
|---|---|
| `claude` on PATH AND `FORCE_LOCAL_LLM` not set | Claude Code CLI (`claude -p --effort max --mcp-config .mcp.json`) |
| Claude unavailable OR `FORCE_LOCAL_LLM=1` | Local `mlx_lm.server` |

For the **local** path, start the server in a separate terminal (~60 s to load
Llama into memory):

```bash
source venv/bin/activate
mlx_lm.server --model mlx-community/Llama-3.3-70B-Instruct-4bit --port 8080
```

For the **Claude** path, just make sure `claude -p --help` works — no server
needed. Calls count toward your Claude Max/Pro subscription.

### `--effort max` + sequential-thinking MCP

When Claude CLI is used, the dispatcher passes `--effort max` (highest
thinking budget) and loads `.mcp.json` at the repo root, which declares
the sequential-thinking MCP server. Prompts get a prefix instructing
Claude to use the `sequentialthinking` tool before answering — meaningful
quality lift on layout and zoom-hint decisions at zero marginal cost for
Claude Max/Pro subscribers.

Gate with `USE_SEQUENTIAL_THINKING=0` if you want faster turnaround or
your `.mcp.json` isn't set up. `CLAUDE_EFFORT` overrides the effort level.
See `.env.example` for the full set of knobs.

### One-command run

```bash
python -m src.cli run \
    --screen ~/Movies/ext2026-04-17\ 01-49-37.mov \
    --webcam ~/Movies/cam2026-04-17\ 01-49-37.mov \
    --audio  ~/Movies/2026-04-17\ 01-49-37.mp4 \
    --cursor ~/sessions/episode_01.csv \
    --output ~/Movies/episode_01_final.mp4
```

The pipeline runs Stage A → E in order, printing progress per stage. Final
output is a ready-to-upload `1920×1080 60fps HEVC` mp4.

### Multi-monitor cursor-zoom calibration

If your external monitor is positioned to the right (or above) your built-in
display in System Settings → Displays, pynput logs cursor coordinates in
macOS **global** space — which don't start at (0,0) for the external. Pass
the external display's dimensions and global origin so cursor zoom normalizes
correctly:

```bash
python -m src.cli run ... \
    --screen-w 2560 --screen-h 1440 \
    --origin-x 0 --origin-y -1440      # external is ABOVE the macbook display
```

Find your origin by running a short test session, looking at the `y` values
in `cursor.csv`, and inferring. Typical values:

| External monitor position | --origin-x | --origin-y |
|---|---|---|
| Default (primary display is external) | 0 | 0 |
| External to the right of macbook | 1440 or 1512 | 0 |
| External above macbook | 0 | -1440 |
| External to the left of macbook | -2560 | 0 |

---

## Run stages individually (debugging / incremental re-run)

Every stage is independently invocable. Outputs go to `./work/` (or `--work DIR`).

```bash
# Stage A — align cursor.csv time → video time via clap cue
python -m src.cli sync \
    --screen ext.mov --webcam cam.mov --cursor cursor.csv
#   → work/sync.json

# Stage B — transcribe + LLM filler/layout analysis
python -m src.cli analyze --webcam cam.mov --audio merged.mp4
#   → work/words.json, filler_cuts.json, layout_plan.json

# Stage C — dead-zone detection
python -m src.cli detect-dead \
    --screen ext.mov --webcam cam.mov --audio merged.mp4
#   → work/dead_zones.json

# Stage D — merge all decisions into canonical edit list
python -m src.cli unify --cursor cursor.csv
#   → work/segments.json

# Stage E — render final
python -m src.cli render \
    --screen ext.mov --webcam cam.mov --audio merged.mp4 \
    --segments work/segments.json \
    --output final.mp4
```

`python -m src.cli verify` runs a sanity check on the environment (binaries,
Python deps, LLM server, model cache).

---

## Running on old recordings (no clap, no cursor)

If your source files predate the clap-cue / cursor-tracker workflow (or you
simply don't want them), skip Stage A auto-sync with `--manual-offset 0`:

```bash
python -m src.cli run \
    --screen ext.mov --webcam cam.mov --audio merged.mp4 \
    --output final.mp4 \
    --manual-offset 0
```

`--manual-offset 0` is safe when `ext.mov` and `cam.mov` come from the same OBS
`source-record` session — they share a wall-clock start time. Omit `--cursor`
too and cursor-zoom is simply disabled. Everything else (filler cuts, layout,
dead zones) runs normally.

## Multi-session recordings (OBS restart mid-session)

If you had to pause OBS and restart mid-recording, you'll have N sets of
`(ext, cam, merged)` files. Stitch each stream back into one continuous file
with ffmpeg's concat demuxer — no re-encoding, seconds to run:

```bash
# List the files in chronological order (one per stream).
cat > /tmp/cam.txt <<EOF
file '/Users/you/Movies/cam2026-04-12 19-03-51.mov'
file '/Users/you/Movies/cam2026-04-12 19-24-07.mov'
file '/Users/you/Movies/cam2026-04-12 19-44-28.mov'
EOF
# Same for ext.txt and merged.txt, then:
for s in cam ext merged; do
    ffmpeg -y -f concat -safe 0 -i /tmp/$s.txt -c copy /Users/you/Movies/stitched/$s.mov
done
```

Then run the pipeline on the stitched files.

## Shortform (portrait 9:16) pipeline

After a long-form render, you can pull portrait clips out of the same
source (or out of the composited `final.mp4`) for YouTube Shorts /
Reels / TikTok:

```bash
# Install the optional deps (~1.5 GB extras, first time):
#   parakeet-mlx, mediapipe, scenedetect, stable-ts,
#   sentence-transformers, ultralytics, OneEuroFilter
pip install -e '.[shortform]'

# Dual-track input (preferred — enables per-clip layout picks):
python -m src.cli shortform \
    --screen ext.mov --webcam cam.mov --audio merged.mp4 \
    --cursor ~/sessions/episode_01.csv \
    --top 3 --min-sec 30 --max-sec 60

# OR single-source (already-composited mp4):
python -m src.cli shortform --composited final.mp4 --top 3
```

**Pipeline stages** (all inside `src/shortform/`):

1. **Transcribe** — parakeet-mlx (CC-BY-4.0, ~110× realtime) if installed,
   else mlx-whisper fallback. Emits word-level timestamps + sentence
   boundaries.
2. **Topical segment** — multi-scale TextTiling on sentence-transformers
   `all-roberta-large-v1` embeddings across 8 window sizes. Produces
   candidate clips at topic boundaries. Falls back to 60 s fixed chunks
   if sentence-transformers isn't installed.
3. **Score + rank** — Claude CLI (`--effort max` + MCP) per candidate
   returns `{score, title, reason, start_offset, end_offset}`; composite
   with audio-energy / punctuation / length heuristics picks top-N.
   Results cached to `shortform_ranked.json` for fast re-runs.
4. **Reframe** — OpenCV Haar cascade per-scene face centroid (webcam) +
   cursor-activity centroid (screen) + OneEuroFilter smoothing. Falls
   back to `PIP_FACE_X/Y` + screen-center when no face / cursor.
5. **Captions** — stable-ts karaoke ASS when installed, else a
   pure-Python ASS writer with the same style (Montserrat ExtraBold,
   yellow sweep, bottom-center, hand-wrapped to ~18 chars/line).
6. **Render** — per-layout ffmpeg filter graph:
   - `cam_full` — webcam fills 1080×1920
   - `screen_full` — screen-only, cursor-centered 9:16 crop
   - `split_vstack` — screen top + webcam face bottom (TikTok style); the
     screen portion is **pre-zoomed** (default 1.6×) so code / UI remains
     legible at phone size
   - `pip` — screen main with circular webcam inset bottom-right

Layout is heuristically picked per clip from transcript cues ("look at",
"notice", "here's the thing", …). Default `split_vstack` unless the
clip has purely narrative cues (goes `cam_full`) or no webcam track
(goes `screen_full`).

### Shortform known issues

- **Cursor-less sources degrade quality.** Without `cursor.csv`, screen
  crops default to center — UI off-center in the 16:9 source gets
  cropped out of `split_vstack`'s top half. Several upgrade paths
  documented in `docs/future-improvements.md §0`. Record with the cursor
  tracker if you care about shortform output on screen content.
- **Scoring time scales with candidate count** (~50 s/clip at
  `--effort max`). Pipeline pre-filters by duration to drop obvious
  non-candidates before LLM cost. See `docs/future-improvements.md §9`
  for further optimization paths.

## Circle PIP (webcam bubble) and face offset

By default the webcam is composited as a **circular talking-head bubble** over
the screen (alpha-masked via `assets/circle_mask.png` + ffmpeg `alphamerge`).
The webcam is cropped to a face-centered square BEFORE scaling into the circle
so facial proportions are preserved even if you sit off-center.

Environment tunables (or override in `.env`):

| Var | Default | Meaning |
|---|---|---|
| `PIP_SHAPE` | `circle` | `circle` (alpha mask) or `rect` (classic rectangle) |
| `PIP_DIAMETER` | `320` | Pixels — diameter of circle PIP |
| `PIP_FACE_X` | `0.5` | Horizontal face position on webcam (0.0–1.0) |
| `PIP_FACE_Y` | `0.5` | Vertical face position on webcam (0.0–1.0) |
| `CIRCLE_MASK_PATH` | `assets/circle_mask.png` | Grayscale alpha-mask override |

If you sit to the left of frame, try `PIP_FACE_X=0.35`; to the right, `0.65`.
See [`docs/future-improvements.md`](docs/future-improvements.md) for the
auto-detection roadmap (Apple Vision / mediapipe / InsightFace options).

## Zoom v2 — three signals feeding cursor-zoom

Cursor-driven click/move zooms were v1. Three additional signals can
emit `ZoomSegment`s that merge into the cursor stream:

| Signal | How it fires | Default | Enable |
|---|---|---|---|
| **Speech-emphasis** | LLM parses transcript for deictic cues ("look at this", "right here", "notice", …) and returns anchored (start, end, strength) windows. Centroid = cursor position at hint start when cursor CSV exists, else screen center. | ON | Via `analyze_zoom_hints` (runs in Stage B). Disable by emptying `work/zoom_hints.json`. |
| **Element-aware OCR snap** | After cursor centroids are computed, paddleocr is run on the frame; if the nearest UI text block is within `ELEMENT_SNAP_MAX_PX` (150 px default), the centroid snaps to that element's midpoint. | OFF | `USE_ELEMENT_AWARE_ZOOM=1` + `pip install '.[zoom-ocr]'` |
| **Zoom-on-scroll** | During cursor-idle windows on the screen track, frame-diff (pixel absdiff thresholded) detects new content appearing. If the changed region is ≥ `SCROLL_ZOOM_DIFF_THRESHOLD` (default 4%) of the frame, emit a zoom centered on the change bbox. | OFF | `USE_SCROLL_ZOOM=1` |

All three are gated behind env vars so the default pipeline stays
predictable. Speech-emphasis is on by default because its LLM call has
no local compute cost.

## Face-visibility + triple-intersection hard-cut

Stage C.2 (`src/stages/face_visibility.py`) samples the webcam at
`FACE_SAMPLE_RATE_HZ` (2 Hz default) through Apple Vision
`VNDetectFaceRectanglesRequest` via PyObjC. Emits `work/face_absent.json`
with contiguous intervals ≥ `FACE_ABSENT_MIN_SEC` where no face was
detected.

Stage D (`src/stages/unify_segments.py`) intersects three signals:

- **Face absent** (≥ 2 s)
- **Cursor idle** (`src/stages/cursor_idle.py`, ≥ 2 s no movements)
- **Narration silent** (`silence_intervals.json` side-artifact from
  Stage C)

Where all three agree, the range becomes a **hard cut regardless of
length** — catches "stepped away from the computer" footage that
freeze+silence alone wouldn't.

**The triple-intersection is dormant when any source is missing** —
specifically, no `cursor.csv` ⇒ no cursor-idle source ⇒ no triple cuts.
By design: two-out-of-three agreement isn't strong enough evidence.
If you're running on old recordings, use the existing dead-zone
speed-ramp / cut pipeline (works unchanged).

## Why chunked rendering

Stage E doesn't build one giant ffmpeg filter graph for the whole edit. It
batches segments (`chunk_size=30` by default), renders each batch to its own
intermediate mp4, and stitches them at the end with the concat **demuxer**
(stream-copy, no re-encode). Reasons:

- On long edits with many `atempo`-chained speed ramps, the concat **filter**
  accumulates timestamp drift per segment — the muxer output queue piles
  up ("N buffers queued in out_#0:0") until OOM kills ffmpeg.
- `hevc_videotoolbox` stalls on very complex graphs (200+ filter chains).
  Keeping each invocation small keeps it reliable.
- The mask PNG is loaded **once per ffmpeg invocation** via `movie=...,split=N`
  rather than `movie=` per segment — cuts 100+ redundant file opens.
- `alphamerge=shortest=1` (required because the mask is infinitely looped)
  ensures each PIP segment EOFs with the webcam so the concat filter can
  close it cleanly.

`RENDER_ENCODER=libx264` is the escape hatch if `hevc_videotoolbox` still
misbehaves on a specific clip.

---

## Architecture

Brief. See [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) for milestone-level
detail and `research/` docs (one level up) for the design rationale.

### Inputs

| File | Role |
|---|---|
| `screen_raw.mkv` | Silent screen recording (no talking head). OBS Display Capture via `obs-source-record`. |
| `webcam_raw.mkv` | Face-cam with mic audio. OBS Video Capture Device via `obs-source-record`. |
| `merged.mp4` _(optional)_ | OBS merged main recording. Video is useless, but its audio track is often the cleanest mic mix. Use its audio via `--audio`. |
| `cursor.csv` _(optional)_ | `cursor-tracker/cursor_logger.py` output — every move/click/clap with timestamps. Enables auto-zoom. |

### What each stage writes to `work/`

```
work/
├── sync.json           (Stage A: cursor↔video offset, detected cue times)
├── words.json          (Stage B: word-level transcript)
├── filler_cuts.json    (Stage B: LLM-identified fillers/false-starts/repeats)
├── layout_plan.json    (Stage B: LLM-chosen cam_full/pip/screen_full segments)
├── dead_zones.json     (Stage C: freezedetect ∩ silencedetect → cut/speed regions)
├── segments.json       (Stage D: canonical merged edit list)
├── transcribe_cache/   (Stage B: cached transcription by audio content hash)
└── final.mp4           (Stage E output, or your --output path)
```

Each stage is idempotent and can be re-run independently. Delete specific
artifacts to force a re-run of that stage.

### Key design choices

- **Dual-track**, not single MKV: OBS doesn't promise sync between
  source-record outputs ([OBS issue 4301](https://github.com/obsproject/obs-studio/issues/4301)),
  so we sync the cursor.csv to video time via the clap cue rather than
  trying to align the two video files to each other (they come from OBS
  sharing a wall-clock).
- **LLM in plain text mode**, not `response_format=json_object`: mlx_lm.server's
  grammar-constrained decoding hangs reproducibly on non-trivial prompts
  against Llama 3.3 70B. Llama at `temperature=0` with a strict system
  suffix (`"Return ONLY a valid JSON object..."`) complies reliably.
- **Claude CLI as preferred LLM backend**: if `claude` is on PATH, the
  dispatcher shells out to `claude -p --model opus --output-format text` for
  filler + layout calls. Uses the existing subscription auth (no API key,
  counts toward Claude Max/Pro quota). Falls back to local MLX gracefully
  on any failure. Measured ~10% quality lift on layout decisions at zero
  marginal cost for subscribers.
- **Stream-copy-then-re-encode-in-render**: Stage A's optional `--trim` uses
  `-c copy` which snaps to keyframes; precise trim accuracy comes from
  Stage E applying the offset on its own via per-segment `trim + setpts`.
- **Dead-zone intersection**, not union: require ≥2 detectors (freezedetect +
  silencedetect) to agree. Catches "silent narrator while screen isn't
  moving" but not false positives from either alone.
- **Cursor zoom split at boundaries**: a Segment with N cursor-zoom windows
  becomes ≤2N+1 RenderSegments, each with at most one static crop. Keeps the
  ffmpeg filter graph simple (no per-frame `crop=x:y` expressions).
- **Smooth zoom via scale + static crop** (not time-varying crop): ffmpeg's
  `crop` filter re-evaluates `x`/`y` per frame but `w`/`h` only at init.
  To animate a zoom, we `scale=w=EXPR:h=EXPR:eval=frame` by a cubic-Hermite
  smoothstep `z(t)` and center-crop a fixed `(in_w × in_h)` window with
  panning `x`/`y` — gives ease-in / hold / ease-out without the snap.
- **Circle PIP via shared alpha mask**: one `movie='assets/circle_mask.png',
  scale=D:D,format=gray,loop=-1:size=1,split=N` at the top of the graph
  produces N per-segment labels consumed by `alphamerge=shortest=1`.
  `shortest=1` is load-bearing — the looped mask never EOFs, so without it
  the concat filter downstream waits forever.
- **Chunked rendering + concat demuxer**: long edits are batched into
  N-segment chunks; each chunk's graph stays small enough for
  `hevc_videotoolbox` to stream through without muxer-queue blowup. Final
  stitch is stream-copy (no re-encode). See
  [Why chunked rendering](#why-chunked-rendering).

---

## Requirements

- **Apple Silicon** (M1 or newer). Uses MLX, VideoToolbox, ScreenCaptureKit.
  Intel Macs and non-Mac platforms are not supported.
- **macOS 13 Ventura or newer.**
- **Python 3.11+** (Homebrew python@3.12 recommended).
- **FFmpeg 6+** with `hevc_videotoolbox` encoder.
- **~40 GB disk** for Llama 3.3 70B 4-bit weights.
- **64 GB unified memory minimum** (for 70B Llama + Whisper in parallel).
  128 GB is roomier and leaves headroom for browsers.

`./scripts/install.sh` installs all system + Python deps.

---

## Limitations (honest)

- **DeepFilterNet denoise is opt-in.** `deep-filter` binary isn't installed
  by the pipeline's install script — grab the `aarch64-apple-darwin` release
  from https://github.com/Rikorose/DeepFilterNet/releases and put it on
  PATH. Without it, Stage F silently skips denoise and still does loudnorm.
- **Short clips may land up to ±2.5 LU from the -14 LUFS target.** EBU R128
  auto-switches to dynamic mode when input LRA exceeds target LRA (common
  on short samples with mixed speech + silence). Dynamic mode is the safer
  choice per the spec — it preserves peaks without clipping. For a full
  30-min tutorial the result usually lands within ±0.5 LU.
- **Long renders lose ~1–2% duration to `atempo` resampling drift.** Each
  speed-ramped segment's audio is slightly shorter than `duration / speed`
  because atempo interpolates. Over 70+ speed-ramped segments this adds up —
  the 34.7 min example above came out to 27:58 vs unify's theoretical
  31:58. Tolerable for tutorial content; if you need frame-exact output,
  bump `CUT_MIN_SEC` so more dead zones are hard-cut instead of sped up.
- **macOS only.** MLX is Apple-silicon-specific; ScreenCaptureKit is macOS-specific.
- **mlx_lm.server must run on `--port 8080`.** Override via `.env`
  (`LLM_SERVER_URL=http://host:port/v1`). Irrelevant if you use the Claude CLI
  backend.
- **Constrained-decoding JSON mode avoided.** If you swap in a different
  backend (vLLM, Ollama) that handles `response_format=json_object` well, the
  `call_llm_json()` helper would benefit from re-enabling it. Currently
  disabled as a reliability measure against mlx_lm.server's hangs.
- **Circle-PIP face position is a static offset**, not per-frame tracking.
  If you move chair-to-chair mid-clip, half your face may end up out of the
  circle. One-pass face-detection upgrade documented in
  [`docs/future-improvements.md`](docs/future-improvements.md).
- **Cursor zoom needs manual origin calibration** on multi-monitor setups.
  See the table above for common positions.
- **Dead-zone detection is heuristic.** 10–20% false positive / false negative
  rate on first runs. Tune `FREEZE_DB`, `SILENCE_DB` in `.env` after 5–10
  real sessions.
- **Cap algorithm port** in `cursor_zoom.py` is an AGPL-3.0 derivation of
  CapSoftware/Cap's Rust source. If you commercialize this pipeline, the
  cursor-zoom component is AGPL — either re-implement the algorithm from
  the documented constants alone or pay Cap's commercial license.
- **Ultralytics YOLO11 is AGPL-3.0** (only used if installed via
  `[shortform]` extras; YOLO-based person-detection code paths aren't
  in reframe.py at the moment — shortform uses Haar cascade). Fine for
  personal use; swap for RT-DETR / YOLOX before commercializing.
- **Shortform quality on cursor-less sources** is visibly worse than on
  sources with `cursor.csv`. Screen crops default to center; triple-
  intersection stays dormant; speech-emphasis zoom centroids fall back
  to screen center. Four upgrade paths tracked in
  `docs/future-improvements.md §0` — ship activity-centered crop first.
- **Shortform scoring is slow at `--effort max`.** ~50 s per candidate,
  and a 34-min source produces ~17 candidates after the duration
  pre-filter ⇒ ~14 min scoring pass. Ranked-cache speeds up re-runs.
  Further optimizations in `docs/future-improvements.md §9`.

---

## Project structure

```
long-form-editor/
├── CLAUDE.md                       # per-session guidance for Claude Code
├── IMPLEMENTATION_PLAN.md          # milestone-by-milestone build plan
├── pyproject.toml                  # deps, dev deps, pytest markers
├── .env.example                    # tunable thresholds + LLM settings
├── assets/
│   └── circle_mask.png             # grayscale alpha mask for circle PIP
├── docs/
│   └── future-improvements.md      # face-aware PIP + feathered edges + more
├── scripts/
│   ├── install.sh                  # one-shot full install
│   ├── install_cursor_tracker.sh   # record-time tracker only (separate venv)
│   ├── download_models.sh          # Whisper + Llama 3.3 70B (+ optional extras)
│   └── verify_env.py               # sanity-check all binaries + deps + models
├── cursor-tracker/                 # standalone recording helper
│   ├── cursor_logger.py            # pynput logger with clap hotkey
│   ├── screen_flash.py             # AppKit multi-display white-flash helper
│   ├── record.sh                   # venv activator + launcher
│   ├── requirements.txt
│   └── README.md
├── src/
│   ├── cli.py                      # argparse entry: sync|analyze|detect-dead|unify|render|polish|run|shortform|verify
│   ├── config.py                   # paths, thresholds, prompts, PIP knobs, --effort, MCP, face/cursor/zoom (env-overridable)
│   ├── pipeline.py                 # run_all orchestrator (A → F + C.2 face-visibility)
│   ├── stages/                     # long-form stages
│   │   ├── sync_clap.py            # Stage A
│   │   ├── transcribe.py           # Stage B.1 (+ analyze dispatch)
│   │   ├── analyze_llm.py          # Stage B.2 — Claude CLI (--effort + MCP) / MLX dispatcher, zoom_hints schema
│   │   ├── dead_zone_detect.py     # Stage C + silence_intervals side-artifact
│   │   ├── face_visibility.py      # Stage C.2 — Apple Vision face absence detection
│   │   ├── cursor_idle.py          # helper for triple-intersection
│   │   ├── unify_segments.py       # Stage D + triple-intersection hard-cut + speech zoom merge
│   │   ├── element_aware.py        # zoom v2 OCR snap (opt-in)
│   │   ├── scroll_zoom.py          # zoom v2 frame-diff (opt-in)
│   │   ├── cursor_zoom.py          # Cap algorithm port + speech-emphasis zoom merge
│   │   ├── render.py               # Stage E (chunked + circle PIP + smooth zoom)
│   │   └── polish.py               # Stage F (loudnorm, optional denoise)
│   ├── shortform/                  # Portrait 9:16 subpackage (v1 2026-04-18)
│   │   ├── __init__.py
│   │   ├── transcribe.py           # parakeet-mlx → mlx-whisper fallback
│   │   ├── segment.py              # multi-scale TextTiling
│   │   ├── score.py                # Claude-CLI + composite heuristics + ranked cache
│   │   ├── reframe.py              # Haar face crops + PySceneDetect + OneEuroFilter
│   │   ├── captions.py             # stable-ts karaoke + pure-Python ASS fallback
│   │   ├── render.py               # 4 layouts, ffmpeg filter_complex builder
│   │   └── pipeline.py             # orchestrator + layout picker + slug
│   └── utils/
│       ├── ffmpeg_helpers.py
│       ├── log_parsers.py
│       └── timecodes.py
└── tests/
    ├── fixtures/real/              # symlinks to real OBS fixtures (gitignored)
    ├── test_sync.py                # 18
    ├── test_analyze_llm.py         # 38
    ├── test_transcribe.py          # 12
    ├── test_dead_zone_detect.py    # 19
    ├── test_unify_segments.py      # 53
    ├── test_render.py              # 56
    ├── test_cursor_zoom.py         # 22
    ├── test_cursor_idle.py         # 8
    ├── test_face_visibility.py     # 7
    ├── test_zoom_v2_speech.py      # 19
    ├── test_zoom_v2_element.py     # 9
    ├── test_zoom_v2_scroll.py      # 12
    ├── test_log_parsers.py         # 8
    ├── test_timecodes.py           # 8
    └── shortform/                  # 88 tests
        ├── test_segment.py         # 15
        ├── test_score.py           # 16
        ├── test_reframe.py         # 15
        ├── test_captions.py        # 12
        ├── test_render.py          # 16
        └── test_pipeline.py        # 14
```

---

## Running tests

```bash
source venv/bin/activate
pytest                              # all (~1 min including slow)
pytest -m "not slow"                # fast only, ~30 s
pytest tests/test_render.py -v      # one module
```

Slow tests exercise:
- real ffmpeg on the real OBS fixtures (`tests/fixtures/real/*.mov`)
- the running `mlx_lm.server` for actual LLM integration

Tests gracefully skip if the server or fixtures are missing — they only fail
on real errors, not environment gaps.

---

## License

Your code: your choice. Dependencies pull in mixed licenses:
- **AGPL-3.0**: Cap algorithm port in `cursor_zoom.py` (see limitation above)
- **GPL-2.0+**: optional rubberband if you enable higher-quality audio stretch
- **CC-BY-NC-4.0**: CrisperWhisper (NOT used — commercial path via mlx-whisper)
- **Apache-2.0 / MIT / BSD**: everything else

See [`IMPLEMENTATION_PLAN.md` §4](IMPLEMENTATION_PLAN.md) for the full dependency
license table.

---

## Credits

- **Cap** (CapSoftware/Cap) — cursor-zoom algorithm, AGPL-3.0
- **mlx-whisper**, **mlx-lm** — Apple
- **Llama 3.3** — Meta
- **obs-source-record** (exeldro) — dual source capture in OBS
- **auto-editor** (WyattBlue) — silence-cut motion predicate reference
- **ClipsAI** — short-form clipper reference algorithm (superseded by this pipeline's LLM layout + dead-zone approach for long-form)
