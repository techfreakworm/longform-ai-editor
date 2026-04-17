# CLAUDE.md — project-specific guidance for Claude Code

This file is a long-lived handoff document for Claude Code sessions working on
this repo. It covers architecture, conventions, gotchas that took real
debugging to find, and the git / authorship rules the repo owner enforces.
Read the whole thing before making changes.

If you're here because the user typed `/clear` or compacted the session —
everything you need to resume is documented below.

---

## 1. Who this is for

The owner (**Mayank Gupta**, `techfreakworm@gmail.com`) is a TPM + AI/ML + financial
engineer with 7+ years of experience. Treats this project as a personal tool
for producing YouTube tutorials end-to-end on his own M5 Max (128 GB unified
memory, macOS Darwin 25.4).

**Collaboration style he expects**:

- **Execute autonomously.** Don't ask for permission on reversible edits —
  just do the work and report. Ask first only for genuinely risky actions
  (force pushes, deleting branches, dropping state). Writing scripts, editing
  files, running tests, rendering test clips = just go.
- **Write `.py` scripts, not notebooks.** Never produce `.ipynb` files.
- **Fresh builds preferred.** Don't copy/adapt code from other projects he has
  — greenfield every time.
- **`venv` not `conda`.** `python3.12 -m venv venv && source venv/bin/activate`.
  Brew for system binaries.
- **`hf` CLI, not `huggingface-cli`.** Models cache under `~/.cache/huggingface`.

---

## 2. What this project is

**100%-local dual-track long-form YouTube editor for Apple Silicon.** No
cloud, no credits, no SaaS. Input: OBS source-record files (`ext.mov` screen +
`cam.mov` webcam + `merged.mp4` for audio) + optional `cursor.csv`. Output:
polished 1920×1080 HEVC mp4, EBU R128 loudness-normalized, with automatic
filler cuts, dead-zone speed ramps, cursor-driven zoom, and a circular
webcam PIP.

The six pipeline stages run in order:

```
Stage A (sync_clap)         clap-cue alignment  →  work/sync.json
Stage B (transcribe)        mlx-whisper         →  work/words.json
Stage B (analyze_llm)       Claude CLI or MLX   →  work/filler_cuts.json, layout_plan.json
Stage C (dead_zone_detect)  freeze ∩ silence    →  work/dead_zones.json
Stage D (unify_segments)    merge decisions     →  work/segments.json
Stage E (render)            ffmpeg composite    →  work/composed.mp4
Stage F (polish)            denoise + loudnorm  →  final.mp4
```

Each stage is **idempotent and independently invocable** via
`python -m src.cli <subcommand>` — delete any output file to force that
stage to re-run.

---

## 3. Source map (what every file does)

```
src/
├── cli.py                    argparse entry; dispatches to stage run() funcs
├── config.py                 env-overridable knobs + FILLER/LAYOUT/ZOOM_HINTS/SHORTFORM_SCORING prompts
├── pipeline.py               run_all — sequences Stage A → F (+ C.2 face-visibility)
├── stages/                   Long-form pipeline stages
│   ├── sync_clap.py          Stage A: flash + audio onset + csv event detection
│   ├── transcribe.py         Stage B.1: mlx-whisper wrapper, content-hash cache
│   ├── analyze_llm.py        Stage B.2: Claude-CLI (--effort + MCP) / MLX-server dispatcher
│   ├── dead_zone_detect.py   Stage C: freeze ∩ silence + silence_intervals side-artifact
│   ├── face_visibility.py    Stage C.2: Apple Vision face detection via PyObjC
│   ├── cursor_idle.py        helper: cursor-CSV → idle intervals (for triple-intersection)
│   ├── unify_segments.py     Stage D: merge decisions + triple-intersection hard-cut
│   ├── element_aware.py      zoom v2: paddleocr snap to UI elements (opt-in via env)
│   ├── scroll_zoom.py        zoom v2: frame-diff on cursor-idle windows (opt-in via env)
│   ├── cursor_zoom.py        Cap algorithm port + speech-emphasis zoom merge
│   ├── render.py             Stage E: chunked filter_complex + circle PIP
│   └── polish.py             Stage F: ffmpeg-normalize + optional deep-filter denoise
└── shortform/                Portrait 9:16 subpackage (v1 shipped 2026-04-17)
    ├── __init__.py
    ├── transcribe.py         parakeet-mlx preferred, mlx-whisper fallback
    ├── segment.py            self-contained multi-scale TextTiling (Hearst 1997)
    ├── score.py              Claude-CLI LLM + composite heuristics (audio/punct/length)
    ├── reframe.py            OpenCV Haar face crops + PySceneDetect scenes + OneEuroFilter
    ├── captions.py           stable-ts karaoke ASS + pure-Python fallback
    ├── render.py             4 layouts: cam_full/screen_full/split_vstack/pip
    └── pipeline.py           orchestrator + CLI subcommand + ranked-cache

cursor-tracker/               separate venv; runs at record time
├── cursor_logger.py          pynput mouse+keyboard+clap hotkey logger
├── screen_flash.py           PyObjC/AppKit multi-display white-flash helper
└── record.sh                 venv activator + launcher

assets/circle_mask.png        pre-baked grayscale alpha mask for circle PIP
.mcp.json                     repo-root MCP config (sequential-thinking) for claude -p
docs/future-improvements.md   roadmap — cursor-less quality, music bed, animated captions
docs/plans/                   per-build plan docs (dated, pinned to the session)
```

Tests mirror the source layout (`tests/test_<module>.py`). Fast tests only
need Python; slow tests (`@pytest.mark.slow`) need real ffmpeg and real
fixtures under `tests/fixtures/real/` (symlinks, gitignored).

---

## 4. Running the pipeline

### LLM backend is auto-dispatched

`src/stages/analyze_llm.py` picks the backend at call time:

1. If `claude -p` works AND `FORCE_LOCAL_LLM` is not set → **Claude CLI** (uses
   the user's existing `claude login` subscription, counts toward Max/Pro
   quota). Preferred — higher quality on layout decisions.
2. Otherwise → **local `mlx_lm.server`** (Llama 3.3 70B 4-bit, must already
   be running on `http://127.0.0.1:8080/v1`). Fallback also on any Claude
   failure (auth, timeout, malformed JSON).

**No `--json` mode on mlx_lm.server.** Grammar-constrained decoding hangs on
non-trivial prompts against Llama 3.3 70B. Plain generation + the suffix
`"Return ONLY a valid JSON object. No markdown fences, no preamble, no
commentary."` is what ships.

### Commands

```bash
# Full pipeline, fresh recording
python -m src.cli run \
    --screen ext.mov --webcam cam.mov --audio merged.mp4 \
    --cursor cursor.csv --output final.mp4

# Old recording, no clap/cursor
python -m src.cli run \
    --screen ext.mov --webcam cam.mov --audio merged.mp4 \
    --output final.mp4 --manual-offset 0

# Individual stage re-run (use after tweaking .env or the prompt)
python -m src.cli analyze --webcam cam.mov --audio merged.mp4
python -m src.cli render --screen ext.mov --webcam cam.mov --audio merged.mp4 \
    --segments work/segments.json --output work/composed.mp4

# Multi-session stitch (user has done this for April 2026 recording)
# Stitch 3 OBS sessions' files into 3 continuous streams via concat demuxer.
# See README "Multi-session recordings" section.
```

---

## 5. Config knobs (`.env` or environment)

All read in `src/config.py`. Defaults are in the source; override via
`.env` or shell.

| Var | Default | What |
|---|---|---|
| `WORK_DIR` | `./work` | Stage artifacts directory |
| `OUTPUT_DIR` | `./output` | Final polished files |
| `LLM_SERVER_URL` | `http://127.0.0.1:8080/v1` | Local MLX server |
| `LLM_MODEL` | `mlx-community/Llama-3.3-70B-Instruct-4bit` | 70B Llama 4-bit |
| `LLM_TIMEOUT_SEC` | `600.0` | Per-call timeout |
| `FORCE_LOCAL_LLM` | (unset) | Set to `1` to skip Claude CLI even if installed |
| `CLAUDE_MODEL` | `opus` | `opus`, `sonnet`, or full ID like `claude-opus-4-7` |
| `WHISPER_MODEL` | `mlx-community/whisper-large-v3-turbo` | Transcription model |
| `FREEZE_DB` / `FREEZE_MIN_SEC` | `-50.0` / `2.0` | Screen stillness |
| `SILENCE_DB` / `SILENCE_MIN_SEC` | `-30.0` / `2.0` | Audio silence |
| `CUT_MIN_SEC` | `10.0` | Dead zones ≥ this are hard-cut |
| `SPEED_8X_MIN_SEC` | `3.0` | Dead zones ≥ this < CUT are sped 8× |
| `VIDEO_BITRATE` | `12M` | Render bitrate |
| `VIDEO_RES_W` / `H` / `FPS` | `1920` / `1080` / `30` | Output geometry |
| `PIP_SHAPE` | `circle` | `circle` or `rect` |
| `PIP_DIAMETER` | `320` | Circle diameter, px |
| `PIP_FACE_X` / `Y` | `0.5` / `0.5` | Face position on webcam (0–1) |
| `CIRCLE_MASK_PATH` | `assets/circle_mask.png` | Grayscale alpha mask file |
| `RENDER_ENCODER` | `hevc_videotoolbox` | Override to `libx264` if VT stalls |
| `LOUDNESS_TARGET` | `-14.0` LUFS | YouTube-matched |
| `TRUE_PEAK` | `-1.5` dBTP | Headroom |
| `LOUDNESS_RANGE` | `11.0` LU | EBU R128 LRA |
| `CLAUDE_EFFORT` | `max` | Claude CLI `--effort` level (low/medium/high/xhigh/max) |
| `USE_SEQUENTIAL_THINKING` | `1` | Prepend MCP sequential-thinking instruction to prompts |
| `CLAUDE_MCP_CONFIG` | `<repo>/.mcp.json` | MCP config path passed to `claude -p` |
| `FACE_SAMPLE_RATE_HZ` | `2.0` | Apple Vision sample rate for face-visibility |
| `FACE_ABSENT_MIN_SEC` | `2.0` | Min continuous absence to emit interval |
| `CURSOR_IDLE_MIN_SEC` | `2.0` | Min no-move window for cursor-idle |
| `USE_ELEMENT_AWARE_ZOOM` | (off) | Snap zoom centroids to OCR'd UI elements (paddleocr) |
| `ELEMENT_SNAP_MAX_PX` | `150.0` | Max snap distance in pixels |
| `USE_SCROLL_ZOOM` | (off) | Frame-diff zoom-on-scroll during cursor idle |
| `SCROLL_ZOOM_SAMPLE_RATE_HZ` | `2.0` | Scroll-zoom sample rate |
| `SCROLL_ZOOM_DIFF_THRESHOLD` | `0.04` | Min fraction of changed pixels to trigger |

---

## 6. Key gotchas (hard-won debugging lessons)

These are the bugs that took real time to diagnose — avoid re-introducing.

### 6.1 `alphamerge` with a looped mask needs `shortest=1`

The circle PIP loads the mask via `movie=...,loop=loop=-1:size=1` (infinite
loop — required so the single-frame PNG spans the segment). ffmpeg's
`alphamerge` inherits framesync defaults, where `eof_action=repeat` means the
output keeps emitting the last frame after the shorter input EOFs. With the
mask looped infinitely, alphamerge never sees the mask EOF → output runs
forever → downstream concat filter waits → muxer queue fills → **OOM kill**.

**Fix** (already in `render.py`): `alphamerge=shortest=1` so the output ends
with the webcam trim. Do not remove this flag.

### 6.2 `mlx_lm.server` + `response_format=json_object` hangs

Grammar-constrained decoding on Llama 3.3 70B reproducibly hangs on non-trivial
prompts. The server sends HTTP 200, logs `Prompt processing progress: 1/1`,
then CPU goes idle and never writes body. **Do not enable `response_format`**.
Use plain generation + strict prompt suffix.

### 6.3 `hevc_videotoolbox` stalls on huge filter graphs

On graphs with 100+ filter chains (typical for a 30-min edit with 100+
segments), the hardware HEVC encoder on VideoToolbox stalls — video frames
pile up in the muxer output queue faster than they can be written:
`N buffers queued in out_#0:0`. Eventually OOMs.

**Fix** (already in `render.py`): chunked rendering. Batch segments into
`chunk_size=30` (configurable in `RenderOptions`), render each chunk to an
intermediate mp4, then concat-demuxer + `-c copy` at the end. The concat
demuxer's stream-copy path is unaffected by graph complexity.

Escape hatch: `RENDER_ENCODER=libx264` forces software encoding (slower but
rock-solid).

### 6.4 `atempo` accumulates drift

atempo is a stateful resampler. Each invocation produces slightly fewer
samples than `input_duration / speed` due to interpolation. Over 70+
speed-ramped segments, this compounds to ~1–2% total duration loss (e.g.,
31:58 theoretical → 27:58 actual on the April 2026 long recording).

Tolerable for tutorials; documented in README limitations. If frame-exact
output is ever required: bump `CUT_MIN_SEC` so more dead zones are hard-cut
instead of ramped, or replace atempo with rubberband (GPL-2.0+).

### 6.5 concat demuxer resolves relative paths from the LIST FILE's dir

Not from cwd. `_concat_chunks_copy` uses `p.resolve()` to write absolute
paths into the list file. Do not change this or the concat will fail with
`Impossible to open 'work/<doubled-path>/_chunk_...'`.

### 6.6 Whisper pads tokens with up to 1.3 s of silence

A word's `end - start` > 1.3 s almost always means Whisper tokenized real
speech + trailing silence into one token. Do not treat such tokens as
fillers — the filler prompt has an explicit carve-out for this.

### 6.7 pynput global hotkeys need Accessibility + Input Monitoring

System Settings → Privacy & Security → grant both to the terminal that runs
`cursor-tracker/cursor_logger.py`. `install_cursor_tracker.sh` prints a
reminder.

### 6.8 Multi-monitor cursor coordinates are in macOS global space

pynput returns coordinates in the global Quartz coordinate system. If your
recorded display isn't the primary, pass `--origin-x` / `--origin-y` to
`run` / `unify` so cursor positions normalize to the recorded-display
coordinate space. README has a table of common positions.

### 6.9 sweep-line interval intersection needs a per-detector Counter

`dead_zone_detect.py`'s sweep-line used a `set()` of active detectors; with
abutting intervals (e.g., two consecutive freeze detections), the same
detector opened and closed at the same timestamp, causing spurious coverage
gaps. Fixed by using `collections.Counter` to track per-detector open counts.

### 6.10 Stage A trim is NOT precise — keyframe-snapped

The optional `--trim` on Stage A uses `-c copy`, which snaps to keyframes.
Real sub-second precision comes from Stage E's per-segment `trim=start=X:end=Y`
filter. Don't rewrite Stage A trim to re-encode — it's stream-copy on purpose.

### 6.11 `mediapipe.solutions` is GONE in 0.10.x

`mediapipe >= 0.10` removed the legacy `mp.solutions.face_detection`,
`mp.solutions.face_mesh`, etc. Only `mediapipe.Image`, `ImageFormat`,
and `tasks` survive. The Tasks API requires downloading a `.task` model
file per detector (e.g. `blaze_face_short_range.tflite`) — no
auto-download like the old solutions API.

**Fix in `src/shortform/reframe.py`**: switched to OpenCV Haar cascade
via `cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'`. No
extra dep (cv2 is base), no model download, plenty accurate for
frontal-face PIP re-centering on a solo creator.

Legacy symbol `_face_centroid_per_scene_mediapipe` is aliased to the
Haar impl so tests that patched the old name keep working.

### 6.12 Bash-tool 10-min timeout kills long background runs

The Bash tool has a hard 10-minute (600000ms) kill timer. Setting
`run_in_background: true` does NOT detach the child — after 10 min the
whole process group gets SIGTERM. A 30-min shortform scoring pass dies
silently with no error in the log.

**Workaround**: wrap with `nohup` + `disown` so the Python survives the
bash-tool's timeout:
```bash
nohup python -m src.cli shortform ... > /tmp/shortform.log 2>&1 &
disown
```
Tail `/tmp/shortform.log` via the Monitor tool to watch progress.

### 6.13 Shortform scoring cost scales with candidate count

A 34-min clip segments into ~26 candidates from multi-scale TextTiling.
Each scored with Claude CLI `--effort max` + sequential-thinking MCP
takes ~50s/candidate → 22 min wall-clock.

Mitigations already in place:
- **Pre-filter by duration** in `src/shortform/pipeline.py` — drops
  candidates outside `[0.5 × min_sec, 2.5 × max_sec]` before LLM cost.
  Typically trims 26 → 15-17.
- **Ranked cache** (`work/<stem>/shortform_ranked.json`) — survives
  late-stage crashes so re-runs skip the LLM pass entirely.

Further optimizations still open (see
`docs/future-improvements.md §9`): heuristic prefilter by audio energy,
two-pass cheap-then-expensive LLM, batch multi-candidate scoring.

### 6.14 Shortform quality degrades on cursor-less sources

Reported 2026-04-17 after the 34-min stitched clip's shortform run.
Without `cursor.csv`, the screen crop for `split_vstack` defaults to
center — missing UI elements that were framed off-center in the 16:9
source. Triple-intersection hard-cut, zoom-on-scroll, and speech-emphasis
zooms also degrade.

Future-improvements doc §0 sketches four upgrade paths (activity-centered
crop, OCR, post-hoc cursor detection, LLM region-of-interest). For now,
RECORD WITH CURSOR if you care about shortform output quality on screen
content.

---

## 7. Testing

- **Run fast tests**: `pytest -m "not slow"` (~30 s, 208 tests).
- **Run all**: `pytest` (~1 min, includes ffmpeg integration + optional LLM).
- **Slow tests**: marked `@pytest.mark.slow`. They exercise real ffmpeg, real
  fixture files, real Claude / MLX server. They **gracefully skip** if
  fixtures or servers are missing — only fail on real errors. Don't change
  that behavior.
- **Real fixtures live under `tests/fixtures/real/`**. Gitignored (large binary
  files). Symlinks are OK to commit.

When you add a feature:

1. Write unit tests in `tests/test_<module>.py`. Match existing style
   (dataclass test helpers like `S()` / `Z()` in `test_render.py`).
2. If behavior involves ffmpeg, add a `@pytest.mark.slow` integration test
   that actually invokes ffmpeg with lavfi sources — catches filter-graph
   syntax errors that unit tests on strings miss.
3. Run `pytest -m "not slow" -q` before committing; `pytest` before pushing.

---

## 8. Git / authorship rules (the user enforces these strictly)

- **Sole author**: `Mayank Gupta <techfreakworm@gmail.com>`. No Co-Authored-By
  lines, no "Generated with Claude Code" footer. He wants the commits to
  show as his.
- **Use HEREDOCs for multi-line commit messages** to preserve formatting:
  ```bash
  git commit -m "$(cat <<'EOF'
  subject line under 70 chars

  body explaining the why (not the what — the diff shows what).
  EOF
  )"
  ```
- **Never use `--no-verify`, `--no-gpg-sign`, or `-c commit.gpgsign=false`** unless
  he specifically asks. If a pre-commit hook fails, investigate. (Exception:
  this repo has no hooks configured as of 2026-04-17, so this is mostly
  precautionary.) — **Actually check this: I pass `-c commit.gpgsign=false`
  in my commits in this session because of a local signing config issue. If
  he tells you to stop, stop.**
- **Remote**: `git@github.com:techfreakworm/longform-ai-editor.git`, branch `main`.
- **Push after each logical unit of work** unless he says otherwise.
- **Never commit `.env`, fixtures, or model files**. `.gitignore` covers
  `venv/`, `work/`, `output/`, `*.mp4/mov/wav`, `tests/fixtures/*.mkv` etc.

---

## 9. Decision log (why is it this way)

In reverse chronological order. Delete entries older than 6 months when they
stop informing current work.

### 2026-04-18

- **Shortform (portrait 9:16) pipeline shipped.** New subpackage
  `src/shortform/`. End-to-end: parakeet-mlx transcribe → multi-scale
  TextTiling segment → Claude-CLI + composite-heuristic score → OpenCV
  Haar face crops + PySceneDetect + OneEuroFilter reframe → stable-ts
  karaoke captions → per-layout ffmpeg (`cam_full`/`screen_full`/
  `split_vstack`/`pip`). Optional deps via `pip install '.[shortform]'`.
  Validated on the 34.7-min stitched clip: 3 × 1080×1920 HEVC shorts,
  23–34s each, full run 2-3 min (post ranked cache).
- **Mediapipe Haar fallback.** `mediapipe.solutions` gone in 0.10.x →
  reframe.py uses OpenCV Haar cascade (no model download, base dep).
  See Gotcha 6.11.
- **Ranked cache for shortform** (`shortform_ranked.json`) — crash-resume
  without repeating 20-min Claude scoring pass.
- **Duration pre-filter** in shortform pipeline cuts LLM candidate count
  ~35% before scoring.

### 2026-04-17

- **Zoom v2** — three new zoom sources merged into the cursor-driven
  ZoomSegment stream before render: (1) **speech-emphasis** via a new
  `ZOOM_HINTS_PROMPT` / `analyze_zoom_hints` LLM call emitting anchored
  windows with `strength ∈ {soft, normal, strong}`; (2) **element-aware**
  OCR snap via `element_aware.py` using paddleocr (opt-in via
  `USE_ELEMENT_AWARE_ZOOM`); (3) **zoom-on-scroll** via frame-diff on
  cursor-idle windows (opt-in via `USE_SCROLL_ZOOM`).
- **Face-visibility stage** (`face_visibility.py`) — Apple Vision
  `VNDetectFaceRectanglesRequest` at 2 Hz over `cam.mov`, emits
  `work/face_absent.json` intervals. Works on macOS only; returns empty
  on other platforms so pipelines degrade cleanly.
- **Cursor-idle intervals** (`cursor_idle.py`) — reads `cursor.csv`,
  emits `≥ CURSOR_IDLE_MIN_SEC` no-move windows.
- **Triple-intersection hard-cut** in `unify_segments.py`: face-absent ∩
  cursor-idle ∩ narration-silent becomes a hard cut regardless of
  length. Dormant when any source is missing — by design.
- **Claude CLI `--effort max` + sequential-thinking MCP.**
  `_call_via_claude_cli` adds `--effort max` and `--mcp-config
  <repo>/.mcp.json`. Gated by `USE_SEQUENTIAL_THINKING` (default on);
  prepends a prompt prefix instructing the sequentialthinking MCP tool.
  `.mcp.json` declares the `@modelcontextprotocol/server-sequential-
  thinking` MCP via npx.
- **Circle PIP** replaced rectangular PIP as the default. Alpha-masked via
  `assets/circle_mask.png` + ffmpeg `alphamerge`. Webcam face-centered crop
  BEFORE scaling prevents squashed faces at 16:9 input; `PIP_FACE_X/Y` let
  the user override off-center seating. Commit `701ad7c`.
- **Chunked rendering** replaced single-pass. Was killing long renders via
  OOM (see gotcha 6.3). Commit `4f99491`.
- **`alphamerge=shortest=1`** added — was the root cause of the long-render
  OOM, compounding with the encoder-stall issue. See gotcha 6.1.
- **Shared circle mask** via `movie=...,split=N`. Cuts file opens from 100+
  to 1 per ffmpeg invocation. Commit `4f99491`.
- **Smooth cursor zoom** replaced the snapping static-crop zoom with
  cubic-Hermite ease in/out via `scale=...:eval=frame,crop`. ffmpeg's
  `crop` re-evaluates `x`/`y` per frame but not `w`/`h`; workaround is to
  zoom on scale and pan on crop. Commit `1f168dc`.
- **LLM prompts tuned for Claude/Llama parity**. Single prompt for both
  backends; stricter quantitative rules (word duration, probability,
  gap thresholds). 208 tests passing. Commit `e08eabf`.
- **Claude CLI dispatcher** (`a350e36`). Preferred backend when available;
  graceful fallback to local MLX on any failure. See §4.

### 2026-04-16 and earlier

- **Dead-zone: intersection not union** of freeze + silence detectors —
  avoids false positives from either alone (reveal-demos with no motion
  but continuous narration; music beds with freeze-frame slides).
- **Claude Max / Pro subscription auth** preferred over API key: uses
  existing `claude login`, no per-call billing. Documented in
  `src/stages/analyze_llm.py`.
- **Dual-track sync via clap cue**, not frame-correlation. OBS source-record
  doesn't promise cross-file sync (OBS issue #4301). Three-way marker
  (visible flash + audible clap + CSV row) gives us a robust anchor point.

---

## 10. Current state (as of 2026-04-18)

- **All 6 long-form milestones done.** M0 / M0b / M1 / M2 / M3 / M4 /
  M5 / M6 all ✅.
- **Long-form v2 improvements done**: face-visibility (C.2), triple-
  intersection hard-cut, zoom v2 (speech-emphasis + element-aware OCR +
  zoom-on-scroll), Claude CLI `--effort max` + sequential-thinking MCP.
- **Shortform pipeline v1 done**: `src/shortform/` subpackage, CLI
  subcommand `python -m src.cli shortform`, 4 layouts (cam_full /
  screen_full / split_vstack / pip), stable-ts karaoke captions.
- **369 fast tests + 11 slow integration tests** passing (+152 new since
  208 baseline).
- **Real-world validated** twice:
  - Long-form on 95 s clip → 44.8 s output with zoom hints + face
    visibility running (commit `351d103`+).
  - Long-form on 34.7-min stitched clip → 27:59 polished output with 9
    zoom hints, 51 face absences, 179 segments.
  - Shortform on same 34.7-min stitched clip → 3 × 1080×1920 HEVC
    shorts, 23–34s each, all `split_vstack` layout.
- **Known quality issue**: shortform (and some long-form zoom features)
  degrade on cursor-less sources. See Gotcha 6.14 + future-improvements
  §0.
- **Open items** — all in `docs/future-improvements.md`:
  0. Cursor-less clip quality (activity / OCR / LLM fallbacks)
  1. Face-aware PIP auto-detection
  2. Feathered circle edge (soft mask)
  3. Circle border / drop shadow
  4. Multi-speaker PIP
  5. Face-aware cursor-zoom avoidance
  6. Shortform music bed + sidechain
  7. Shortform Submagic-style captions
  8. Shortform multi-speaker ASD
  9. Shortform candidate prefilter cost optimization
  10. Misleading log line (fixed in a follow-up)
  11. Bash/subprocess timeouts

---

## 11. Quick sanity checks when starting a session

1. `git status` — ensure clean / know what's in flight.
2. `git log --oneline -5` — what was the last commit about.
3. `pytest -m "not slow" -q` — baseline green.
4. Check `work/` for leftover artifacts from a previous run — they'll be
   reused (each stage is idempotent, content-hash-cached where expensive).
5. For any render debugging, `ls work/*.mp4 tmp*.txt` — failed renders leave
   the filter script (`tmp*.txt`) and chunk files (`_chunk_*.mp4`) for
   inspection.

---

## 12. Code style

- **Default to writing NO comments.** Only write a comment when the WHY is
  non-obvious (hidden constraint, subtle invariant, workaround for a known
  bug). Don't comment WHAT the code does — the identifiers should.
- **Docstrings on public functions, stages, and data classes.** The existing
  style is a one-line summary + a blank line + a paragraph on the subtle
  behavior (e.g., why shortest=1 matters, why the mask is loaded once).
  Keep that style.
- **Prefer editing existing files to creating new ones.** The test/src layout
  is stable.
- **Type hints everywhere.** `from __future__ import annotations` at the top.
- **pydantic v2** for LLM response validation. `tenacity` for retry.
- **Paths via `pathlib.Path`**, never raw strings.
- **Logging via `log = logging.getLogger(__name__)`** inside modules;
  `print(... flush=True)` only for end-user progress in CLI entry points
  (looks nice in the live log tail).
