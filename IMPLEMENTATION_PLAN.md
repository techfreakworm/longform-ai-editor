# Implementation Plan — `long-form-editor`

**Target:** concrete, milestone-by-milestone build plan for the dual-track long-form YouTube editor, matching the architecture in [`../plans/long-form-pipeline.md`](../plans/long-form-pipeline.md).

**Host:** Apple M5 Max, 128 GB unified memory, macOS Darwin 25.4 (arm64).
**Constraint:** 100% local, OSS. No cloud credits, no per-render fees.

---

## 1. Inputs & output contract

```
INPUT:
  screen_raw.mkv  (silent, from OBS Display Capture via obs-source-record)
  webcam_raw.mkv  (face-cam + mic, from OBS Video Capture Device via obs-source-record)
  cursor.csv      (optional — from cursor-tracker/, enables auto-zoom)

OUTPUT:
  final.mp4       (1080p, HEVC + AAC, -14 LUFS EBU R128, faststart)
  work/           (all intermediate artifacts, for debugging + caching)
    offset.json
    words.json
    filler_cuts.json
    layout_plan.json
    dead_zones.json
    segments.json
    composed.mp4
```

Every stage writes to `work/` and can be re-run independently. Stages are idempotent where possible.

---

## 2. Project layout

```
long-form-editor/
├── README.md                      # quick start
├── IMPLEMENTATION_PLAN.md         # this file
├── pyproject.toml                 # deps, dev deps
├── .env.example                   # LLM server URL, paths
├── scripts/
│   ├── install.sh                 # full dev env bootstrap
│   ├── install_cursor_tracker.sh  # cursor tracker only (recording sessions)
│   └── verify_env.py              # check all binaries + models present
├── src/
│   ├── __init__.py
│   ├── cli.py                     # CLI entry — python -m src.cli
│   ├── config.py                  # paths, thresholds, model IDs
│   ├── pipeline.py                # orchestrator, calls stages in order
│   ├── stages/
│   │   ├── __init__.py
│   │   ├── sync_clap.py           # Stage A — clap cue sync
│   │   ├── transcribe.py          # Stage B.1 — mlx-whisper wrapper
│   │   ├── analyze_llm.py         # Stage B.2 — filler + layout via Qwen3
│   │   ├── dead_zone_detect.py    # Stage C — multi-signal screen dead-zone
│   │   ├── unify_segments.py      # Stage D — merge all decisions
│   │   ├── render.py              # Stage E — ffmpeg filter_complex builder
│   │   ├── polish.py              # Stage F — denoise + loudnorm
│   │   └── cursor_zoom.py         # Port of Cap's algorithm
│   └── utils/
│       ├── __init__.py
│       ├── ffmpeg_helpers.py      # subprocess wrapper, ffprobe, remux
│       ├── log_parsers.py         # freezedetect/silencedetect stderr → JSON
│       └── timecodes.py           # second ↔ frame conversions
├── cursor-tracker/                # recording-time companion (standalone)
│   ├── README.md
│   ├── cursor_logger.py           # pynput-based cursor/click logger
│   ├── screen_flash.py            # tkinter clap-cue helper
│   ├── record.sh                  # launcher
│   └── requirements.txt
├── tests/
│   ├── fixtures/                  # sample MKVs (to be created)
│   ├── test_sync.py
│   ├── test_dead_zone.py
│   ├── test_unify.py
│   └── test_render.py
└── examples/
    └── sample_session_config.toml
```

---

## 3. Milestones

Seven milestones, M0 through M6. Each milestone has:
- **DoD** (definition of done) — what passing looks like
- **Files touched**
- **Test(s) to pass**
- **Gotchas** — known risks lifted from research

Estimate 1–3 evenings per milestone on M5 Max.

### Status as of last commit

| Milestone | Status | Tests |
|---|---|---|
| M0 — scaffold | ✅ done | — |
| M0b — cursor tracker | ✅ done | — |
| M1 — Stage A (sync) | ✅ done | 18 |
| cursor_zoom (Cap port) | ✅ done | 22 |
| M2 — Stage B (transcribe + LLM) | ✅ done | 32 |
| M3 — Stage C (dead-zone detect) | ✅ done | 32 |
| M4 — Stage D (unify segments) | ✅ done | 46 |
| M5 — Stage E (render) | ✅ done | 30 |
| M6 — Stage F (polish) | ✅ done | 10 |

End-to-end verified against real OBS fixtures: 95.75 s source →
42.99 s 1920×1080 HEVC 60fps in 7.6 s render wall-clock.

### M0 — Skeleton + environment verified  ✅ DONE

**DoD.** `scripts/install.sh` runs to completion. `scripts/verify_env.py` passes. Empty `python -m src.cli --help` prints usage. Cursor tracker works (see M0b).

**Files:**
- `pyproject.toml` — deps list
- `scripts/install.sh` — brew + pip + model pre-pull
- `scripts/verify_env.py` — checks ffmpeg/auto-editor/mlx_lm/hf CLI present, MLX server reachable, Qwen3 model cached
- `src/__init__.py`, `src/cli.py` with Typer/argparse stub
- `src/config.py` with PATHS, THRESHOLDS, MODEL_IDS
- `.env.example`

**Test:** `python scripts/verify_env.py` exits 0 on a fresh clone after install.

**Gotchas:**
- `obs-source-record` plugin installs outside pip — script downloads the `.pkg` but user must run it manually (macOS Gatekeeper).
- `hf auth login` needs interactive token entry — script detects if already authed and skips.

### M0b — Cursor tracker (parallel, usable immediately)  ✅ DONE

**DoD.** Run `cursor-tracker/record.sh` and it logs moves/clicks to CSV. Hotkey ⌃⌥⌘K flashes screen + marks CSV with `event=clap`.

**Files:**
- `cursor-tracker/cursor_logger.py` — pynput Listener + hotkey
- `cursor-tracker/screen_flash.py` — tkinter fullscreen white window for 80ms
- `cursor-tracker/record.sh` — activates venv, runs logger
- `cursor-tracker/requirements.txt` — pynput
- `scripts/install_cursor_tracker.sh` — creates cursor-tracker/venv, installs pynput, prints macOS accessibility instructions

**Test:** manually record 10 s of cursor movement, verify CSV has ≥ 100 move events, one clap row at the hotkey moment, and that a visible flash was captured by OBS.

**Gotchas:**
- macOS requires **Accessibility** + **Input Monitoring** permissions for pynput. Script documents this; cannot be automated.
- tkinter `Tk()` on macOS sometimes needs `root.update()` before `root.after()` for the flash to actually paint. Covered in `screen_flash.py`.
- The `<cmd>` key in pynput hotkeys is the Command key; combining with `<ctrl>+<alt>+<cmd>` is a 4-key chord that won't collide with anything common.

This milestone can ship independently of M1–M6 so you can start recording with the tracker immediately.

### M1 — Stage A: sync working end-to-end  ✅ DONE

**DoD.** Given two raw mkv inputs with a recorded clap cue, `python -m src.cli sync --screen <path> --webcam <path>` emits `offset.json` and trimmed `screen_synced.mkv` + `webcam_synced.mkv` that align to within ±1 frame @ 60 fps.

**Files:**
- `src/stages/sync_clap.py` — librosa onset on webcam, OpenCV luminance-diff on screen, compute offset, trim both with ffmpeg
- `tests/fixtures/` — two 30 s test mkvs with a known clap at t=2.0 s (human-generated once, committed)
- `tests/test_sync.py`

**Test:** `pytest tests/test_sync.py` — verifies detected offset is within 1 frame of ground truth.

**Gotchas:**
- Librosa onset_detect returns candidates sorted by time, not amplitude. Rank by amplitude in a 50 ms window and take top-1.
- If OBS dropped a frame at record start, the luminance-diff index needs `/fps` conversion — use `cv2.CAP_PROP_FPS` not a hardcoded 60.
- When `offset < 0`, screen started after webcam — trim screen instead of webcam. Handle both signs.
- Fall back to manual offset entry via `--manual-offset 0.163` flag if auto-detection fails.

### M2 — Stage B: transcribe + analyze  ✅ DONE

**DoD.** `python -m src.cli analyze --webcam webcam_synced.mkv` emits:
- `words.json` (mlx-whisper output with word timestamps)
- `filler_cuts.json` ([{start, end, reason}, ...])
- `layout_plan.json` ([{start, end, layout ∈ {cam_full,pip,screen_full}}, ...])

**Files:**
- `src/stages/transcribe.py` — `mlx_whisper.transcribe()` wrapper, caches by audio hash
- `src/stages/analyze_llm.py` — two prompts, retries, JSON schema validation
- `src/config.py` — prompt templates
- `tests/test_analyze.py`

**Test:** on a 60 s test clip with known fillers, verify ≥ 80% filler recall. Layout plan covers entire duration with no gaps and only valid enum values.

**Gotchas:**
- mlx_lm.server must be running (`mlx_lm.server --model mlx-community/Llama-3.3-70B-Instruct-4bit --port 8080`). Script checks `/v1/models` endpoint first.
- `response_format={"type": "json_object"}` isn't always respected — validate with `jsonschema` and retry up to 3 times with stricter prompt.
- Qwen3's thinking mode can inject `<think>...</think>` tags into the output. Strip before json.loads.
- Layout plan must cover entire duration — if LLM returns gaps, post-process to fill with previous segment's layout.

### M3 — Stage C: dead-zone detect  ✅ DONE

**DoD.** `python -m src.cli detect-dead --screen screen_synced.mkv --webcam webcam_synced.mkv --words words.json` emits `dead_zones.json = [{start, end, action ∈ {cut, speed@4x, speed@8x}}, ...]`.

**Files:**
- `src/stages/dead_zone_detect.py` — runs 4 detectors in parallel, intersects, classifies
- `src/utils/log_parsers.py` — parse ffmpeg stderr (freezedetect/silencedetect) to JSON
- `tests/test_dead_zone.py`

**Detectors (each returns list of `(start, end)` intervals):**
1. `freezedetect n=-50dB d=2` on screen
2. `silencedetect noise=-30dB d=2` on webcam audio
3. `auto-editor motion:threshold=0.02,blur=9,width=400 --export premiere` → parse XML
4. LLM transcript cues ("while that installs", "let me skip") via Qwen3

**Intersection rule:** keep ranges where ≥ 2 detectors agree and duration > 2 s.

**Classification:**
- duration > 10 s → action="cut"
- 3 s < duration ≤ 10 s → action="speed@8x"
- 2 s < duration ≤ 3 s → action="speed@4x"

**Test:** on a hand-crafted fixture with 3 known dead zones (silent+frozen, silent+scrolling install, speaking+nothing), verify detector labels each correctly.

**Gotchas:**
- `freezedetect` threshold tuning is the biggest variable. `-60 dB` is too sensitive (cursor blink triggers), `-40 dB` misses subtle motion. `-50 dB` is the documented default but tune per display (Retina displays have different text rendering).
- `auto-editor --export premiere` emits FCP7 XML; parse with ElementTree, extract `<clipitem><start>` values.
- Expect 10–20% false positives on first sessions. Add a `--tune` mode that surfaces candidates for human review.

### M4 — Stage D: unify segments  ✅ DONE

**DoD.** `python -m src.cli unify` consumes all Stage B+C outputs and emits `segments.json` — a complete edit list covering every second of the final timeline with resolved `(in, out, speed, layout, cursor_zoom?)` tuples.

**Files:**
- `src/stages/unify_segments.py`
- `tests/test_unify.py` (heaviest unit test coverage — this is the decision brain)

**Algorithm:**
1. Start with `layout_plan.json` as the backbone (covers entire duration).
2. Remove intervals marked in `filler_cuts.json` (ripple delete from timeline).
3. For each `dead_zones.json` entry:
   - action="cut" → remove from timeline
   - action="speed@N" → keep, set `speed=N` on the segment
4. Merge adjacent same-layout same-speed segments.
5. If `cursor.csv` present, annotate each screen-visible segment with zoom segments from `cursor_zoom.py` (Cap algorithm port).
6. Validate: no gaps, no overlaps, every segment has all required fields.

**Test:** golden file — hand-constructed input triplet + expected segments.json. Byte-compare.

**Gotchas:**
- Ordering matters: apply layout → cuts → dead-zones in that order. Applying dead-zones before cuts can produce incorrect speed assignments on segments that filler-removal would otherwise delete entirely.
- Float arithmetic on timestamps drifts — use `Decimal` or round to millisecond before comparing.
- The golden test needs to survive small floating-point differences — use `assertAlmostEqual` with 3 decimal places for times.

### M5 — Stage E: render  ✅ DONE

**DoD.** `python -m src.cli render --segments segments.json` invokes a single `ffmpeg` call that produces `composed.mp4` — 1080p HEVC with all cuts, speed ramps, layout switches, and cursor zooms applied.

**Files:**
- `src/stages/render.py` — filter_complex builder
- `src/stages/cursor_zoom.py` — Cap algorithm port, emits crop+scale expressions
- `src/utils/ffmpeg_helpers.py` — subprocess runner with proper quoting
- `tests/test_render.py`

**Filter graph structure (per segment):**
```
# Screen branch
[0:v]trim=s:e,setpts=(PTS-STARTPTS)/speed,crop={cursor_zoom_x}[sN]

# Webcam video branch
[1:v]trim=s:e,setpts=(PTS-STARTPTS)/speed[wN]

# Webcam audio branch (pitch-preserving)
[1:a]atrim=s:e,asetpts=PTS-STARTPTS[aN_raw]
# speed ∈ {1, 4, 8} → 0, 2, 3 atempo filters chained
[aN_raw]atempo=2.0,atempo=2.0[aN]   # for speed=4

# Layout composition via overlay enable='between(t,...)'
[sN]scale=1920:1080[s_scaled]
[wN]scale=480:270[w_pip]          # corner
[wN]scale=1920:1080[w_full]       # full-frame
[s_scaled][w_pip]overlay=W-w-32:H-h-32:enable='between(t,T_pip_start,T_pip_end)'[...]

# Concat all segments
[v0][v1]...[vN]concat=n=N:v=1:a=0[vout]
[a0][a1]...[aN]concat=n=N:v=0:a=1[aout]
```

**Test:** render a 10 s fixture with 2 segments of different layouts + 1 speed ramp. Verify output duration matches expected, pixel-sample check that layout transition happens at the marked boundary.

**Gotchas:**
- ffmpeg filter syntax is finicky with newlines and shell quoting — build the filter string in Python, write to a file, pass with `-filter_complex_script`.
- `atempo` caps at 2.0 per filter; chain 2 for 4×, 3 for 8×. Max meaningful audio speed is 4× (beyond that it's unintelligible anyway).
- VideoToolbox encoder doesn't like some pixel formats — force `format=yuv420p` on the final chain.
- For cursor zoom, use pre-segment `crop` with `scale` to 1080p rather than per-frame expressions — simpler and faster.

### M6 — Stage F: polish + integration test  ✅ DONE

**DoD.** `python -m src.cli run --screen raw_screen.mkv --webcam raw_webcam.mkv` runs the full pipeline end-to-end. Output `final.mp4` is -14 LUFS compliant (verified via `ffmpeg-normalize --print-stats`).

**Files:**
- `src/stages/polish.py` — DeepFilterNet (optional) + ffmpeg-normalize wrapper
- `src/pipeline.py` — orchestrator tying all stages
- `src/cli.py` — `run` subcommand for full pipeline
- `tests/test_integration.py` — runs entire pipeline on a 2-min fixture

**Test:** integration test completes in < 5 min on M5 Max, output passes:
- duration > 0
- audio is -14 LUFS ± 0.5
- video is 1080p HEVC
- file is playable (`ffprobe` doesn't error)

**Gotchas:**
- Integration test fixture MP4 should have variety: at least one filler, one dead-zone, one layout change, one cursor zoom. Small but representative.
- Allow `--skip-denoise` for CI — DeepFilterNet binary may not install in sandboxed environments.
- Log timing per stage — useful for future optimization.

---

## 4. Dependencies

All installed via `scripts/install.sh`:

### System (brew)
- `ffmpeg` 8.1 — main workhorse
- `auto-editor` 30.1.2 — motion predicate
- `jq` — shell JSON parsing
- `rubberband` — higher-quality audio time-stretch (optional, GPL-2.0+)

### Python (venv, pip)
- `mlx-whisper` — transcription
- `mlx-lm` — LLM server
- `librosa` — clap onset detection
- `opencv-python-headless` — clap flash detection
- `numpy` — array ops
- `typer` or `argparse` — CLI (TBD, leaning toward argparse for zero extra deps)
- `pydantic` — schema validation for JSON I/O
- `tenacity` — retry logic for LLM calls
- `ffmpeg-normalize` — loudness conformance
- `python-dotenv` — config loading
- `pytest` + `pytest-asyncio` — testing

### OBS plugin (one-time manual install)
- `obs-source-record` 0.4.8 `.pkg` from [release page](https://github.com/exeldro/obs-source-record/releases/tag/0.4.8)

### HuggingFace models (one-time pull to ~/.cache/huggingface)
- `mlx-community/whisper-large-v3-turbo` — 1.6 GB
- `mlx-community/Llama-3.3-70B-Instruct-4bit` — 39.7 GB (primary LLM)
- `mlx-community/Qwen3-30B-A3B-4bit` — 17.2 GB (optional fast fallback)

**Note:** the original research doc suggested Qwen3-235B-A22B-MLX-4bit (~125 GB full
repo). That was measured and rejected — on 128 GB unified memory the
model + KV cache + OS would exceed physical RAM. Llama 3.3 70B 4-bit is
the right primary for this host class.

### Cursor tracker (separate venv, can install alone)
- `pynput` — global mouse/keyboard hooks (requires macOS Accessibility + Input Monitoring perms)
- `tkinter` — stdlib, clap flash

---

## 5. Configuration

All tunables in `src/config.py` + overrideable via `.env`:

```python
# src/config.py
from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()

# Paths
WORK_DIR = Path(os.getenv("WORK_DIR", "./work"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./output"))

# LLM server
LLM_SERVER_URL = os.getenv("LLM_SERVER_URL", "http://127.0.0.1:8080/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "mlx-community/Llama-3.3-70B-Instruct-4bit")

# Transcription
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo")

# Detection thresholds (tune per creator)
FREEZE_DB = float(os.getenv("FREEZE_DB", "-50"))
FREEZE_MIN_SEC = float(os.getenv("FREEZE_MIN_SEC", "2.0"))
SILENCE_DB = float(os.getenv("SILENCE_DB", "-30"))
SILENCE_MIN_SEC = float(os.getenv("SILENCE_MIN_SEC", "2.0"))
MOTION_THRESHOLD = float(os.getenv("MOTION_THRESHOLD", "0.02"))

# Dead zone classification
CUT_MIN_SEC = float(os.getenv("CUT_MIN_SEC", "10.0"))
SPEED_8X_MIN_SEC = float(os.getenv("SPEED_8X_MIN_SEC", "3.0"))

# Render
VIDEO_BITRATE = os.getenv("VIDEO_BITRATE", "12M")
LOUDNESS_TARGET = float(os.getenv("LOUDNESS_TARGET", "-14.0"))
TRUE_PEAK = float(os.getenv("TRUE_PEAK", "-1.5"))
LOUDNESS_RANGE = float(os.getenv("LOUDNESS_RANGE", "11.0"))
```

---

## 6. Testing strategy

### Fixtures (in `tests/fixtures/`)

Create once, commit to git:
- `mini_screen.mkv` — 30 s silent screen recording with known events (type a command, scroll, wait, type more)
- `mini_webcam.mkv` — 30 s face-cam with a known clap at t=2.0, one "umm" at t=15.0, one silent wait at t=22–25
- `mini_cursor.csv` — matching cursor log
- `expected_*.json` — golden outputs for each stage

### Unit tests (pytest)

One test module per stage:
- `test_sync.py` — offset detection accuracy
- `test_analyze.py` — filler recall, layout coverage
- `test_dead_zone.py` — detector agreement on known fixture
- `test_unify.py` — golden segment list match
- `test_render.py` — output duration + format sanity check
- `test_integration.py` — full pipeline on fixture

### CI / local reproducibility

- Tests run with `pytest` from repo root
- `--skip-slow` flag to skip LLM and render tests in quick runs
- `pytest-recording` or VCR for LLM responses (deterministic)

---

## 7. Risk register

Lifted from research 08 §8.4 and extended:

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Clap-cue missed or weak | Medium | High | Fallback `--manual-offset` flag. Train yourself to clap loudly. |
| Dead-zone detector false positive cuts real content | High (first 10 sessions) | Medium | `--tune` mode surfaces candidates before applying. Per-creator threshold tuning. |
| LLM returns malformed JSON | Low | Low | jsonschema validation + retry loop + strict prompt. |
| OBS source-record sync drift > 200 ms | Medium | High | Clap-cue corrects for it. If clap missed, session is lost (retake). |
| MLX Qwen3-235B OOM on 128 GB | Low | Medium | Fallback to Llama-3.3-70B-4bit (39.7 GB). Detect OOM, retry with smaller model. |
| ffmpeg filter_complex > 8 KB shell arg limit | Medium (long sessions) | Medium | Always use `-filter_complex_script path.txt`. |
| auto-editor motion export format change between versions | Low | Low | Pin version in pyproject.toml. Schema-validate parsed output. |
| DeepFilterNet binary ABI-incompatible on new macOS | Low | Medium | Make denoise optional via `--skip-denoise`. |
| Cursor tracker missing Accessibility perm | Medium (first time) | Low | install script prints clear instructions. Logger aborts with helpful error. |
| tkinter flash not captured by OBS (different display) | Medium | Medium | Document: OBS Display Capture must be on primary display. Flash goes to primary. |

---

## 8. Out of scope (v1)

Explicitly deferred to avoid scope creep:

- Color grading — add DaVinci Resolve Studio ([../plans/resolve-studio-hybrid.md](../plans/resolve-studio-hybrid.md)) or manual ffmpeg `eq`/`lut3d` later.
- Music bed + sidechain ducking — not needed for screen tutorials.
- Multi-speaker diarization — solo creator use case only.
- YouTube upload automation — out of pipeline; use `yt-dlp`'s upload tools if needed.
- B-roll insertion — manual.
- Chapter marker generation — phase 2 polish.
- Web UI — CLI first. Phase 2+.
- Windows/Linux support — macOS-only by design (ScreenCaptureKit + VideoToolbox + MLX).

---

## 9. Acceptance criteria (v1 "done")

A successful M6 means all of:

1. Record a 30-min session in OBS (dual source-record) + cursor tracker.
2. Run `python -m src.cli run --screen ... --webcam ... --cursor ...`.
3. Pipeline completes in < 10 min on M5 Max.
4. Output `final.mp4` plays cleanly in QuickTime.
5. Fillers visibly removed (spot check 5 random positions).
6. At least one dead-zone speed-ramp visible at 4× or 8×.
7. Layout switches from cam_full intro → pip demo → cam_full outro.
8. Cursor-anchored zoom visible during at least one click event.
9. Loudness verified at -14 LUFS ± 0.5.
10. Total user clicks: ~3 (start OBS, hit record, run pipeline). The rest is automated.

---

## 10. Rollout timeline (suggested)

Solo evening sessions:

- **Evening 1** — M0 + M0b (scaffold + cursor tracker working) → can start recording with the tracker the next day
- **Evening 2–3** — M1 (sync)
- **Evening 4–5** — M2 (transcribe + analyze)
- **Evening 6–7** — M3 (dead zone)
- **Evening 8** — M4 (unify)
- **Evening 9–10** — M5 (render)
- **Evening 11–12** — M6 (polish + integration)

Total: ~2–3 weeks at 2 hours/evening. Cursor tracker alone is usable after evening 1.
