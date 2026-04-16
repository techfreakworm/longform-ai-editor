# long-form-editor

**100%-local dual-track long-form YouTube editor for Apple Silicon.**
No cloud credits, no per-render fees, no SaaS. Runs on your machine end-to-end.

```
OBS session (ext.mov + cam.mov + merged.mp4 + cursor.csv)
                         ↓
   ┌───────────────────────────────────────────────────┐
   │ sync  →  transcribe+LLM  →  dead-zone detect      │
   │         →  unify  →  render  (→ polish [TODO])    │
   └───────────────────────────────────────────────────┘
                         ↓
                     final.mp4
```

---

## What it actually does (real measurements)

On a 95.75-second test session the pipeline produced a polished **42.99-second**
`1920×1080 HEVC 60 fps` final in **7.6 seconds of render wall-clock** on an
M5 Max. The pipeline:

- transcribed the webcam audio (47 words, mlx-whisper large-v3-turbo)
- detected **7 filler cuts** (fillers / false-starts / repeats) via Llama 3.3 70B
- laid out **5 tutorial sections** (cam_full intro → pip → screen_full demo → pip → cam_full outro)
- detected **3 dead zones** on screen + audio (one 42-second cut, two speed@8x ramps)
- composited webcam as PiP over the screen, respecting layout switches per segment
- applied speed ramps via `setpts` + pitch-preserving `atempo` chains

| Stage | Role | Status | Tests |
|---|---|---|---|
| M0 · scaffold | install scripts, venv, CLI | ✅ | — |
| M0b · cursor tracker | record-time cursor/click/clap log | ✅ | — |
| M1 · Stage A sync | clap-cue alignment, CSV ↔ video time | ✅ | 18 |
| cursor_zoom (Cap port) | AGPL algorithm port for auto-zoom | ✅ | 22 |
| M2 · Stage B transcribe + LLM | mlx-whisper + Llama 3.3 filler/layout | ✅ | 32 |
| M3 · Stage C dead-zone detect | freezedetect + silencedetect intersection | ✅ | 32 |
| M4 · Stage D unify segments | merge all decisions → canonical edit list | ✅ | 46 |
| M5 · Stage E render | single-pass ffmpeg filter_complex | ✅ | 30 |
| M6 · Stage F polish | (optional) DeepFilterNet denoise + ffmpeg-normalize EBU R128 | ✅ | 10 |

**190 tests passing** (179 fast + 11 integration against real ffmpeg / LLM / fixtures).

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

In a separate terminal, start the LLM server (takes ~60 s to load Llama into memory):

```bash
source venv/bin/activate
mlx_lm.server --model mlx-community/Llama-3.3-70B-Instruct-4bit --port 8080
```

Then one command:

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
- **Stream-copy-then-re-encode-in-render**: Stage A's optional `--trim` uses
  `-c copy` which snaps to keyframes; precise trim accuracy comes from
  Stage E applying the offset on its own via per-segment `trim + setpts`.
- **Dead-zone intersection**, not union: require ≥2 detectors (freezedetect +
  silencedetect) to agree. Catches "silent narrator while screen isn't
  moving" but not false positives from either alone.
- **Cursor zoom split at boundaries**: a Segment with N cursor-zoom windows
  becomes ≤2N+1 RenderSegments, each with at most one static crop. Keeps the
  ffmpeg filter graph simple (no per-frame `crop=x:y` expressions).

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
- **macOS only.** MLX is Apple-silicon-specific; ScreenCaptureKit is macOS-specific.
- **mlx_lm.server must run on `--port 8080`.** Override via `.env`
  (`LLM_SERVER_URL=http://host:port/v1`).
- **Constrained-decoding JSON mode avoided.** If you swap in a different
  backend (vLLM, Ollama) that handles `response_format=json_object` well, the
  `call_llm_json()` helper would benefit from re-enabling it. Currently
  disabled as a reliability measure against mlx_lm.server's hangs.
- **Cursor zoom needs manual origin calibration** on multi-monitor setups.
  See the table above for common positions.
- **Dead-zone detection is heuristic.** 10–20% false positive / false negative
  rate on first runs. Tune `FREEZE_DB`, `SILENCE_DB` in `.env` after 5–10
  real sessions.
- **Cap algorithm port** in `cursor_zoom.py` is an AGPL-3.0 derivation of
  CapSoftware/Cap's Rust source. If you commercialize this pipeline, the
  cursor-zoom component is AGPL — either re-implement the algorithm from
  the documented constants alone or pay Cap's commercial license.

---

## Project structure

```
long-form-editor/
├── IMPLEMENTATION_PLAN.md          # milestone-by-milestone build plan
├── pyproject.toml                  # deps, dev deps, pytest markers
├── .env.example                    # tunable thresholds + LLM settings
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
│   ├── cli.py                      # argparse entry: sync|analyze|detect-dead|unify|render|run|verify
│   ├── config.py                   # paths, thresholds, prompts (env-overridable)
│   ├── pipeline.py                 # run_all orchestrator
│   ├── stages/
│   │   ├── sync_clap.py            # Stage A
│   │   ├── transcribe.py           # Stage B.1
│   │   ├── analyze_llm.py          # Stage B.2
│   │   ├── dead_zone_detect.py     # Stage C
│   │   ├── unify_segments.py       # Stage D
│   │   ├── render.py               # Stage E
│   │   ├── polish.py               # Stage F (skeleton, M6 TODO)
│   │   └── cursor_zoom.py          # Cap algorithm port
│   └── utils/
│       ├── ffmpeg_helpers.py       # subprocess, ffprobe wrappers
│       ├── log_parsers.py          # freezedetect/silencedetect regex
│       └── timecodes.py            # seconds ↔ frames + interval math
└── tests/
    ├── fixtures/real/              # symlinks to real OBS fixtures (gitignored)
    ├── test_sync.py                # 18
    ├── test_analyze_llm.py         # 25
    ├── test_transcribe.py          # 12
    ├── test_dead_zone_detect.py    # 19
    ├── test_unify_segments.py      # 46
    ├── test_render.py              # 30
    ├── test_cursor_zoom.py         # 22
    ├── test_log_parsers.py         # 8
    └── test_timecodes.py           # 8
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
