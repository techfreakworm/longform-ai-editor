# Build plan — 2026-04-17 ("Everything at once")

Scope locked from the questionnaire on 2026-04-17. Track progress via the
TaskCreate list in the active Claude session; this doc is the reference.

## Decisions captured

| Area | Decision |
|---|---|
| Build order | Everything at once |
| Face absence | **Hard-cut only when** narration silent **AND** no face detected **AND** no cursor movement |
| Shortform repo | Subpackage `src/shortform/` inside long-form-editor |
| LLM strategy | Claude-preferred dispatcher everywhere + Qwen3/Llama fallback |
| Claude CLI | Pass `--effort max` (verified flag exists) |
| Sequential-thinking MCP | Applied to FILLER + LAYOUT + shortform scoring prompts, gated by `USE_SEQUENTIAL_THINKING` |
| Zoom improvements | All three: speech-emphasis + element-aware OCR + zoom-on-scroll |
| Transcription | Keep mlx-whisper for long-form; add parakeet-mlx for shortform only |
| Shortform captions | Plain stable-ts karaoke (yellow sweep, Montserrat ExtraBold 88px) |
| Shortform defaults | 3 clips, 30–60s each |
| Music bed | **Deferred** — document in `docs/future-improvements.md` |
| Deps | `pip install '.[shortform]'` optional extra |
| Default shortform layout | `split_vstack` **with intelligent screen-portion zoom** so UI elements are legible on phones |

## Architecture

### Long-form changes

1. **MCP + effort wiring.** `.mcp.json` at repo root (sequential-thinking
   server); `_call_via_claude_cli` adds `--effort max` and `--mcp-config`
   flags; `USE_SEQUENTIAL_THINKING` env gates the prompt prefix:
   *"Before responding, use the `sequentialthinking` MCP tool to reason
   through the input in multiple steps."*
2. **Face-visibility stage** (`src/stages/face_visibility.py`). Apple
   Vision `VNDetectFaceRectanglesRequest` via PyObjC at 2 Hz over
   `cam.mov`. Emits `work/face_absent.json` intervals (≥ 2s absence).
   Already have similar PyObjC usage pattern in `screen_flash.py`.
3. **Cursor-idle intervals.** New helper in
   `src/stages/dead_zone_detect.py` (or a new `cursor_idle.py`): reads
   `cursor.csv`, emits intervals of no-movement ≥ 2s.
4. **Triple-intersection hard-cut.** `src/stages/unify_segments.py` grows
   a new cut source: intersect face-absent ∩ narration-silent ∩
   cursor-idle. These join the existing `dead_zones` pipeline and are
   hard-cut regardless of length.
5. **Zoom v2** (`src/stages/cursor_zoom.py` grows three new zoom
   sources). All feed into the existing ZoomSegment list:
   - `speech_emphasis_zooms()` — LLM in `analyze_llm.py` extended to emit
     `zoom_hints: [{start, end, anchor_word_idx, reason}]`. Anchor
     words: "look at", "right here", "notice", "see this", etc.
   - `element_aware_target()` — OCR via `paddleocr` on the frame at the
     zoom center timestamp; snap zoom centroid to the nearest text block's
     bounding box if within 150 px.
   - `zoom_on_scroll()` — frame-diff (SSIM) on screen track during
     cursor-idle; if change region > threshold, emit zoom centered on
     that region.

### Shortform (new subpackage)

```
src/shortform/
├── __init__.py
├── transcribe.py       # parakeet-mlx wrapper, falls back to mlx-whisper
├── segment.py          # vendored ClipsAI TextTiler, multi-scale windows
├── score.py            # LLM (Claude CLI preferred) + composite heuristics
├── reframe.py          # YOLO11 + ByteTrack + MediaPipe + OneEuroFilter
├── captions.py         # stable-ts karaoke ASS
├── render.py           # per-layout ffmpeg (cam_full/screen_full/split_vstack/pip)
└── pipeline.py         # orchestrator: run_all(source) -> list[Path]

assets/clipsai/         # vendored TextTiler + ClipFinder, MIT
```

CLI: `python -m src.cli shortform --source <long.mp4> --top 3`

**split_vstack with screen zoom** (user refinement): when LLM picks
`split_vstack`, the screen 1080×960 top portion is further cropped/zoomed
around the cursor centroid for that clip window — so code/UI is readable
on a phone. Reuses `cursor_zoom.py`'s centroid computation.

## Dependencies

Added to `pyproject.toml [project.optional-dependencies.shortform]`:

```
parakeet-mlx
ultralytics          # AGPL-3.0 — personal use OK, flag if commercialized
mediapipe
scenedetect
OneEuroFilter
stable-ts
paddleocr            # for element-aware zoom — also usable by shortform
```

Installs via `pip install '.[shortform]'`. Fast tests still run against
base requirements; shortform tests skip if extras not installed.

## Test plan

- Unit tests per new module (mirror `tests/test_<module>.py` style).
- Mark shortform integration tests `@pytest.mark.slow` — they need
  YOLO/MediaPipe weights (~300 MB) and a real mp4.
- **End-to-end validation**: re-run the April 2026 long clip through
  the updated long-form pipeline, then pipe the composed output into the
  shortform CLI and produce 3 shorts. Verify face-absence cuts land
  correctly, zooms trigger on "look at this"-style phrases, captions
  burn in.

## Out of scope / future

- Music bed + sidechain ducking → future-improvements.md
- Submagic-style animated caption templates → future-improvements.md
- TalkNet-ASD for multi-speaker diarization → future-improvements.md
- Remotion overlay compositing → future-improvements.md
