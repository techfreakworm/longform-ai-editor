# long-form-editor

100%-local dual-track long-form YouTube editor for Apple M5 Max.

**Inputs:** `screen_raw.mkv` (silent screen recording) + `webcam_raw.mkv` (face + mic) + optional `cursor.csv`
**Output:** `final.mp4` — 1080p HEVC, -14 LUFS, loudness-conformed YouTube master
**Constraints:** no cloud credits, no per-render fees, 100% OSS, runs on Apple Silicon.

## Status

Scaffold + **cursor tracker working**. Pipeline stages are skeleton-only — see [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the milestone-by-milestone build plan (M0–M6).

## Quick start — cursor tracker only (ready to use for recording)

```bash
./scripts/install_cursor_tracker.sh
# Follow the macOS permissions prompt, then:
./cursor-tracker/record.sh ~/sessions/episode_01.csv
```

- Hotkey **Ctrl+Option+Cmd+K** = clap cue (flash + CSV marker, for sync)
- Ctrl+C to stop

See [`cursor-tracker/README.md`](cursor-tracker/README.md) for details.

## Full pipeline (once implemented — M6 complete)

```bash
# One-time
./scripts/install.sh
python scripts/verify_env.py      # checks ffmpeg, auto-editor, mlx_lm, models

# Per session
mlx_lm.server --model Qwen/Qwen3-235B-A22B-MLX-4bit --port 8080 &
python -m src.cli run \
    --screen ~/sessions/ep01/screen_raw.mkv \
    --webcam ~/sessions/ep01/webcam_raw.mkv \
    --cursor ~/sessions/ep01/cursor.csv \
    --output ~/sessions/ep01/final.mp4
```

Individual stages can also be run independently:
```bash
python -m src.cli sync       --screen ... --webcam ...
python -m src.cli analyze    --webcam webcam_synced.mkv
python -m src.cli detect-dead --screen ... --webcam ...
python -m src.cli unify
python -m src.cli render     --segments segments.json
python -m src.cli polish     --input composed.mp4
```

## Architecture

6 stages (A–F). Full description in [`../plans/long-form-pipeline.md`](../plans/long-form-pipeline.md) and the research documents in [`../research/`](../research/).

```
RECORD (OBS + obs-source-record, cursor_logger.py)
   ├─► screen_raw.mkv + webcam_raw.mkv + cursor.csv
   ▼
A. SYNC        — clap cue (librosa onset + OpenCV luminance diff)
B. ANALYZE     — mlx-whisper transcription + Qwen3 filler cuts + layout plan
C. DEAD-ZONE   — freezedetect + silencedetect + auto-editor motion + LLM cues
D. UNIFY       — merge all decisions into segments.json
E. RENDER      — one ffmpeg filter_complex pass (trim + setpts + overlay)
F. POLISH      — DeepFilterNet (optional) + ffmpeg-normalize -14 LUFS
   ▼
final.mp4
```

## License

Your code: pick your license.

Dependencies pull in mixed licenses — see [IMPLEMENTATION_PLAN.md §4](IMPLEMENTATION_PLAN.md) for the full list. Notable: AGPL-3.0 (Ultralytics YOLO, via cursor_zoom), GPL-2.0+ (rubberband if used for audio stretch), CC-BY-NC-4.0 (CrisperWhisper — avoided in favor of mlx-whisper for commercial-safe pipeline).
