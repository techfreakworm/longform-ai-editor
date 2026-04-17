"""Short-form (portrait 9:16) pipeline.

Consumes a long-form source (either dual-track `cam.mov` + `ext.mov` or
an already-composited mp4), produces N portrait clips with karaoke
captions. All components stay within this repo's conventions —
mlx-whisper / Claude-CLI / ffmpeg — plus an optional extras group
(`pip install '.[shortform]'`) for parakeet-mlx, YOLO11, MediaPipe,
stable-ts, and sentence-transformers.

Stages (see docs/plans/build-plan-2026-04-17.md for the design doc):

    transcribe → segment (TextTiling) → score (LLM + heuristics) →
    cut → reframe (YOLO + MediaPipe + OneEuroFilter) →
    captions (stable-ts karaoke) → render (4 layouts) → N mp4s
"""
