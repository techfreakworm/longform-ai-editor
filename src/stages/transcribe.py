"""Stage B.1 — transcribe webcam audio with mlx-whisper.

Outputs `work/words.json` — a flat list of word-level timestamps:
  [{"word": "hello", "start": 1.234, "end": 1.402, "probability": 0.99}, ...]

Cached by audio file hash (sha256 of first + last 1 MB + size) — re-running
skips if cache hit.

TODO(M2): implement. See IMPLEMENTATION_PLAN.md §M2.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def transcribe(webcam_path: Path, model: str | None = None) -> dict[str, Any]:
    """Run mlx_whisper.transcribe with word_timestamps=True.

    Returns the full mlx-whisper dict: {"text", "segments", "language"}.
    Each segment has a "words" list with per-word timestamps.

    TODO(M2): implement.
    """
    raise NotImplementedError


def flatten_words(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten nested segments into a single list of word entries."""
    raise NotImplementedError


def run_analyze(args: argparse.Namespace) -> int:
    """CLI entry for Stage B — transcribe then analyze via LLM."""
    raise NotImplementedError(
        "transcribe.run_analyze — see IMPLEMENTATION_PLAN.md §M2"
    )


__all__ = ["transcribe", "flatten_words", "run_analyze"]
