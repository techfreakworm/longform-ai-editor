"""Stage F — final polish: optional DeepFilterNet denoise + EBU R128 loudnorm.

TODO(M6): implement. See IMPLEMENTATION_PLAN.md §M6.

Pipeline:
  1. If denoise enabled and `deep-filter` binary present:
     - Extract mic audio to 48 kHz WAV
     - Run deep-filter on it
     - Mux cleaned audio back over the video (ffmpeg -c:v copy)
  2. ffmpeg-normalize -nt ebu -t -14 -tp -1.5 -lra 11 → final.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path


def has_deep_filter() -> bool:
    import shutil
    return shutil.which("deep-filter") is not None


def denoise(input_mp4: Path, output_mp4: Path) -> None:
    """Extract audio → deep-filter → mux back with stream-copied video.
    TODO(M6).
    """
    raise NotImplementedError


def loudnorm(
    input_mp4: Path,
    output_mp4: Path,
    target: float = -14.0,
    true_peak: float = -1.5,
    lra: float = 11.0,
) -> None:
    """Two-pass EBU R128 via ffmpeg-normalize.
    TODO(M6).
    """
    raise NotImplementedError


def run(args: argparse.Namespace) -> int:
    raise NotImplementedError(
        "polish.run — see IMPLEMENTATION_PLAN.md §M6"
    )


__all__ = ["has_deep_filter", "denoise", "loudnorm", "run"]
