"""Stage E — build one ffmpeg filter_complex pass that renders the final video.

Input: screen_synced.mkv, webcam_synced.mkv, segments.json
Output: composed.mp4 — 1080p HEVC VideoToolbox

Strategy: generate the entire filter graph in Python, write to a
`filter_complex.txt` file, pass to ffmpeg via `-filter_complex_script`
(avoids shell arg length limits on long sessions).

Per segment, three branches are assembled:
  - Screen video:  trim + setpts(speed) + optional cursor-zoom crop+scale
  - Webcam video:  trim + setpts(speed) + scale (pip or full)
  - Webcam audio:  atrim + asetpts + atempo chain for speed>1

Then the time-varying overlay composites webcam onto screen according to
each segment's layout, and concat joins all segments.

TODO(M5): implement. See IMPLEMENTATION_PLAN.md §M5.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def build_filter_complex(segments: list, screen_res: tuple[int, int]) -> str:
    """Assemble the entire filter graph string.

    TODO(M5): implement. Key concerns:
      - atempo caps at 2.0 per filter; chain 2 for 4×, 3 for 8×
      - setpts=(PTS-STARTPTS)/speed on video
      - overlay with enable='between(t,...)' for layout switching
      - scale=W:H on both branches to force consistent output resolution
      - concat filter joins all segments
      - format=yuv420p at end for VideoToolbox compatibility
    """
    raise NotImplementedError


def render(
    screen: Path,
    webcam: Path,
    segments_json: Path,
    output: Path,
    bitrate: str = "12M",
) -> None:
    """Build filter graph, write to temp file, invoke ffmpeg.
    TODO(M5).
    """
    raise NotImplementedError


def run(args: argparse.Namespace) -> int:
    raise NotImplementedError(
        "render.run — see IMPLEMENTATION_PLAN.md §M5"
    )


__all__ = ["build_filter_complex", "render", "run"]
