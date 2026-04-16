"""Top-level orchestrator — runs Stages A through F in sequence.

Each stage writes its outputs to `config.WORK_DIR` and is idempotent (re-running
skips work if the output already exists). Deletes `work/` to force a full redo.
"""
from __future__ import annotations

import argparse
import logging

log = logging.getLogger(__name__)


def run_all(args: argparse.Namespace) -> int:
    """Full pipeline: sync → analyze → detect-dead → unify → render → polish.

    TODO(M6): implement orchestration. See IMPLEMENTATION_PLAN.md §M6.

    Intended flow:
      1. sync_clap.run(screen, webcam) → work/{offset.json,screen_synced.mkv,webcam_synced.mkv}
      2. transcribe.run(work/webcam_synced.mkv) → work/words.json
      3. analyze_llm.run(work/words.json) → work/{filler_cuts.json,layout_plan.json}
      4. dead_zone_detect.run(...) → work/dead_zones.json
      5. unify_segments.run(...) → work/segments.json
      6. render.run(...) → work/composed.mp4
      7. polish.run(work/composed.mp4, args.output) → final.mp4
    """
    raise NotImplementedError(
        "pipeline.run_all — see IMPLEMENTATION_PLAN.md §M6. "
        "Until then, invoke stages individually via `lfe <subcommand>`."
    )
