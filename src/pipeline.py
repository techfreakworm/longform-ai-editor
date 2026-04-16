"""Top-level orchestrator — runs stages A → E (and F when available) in sequence.

Each stage's CLI `run()` function reads its inputs from `config.WORK_DIR`
(or the `--work` override) and writes its outputs to the same directory.
The orchestrator just builds the right argparse.Namespace per stage and
calls them in order.

If any stage returns non-zero, the orchestrator bails immediately and
returns the first non-zero exit code it saw. Partial work on disk is
preserved so the user can re-run from the broken stage individually.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src import config

log = logging.getLogger(__name__)


def run_all(args: argparse.Namespace) -> int:
    """Stages A → E end-to-end. Stage F (polish) will be added with M6."""
    work_dir: Path = Path(args.work) if getattr(args, "work", None) else config.WORK_DIR
    work_dir.mkdir(parents=True, exist_ok=True)

    screen: Path = args.screen
    webcam: Path = args.webcam
    audio: Path | None = getattr(args, "audio", None)
    cursor: Path | None = getattr(args, "cursor", None)
    output: Path = args.output

    # Stage-specific args (constructed here — each stage only reads what it needs)
    common = dict(work=work_dir, verbose=getattr(args, "verbose", 0))

    # -----------------------------------------------------------
    # Stage A — sync
    # -----------------------------------------------------------
    print("\n========= Stage A — sync =========")
    from src.stages.sync_clap import run as sync_run
    sync_args = argparse.Namespace(
        screen=screen,
        webcam=webcam,
        cursor=cursor,
        trim=False,
        manual_offset=getattr(args, "manual_offset", None),
        **common,
    )
    rc = sync_run(sync_args)
    if rc != 0:
        return rc

    # -----------------------------------------------------------
    # Stage B — transcribe + LLM analyze
    # -----------------------------------------------------------
    print("\n========= Stage B — transcribe + analyze =========")
    from src.stages.transcribe import run_analyze
    analyze_args = argparse.Namespace(
        webcam=webcam,
        audio=audio,
        **common,
    )
    rc = run_analyze(analyze_args)
    if rc != 0:
        return rc

    # -----------------------------------------------------------
    # Stage C — dead-zone detect
    # -----------------------------------------------------------
    print("\n========= Stage C — dead-zone detect =========")
    from src.stages.dead_zone_detect import run as dead_run
    dead_args = argparse.Namespace(
        screen=screen,
        webcam=webcam,
        audio=audio,
        words=work_dir / "words.json",
        **common,
    )
    rc = dead_run(dead_args)
    if rc != 0:
        return rc

    # -----------------------------------------------------------
    # Stage D — unify segments
    # -----------------------------------------------------------
    print("\n========= Stage D — unify =========")
    from src.stages.unify_segments import run as unify_run
    unify_args = argparse.Namespace(
        cursor=cursor,
        screen_w=getattr(args, "screen_w", 2560),
        screen_h=getattr(args, "screen_h", 1440),
        origin_x=getattr(args, "origin_x", 0.0),
        origin_y=getattr(args, "origin_y", 0.0),
        **common,
    )
    rc = unify_run(unify_args)
    if rc != 0:
        return rc

    # -----------------------------------------------------------
    # Stage E — render to an intermediate; Stage F produces the final.
    # -----------------------------------------------------------
    print("\n========= Stage E — render =========")
    from src.stages.render import run as render_run
    composed = work_dir / "composed.mp4"
    render_args = argparse.Namespace(
        screen=screen,
        webcam=webcam,
        audio=audio,
        segments=work_dir / "segments.json",
        output=composed,
        **common,
    )
    rc = render_run(render_args)
    if rc != 0:
        return rc

    # -----------------------------------------------------------
    # Stage F — polish (denoise + EBU R128 loudnorm)
    # -----------------------------------------------------------
    print("\n========= Stage F — polish =========")
    from src.stages.polish import run as polish_run
    polish_args = argparse.Namespace(
        input=composed,
        output=output,
        skip_denoise=getattr(args, "skip_denoise", False),
        **common,
    )
    rc = polish_run(polish_args)
    if rc != 0:
        return rc

    print(f"\n✓ pipeline complete: {output}")
    return 0
