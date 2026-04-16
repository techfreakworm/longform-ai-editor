"""CLI entry — `python -m src.cli [subcommand] [options]`.

Subcommands map 1:1 to pipeline stages so each can be re-run independently.

  sync          — Stage A: clap-cue sync
  analyze       — Stage B: transcribe + LLM filler + layout
  detect-dead   — Stage C: dead-zone detection on screen
  unify         — Stage D: merge all decisions into segments.json
  render        — Stage E: ffmpeg filter_complex render
  polish        — Stage F: denoise + loudnorm
  run           — run all stages end-to-end
  verify        — environment sanity check (calls scripts/verify_env.py)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_common_paths(p: argparse.ArgumentParser) -> None:
    p.add_argument("--work", type=Path, default=None, help="Work directory (default: config.WORK_DIR)")
    p.add_argument("--verbose", "-v", action="count", default=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lfe",
        description="Dual-track long-form YouTube editor. See IMPLEMENTATION_PLAN.md.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_sync = sub.add_parser("sync", help="Stage A — clap-cue sync")
    p_sync.add_argument("--screen", type=Path, required=True,
                        help="Screen recording (silent or with audio)")
    p_sync.add_argument("--webcam", type=Path, required=True,
                        help="Webcam recording (used for audio clap detection)")
    p_sync.add_argument("--cursor", type=Path, default=None,
                        help="cursor.csv from cursor-tracker (for CSV alignment)")
    p_sync.add_argument("--trim", action="store_true",
                        help="Also write trimmed screen_synced.mkv + webcam_synced.mkv "
                             "(needed only when source files are NOT already OBS-synced)")
    p_sync.add_argument("--manual-offset", type=float, default=None,
                        help="Skip auto-detection, apply this video-to-video offset (seconds)")
    _add_common_paths(p_sync)

    p_analyze = sub.add_parser("analyze", help="Stage B — transcribe + LLM")
    p_analyze.add_argument("--webcam", type=Path, required=True,
                           help="Synced webcam from Stage A")
    p_analyze.add_argument("--audio", type=Path, default=None,
                           help="Override audio source (default: --webcam)")
    _add_common_paths(p_analyze)

    p_dead = sub.add_parser("detect-dead", help="Stage C — screen dead-zone detect")
    p_dead.add_argument("--screen", type=Path, required=True,
                        help="Screen recording (for freezedetect)")
    p_dead.add_argument("--webcam", type=Path, required=True,
                        help="Webcam recording (default audio source for silencedetect)")
    p_dead.add_argument("--audio", type=Path, default=None,
                        help="Audio source override (default: --webcam)")
    p_dead.add_argument("--words", type=Path, default=None,
                        help="words.json — optional, enables LLM transcript cues (not yet implemented)")
    _add_common_paths(p_dead)

    p_unify = sub.add_parser("unify", help="Stage D — merge decisions into segments.json")
    p_unify.add_argument("--cursor", type=Path, default=None,
                         help="Cursor CSV for auto-zoom (optional)")
    p_unify.add_argument("--screen-w", type=int, default=2560,
                         help="Recorded display width in pixels (for cursor normalization)")
    p_unify.add_argument("--screen-h", type=int, default=1440,
                         help="Recorded display height in pixels")
    p_unify.add_argument("--origin-x", type=float, default=0.0,
                         help="Recorded display origin X in macOS global coords")
    p_unify.add_argument("--origin-y", type=float, default=0.0,
                         help="Recorded display origin Y in macOS global coords")
    _add_common_paths(p_unify)

    p_render = sub.add_parser("render", help="Stage E — ffmpeg render")
    p_render.add_argument("--screen", type=Path, required=True,
                          help="Screen recording source (e.g. ext.mov)")
    p_render.add_argument("--webcam", type=Path, required=True,
                          help="Webcam video source (e.g. cam.mov)")
    p_render.add_argument("--audio", type=Path, default=None,
                          help="Audio source (defaults to --webcam's audio)")
    p_render.add_argument("--segments", type=Path, required=True,
                          help="segments.json from Stage D")
    p_render.add_argument("--output", type=Path, required=True)
    _add_common_paths(p_render)

    p_polish = sub.add_parser("polish", help="Stage F — denoise + loudnorm")
    p_polish.add_argument("--input", type=Path, required=True)
    p_polish.add_argument("--output", type=Path, required=True)
    p_polish.add_argument("--skip-denoise", action="store_true")
    _add_common_paths(p_polish)

    p_run = sub.add_parser("run", help="Run all stages end-to-end")
    p_run.add_argument("--screen", type=Path, required=True,
                       help="Screen recording (e.g. ext.mov from OBS source-record)")
    p_run.add_argument("--webcam", type=Path, required=True,
                       help="Webcam recording (e.g. cam.mov from OBS source-record)")
    p_run.add_argument("--audio", type=Path, default=None,
                       help="Audio source (defaults to --webcam's audio; use merged.mp4 for cleaner mix)")
    p_run.add_argument("--cursor", type=Path, default=None,
                       help="cursor.csv from cursor-tracker (enables auto-zoom)")
    p_run.add_argument("--output", type=Path, required=True,
                       help="Final .mp4 path")
    p_run.add_argument("--manual-offset", type=float, default=None,
                       help="Skip Stage A auto-sync and use this CSV offset (seconds)")
    p_run.add_argument("--screen-w", type=int, default=2560,
                       help="Recorded display width in pixels (cursor-zoom normalization)")
    p_run.add_argument("--screen-h", type=int, default=1440,
                       help="Recorded display height")
    p_run.add_argument("--origin-x", type=float, default=0.0,
                       help="Recorded display origin X in macOS global coords")
    p_run.add_argument("--origin-y", type=float, default=0.0,
                       help="Recorded display origin Y in macOS global coords")
    p_run.add_argument("--skip-denoise", action="store_true",
                       help="Skip M6 denoise stage (currently always skipped until M6 lands)")
    _add_common_paths(p_run)

    sub.add_parser("verify", help="Environment sanity check")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Stages are imported lazily so `--help` works without full deps installed.
    if args.command == "sync":
        from src.stages.sync_clap import run as sync_run
        return sync_run(args)
    elif args.command == "analyze":
        from src.stages.transcribe import run_analyze
        return run_analyze(args)
    elif args.command == "detect-dead":
        from src.stages.dead_zone_detect import run as dead_run
        return dead_run(args)
    elif args.command == "unify":
        from src.stages.unify_segments import run as unify_run
        return unify_run(args)
    elif args.command == "render":
        from src.stages.render import run as render_run
        return render_run(args)
    elif args.command == "polish":
        from src.stages.polish import run as polish_run
        return polish_run(args)
    elif args.command == "run":
        from src.pipeline import run_all
        return run_all(args)
    elif args.command == "verify":
        import subprocess
        script = Path(__file__).parent.parent / "scripts" / "verify_env.py"
        return subprocess.call([sys.executable, str(script)])
    else:
        parser.print_help()
        return 2


if __name__ == "__main__":
    sys.exit(main())
