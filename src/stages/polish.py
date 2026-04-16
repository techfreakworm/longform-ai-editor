"""Stage F — final polish: optional DeepFilterNet denoise + EBU R128 loudnorm.

Pipeline (in order):
  1. If `deep-filter` binary is on PATH AND --skip-denoise was not passed:
     extract the mic audio, run deep-filter on it, mux it back into the
     video (video bitstream stream-copied).
  2. Run ffmpeg-normalize two-pass EBU R128 to the YouTube -14 LUFS target.

If deep-filter is missing, step 1 is silently skipped — loudnorm alone is
enough to produce a shippable output, just without the optional denoise
pass. Install deep-filter from
https://github.com/Rikorose/DeepFilterNet/releases (aarch64-apple-darwin
binary) to enable it.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from src import config

log = logging.getLogger(__name__)


# ---------- capability checks -----------------------------------------

def has_deep_filter() -> bool:
    return shutil.which("deep-filter") is not None


# ---------- denoise ---------------------------------------------------

def denoise(input_mp4: Path, output_mp4: Path) -> None:
    """Extract audio → run deep-filter → mux back over the original video.

    Video stream is copied bit-exactly (`-c:v copy`); only audio is
    re-encoded (as AAC 192 kbit to match the rest of the pipeline).
    """
    if not has_deep_filter():
        raise RuntimeError(
            "deep-filter binary not found on PATH — install from "
            "https://github.com/Rikorose/DeepFilterNet/releases or skip "
            "denoise with --skip-denoise"
        )

    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="polish_") as td:
        td_path = Path(td)
        raw_wav = td_path / "audio.wav"
        clean_dir = td_path / "clean"
        clean_dir.mkdir()

        # 1. Extract audio as 48 kHz PCM WAV (DeepFilterNet's native rate)
        log.info("extracting audio to %s", raw_wav)
        subprocess.run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(input_mp4),
            "-vn", "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "1",
            str(raw_wav),
        ], check=True)

        # 2. Run deep-filter; output lands at clean_dir/audio.wav
        log.info("running deep-filter on %s", raw_wav)
        subprocess.run([
            "deep-filter",
            "-o", str(clean_dir),
            str(raw_wav),
        ], check=True)
        clean_wav = clean_dir / raw_wav.name
        if not clean_wav.exists():
            # Some deep-filter versions add a suffix; find the WAV produced.
            candidates = list(clean_dir.glob("*.wav"))
            if not candidates:
                raise RuntimeError(
                    f"deep-filter wrote nothing to {clean_dir}"
                )
            clean_wav = candidates[0]

        # 3. Mux cleaned audio back over the original video
        log.info("muxing cleaned audio back into %s", output_mp4)
        subprocess.run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(input_mp4),
            "-i", str(clean_wav),
            "-map", "0:v", "-map", "1:a",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            str(output_mp4),
        ], check=True)


# ---------- loudnorm --------------------------------------------------

def loudnorm(
    input_mp4: Path,
    output_mp4: Path,
    target: float | None = None,
    true_peak: float | None = None,
    lra: float | None = None,
    audio_bitrate: str = "192k",
    sample_rate: int = 48000,
) -> None:
    """Two-pass EBU R128 normalization via ffmpeg-normalize.

    Defaults to YouTube's -14 LUFS target (from config). ffmpeg-normalize
    handles the two-pass internally: pass 1 measures integrated loudness
    + true peak + LRA, pass 2 applies the derived gain.
    """
    target = target if target is not None else config.LOUDNESS_TARGET
    true_peak = true_peak if true_peak is not None else config.TRUE_PEAK
    lra = lra if lra is not None else config.LOUDNESS_RANGE

    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    # ffmpeg-normalize 1.x renamed `-lra` → `-lrt` / `--loudness-range-target`.
    cmd = [
        "ffmpeg-normalize", str(input_mp4),
        "-nt", "ebu",
        "-t", str(target),
        "-tp", str(true_peak),
        "-lrt", str(lra),
        "-c:a", "aac",
        "-b:a", audio_bitrate,
        "-ar", str(sample_rate),
        "-o", str(output_mp4),
        "-f",  # overwrite existing output
    ]
    log.info("running ffmpeg-normalize: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


# ---------- CLI entry -------------------------------------------------

def run(args: argparse.Namespace) -> int:
    """Stage F CLI entry.

    Workflow:
      (optional) denoise → loudnorm → args.output
    """
    input_mp4: Path = args.input
    output_mp4: Path = args.output
    skip_denoise: bool = bool(getattr(args, "skip_denoise", False))

    if not input_mp4.exists():
        print(f"[polish] ERROR: input not found: {input_mp4}", file=sys.stderr)
        return 1

    try:
        # 1. (optional) denoise to an intermediate
        loudnorm_input = input_mp4
        intermediate: Path | None = None
        if skip_denoise:
            print("[polish] skipping denoise (--skip-denoise)")
        elif not has_deep_filter():
            print("[polish] skipping denoise — deep-filter not on PATH")
            print("[polish]   install: https://github.com/Rikorose/DeepFilterNet/releases")
        else:
            intermediate = input_mp4.parent / f"{input_mp4.stem}.dn.mp4"
            print(f"[polish] denoising → {intermediate}")
            denoise(input_mp4, intermediate)
            loudnorm_input = intermediate

        # 2. Always loudnorm to the final output
        print(
            f"[polish] loudnorm → {output_mp4} "
            f"(target {config.LOUDNESS_TARGET} LUFS, TP {config.TRUE_PEAK}, LRA {config.LOUDNESS_RANGE})"
        )
        loudnorm(loudnorm_input, output_mp4)

        # Clean up intermediate if we made one
        if intermediate is not None and intermediate.exists():
            intermediate.unlink()

        print(f"[polish] wrote {output_mp4}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[polish] ERROR: {exc}", file=sys.stderr)
        return 1


__all__ = ["has_deep_filter", "denoise", "loudnorm", "run"]
