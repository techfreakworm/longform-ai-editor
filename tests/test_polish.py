"""Tests for Stage F — polish (denoise + loudnorm).

Unit tests mock subprocess so we don't shell out to real ffmpeg /
ffmpeg-normalize / deep-filter. One integration test runs the real
ffmpeg-normalize against the already-rendered work/final.mp4 from a
previous pipeline run and verifies the loudness target was achieved.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.stages.polish import (
    denoise,
    has_deep_filter,
    loudnorm,
    run,
)


# --- has_deep_filter ----------------------------------------------------

def test_has_deep_filter_matches_which(monkeypatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/deep-filter")
    assert has_deep_filter() is True


def test_has_deep_filter_false_when_missing(monkeypatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert has_deep_filter() is False


# --- denoise ------------------------------------------------------------

def test_denoise_raises_if_deep_filter_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="deep-filter binary not found"):
        denoise(tmp_path / "in.mp4", tmp_path / "out.mp4")


def test_denoise_invokes_ffmpeg_then_deepfilter_then_mux(tmp_path, monkeypatch) -> None:
    """With deep-filter present, denoise runs three subprocess calls:
    extract WAV, deep-filter, mux. Pydantic-ish sequence check.
    """
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/deep-filter")

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        # deep-filter "produces" its output: create an empty wav where expected
        if cmd[0] == "deep-filter":
            out_dir = Path(cmd[cmd.index("-o") + 1])
            # Input wav path is the last positional arg
            in_wav = Path(cmd[-1])
            (out_dir / in_wav.name).write_bytes(b"")
        return MagicMock(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    input_mp4 = tmp_path / "in.mp4"
    input_mp4.write_bytes(b"fake")
    output_mp4 = tmp_path / "out.mp4"

    denoise(input_mp4, output_mp4)

    # Expected sequence: ffmpeg extract, deep-filter, ffmpeg mux
    assert len(calls) == 3
    assert calls[0][0] == "ffmpeg"
    assert "-vn" in calls[0]  # audio-only extract
    assert calls[1][0] == "deep-filter"
    assert calls[2][0] == "ffmpeg"
    # Mux should copy video bitstream
    assert "copy" in calls[2]


# --- loudnorm -----------------------------------------------------------

def test_loudnorm_builds_correct_ffmpeg_normalize_cmd(tmp_path, monkeypatch) -> None:
    captured: list[list[str]] = []
    monkeypatch.setattr(
        subprocess, "run",
        lambda cmd, **kwargs: captured.append(cmd) or MagicMock(returncode=0),
    )
    loudnorm(
        tmp_path / "in.mp4", tmp_path / "out.mp4",
        target=-14, true_peak=-1.5, lra=11,
    )
    cmd = captured[0]
    assert cmd[0] == "ffmpeg-normalize"
    # EBU R128 two-pass
    assert "-nt" in cmd and cmd[cmd.index("-nt") + 1] == "ebu"
    # Target, true peak, LRA (ffmpeg-normalize 1.x uses -lrt for LRA)
    assert cmd[cmd.index("-t") + 1] == "-14"
    assert cmd[cmd.index("-tp") + 1] == "-1.5"
    assert cmd[cmd.index("-lrt") + 1] == "11"
    # Audio encoded AAC 48k
    assert cmd[cmd.index("-c:a") + 1] == "aac"
    assert cmd[cmd.index("-ar") + 1] == "48000"


def test_loudnorm_uses_config_defaults(tmp_path, monkeypatch) -> None:
    from src import config
    captured: list[list[str]] = []
    monkeypatch.setattr(
        subprocess, "run",
        lambda cmd, **kwargs: captured.append(cmd) or MagicMock(returncode=0),
    )
    loudnorm(tmp_path / "in.mp4", tmp_path / "out.mp4")
    cmd = captured[0]
    assert cmd[cmd.index("-t") + 1] == str(config.LOUDNESS_TARGET)
    assert cmd[cmd.index("-tp") + 1] == str(config.TRUE_PEAK)
    assert cmd[cmd.index("-lrt") + 1] == str(config.LOUDNESS_RANGE)


# --- run (CLI) ----------------------------------------------------------

def test_run_skips_denoise_when_binary_missing(tmp_path, monkeypatch) -> None:
    """No deep-filter on PATH: denoise is silently skipped, loudnorm still runs."""
    monkeypatch.setattr(shutil, "which", lambda name: None)

    calls: list[list[str]] = []
    monkeypatch.setattr(
        subprocess, "run",
        lambda cmd, **kw: calls.append(cmd) or MagicMock(returncode=0),
    )

    input_mp4 = tmp_path / "in.mp4"
    input_mp4.write_bytes(b"fake")
    output_mp4 = tmp_path / "out.mp4"

    args = argparse.Namespace(
        input=input_mp4, output=output_mp4, skip_denoise=False,
        work=tmp_path, verbose=0,
    )
    rc = run(args)
    assert rc == 0
    # Exactly one subprocess call = ffmpeg-normalize (no denoise attempted)
    assert len(calls) == 1
    assert calls[0][0] == "ffmpeg-normalize"


def test_run_skips_denoise_with_flag_even_if_binary_present(tmp_path, monkeypatch) -> None:
    """--skip-denoise trumps deep-filter availability."""
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/deep-filter")

    calls: list[list[str]] = []
    monkeypatch.setattr(
        subprocess, "run",
        lambda cmd, **kw: calls.append(cmd) or MagicMock(returncode=0),
    )

    input_mp4 = tmp_path / "in.mp4"
    input_mp4.write_bytes(b"fake")
    output_mp4 = tmp_path / "out.mp4"

    args = argparse.Namespace(
        input=input_mp4, output=output_mp4, skip_denoise=True,
        work=tmp_path, verbose=0,
    )
    rc = run(args)
    assert rc == 0
    assert len(calls) == 1
    assert calls[0][0] == "ffmpeg-normalize"


def test_run_does_denoise_then_loudnorm_when_available(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/deep-filter")

    def fake_run(cmd, **kwargs):
        # deep-filter must produce its output file
        if cmd[0] == "deep-filter":
            out_dir = Path(cmd[cmd.index("-o") + 1])
            in_wav = Path(cmd[-1])
            (out_dir / in_wav.name).write_bytes(b"")
        return MagicMock(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    input_mp4 = tmp_path / "in.mp4"
    input_mp4.write_bytes(b"fake")
    output_mp4 = tmp_path / "out.mp4"

    args = argparse.Namespace(
        input=input_mp4, output=output_mp4, skip_denoise=False,
        work=tmp_path, verbose=0,
    )
    rc = run(args)
    assert rc == 0

    # Intermediate denoised file should have been created and then removed
    intermediate = input_mp4.parent / f"{input_mp4.stem}.dn.mp4"
    assert not intermediate.exists(), "intermediate should be cleaned up"


def test_run_missing_input_returns_error(tmp_path) -> None:
    args = argparse.Namespace(
        input=tmp_path / "does_not_exist.mp4",
        output=tmp_path / "out.mp4",
        skip_denoise=False,
        work=tmp_path, verbose=0,
    )
    rc = run(args)
    assert rc == 1


# --- real integration ---------------------------------------------------

def _ffprobe_has_audio(path: Path) -> bool:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_name",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True,
    )
    return bool(r.stdout.strip())


def _measure_loudness(path: Path) -> float:
    """Return integrated loudness (LUFS) via ffmpeg's loudnorm analysis pass."""
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-i", str(path),
         "-af", "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json",
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    # loudnorm prints a JSON block on stderr between `{` and `}` at the end.
    err = r.stderr
    start = err.rfind("{")
    end = err.rfind("}")
    if start < 0 or end < 0 or end < start:
        raise RuntimeError(f"could not parse loudnorm output:\n{err[-400:]}")
    stats = json.loads(err[start:end + 1])
    return float(stats["input_i"])


@pytest.mark.slow
def test_loudnorm_against_real_rendered_mp4(tmp_path) -> None:
    """Polish the previously-rendered work/final.mp4 (left by the E2E pipeline
    integration run) and verify the output really is at the loudness target.
    Skipped if that file is not present (e.g. on CI).
    """
    rendered = Path("work/final.mp4")
    if not rendered.exists():
        pytest.skip(
            "no work/final.mp4 to polish — run the full pipeline first "
            "(python -m src.cli run ...)"
        )
    output = tmp_path / "polished.mp4"
    loudnorm(rendered, output, target=-14, true_peak=-1.5, lra=11)

    assert output.exists()
    assert _ffprobe_has_audio(output)

    # Re-measure the output. Tolerance depends on whether ffmpeg-normalize
    # could do linear normalization or had to fall back to dynamic mode
    # (triggered when input LRA > target LRA, common on short clips with
    # mixed speech + silence). Dynamic mode lands within ~±2.5 LU of
    # target on a short fixture; linear lands within ±0.5 LU. We only
    # assert the more permissive spec so the test is robust against
    # expected-behavior fallback warnings.
    measured = _measure_loudness(output)
    assert abs(measured - (-14.0)) <= 2.5, (
        f"measured loudness {measured:.2f} LUFS deviates from target -14 "
        f"by more than 2.5 LU (EBU R128 dynamic-mode tolerance for short "
        f"samples)"
    )
