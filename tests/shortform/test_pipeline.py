"""Tests for src/shortform/pipeline.py — orchestrator + layout picker."""
from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.shortform.pipeline import (
    DEFAULT_MAX_SEC,
    DEFAULT_MIN_SEC,
    DEFAULT_TOP_N,
    _find_crop_at,
    _slugify,
    pick_layout_for_clip,
    run_all,
)
from src.shortform.reframe import CropWindow


# --- layout picker --------------------------------------------------

def test_layout_single_source_is_screen_full() -> None:
    assert pick_layout_for_clip("look at this", dual_track=False) == "screen_full"


def test_layout_strong_demo_cues_goes_split_vstack() -> None:
    """≥ 2 demo-pointing phrases → split_vstack."""
    out = pick_layout_for_clip(
        "look at this and see how the cursor moves right here",
        dual_track=True,
    )
    assert out == "split_vstack"


def test_layout_pure_narrative_hook_goes_cam_full() -> None:
    out = pick_layout_for_clip(
        "what if I told you I replaced my entire editor this week",
        dual_track=True,
    )
    assert out == "cam_full"


def test_layout_default_is_split_vstack() -> None:
    out = pick_layout_for_clip("generic narration with no specific cues",
                               dual_track=True)
    assert out == "split_vstack"


def test_layout_default_override() -> None:
    out = pick_layout_for_clip("generic narration", dual_track=True,
                               default="pip")
    assert out == "pip"


# --- slugify -------------------------------------------------------

def test_slugify_basic() -> None:
    assert _slugify("Why I stopped writing tests") == "why_i_stopped_writing_tests"


def test_slugify_strips_punct_and_caps() -> None:
    assert _slugify("Hello, World! How?") == "hello_world_how"


def test_slugify_truncates() -> None:
    long = "a_" * 40
    out = _slugify(long, max_len=10)
    assert len(out) <= 10


def test_slugify_empty_fallback() -> None:
    assert _slugify("") == "clip"
    assert _slugify("!@#$") == "clip"


# --- crop lookup ----------------------------------------------------

def test_find_crop_exact_match() -> None:
    crops = [CropWindow(0.0, 10.0, 0.3, 0.3), CropWindow(10.0, 20.0, 0.7, 0.7)]
    c = _find_crop_at(crops, 15.0, 0.5, 0.5)
    assert c.cx == 0.7


def test_find_crop_empty_returns_fallback() -> None:
    c = _find_crop_at([], 5.0, 0.2, 0.9)
    assert c.cx == 0.2 and c.cy == 0.9


def test_find_crop_nearest_when_no_exact() -> None:
    crops = [CropWindow(0.0, 5.0, 0.2, 0.2), CropWindow(10.0, 15.0, 0.8, 0.8)]
    c = _find_crop_at(crops, 6.0, 0.5, 0.5)
    # t=6 is 1s from first scene's end, 4s from second's start → nearest is first
    assert c.cx == 0.2


# --- run_all wiring -------------------------------------------------

def _make_args(tmp_path, **extra):
    base = dict(
        screen=tmp_path / "ext.mov",
        webcam=tmp_path / "cam.mov",
        audio=tmp_path / "audio.mp4",
        composited=None,
        cursor=None,
        output_dir=tmp_path / "out",
        top=2, min_sec=30.0, max_sec=60.0,
        work=tmp_path / "work",
    )
    base.update(extra)
    return argparse.Namespace(**base)


def test_run_all_requires_some_input(tmp_path) -> None:
    args = argparse.Namespace(
        screen=None, webcam=None, audio=None, composited=None,
        cursor=None, output_dir=None, top=2, min_sec=30.0, max_sec=60.0,
        work=tmp_path,
    )
    rc = run_all(args)
    assert rc == 1


def test_run_all_happy_path(tmp_path) -> None:
    """End-to-end wiring with every heavy call mocked."""
    for name in ("ext.mov", "cam.mov", "audio.mp4"):
        (tmp_path / name).write_bytes(b"")

    fake_words = [
        {"word": "look", "start": 0.0, "end": 0.3, "probability": 0.9},
        {"word": "here", "start": 0.3, "end": 0.6, "probability": 0.9},
    ]
    fake_sentences = [
        {"text": "look here.", "start": 0.0, "end": 0.6},
    ]

    from src.shortform.segment import Candidate
    from src.shortform.score import RankedClip, ShortformScore

    fake_ranked = [
        RankedClip(
            candidate=Candidate(start=0.0, end=35.0, sentence_idx_range=(0, 1)),
            llm=ShortformScore(score=8.0, title="Test Title"),
            audio_energy=0.6, punctuation=0.2, length_score=1.0,
            composite=0.74,
            effective_start=0.0, effective_end=35.0,
        ),
    ]

    def fake_render(spec, screen_path, webcam_path, audio_path, out_path, **kw):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"fakefile")
        return out_path

    with patch("src.shortform.pipeline.transcribe", return_value={
             "words": fake_words,
             "sentences": [type("S", (), s) for s in fake_sentences],
             "backend": "fake",
         }), \
         patch("src.shortform.pipeline.segment_topics",
               return_value=[fake_ranked[0].candidate]), \
         patch("src.shortform.pipeline.score_candidates",
               return_value=fake_ranked), \
         patch("src.shortform.pipeline.build_webcam_crops", return_value=[]), \
         patch("src.shortform.pipeline.build_screen_crops", return_value=[]), \
         patch("src.shortform.pipeline.build_ass",
               side_effect=lambda w, p: p.write_text("ASS")) , \
         patch("src.shortform.pipeline.render_clip", side_effect=fake_render):
        rc = run_all(_make_args(tmp_path))
    assert rc == 0
    # One short + one plan file written
    shorts_root = tmp_path / "out" / "ext"
    assert any(p.suffix == ".mp4" for p in shorts_root.iterdir())
    assert (tmp_path / "work" / "shortform_plan.json").exists()
