"""Tests for src/shortform/score.py — LLM + composite scoring."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.shortform.score import (
    RankedClip,
    ShortformScore,
    _audio_energy_score,
    _length_preference,
    _punctuation_density,
    score_candidates,
    score_clip_llm,
)
from src.shortform.segment import Candidate


# --- schema ----------------------------------------------------------

def test_shortform_score_defaults() -> None:
    s = ShortformScore(score=7.0)
    assert s.score == 7.0
    assert s.title == ""
    assert s.start_offset == 0.0


def test_shortform_score_clamps_via_pydantic() -> None:
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ShortformScore(score=11.0)
    with pytest.raises(ValidationError):
        ShortformScore(score=-0.5)


# --- heuristics ------------------------------------------------------

def test_punctuation_density_empty_string() -> None:
    assert _punctuation_density("") == 0.0


def test_punctuation_density_no_marks() -> None:
    assert _punctuation_density("this is a calm sentence") == 0.0


def test_punctuation_density_many_marks() -> None:
    # 5 marks in ~60 chars → well above the 5-per-100 saturation point.
    # But density = 5 marks / (60/100) = ~8.3 per_100 → capped at 1.0.
    s = "wow! really? amazing! how! why?"
    assert _punctuation_density(s) == pytest.approx(1.0, abs=0.01)


def test_length_preference_inside_band_is_one() -> None:
    assert _length_preference(45.0, 30.0, 60.0) == 1.0
    assert _length_preference(30.0, 30.0, 60.0) == 1.0
    assert _length_preference(60.0, 30.0, 60.0) == 1.0


def test_length_preference_below_band() -> None:
    # 15 s with sweet_min 30 → 50% below → drop 50%
    assert _length_preference(15.0, 30.0, 60.0) == pytest.approx(0.5, abs=0.01)


def test_length_preference_above_band() -> None:
    # 90 s with sweet_max 60 → 50% over → drop 50%
    assert _length_preference(90.0, 30.0, 60.0) == pytest.approx(0.5, abs=0.01)


def test_length_preference_way_off_zero() -> None:
    assert _length_preference(300.0, 30.0, 60.0) == 0.0


# --- LLM call --------------------------------------------------------

def test_score_clip_llm_happy_path() -> None:
    fake = {
        "score": 8.2,
        "title": "Why I stopped writing tests myself",
        "reason": "provocative hook, self-contained 30s demo",
        "start_offset": 0.0, "end_offset": 0.0,
    }
    with patch("src.stages.analyze_llm.call_llm_json", return_value=fake):
        out = score_clip_llm(0.0, 45.0, "some transcript here")
    assert out.score == 8.2
    assert out.title.startswith("Why I stopped")


# --- score_candidates end-to-end ------------------------------------

def test_score_candidates_ranks_by_composite(tmp_path) -> None:
    """Two candidates with different synthetic scores → high first."""
    audio_path = tmp_path / "a.wav"
    audio_path.write_bytes(b"")

    cands = [
        Candidate(start=0.0, end=45.0, sentence_idx_range=(0, 1)),     # sweet spot
        Candidate(start=50.0, end=250.0, sentence_idx_range=(1, 2)),   # far too long
    ]
    sentences = [
        {"text": "Amazing! You have to see this.", "start": 0.0, "end": 45.0},
        {"text": "And then we talked for ages about unrelated topics.",
         "start": 50.0, "end": 250.0},
    ]

    fake_scores = iter([
        ShortformScore(score=8.0, title="t1"),
        ShortformScore(score=6.0, title="t2"),
    ])

    with patch("src.shortform.score.score_clip_llm",
               side_effect=lambda *_a, **_kw: next(fake_scores)), \
         patch("src.shortform.score._audio_energy_score", return_value=0.7):
        ranked = score_candidates(
            cands, sentences, audio_path,
            sweet_min=30.0, sweet_max=60.0,
        )
    assert len(ranked) == 2
    assert ranked[0].candidate.start == 0.0   # sweet-spot wins
    assert ranked[0].composite > ranked[1].composite


def test_score_candidates_respects_top_n(tmp_path) -> None:
    audio_path = tmp_path / "a.wav"
    audio_path.write_bytes(b"")
    cands = [
        Candidate(start=i * 60.0, end=i * 60.0 + 45.0, sentence_idx_range=(i, i + 1))
        for i in range(5)
    ]
    sentences = [
        {"text": f"clip {i}", "start": i * 60.0, "end": i * 60.0 + 45.0}
        for i in range(5)
    ]
    with patch("src.shortform.score.score_clip_llm",
               return_value=ShortformScore(score=5.0)), \
         patch("src.shortform.score._audio_energy_score", return_value=0.5):
        ranked = score_candidates(
            cands, sentences, audio_path, top_n=2,
        )
    assert len(ranked) == 2


def test_score_candidates_handles_llm_failure(tmp_path) -> None:
    """LLM raises → candidate still ranked using heuristics only."""
    audio_path = tmp_path / "a.wav"
    audio_path.write_bytes(b"")
    cands = [Candidate(start=0.0, end=45.0, sentence_idx_range=(0, 1))]
    sentences = [{"text": "boring", "start": 0.0, "end": 45.0}]

    with patch("src.shortform.score.score_clip_llm",
               side_effect=RuntimeError("claude down")), \
         patch("src.shortform.score._audio_energy_score", return_value=0.5):
        ranked = score_candidates(cands, sentences, audio_path)
    assert len(ranked) == 1
    assert ranked[0].llm is None
    assert 0.0 < ranked[0].composite <= 1.0


def test_score_candidates_trim_offsets_applied(tmp_path) -> None:
    audio_path = tmp_path / "a.wav"
    audio_path.write_bytes(b"")
    cands = [Candidate(start=100.0, end=160.0, sentence_idx_range=(0, 1))]
    sentences = [{"text": "hello", "start": 100.0, "end": 160.0}]

    with patch("src.shortform.score.score_clip_llm",
               return_value=ShortformScore(
                   score=7.0, start_offset=2.0, end_offset=1.5,
               )), \
         patch("src.shortform.score._audio_energy_score", return_value=0.5):
        ranked = score_candidates(cands, sentences, audio_path, call_llm=True)
    assert ranked[0].effective_start == pytest.approx(102.0)
    assert ranked[0].effective_end == pytest.approx(158.5)


def test_score_candidates_skip_llm_uses_heuristics_only(tmp_path) -> None:
    audio_path = tmp_path / "a.wav"
    audio_path.write_bytes(b"")
    cands = [Candidate(start=0.0, end=45.0, sentence_idx_range=(0, 1))]
    sentences = [{"text": "hi!", "start": 0.0, "end": 45.0}]

    with patch("src.shortform.score._audio_energy_score", return_value=0.5):
        ranked = score_candidates(cands, sentences, audio_path, call_llm=False)
    assert ranked[0].llm is None


# --- audio energy (skipped if librosa unavailable) ------------------

def test_audio_energy_returns_default_on_missing_file(tmp_path) -> None:
    # librosa will fail on an empty/nonexistent file → defaults to 0.5
    out = _audio_energy_score(tmp_path / "nope.wav", 0.0, 10.0)
    assert out == 0.5
