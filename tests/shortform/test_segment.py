"""Tests for src/shortform/segment.py — multi-scale TextTiling."""
from __future__ import annotations

import numpy as np
import pytest

from src.shortform.segment import (
    DEDUP_GAP_SEC,
    FALLBACK_CHUNK_SEC,
    Candidate,
    _depth_scores,
    _find_boundaries_at_scale,
    _fixed_length_candidates,
    _gap_scores,
    _smooth,
    segment_topics,
)


# --- gap_scores -------------------------------------------------------

def test_gap_scores_identical_windows_zero() -> None:
    """Two identical-content windows → cosine ≈ 1 → gap ≈ 0."""
    emb = np.tile(np.array([1.0, 0.0, 0.0]), (10, 1))  # all the same
    out = _gap_scores(emb, k=3)
    assert out.shape == (10 - 2 * 3 + 1,)
    assert np.allclose(out, 0.0, atol=1e-6)


def test_gap_scores_orthogonal_windows_max() -> None:
    """Orthogonal windows → cosine ≈ 0 → gap ≈ 1."""
    emb = np.zeros((10, 2), dtype=np.float32)
    emb[:5] = np.array([1.0, 0.0])
    emb[5:] = np.array([0.0, 1.0])
    out = _gap_scores(emb, k=5)
    # Only one valid gap position at i=5; windows [0,5) and [5,10) fully split
    assert out.shape == (1,)
    assert out[0] == pytest.approx(1.0, abs=0.01)


def test_gap_scores_too_few_sentences_empty() -> None:
    emb = np.random.rand(3, 8)
    assert _gap_scores(emb, k=5).size == 0


# --- smoothing + depth -----------------------------------------------

def test_smooth_preserves_constant_signal() -> None:
    x = np.ones(10, dtype=np.float32) * 3.0
    out = _smooth(x, window=3)
    # mode="same" convolution with 1/3 kernel zero-pads the edges, so
    # first/last elements drop to 2.0 (two 1s + one 0 in-window).
    assert np.allclose(out[1:-1], 3.0, atol=1e-6)


def test_smooth_noop_on_empty() -> None:
    x = np.zeros(0, dtype=np.float32)
    assert _smooth(x).size == 0


def test_depth_scores_flat_returns_zeros() -> None:
    x = np.ones(10, dtype=np.float32) * 0.5
    out = _depth_scores(x)
    assert np.allclose(out, 0.0)


def test_depth_scores_single_valley_peaks_depth() -> None:
    """A single valley [1, 0, 1] should have a positive depth at index 1."""
    x = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    out = _depth_scores(x)
    assert out[1] > out[0]
    assert out[1] > out[2]


# --- boundary finding -------------------------------------------------

def test_find_boundaries_at_scale_handles_small_input() -> None:
    """Fewer than 2k sentences → no boundaries."""
    emb = np.random.rand(5, 4).astype(np.float32)
    assert _find_boundaries_at_scale(emb, k=5) == []


def test_find_boundaries_at_scale_detects_abrupt_topic_shift() -> None:
    """Synthetic half-A/half-B embeddings → boundary near the midpoint."""
    np.random.seed(0)
    dim = 8
    a = np.random.rand(10, dim).astype(np.float32)
    b = np.random.rand(10, dim).astype(np.float32) + 2.0  # shifted far
    emb = np.concatenate([a, b], axis=0)
    boundaries = _find_boundaries_at_scale(emb, k=3, threshold_std=0.0)
    assert any(8 <= b <= 12 for b in boundaries)


# --- fallback --------------------------------------------------------

def test_fixed_length_candidates_chunks_at_target() -> None:
    sentences = [{"text": f"s{i}", "start": float(i), "end": float(i) + 0.9}
                 for i in range(180)]  # 180 s worth
    out = _fixed_length_candidates(sentences, target_sec=60.0)
    assert len(out) == 3
    assert out[0].start == pytest.approx(0.0)
    assert out[-1].end == pytest.approx(179.9)


def test_fixed_length_candidates_empty() -> None:
    assert _fixed_length_candidates([]) == []


def test_fixed_length_candidates_short_input_single_chunk() -> None:
    sentences = [{"text": "a", "start": 0.0, "end": 3.0}]
    out = _fixed_length_candidates(sentences, target_sec=60.0)
    assert len(out) == 1


# --- public API ------------------------------------------------------

def test_segment_topics_falls_back_when_sentences_too_few() -> None:
    """Smallest window is 5 → need ≥ 10 sentences for real tiling."""
    sentences = [{"text": f"s{i}", "start": float(i), "end": float(i) + 0.9}
                 for i in range(6)]
    out = segment_topics(sentences)
    # Fallback → at least one chunk covering the input
    assert len(out) >= 1
    assert out[0].start == 0.0


def test_segment_topics_returns_empty_on_empty_input() -> None:
    assert segment_topics([]) == []


def test_candidate_dataclass_fields() -> None:
    c = Candidate(start=1.0, end=2.0, sentence_idx_range=(0, 3))
    assert c.start == 1.0
    assert c.sentence_idx_range == (0, 3)
