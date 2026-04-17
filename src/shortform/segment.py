"""Topical segmentation — multi-scale TextTiling on sentence embeddings.

Implements the Hearst (1997) TextTiling algorithm adapted for
RoBERTa-style sentence embeddings with a multi-scale sliding-window
pass (inspired by ClipsAI's implementation; not vendored to avoid
pulling in their filesystem + config manager stack).

Algorithm:
  1. Split transcript into sentences (already the case with Parakeet).
  2. Compute embedding per sentence via sentence-transformers.
  3. For each window size k in {5, 7, 11, 17, 37, 53, 73, 97}:
       a. Slide a pair of k-sentence windows across the text.
       b. Gap score at position i = 1 − cos(mean_emb(L), mean_emb(R))
          where L = sentences [i-k, i) and R = sentences [i, i+k).
       c. Smooth gap scores with a 3-tap moving average.
       d. Compute depth scores: for each gap score, subtract the max
          of (left-neighbor peak, right-neighbor peak).
       e. Threshold depth score > mean + 0.5*std → boundary.
  4. Union boundaries across all scales; dedupe within 15 s.
  5. Convert boundaries (sentence indices) into (start_s, end_s)
     windows bounded by the sentences' actual timestamps.

Graceful degradation: if sentence-transformers isn't installed, or
there are fewer than 2k sentences for the smallest k, we fall back to
fixed-length chunking. Callers can still work — candidate selection
just becomes less topical.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_WINDOW_SIZES = (5, 7, 11, 17, 37, 53, 73, 97)
DEFAULT_MODEL = "sentence-transformers/all-roberta-large-v1"
DEDUP_GAP_SEC = 15.0
FALLBACK_CHUNK_SEC = 60.0


@dataclass
class Candidate:
    start: float
    end: float
    sentence_idx_range: tuple[int, int]  # [start, end)


# ----- embedder ------------------------------------------------------

_embedder_cache: dict[str, Any] = {}


def _get_embedder(model_name: str):
    """Lazy-load the sentence-transformers model; memoized per name."""
    if model_name in _embedder_cache:
        return _embedder_cache[model_name]
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        _embedder_cache[model_name] = model
        return model
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "shortform.segment: sentence-transformers unavailable (%s) — "
            "falling back to fixed-length chunking. "
            "Install with `pip install '.[shortform]'`.",
            exc,
        )
        _embedder_cache[model_name] = None
        return None


# ----- core TextTiling ----------------------------------------------

def _gap_scores(embeddings: np.ndarray, k: int) -> np.ndarray:
    """Compute 1 − cosine similarity between paired k-sentence windows.

    Returns an array of length max(0, n - 2k + 1).
    """
    n = embeddings.shape[0]
    if n < 2 * k:
        return np.zeros(0, dtype=np.float32)

    # L2-normalize for stable cosine.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    unit = embeddings / norms

    # Sliding window mean via cumulative sum.
    csum = np.concatenate([np.zeros((1, unit.shape[1])), unit.cumsum(axis=0)], axis=0)
    window_means = (csum[k:] - csum[:-k]) / k  # (n - k + 1, d)

    # Pair means at position i: L = window_means[i-k], R = window_means[i]
    # Valid i: k <= i <= n - k  → left idx i-k, right idx i
    left = window_means[: -k]    # (n - 2k + 1, d)
    right = window_means[k:]     # (n - 2k + 1, d)

    cos = (left * right).sum(axis=1) / (
        np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1) + 1e-9
    )
    return 1.0 - cos


def _smooth(x: np.ndarray, window: int = 3) -> np.ndarray:
    """Moving-average smoothing. Edge-preserving via np.convolve with
    symmetric padding."""
    if x.size == 0 or window <= 1:
        return x
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(x, kernel, mode="same")


def _depth_scores(gap_scores: np.ndarray) -> np.ndarray:
    """Peak-depth of each gap position vs. its surrounding peaks.

    For each index i, find the nearest local maxima to the left and
    right, and depth = (peak_left - gap[i]) + (peak_right - gap[i]).
    A deep depth means gap[i] sits in a valley between two peaks.
    """
    n = gap_scores.size
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    left_peak = np.zeros(n, dtype=np.float32)
    right_peak = np.zeros(n, dtype=np.float32)

    # Left-running peak
    peak = gap_scores[0]
    for i in range(n):
        peak = max(peak, gap_scores[i])
        left_peak[i] = peak
    # Reset at valleys
    for i in range(1, n):
        if gap_scores[i] < gap_scores[i - 1]:
            # start of descent — reset peak to this position's LHS peak
            left_peak[i] = max(gap_scores[i], left_peak[i - 1])

    # Right-running peak
    peak = gap_scores[-1]
    for i in range(n - 1, -1, -1):
        peak = max(peak, gap_scores[i])
        right_peak[i] = peak
    for i in range(n - 2, -1, -1):
        if gap_scores[i] < gap_scores[i + 1]:
            right_peak[i] = max(gap_scores[i], right_peak[i + 1])

    return (left_peak - gap_scores) + (right_peak - gap_scores)


def _find_boundaries_at_scale(
    embeddings: np.ndarray, k: int, threshold_std: float = 0.5,
) -> list[int]:
    """Return sentence indices where a topic boundary is detected at
    window size k. Indices are in the ORIGINAL sentence list space."""
    gaps = _gap_scores(embeddings, k)
    if gaps.size == 0:
        return []
    gaps = _smooth(gaps)
    depths = _depth_scores(gaps)
    if depths.size == 0:
        return []
    cutoff = float(depths.mean() + threshold_std * depths.std())
    # Convert gap-score index i back to sentence index: that gap lies
    # between sentences (i-k + k) and (i-k + k), i.e. sentence i + k.
    # We recorded gaps at indices [0, n - 2k + 1); original sentence
    # boundary offset = i + k.
    return [int(i + k) for i in range(depths.size) if depths[i] >= cutoff]


# ----- fallback chunker ---------------------------------------------

def _fixed_length_candidates(
    sentences: list[dict], target_sec: float = FALLBACK_CHUNK_SEC,
) -> list[Candidate]:
    """Simple fixed-duration windows when embeddings aren't available.

    Sentences expected to carry `start` and `end` in seconds.
    """
    if not sentences:
        return []
    out: list[Candidate] = []
    chunk_start = float(sentences[0]["start"])
    chunk_start_idx = 0
    for i, s in enumerate(sentences):
        if float(s["end"]) - chunk_start >= target_sec:
            out.append(Candidate(
                start=chunk_start, end=float(s["end"]),
                sentence_idx_range=(chunk_start_idx, i + 1),
            ))
            chunk_start = float(s["end"])
            chunk_start_idx = i + 1
    if chunk_start_idx < len(sentences):
        out.append(Candidate(
            start=chunk_start, end=float(sentences[-1]["end"]),
            sentence_idx_range=(chunk_start_idx, len(sentences)),
        ))
    return out


# ----- public API ---------------------------------------------------

def segment_topics(
    sentences: list[dict],
    *,
    window_sizes: tuple[int, ...] = DEFAULT_WINDOW_SIZES,
    model_name: str = DEFAULT_MODEL,
    dedup_gap_sec: float = DEDUP_GAP_SEC,
) -> list[Candidate]:
    """Multi-scale TextTiling on sentence-level transcript.

    Args:
        sentences: list of dicts with "text", "start", "end".
        window_sizes: list of k values to try.
        model_name: sentence-transformers model.
        dedup_gap_sec: merge boundaries within this many seconds.

    Returns:
        Candidate clips, sorted by start. When embedding isn't
        available, returns fixed-length chunks.
    """
    if len(sentences) < 2 * min(window_sizes):
        # Not enough sentences for the smallest window → fall back.
        return _fixed_length_candidates(sentences)

    model = _get_embedder(model_name)
    if model is None:
        return _fixed_length_candidates(sentences)

    texts = [s["text"] for s in sentences]
    try:
        embeddings = np.asarray(model.encode(texts, show_progress_bar=False))
    except Exception as exc:  # noqa: BLE001
        log.warning("shortform.segment: encode failed (%s) — fallback", exc)
        return _fixed_length_candidates(sentences)

    # Boundaries in sentence-index space.
    boundary_set: set[int] = {0, len(sentences)}
    for k in window_sizes:
        boundary_set.update(_find_boundaries_at_scale(embeddings, k))

    # Sort + dedup close-in-time boundaries.
    boundary_sorted = sorted(boundary_set)
    kept: list[int] = [boundary_sorted[0]]
    for b in boundary_sorted[1:]:
        if b <= kept[-1]:
            continue
        b = min(b, len(sentences))
        prev_t = float(sentences[kept[-1]]["start"]) if kept[-1] < len(sentences) \
            else float(sentences[-1]["end"])
        this_t = float(sentences[b]["start"]) if b < len(sentences) \
            else float(sentences[-1]["end"])
        if (this_t - prev_t) >= dedup_gap_sec:
            kept.append(b)
    if kept[-1] != len(sentences):
        kept.append(len(sentences))

    candidates: list[Candidate] = []
    for i in range(len(kept) - 1):
        s_idx = kept[i]
        e_idx = kept[i + 1]
        if e_idx <= s_idx:
            continue
        candidates.append(Candidate(
            start=float(sentences[s_idx]["start"]),
            end=float(sentences[e_idx - 1]["end"]),
            sentence_idx_range=(s_idx, e_idx),
        ))
    return candidates


__all__ = [
    "Candidate",
    "DEFAULT_MODEL",
    "DEFAULT_WINDOW_SIZES",
    "DEDUP_GAP_SEC",
    "FALLBACK_CHUNK_SEC",
    "segment_topics",
]
