"""Shortform candidate scoring — LLM + composite heuristics.

Each Candidate emitted by `segment.py` is scored by:
  1. An LLM call (Claude CLI preferred, MLX fallback — same dispatcher
     as long-form `analyze_llm.py`) that returns score/title/reason
     plus optional start/end trim offsets.
  2. Cheap local heuristics — audio energy peak-to-median RMS,
     punctuation density, length preference — that combine with the
     LLM score into a composite rank.

Weights come from the plan doc:
    composite = 0.45 * (LLM/10)
              + 0.20 * audio_energy_norm
              + 0.15 * punctuation_density_norm
              + 0.10 * topic_depth_placeholder
              + 0.10 * length_preference

Graceful degradation:
  - LLM unreachable → composite skips the LLM term and renormalizes.
  - librosa not available (never in practice; it's a base dep) →
    audio_energy term defaults to 0.5.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError
from tenacity import retry, stop_after_attempt, wait_fixed

from src import config
from src.shortform.segment import Candidate

log = logging.getLogger(__name__)


# ----- pydantic schema -----------------------------------------------

class ShortformScore(BaseModel):
    score: float = Field(ge=0.0, le=10.0)
    title: str = ""
    reason: str = ""
    start_offset: float = 0.0
    end_offset: float = 0.0


@dataclass
class RankedClip:
    candidate: Candidate
    llm: ShortformScore | None
    audio_energy: float          # peak-to-median RMS, normalized
    punctuation: float           # punctuation density (0 = none, 1 = dense)
    length_score: float          # preference: 30–60 s peaks at 1.0
    composite: float
    effective_start: float       # candidate.start + llm.start_offset
    effective_end: float         # candidate.end   - llm.end_offset


# ----- LLM call ------------------------------------------------------

@retry(
    stop=stop_after_attempt(config.LLM_MAX_RETRIES),
    wait=wait_fixed(1),
    reraise=True,
)
def score_clip_llm(start: float, end: float, transcript: str) -> ShortformScore:
    """LLM-score one clip. Routes through the same dispatcher as
    long-form analysis (Claude CLI preferred, MLX fallback)."""
    from src.stages.analyze_llm import call_llm_json

    raw = call_llm_json(
        config.SHORTFORM_SCORING_PROMPT,
        {"start": start, "end": end, "transcript": transcript},
    )
    try:
        return ShortformScore(**raw)
    except ValidationError as e:
        log.warning("shortform score failed schema: %s — retrying", e)
        raise


# ----- heuristic components -----------------------------------------

def _audio_energy_score(audio_path: Path, start: float, end: float) -> float:
    """Peak-to-median RMS on the [start, end] window, squashed to [0, 1].

    High values = dynamic range (speaker raises voice, laughs, gestures
    vocally). Quiet monotone narrations score near 0.
    """
    try:
        import librosa
        import numpy as np

        duration = max(end - start, 0.1)
        y, _sr = librosa.load(str(audio_path), sr=22050, offset=start,
                              duration=duration, mono=True)
        if y.size == 0:
            return 0.5
        rms = librosa.feature.rms(y=y)[0]
        denom = float(np.std(rms)) + 1e-6
        energy = (float(np.max(rms)) - float(np.median(rms))) / denom
        return float(np.clip(energy / 5.0, 0.0, 1.0))
    except Exception as exc:  # noqa: BLE001
        log.warning("audio_energy failed (%s) — using 0.5", exc)
        return 0.5


def _punctuation_density(transcript: str) -> float:
    """Fraction of characters that are question/exclamation marks,
    normalized so ~5 marks per 100 chars reaches 1.0."""
    if not transcript:
        return 0.0
    hits = transcript.count("?") + transcript.count("!")
    per_100 = hits / max(len(transcript) / 100, 1.0)
    return min(per_100 / 5.0, 1.0)


def _length_preference(duration_s: float, sweet_min: float, sweet_max: float) -> float:
    """1.0 inside [sweet_min, sweet_max], dropping off linearly past ±50% of the band."""
    if sweet_min <= duration_s <= sweet_max:
        return 1.0
    if duration_s < sweet_min:
        drop = (sweet_min - duration_s) / sweet_min
        return max(0.0, 1.0 - drop)
    # duration_s > sweet_max
    drop = (duration_s - sweet_max) / sweet_max
    return max(0.0, 1.0 - drop)


# ----- main entry ----------------------------------------------------

def score_candidates(
    candidates: list[Candidate],
    sentences: list[dict[str, Any]],
    audio_path: Path,
    *,
    sweet_min: float = 30.0,
    sweet_max: float = 60.0,
    top_n: int | None = None,
    call_llm: bool = True,
    progress: bool = True,
) -> list[RankedClip]:
    """Rank candidates by composite score.

    Args:
        candidates: output of segment_topics.
        sentences: full sentence list — used to build per-clip transcripts
            from the sentence_idx_range on each Candidate.
        audio_path: source audio for librosa energy.
        sweet_min, sweet_max: preferred clip duration band.
        top_n: if set, truncate to this many after scoring.
        call_llm: set False to skip LLM calls entirely (tests / dry run).

    Returns:
        RankedClip list sorted by composite score desc.
    """
    ranked: list[RankedClip] = []
    total = len(candidates)
    for i, cand in enumerate(candidates, start=1):
        if progress:
            print(f"[shortform]   scoring {i}/{total} "
                  f"[{cand.start:.1f}-{cand.end:.1f}s]",
                  flush=True)
        s_idx, e_idx = cand.sentence_idx_range
        transcript = " ".join(
            str(s.get("text", "")).strip()
            for s in sentences[s_idx:e_idx]
        )

        llm: ShortformScore | None = None
        if call_llm:
            try:
                llm = score_clip_llm(cand.start, cand.end, transcript)
            except Exception as exc:  # noqa: BLE001
                log.warning("LLM score failed for [%s, %s]: %s — using neutral",
                            cand.start, cand.end, exc)

        audio = _audio_energy_score(audio_path, cand.start, cand.end)
        punct = _punctuation_density(transcript)
        length = _length_preference(cand.end - cand.start, sweet_min, sweet_max)

        if llm is not None:
            composite = (
                0.45 * (llm.score / 10.0)
                + 0.20 * audio
                + 0.15 * punct
                + 0.10 * 0.5   # topic-depth placeholder; future work
                + 0.10 * length
            )
        else:
            # Renormalize the non-LLM terms to sum to 1.0 so composite
            # stays comparable across candidates.
            composite = (
                0.20 * audio + 0.15 * punct + 0.10 * 0.5 + 0.10 * length
            ) / 0.55

        eff_start = cand.start + (llm.start_offset if llm else 0.0)
        eff_end = cand.end - (llm.end_offset if llm else 0.0)
        if eff_end <= eff_start:
            eff_start, eff_end = cand.start, cand.end

        ranked.append(RankedClip(
            candidate=cand, llm=llm,
            audio_energy=audio, punctuation=punct, length_score=length,
            composite=composite,
            effective_start=eff_start, effective_end=eff_end,
        ))

    ranked.sort(key=lambda r: r.composite, reverse=True)
    if top_n is not None:
        ranked = ranked[:top_n]
    return ranked


__all__ = [
    "ShortformScore",
    "RankedClip",
    "score_clip_llm",
    "score_candidates",
]
