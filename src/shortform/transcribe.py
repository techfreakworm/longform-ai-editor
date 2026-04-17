"""Shortform transcription — parakeet-mlx preferred, mlx-whisper fallback.

Parakeet TDT 0.6B v3 runs ~110× realtime on M-class hardware via MLX
(per Simon Willison 2025-11-14). Apache-2.0 / CC-BY-4.0 license. About
4× faster than mlx-whisper, which matters when the source is a full
1-hour tutorial.

Output shape matches long-form Stage B's `work/words.json`:
    [{"word": str, "start": float, "end": float, "probability": float}, ...]

Also emits a sentence-level view used by the topical segmenter:
    [{"text": str, "start": float, "end": float}, ...]

If parakeet-mlx isn't installed (not in base extras), falls through
transparently to mlx-whisper.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class Sentence:
    text: str
    start: float
    end: float


# ----- parakeet path -------------------------------------------------

def _try_import_parakeet():
    try:
        import parakeet_mlx  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def _transcribe_parakeet(audio_path: Path, model_id: str) -> dict[str, Any]:
    """Run parakeet-mlx via its Python API. Returns a dict with "words"
    (list of {word, start, end, probability}) and "sentences" (list of
    Sentence)."""
    from parakeet_mlx import from_pretrained  # type: ignore[import-not-found]

    model = from_pretrained(model_id)
    result = model.transcribe(str(audio_path))

    words: list[dict[str, Any]] = []
    sentences: list[Sentence] = []

    # parakeet-mlx >= 0.3 returns an object with .sentences containing
    # .tokens (AlignedToken with .text .start .end). API names may
    # change — keep the extraction permissive.
    segs = getattr(result, "sentences", None) or getattr(result, "segments", [])
    for seg in segs:
        seg_text = (getattr(seg, "text", None) or "").strip()
        seg_start = float(getattr(seg, "start", 0.0))
        seg_end = float(getattr(seg, "end", seg_start))
        if seg_text and seg_end > seg_start:
            sentences.append(Sentence(text=seg_text, start=seg_start, end=seg_end))
        tokens = getattr(seg, "tokens", None) or []
        for tok in tokens:
            w = (getattr(tok, "text", None) or "").strip()
            if not w:
                continue
            words.append({
                "word": w,
                "start": float(getattr(tok, "start", seg_start)),
                "end": float(getattr(tok, "end", seg_end)),
                "probability": float(getattr(tok, "probability", 1.0)),
            })

    return {"words": words, "sentences": sentences}


# ----- whisper fallback ---------------------------------------------

def _transcribe_whisper(audio_path: Path, model_id: str) -> dict[str, Any]:
    """Fallback: run mlx-whisper (already a base dep). Builds sentences
    by splitting on punctuation / long silences."""
    from src.stages.transcribe import transcribe as lf_transcribe, flatten_words

    raw = lf_transcribe(audio_path, model_id)
    words = flatten_words(raw)

    # Derive sentences by splitting on terminal punctuation or a long
    # inter-word gap (> 0.8 s).
    sentences: list[Sentence] = []
    buf: list[str] = []
    s_start = None
    s_end = 0.0
    prev_end = 0.0
    for w in words:
        tok = w["word"].strip()
        if not tok:
            continue
        if s_start is None:
            s_start = float(w["start"])
        gap = float(w["start"]) - prev_end if prev_end else 0.0
        buf.append(tok)
        s_end = float(w["end"])
        prev_end = s_end
        if tok.endswith((".", "?", "!")) or gap > 0.8:
            text = " ".join(buf).strip()
            if text and s_start is not None and s_end > s_start:
                sentences.append(Sentence(text=text, start=s_start, end=s_end))
            buf, s_start = [], None
    if buf and s_start is not None:
        sentences.append(Sentence(text=" ".join(buf).strip(),
                                  start=s_start, end=s_end))

    return {"words": words, "sentences": sentences}


# ----- public API ---------------------------------------------------

DEFAULT_PARAKEET_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"


def transcribe(
    audio_path: Path,
    *,
    parakeet_model: str = DEFAULT_PARAKEET_MODEL,
    whisper_model: str | None = None,
) -> dict[str, Any]:
    """Prefer parakeet-mlx; fall back to mlx-whisper.

    Returns:
        {"words": [...], "sentences": [Sentence(...)], "backend": "parakeet"|"whisper"}
    """
    if _try_import_parakeet():
        try:
            result = _transcribe_parakeet(audio_path, parakeet_model)
            result["backend"] = "parakeet"
            return result
        except Exception as exc:  # noqa: BLE001
            log.warning("parakeet-mlx failed (%s) — falling back to whisper", exc)

    from src import config
    result = _transcribe_whisper(audio_path, whisper_model or config.WHISPER_MODEL)
    result["backend"] = "whisper"
    return result


__all__ = [
    "Sentence",
    "DEFAULT_PARAKEET_MODEL",
    "transcribe",
]
