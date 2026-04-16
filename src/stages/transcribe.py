"""Stage B.1 — transcribe webcam audio with mlx-whisper.

Produces `work/words.json` — a flat list of word-level timestamps:
  [{"word": "hello", "start": 1.234, "end": 1.402, "probability": 0.99}, ...]

Cache is keyed by an audio content fingerprint (sha256 of first MB +
last MB + size). Re-running with the same audio file is a fast no-op.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

from src import config
from src.utils.ffmpeg_helpers import probe_duration

log = logging.getLogger(__name__)


def audio_fingerprint(path: Path) -> str:
    """sha256 of the first 1 MB + last 1 MB + size. Fast and unique enough
    to cache transcription output — don't re-hash GB of audio."""
    size = path.stat().st_size
    h = hashlib.sha256()
    h.update(str(size).encode())
    chunk = 1 << 20  # 1 MB
    with path.open("rb") as f:
        head = f.read(chunk)
        h.update(head)
        if size > 2 * chunk:
            f.seek(max(0, size - chunk))
            tail = f.read(chunk)
            h.update(tail)
    return h.hexdigest()[:16]  # 64-bit truncation is plenty


def cache_path_for(audio_path: Path, cache_dir: Path) -> Path:
    return cache_dir / f"words-{audio_fingerprint(audio_path)}.json"


def transcribe(
    audio_path: Path,
    model: str,
    *,
    word_timestamps: bool = True,
) -> dict[str, Any]:
    """Run mlx_whisper.transcribe. Returns the full dict: {text, segments, language}."""
    import mlx_whisper
    log.info("transcribing %s with %s", audio_path, model)
    return mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=model,
        word_timestamps=word_timestamps,
        verbose=None,
    )


def flatten_words(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the nested segments → a flat list of word entries.

    mlx-whisper emits segments containing a `words` list; we want a
    single timeline-ordered list of {word, start, end, probability}.
    """
    words: list[dict[str, Any]] = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            # Defensive: some words lack probability or have None timestamps
            if w.get("start") is None or w.get("end") is None:
                continue
            words.append({
                "word": w.get("word", "").strip(),
                "start": float(w["start"]),
                "end": float(w["end"]),
                "probability": float(w.get("probability", 0.0)),
            })
    return words


def transcribe_and_cache(
    audio_path: Path,
    model: str,
    cache_dir: Path,
    *,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Transcribe and cache. If the cache file exists and force is False,
    return its contents without invoking the model.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_path_for(audio_path, cache_dir)
    if cached.exists() and not force:
        log.info("using cached transcription %s", cached)
        return json.loads(cached.read_text())["words"]

    result = transcribe(audio_path, model)
    words = flatten_words(result)

    cached.write_text(json.dumps(
        {
            "source": str(audio_path),
            "model": model,
            "language": result.get("language"),
            "words": words,
        },
        indent=2,
    ))
    return words


def run_analyze(args: argparse.Namespace) -> int:
    """Stage B CLI entry: transcribe → analyze via LLM → write 3 JSON artifacts."""
    work_dir = Path(args.work) if getattr(args, "work", None) else config.WORK_DIR
    work_dir.mkdir(parents=True, exist_ok=True)

    webcam: Path = args.webcam
    audio: Path = getattr(args, "audio", None) or webcam

    words_path = work_dir / "words.json"
    filler_path = work_dir / "filler_cuts.json"
    layout_path = work_dir / "layout_plan.json"

    try:
        # --- B.1 transcribe (with cache) ----------------------------
        cache_dir = work_dir / "transcribe_cache"
        print(f"[analyze] transcribing audio: {audio.name}")
        words = transcribe_and_cache(audio, config.WHISPER_MODEL, cache_dir)
        words_path.write_text(json.dumps({"words": words}, indent=2))
        print(f"[analyze]   {len(words)} words → {words_path}")

        # --- B.2 LLM analysis ---------------------------------------
        # Imported lazily so CLI --help works without httpx/tenacity installed.
        from src.stages.analyze_llm import analyze_fillers, analyze_layout

        print(f"[analyze] calling LLM for filler cuts (model={config.LLM_MODEL})")
        fillers = analyze_fillers(words)
        filler_path.write_text(json.dumps(
            {"cuts": [c.model_dump() for c in fillers.cuts]},
            indent=2,
        ))
        print(f"[analyze]   {len(fillers.cuts)} filler cuts → {filler_path}")

        duration = probe_duration(audio)
        print(f"[analyze] calling LLM for layout plan (duration={duration:.1f}s)")
        layout = analyze_layout(words, total_duration=duration)
        layout_path.write_text(json.dumps(
            {"segments": [s.model_dump() for s in layout.segments]},
            indent=2,
        ))
        print(f"[analyze]   {len(layout.segments)} layout segments → {layout_path}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[analyze] ERROR: {exc}", file=sys.stderr)
        return 1


__all__ = [
    "audio_fingerprint",
    "cache_path_for",
    "transcribe",
    "flatten_words",
    "transcribe_and_cache",
    "run_analyze",
]
