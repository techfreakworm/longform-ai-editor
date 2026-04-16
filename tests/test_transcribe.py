"""Tests for Stage B.1 — mlx-whisper transcription wrapper.

Unit tests mock `mlx_whisper.transcribe` to avoid loading the Whisper
model (~1.6 GB) on every test run. One @slow integration test actually
runs Whisper against the real fixture.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.stages.transcribe import (
    audio_fingerprint,
    cache_path_for,
    flatten_words,
    transcribe_and_cache,
)


# --- audio_fingerprint --------------------------------------------------

def test_fingerprint_changes_with_content(tmp_path: Path) -> None:
    p1 = tmp_path / "a.bin"
    p2 = tmp_path / "b.bin"
    p1.write_bytes(b"hello world" * 100)
    p2.write_bytes(b"goodbye world" * 100)
    assert audio_fingerprint(p1) != audio_fingerprint(p2)


def test_fingerprint_stable_for_same_content(tmp_path: Path) -> None:
    p = tmp_path / "a.bin"
    p.write_bytes(b"same content" * 1000)
    assert audio_fingerprint(p) == audio_fingerprint(p)


def test_fingerprint_small_file(tmp_path: Path) -> None:
    """Fingerprint works for files smaller than the 1 MB chunk size."""
    p = tmp_path / "tiny.bin"
    p.write_bytes(b"x" * 1024)
    assert len(audio_fingerprint(p)) == 16


# --- flatten_words ------------------------------------------------------

def test_flatten_words_preserves_order() -> None:
    result = {
        "segments": [
            {"words": [
                {"word": "hello", "start": 0.0, "end": 0.5, "probability": 0.99},
                {"word": "world", "start": 0.5, "end": 1.0, "probability": 0.95},
            ]},
            {"words": [
                {"word": "foo", "start": 2.0, "end": 2.3, "probability": 0.90},
            ]},
        ],
    }
    flat = flatten_words(result)
    assert [w["word"] for w in flat] == ["hello", "world", "foo"]
    assert flat[1]["start"] == 0.5


def test_flatten_words_drops_null_timestamps() -> None:
    result = {
        "segments": [{"words": [
            {"word": "x", "start": None, "end": 1.0},
            {"word": "y", "start": 1.0, "end": None},
            {"word": "z", "start": 1.0, "end": 2.0},
        ]}],
    }
    assert [w["word"] for w in flatten_words(result)] == ["z"]


def test_flatten_words_defaults_probability() -> None:
    result = {
        "segments": [{"words": [{"word": "a", "start": 0, "end": 1}]}],
    }
    out = flatten_words(result)
    assert out[0]["probability"] == 0.0


def test_flatten_words_empty_input() -> None:
    assert flatten_words({}) == []
    assert flatten_words({"segments": []}) == []
    assert flatten_words({"segments": [{"words": []}]}) == []


# --- transcribe_and_cache with mocked mlx_whisper ---------------------

def test_transcribe_and_cache_writes_and_reads(tmp_path: Path, monkeypatch) -> None:
    """First call hits the model; second call hits the cache."""
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake audio content" * 1000)
    cache_dir = tmp_path / "cache"

    call_count = {"n": 0}
    fake_result = {
        "segments": [{"words": [
            {"word": "test", "start": 0.0, "end": 0.5, "probability": 0.9},
        ]}],
        "language": "en",
    }

    import mlx_whisper  # module exists in venv

    def fake_transcribe(path, **kwargs):
        call_count["n"] += 1
        return fake_result

    monkeypatch.setattr(mlx_whisper, "transcribe", fake_transcribe)

    # First call: model invoked
    words = transcribe_and_cache(audio, "dummy-model", cache_dir)
    assert call_count["n"] == 1
    assert [w["word"] for w in words] == ["test"]
    cache_file = cache_path_for(audio, cache_dir)
    assert cache_file.exists()

    # Cache file has expected shape
    cached = json.loads(cache_file.read_text())
    assert cached["model"] == "dummy-model"
    assert cached["language"] == "en"
    assert cached["words"] == words

    # Second call: cache hit, model NOT invoked
    words2 = transcribe_and_cache(audio, "dummy-model", cache_dir)
    assert call_count["n"] == 1  # unchanged
    assert words2 == words


def test_transcribe_and_cache_force_bypasses_cache(tmp_path: Path, monkeypatch) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"a" * 100)
    cache_dir = tmp_path / "cache"

    call_count = {"n": 0}
    import mlx_whisper

    def fake_transcribe(path, **kwargs):
        call_count["n"] += 1
        return {"segments": [{"words": []}]}

    monkeypatch.setattr(mlx_whisper, "transcribe", fake_transcribe)

    transcribe_and_cache(audio, "m", cache_dir)
    transcribe_and_cache(audio, "m", cache_dir, force=True)
    assert call_count["n"] == 2


# --- real fixture integration ----------------------------------------

@pytest.mark.slow
def test_transcribe_real_merged_mp4(real_merged: Path, tmp_path: Path) -> None:
    """Runs real mlx-whisper on real audio. Needs Whisper model in HF cache."""
    from src import config
    words = transcribe_and_cache(
        real_merged, config.WHISPER_MODEL, tmp_path / "cache",
    )
    # The session is 95 s of speech — expect at least some words
    assert len(words) > 0
    # Every word should be a valid timestamped entry
    for w in words:
        assert w["end"] > w["start"]
        assert w["start"] >= 0
        assert isinstance(w["word"], str)
        assert w["word"].strip()  # non-empty after strip
