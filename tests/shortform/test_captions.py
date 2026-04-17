"""Tests for src/shortform/captions.py."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.shortform.captions import (
    DEFAULT_FONT,
    StyleOptions,
    _build_karaoke_line,
    _seconds_to_ass,
    _split_into_lines,
    _write_ass_pure_python,
    build_ass,
)


def _w(word: str, start: float, end: float) -> dict:
    return {"word": word, "start": start, "end": end, "probability": 0.95}


# --- helpers ---------------------------------------------------------

def test_seconds_to_ass_format() -> None:
    assert _seconds_to_ass(3.5) == "0:00:03.50"
    assert _seconds_to_ass(65.12) == "0:01:05.12"
    assert _seconds_to_ass(3723.4) == "1:02:03.40"


def test_split_into_lines_breaks_on_char_limit() -> None:
    words = [_w("hello", 0, 0.5), _w("world", 0.5, 1.0),
             _w("again", 1.0, 1.5), _w("and", 1.5, 1.8),
             _w("again", 1.8, 2.3)]
    lines = _split_into_lines(words, max_chars=10)
    # "hello world" (11 chars) already over 10, so break after "hello"
    assert len(lines) >= 2
    # All words accounted for
    total = sum(len(l) for l in lines)
    assert total == len(words)


def test_split_into_lines_breaks_on_gap() -> None:
    words = [_w("hello", 0, 0.5), _w("world", 10.0, 10.5)]  # 9.5s gap
    lines = _split_into_lines(words, max_chars=99)
    assert len(lines) == 2


def test_split_into_lines_empty_words_skipped() -> None:
    words = [_w("", 0, 0.5), _w("hi", 0.5, 1.0)]
    lines = _split_into_lines(words, max_chars=99)
    assert len(lines) == 1
    assert lines[0][0]["word"] == "hi"


# --- karaoke line building ------------------------------------------

def test_build_karaoke_line_tags_each_word() -> None:
    words = [_w("hi", 0.0, 0.3), _w("world", 0.3, 0.8)]
    start, end, text = _build_karaoke_line(words)
    assert start == 0.0
    assert end == 0.8
    assert "\\kf30" in text   # 0.3s = 30 cs
    assert "\\kf50" in text   # 0.5s = 50 cs
    assert "hi" in text
    assert "world" in text


def test_build_karaoke_line_minimum_duration() -> None:
    """Zero-duration word gets clamped to 1 cs so the tag is valid."""
    words = [_w("a", 0.0, 0.0)]
    _s, _e, text = _build_karaoke_line(words)
    assert "\\kf1" in text


# --- pure-python ASS writer -----------------------------------------

def test_write_ass_pure_python_writes_valid_file(tmp_path) -> None:
    words = [
        _w("hello", 0.0, 0.5),
        _w("world,", 0.6, 1.0),
        _w("how", 1.1, 1.3),
        _w("are", 1.3, 1.5),
        _w("you?", 1.5, 1.9),
    ]
    out = tmp_path / "c.ass"
    _write_ass_pure_python(words, out, StyleOptions())
    text = out.read_text()
    assert "[Script Info]" in text
    assert "PlayResX: 1080" in text
    assert "PlayResY: 1920" in text
    assert "Dialogue:" in text
    assert "\\kf" in text


def test_write_ass_custom_style(tmp_path) -> None:
    words = [_w("hi", 0.0, 0.5)]
    style = StyleOptions(font="Arial Black", font_size=100, resolution_w=720)
    out = tmp_path / "c.ass"
    _write_ass_pure_python(words, out, style)
    text = out.read_text()
    assert "Arial Black" in text
    assert "PlayResX: 720" in text


# --- build_ass public entry -----------------------------------------

def test_build_ass_uses_fallback_when_stable_ts_missing(tmp_path) -> None:
    words = [_w("test", 0.0, 0.5)]
    out = tmp_path / "c.ass"
    with patch("src.shortform.captions._try_stable_ts_from_words",
               return_value=False):
        result = build_ass(words, out)
    assert result == out
    assert out.exists()
    assert "Dialogue:" in out.read_text()


def test_build_ass_uses_stable_ts_when_available(tmp_path) -> None:
    words = [_w("test", 0.0, 0.5)]
    out = tmp_path / "c.ass"

    def _fake_stable_write(w, p, s):
        p.write_text("[Script Info]\n# stable-ts fake output\n")
        return True

    with patch("src.shortform.captions._try_stable_ts_from_words",
               side_effect=_fake_stable_write):
        result = build_ass(words, out)
    assert result == out
    assert "stable-ts fake output" in out.read_text()


def test_build_ass_creates_parent_dir(tmp_path) -> None:
    out = tmp_path / "nested" / "dir" / "c.ass"
    with patch("src.shortform.captions._try_stable_ts_from_words",
               return_value=False):
        build_ass([_w("x", 0.0, 0.5)], out)
    assert out.exists()


def test_default_font_is_montserrat_extrabold() -> None:
    assert DEFAULT_FONT == "Montserrat ExtraBold"
