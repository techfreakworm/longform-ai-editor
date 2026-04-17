"""Karaoke-style captions for shortform output.

Two supported paths:

1. **Preferred** — `stable-ts` (if installed). Takes the word-level
   transcript we already have and emits an ASS file with per-word
   `\\kf` karaoke tags so the active word highlights yellow as the
   presenter speaks it.

2. **Fallback** — hand-rolled ASS writer with the same visual language.
   No third-party dep; written in pure Python. Less polished kerning
   than stable-ts but works without the `[shortform]` extra.

Style choices (from plan doc):
  * Montserrat ExtraBold 88 px
  * Primary: white, Karaoke sweep: yellow, Outline: 4 px, Shadow: 1 px
  * Bottom-center alignment
  * Max line length ≈ 18 chars so long sentences wrap naturally on
    9:16 phone aspect
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


DEFAULT_FONT = "Montserrat ExtraBold"
DEFAULT_FONT_SIZE = 88
DEFAULT_MAX_LINE_CHARS = 18
DEFAULT_PRIMARY_BGR = "&H00FFFFFF"   # white  (ASS is BGR hex)
DEFAULT_SECONDARY_BGR = "&H0000FFFF"  # yellow (karaoke sweep)
DEFAULT_OUTLINE_PX = 4
DEFAULT_SHADOW_PX = 1


@dataclass
class StyleOptions:
    font: str = DEFAULT_FONT
    font_size: int = DEFAULT_FONT_SIZE
    primary_bgr: str = DEFAULT_PRIMARY_BGR
    secondary_bgr: str = DEFAULT_SECONDARY_BGR
    outline_px: int = DEFAULT_OUTLINE_PX
    shadow_px: int = DEFAULT_SHADOW_PX
    max_line_chars: int = DEFAULT_MAX_LINE_CHARS
    resolution_w: int = 1080
    resolution_h: int = 1920


# ----- stable-ts path -----------------------------------------------

def _try_stable_ts_from_words(
    words: list[dict], out_path: Path, style: StyleOptions,
) -> bool:
    """Build a stable-ts WhisperResult from raw words and write ASS.

    stable-ts accepts word-level dicts if we feed them into the
    Segment/WordTiming API it exposes. We degrade to a simpler build
    path if the specific API isn't available — the hand-rolled writer
    below handles both.
    """
    try:
        import stable_whisper  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        log.info("stable-ts not installed — using pure-Python ASS writer (%s)", exc)
        return False

    # Construct a minimal WhisperResult. Schema: list of segments; each
    # has .words (list of {word, start, end}). stable-ts 2.x accepts
    # either `from_words()` or a dict payload via `WhisperResult`.
    try:
        from stable_whisper.result import WhisperResult

        segments: list[dict] = []
        current: dict | None = None
        for w in words:
            text = w["word"].strip()
            if not text:
                continue
            if current is None:
                current = {"start": float(w["start"]), "end": float(w["end"]),
                           "text": "", "words": []}
            current["words"].append({
                "word": text,
                "start": float(w["start"]),
                "end": float(w["end"]),
                "probability": float(w.get("probability", 1.0)),
            })
            current["text"] = (current["text"] + " " + text).strip()
            current["end"] = float(w["end"])
            if text.endswith((".", "!", "?")):
                segments.append(current)
                current = None
        if current is not None:
            segments.append(current)

        result = WhisperResult({"segments": segments, "language": "en"})
        result.to_ass(
            str(out_path),
            karaoke=True,
            font=style.font,
            font_size=style.font_size,
        )
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("stable-ts ASS write failed (%s) — falling back", exc)
        return False


# ----- pure-Python ASS writer ---------------------------------------

_ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
ScaledBorderAndShadow: yes
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{size},{primary},{secondary},&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{outline},{shadow},2,40,40,120,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _seconds_to_ass(t: float) -> str:
    """0:01:23.45 format with centiseconds."""
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t - h * 3600 - m * 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _split_into_lines(
    words: list[dict], max_chars: int,
) -> list[list[dict]]:
    """Group words into caption lines. Break on line-length or 1 s gap."""
    lines: list[list[dict]] = []
    cur: list[dict] = []
    cur_len = 0
    prev_end = 0.0
    for w in words:
        tok = w["word"].strip()
        if not tok:
            continue
        gap = float(w["start"]) - prev_end if prev_end else 0.0
        projected = cur_len + len(tok) + (1 if cur else 0)
        if cur and (projected > max_chars or gap > 1.0):
            lines.append(cur)
            cur, cur_len = [], 0
        cur.append(w)
        cur_len += len(tok) + (1 if cur else 0)
        prev_end = float(w["end"])
    if cur:
        lines.append(cur)
    return lines


def _build_karaoke_line(word_line: list[dict]) -> tuple[float, float, str]:
    """Return (line_start, line_end, "{\\kf<cs>}word{\\kf<cs>}word..." text)."""
    line_start = float(word_line[0]["start"])
    line_end = float(word_line[-1]["end"])
    parts: list[str] = []
    for w in word_line:
        dur_cs = int(round((float(w["end"]) - float(w["start"])) * 100))
        dur_cs = max(1, dur_cs)
        text = w["word"].strip()
        parts.append(f"{{\\kf{dur_cs}}}{text}")
    return line_start, line_end, " ".join(parts)


def _write_ass_pure_python(
    words: list[dict], out_path: Path, style: StyleOptions,
) -> None:
    header = _ASS_HEADER.format(
        w=style.resolution_w, h=style.resolution_h,
        font=style.font, size=style.font_size,
        primary=style.primary_bgr, secondary=style.secondary_bgr,
        outline=style.outline_px, shadow=style.shadow_px,
    )
    events: list[str] = []
    for line in _split_into_lines(words, style.max_line_chars):
        start_s, end_s, text = _build_karaoke_line(line)
        events.append(
            f"Dialogue: 0,{_seconds_to_ass(start_s)},{_seconds_to_ass(end_s)},"
            f"Default,,0,0,0,,{text}"
        )
    out_path.write_text(header + "\n".join(events) + "\n")


# ----- public API ---------------------------------------------------

def build_ass(
    words: list[dict], out_path: Path, *, style: StyleOptions | None = None,
) -> Path:
    """Write an ASS caption file for `words` to `out_path`.

    Prefers stable-ts when installed; falls back to a pure-Python writer
    that produces visually-equivalent output for the default style.
    """
    style = style or StyleOptions()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not _try_stable_ts_from_words(words, out_path, style):
        _write_ass_pure_python(words, out_path, style)
    return out_path


__all__ = [
    "StyleOptions",
    "DEFAULT_FONT",
    "DEFAULT_FONT_SIZE",
    "build_ass",
]
