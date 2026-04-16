"""Config — paths, thresholds, model IDs, prompt templates.

Values fall back to sane defaults. Override via `.env` (copied from
`.env.example`). All floats are seconds unless suffixed otherwise.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _f(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _i(name: str, default: int) -> int:
    return int(os.getenv(name, default))


# ----- Paths ------------------------------------------------------------
WORK_DIR = Path(os.getenv("WORK_DIR", "./work")).expanduser().resolve()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./output")).expanduser().resolve()

# ----- LLM --------------------------------------------------------------
LLM_SERVER_URL = os.getenv("LLM_SERVER_URL", "http://127.0.0.1:8080/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen3-235B-A22B-MLX-4bit")
LLM_TIMEOUT_SEC = _f("LLM_TIMEOUT_SEC", 120.0)
LLM_MAX_RETRIES = _i("LLM_MAX_RETRIES", 3)

# ----- Transcription ----------------------------------------------------
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo")

# ----- Detection thresholds --------------------------------------------
FREEZE_DB = _f("FREEZE_DB", -50.0)
FREEZE_MIN_SEC = _f("FREEZE_MIN_SEC", 2.0)
SILENCE_DB = _f("SILENCE_DB", -30.0)
SILENCE_MIN_SEC = _f("SILENCE_MIN_SEC", 2.0)
MOTION_THRESHOLD = _f("MOTION_THRESHOLD", 0.02)
MOTION_BLUR = _i("MOTION_BLUR", 9)
MOTION_WIDTH = _i("MOTION_WIDTH", 400)

# ----- Dead-zone classification ----------------------------------------
CUT_MIN_SEC = _f("CUT_MIN_SEC", 10.0)
SPEED_8X_MIN_SEC = _f("SPEED_8X_MIN_SEC", 3.0)

# ----- Render ----------------------------------------------------------
VIDEO_BITRATE = os.getenv("VIDEO_BITRATE", "12M")
VIDEO_RES_W = _i("VIDEO_RES_W", 1920)
VIDEO_RES_H = _i("VIDEO_RES_H", 1080)
VIDEO_FPS = _i("VIDEO_FPS", 30)

# ----- Loudness --------------------------------------------------------
LOUDNESS_TARGET = _f("LOUDNESS_TARGET", -14.0)
TRUE_PEAK = _f("TRUE_PEAK", -1.5)
LOUDNESS_RANGE = _f("LOUDNESS_RANGE", 11.0)


# ----- Prompts ---------------------------------------------------------
# Kept here so they're version-controlled + unit-testable as strings.

FILLER_PROMPT = """You are a tight script editor for a YouTube tutorial channel.
The user provides a JSON list of words from a Whisper transcript with
second-precise start/end times. Identify spans to CUT for these reasons:
  - filler: "um", "uh", "like", "you know", "i mean" used as filler
  - false_start: speaker restarts a sentence mid-thought
  - repeat: same word/phrase repeated within 500 ms
Skip brief (< 150 ms) instances that read as natural cadence.

Return strict JSON:
{"cuts": [{"start": 5.42, "end": 5.83, "reason": "um"}, ...]}
"""

LAYOUT_PROMPT = """You are an editor for a screen-recording tutorial.
Given the transcript with word timestamps, assign a LAYOUT to each segment:
  - "cam_full": webcam fills the frame. Use for:
    - Opening hook / intro (first 15-45 s)
    - Direct-address moments ("hey everyone", "let me tell you")
    - Outro / CTA (last 15-30 s)
  - "pip": screen main, webcam inset in a corner. Use for:
    - Demonstrations, walkthroughs, explainer-over-screen content
  - "screen_full": screen fills the frame, webcam hidden. Use for:
    - Moments the narrator says "let's focus on the screen"
    - Dense demo where the face-cam would distract

Return strict JSON covering the ENTIRE duration with no gaps:
{"segments": [{"start": 0.0, "end": 12.3, "layout": "cam_full"}, ...]}
"""

DEAD_ZONE_CUES_PROMPT = """You are an editor for a tech tutorial. Given the
transcript with timestamps, return a JSON list of ranges where the narrator
signals BORING or SKIPPABLE on-screen content. Cue phrases include:
  - "let me skip this" / "we'll skip ahead"
  - "this will take a minute" / "give it a second"
  - "while that installs" / "while that compiles"
  - "you can see here" followed by > 5 s of silence
  - "alright, so now" (restart marker after a long wait)
Only return ranges where silence or inactivity on screen is expected.

Return strict JSON:
{"cues": [{"start": 120.5, "end": 148.3, "reason": "while installs"}, ...]}
"""
