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
# 70B @ 4-bit = 39.7 GB on disk; comfortable in 128 GB unified memory with
# ~20 GB OS + browser + ~10 GB KV cache at 32k context. Strong JSON mode.
# Qwen3-235B-A22B considered and rejected (full repo = 125 GB, would cause
# memory pressure or swap). Qwen3-30B-A3B available as opt-in fallback.
LLM_MODEL = os.getenv("LLM_MODEL", "mlx-community/Llama-3.3-70B-Instruct-4bit")
# mlx_lm.server's JSON-mode constrained decoding is much slower than plain
# generation on 70B models — budget generous defaults. Real-world measured:
# a 47-word transcript with the full filler prompt takes ~60–180s.
LLM_TIMEOUT_SEC = _f("LLM_TIMEOUT_SEC", 600.0)
LLM_MAX_RETRIES = _i("LLM_MAX_RETRIES", 3)

# ----- Claude Code CLI fallback ----------------------------------------
# If `claude` is on PATH and FORCE_LOCAL_LLM is not set, analyze_llm
# prefers Claude Code CLI over the local MLX server. Graceful degradation:
# on any failure (auth, network, rate limit) we log a warning and fall
# back to the local path. Counts toward your Claude Max/Pro quota — no
# per-call billing unless you explicitly use an API key.
FORCE_LOCAL_LLM = os.getenv("FORCE_LOCAL_LLM", "").lower() in {"1", "true", "yes"}
# Model alias understood by `claude --model <alias>`. "opus" resolves to
# the latest Opus; "sonnet" to the latest Sonnet. Full IDs also accepted
# (e.g. "claude-opus-4-7").
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "opus")

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
Your job is to find SHORT spans worth cutting so the final video feels crisp.
Be decisive — find EVERY clear instance, but do not invent cuts that are
not supported by the exact transcript.

Input: a JSON list of Whisper words with word/start/end fields. Gaps
between consecutive words are silence, NOT part of any word.

Flag a span as a cut when one of these is true. Walk the transcript
word-by-word and check each pattern:

  1. filler — a word spoken to stall or fill space, not to convey
     meaning. Two sub-cases:
     (a) classic verbal fillers: "um", "uh", "ah", "er", "hmm", "mm",
         "like" (as verbal tic only), "you know", "i mean", "sort of",
         "kind of", "basically" / "literally" (as tic).
     (b) dragged-out openers: a sentence-opening "so", "well",
         "okay", "right" that is unusually long for a monosyllable
         (>= 700 ms AND <= 1300 ms) with low Whisper probability
         (< 0.60). These are vocal drags speakers use while thinking —
         they read on camera as hesitation and a good editor trims them.
     Cut just that single word's start/end — never span into the
     following silence.
     NOTE 1: "and" is NOT in the dragged-opener list — inside a clause
     it is a necessary conjunction. Never cut "and" as filler.
     NOTE 2: if a word's end timestamp minus its start is greater
     than 1.3 s, Whisper has padded the token with trailing silence.
     Do NOT cut such tokens as fillers — that silence is not speech.
     NOTE 3: a short, crisp "so" or "and" at the start of a new clause
     (duration < 500 ms, probability >= 0.70) is a clean connector —
     do NOT cut those.

  2. false_start — speaker begins a clause, abandons it, then restarts
     with different or fuller wording. Tell-tale signs:
       - a short fragment followed by a longer clause that re-states
         the same idea (e.g. "Here the— Here the cursor will not work")
       - an incomplete noun phrase followed by a restart
         (e.g. "The input— is an example" where "The input" is
         quieter/lower-confidence and immediately re-stated)
       - a very low Whisper probability (< 0.05) on a word that looks
         like a self-correction
     Cut the abandoned fragment, NOT the completed version.
     REQUIRES a clear restart. If the speaker finishes the clause,
     do not flag it.

  3. repeat — the SAME short word or phrase appears twice in close
     succession (within ~2 seconds and no completed clause between).
     This includes adjacent identical tokens like "is is", "so so",
     "the the", and short phrase repeats like "that is ... that is"
     where the speaker loops back and re-says the connector before
     continuing. Cut ONE of the two occurrences — typically the
     shorter / lower-confidence one. Prefer cutting the earlier
     occurrence when the later one leads into the finished thought.
     If the two tokens are separated by a complete clause or > 2 s
     of unrelated content, it is NOT a repeat.

Do NOT flag:
  - Silence or pauses. A 2-second gap inside a sentence is a pause,
    not a false_start and not something you can cut.
  - Natural discourse markers that connect clauses: "so", "and",
    "okay", "right", "well" at the START of a new thought.
  - Content words (nouns, verbs, adjectives) in ordinary sentences.
  - Words with matching surface form but separated by a complete
    other clause (that is a new sentence, not a repeat).

Quality bar:
  - Aim for 3-6 cuts on a typical 1-2 minute tutorial clip. Raw
    tutorial footage almost always contains this many stumbles;
    returning fewer than 3 usually means you missed a duplicate or
    a low-confidence fragment. More than 8 often means you are
    cutting normal speech.
  - Each cut is typically 100-600 ms. A cut longer than 1.5 s almost
    always means you are trying to cut silence — do not do that.
  - Before finalizing, scan the transcript for each of these
    high-signal patterns and include any you find:
      a. Any word pair with identical surface form (same token) that
         are within 2 s of each other — cut one of them.
      b. Any word whose Whisper probability is below 0.05 — these
         are almost always artifacts or stumbles, cut them.
      c. Any short (1-2 word) fragment immediately followed by a
         longer restatement of the same idea — cut the fragment.
      d. Any standalone "um" / "uh" / "ah" / "er" / "hmm" token.
  - If after scanning all four patterns you still find zero cuts on
    a clip longer than 30 seconds, return the empty list — but this
    should be rare for real tutorial footage.

Return strict JSON:
{"cuts": [{"start": 5.42, "end": 5.83, "reason": "filler"}, ...]}

reason must be one of: "filler", "false_start", "repeat".
start/end are the exact Whisper timestamps of the word(s) being cut.
"""

LAYOUT_PROMPT = """You are an editor for a screen-recording tutorial.
Decide the on-screen LAYOUT for each moment. The final video always
needs a narrative arc: open on the presenter, go to screen for the
demo, return to the presenter to close.

Input: word-level transcript with start/end timestamps, plus
total_duration_s (the clip length in seconds). You MUST cover the
entire duration [0, total_duration_s] with no gaps and no overlaps.

Choose one of three layouts per segment:

  - "cam_full" — webcam fills the frame, screen hidden.
    Use for opening, outro, and direct-address talking head.

  - "pip"      — screen main with webcam inset in a corner.
    Use for narrated demo content: presenter talks WHILE the screen
    shows something. This is the default for the demo body.

  - "screen_full" — screen fills the frame, webcam hidden.
    Use when the narrator points viewers AT the screen with phrases
    like "look at this", "on the screen", "the cursor", "you can see",
    or during dense demo where the face would distract.

Structural rules you MUST follow:

  1. Open with "cam_full" for the first 10–30 seconds (intro / hook).
     This includes any silence before the first word.
  2. Close with "cam_full" for the last 10–25 seconds (outro / CTA).
     If the clip is short (< 60 s), shrink these to ~5 s each.
  3. Between intro and outro, alternate "pip" (default demo) and
     "screen_full" (explicit "look at the screen" moments).
  4. Aim for 4–6 segments total on a 1–3 minute clip. Fewer than 3
     segments feels static; more than 8 feels choppy.
  5. Every non-intro/outro segment should last at least 6 seconds —
     shorter segments cause visual whiplash.
  6. Do NOT end with "pip" or "screen_full" unless the clip runs
     all the way to its final word with no trailing silence. A
     trailing silence of > 3 s belongs in the outro "cam_full".

Return strict JSON covering [0, total_duration_s] contiguously:
{"segments": [
  {"start": 0.0,  "end": 12.3, "layout": "cam_full"},
  {"start": 12.3, "end": 48.0, "layout": "pip"},
  ...
]}
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
