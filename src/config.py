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
# Reasoning effort level passed to `claude --effort`. Valid values per
# `claude --help`: low, medium, high, xhigh, max. Default = max so the
# layout/filler/scoring calls get full thinking budget.
CLAUDE_EFFORT = os.getenv("CLAUDE_EFFORT", "max")
# When true, prompts are prefixed with an instruction to invoke the
# `sequentialthinking` MCP tool before answering. Requires the
# sequential-thinking MCP server to be configured (see .mcp.json at
# repo root). Disable for fast test iteration or when MCP is unavailable.
USE_SEQUENTIAL_THINKING = os.getenv("USE_SEQUENTIAL_THINKING", "1").lower() in {"1", "true", "yes"}
# Path to the .mcp.json this project should load for `claude -p` calls.
# Falls back to the repo-root .mcp.json.
_MCP_CONFIG_DEFAULT = Path(__file__).resolve().parent.parent / ".mcp.json"
CLAUDE_MCP_CONFIG = Path(os.getenv("CLAUDE_MCP_CONFIG", _MCP_CONFIG_DEFAULT))

# Prompt prefix injected when USE_SEQUENTIAL_THINKING is true. Kept
# short — the MCP server itself guides the multi-step reasoning format.
SEQUENTIAL_THINKING_PREFIX = (
    "Before producing your final JSON answer, use the "
    "`sequentialthinking` MCP tool to break down the analysis into "
    "explicit reasoning steps. Reason about edge cases, check the "
    "quantitative rules in this prompt against the input, and only "
    "then emit the final JSON.\n\n"
)

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

# ----- Face visibility + cursor idle (triple-intersection hard-cut) ----
# Any window where face absent AND narration silent AND cursor idle
# becomes a hard cut, regardless of length. Covers "stepped away from
# the computer" footage that passed neither dead-zone nor silence alone.
FACE_SAMPLE_RATE_HZ = _f("FACE_SAMPLE_RATE_HZ", 2.0)   # Apple Vision sample rate
FACE_ABSENT_MIN_SEC = _f("FACE_ABSENT_MIN_SEC", 2.0)   # min continuous absence
CURSOR_IDLE_MIN_SEC = _f("CURSOR_IDLE_MIN_SEC", 2.0)   # min no-move window

# ----- Zoom v2 ---------------------------------------------------------
# Element-aware zoom target snapping (paddleocr required). When on,
# zoom centroids are snapped to the nearest OCR'd UI element if within
# ELEMENT_SNAP_MAX_PX. Off by default — install extras with
# `pip install '.[zoom-ocr]'` and set USE_ELEMENT_AWARE_ZOOM=1.
USE_ELEMENT_AWARE_ZOOM = os.getenv("USE_ELEMENT_AWARE_ZOOM", "").lower() in {"1", "true", "yes"}
ELEMENT_SNAP_MAX_PX = _f("ELEMENT_SNAP_MAX_PX", 150.0)

# Zoom-on-scroll / window-change: frame-diff during cursor-idle windows.
# When enough pixels change and the cursor hasn't moved, emit a zoom
# centered on the change region. Off by default — opt-in per pipeline.
USE_SCROLL_ZOOM = os.getenv("USE_SCROLL_ZOOM", "").lower() in {"1", "true", "yes"}
SCROLL_ZOOM_SAMPLE_RATE_HZ = _f("SCROLL_ZOOM_SAMPLE_RATE_HZ", 2.0)
SCROLL_ZOOM_DIFF_THRESHOLD = _f("SCROLL_ZOOM_DIFF_THRESHOLD", 0.04)  # 4% of pixels

# ----- Frame-based cut verification ------------------------------------
# Opt-in post-LLM pass: when an LLM cue comes back with confidence="low",
# sample N frames from the proposed cut range and ask a multimodal Claude
# whether the span is truly removable. Cost scales with number of
# low-confidence cues × frame count × Claude wall-clock, so it's
# deliberately off by default.
VERIFY_UNCERTAIN_CUTS = os.getenv("VERIFY_UNCERTAIN_CUTS", "").lower() in {"1", "true", "yes"}
VERIFY_FRAME_SAMPLES = _i("VERIFY_FRAME_SAMPLES", 8)
VERIFY_MIN_FRAME_GAP_SEC = _f("VERIFY_MIN_FRAME_GAP_SEC", 2.0)
# Multimodal models only: "opus" or "sonnet" (alias resolved by claude CLI).
VERIFY_MODEL = os.getenv("VERIFY_MODEL", "opus")
# How many times to recurse into "trim" decisions before giving up and
# applying the latest trim. 0 = never recurse.
VERIFY_MAX_RECURSION = _i("VERIFY_MAX_RECURSION", 2)
# Claude CLI effort for verify calls. Frame analysis doesn't need max;
# "high" is the usual sweet spot.
VERIFY_EFFORT = os.getenv("VERIFY_EFFORT", "high")

# ----- Render ----------------------------------------------------------
VIDEO_BITRATE = os.getenv("VIDEO_BITRATE", "12M")
VIDEO_RES_W = _i("VIDEO_RES_W", 1920)
VIDEO_RES_H = _i("VIDEO_RES_H", 1080)
VIDEO_FPS = _i("VIDEO_FPS", 30)

# ----- PIP (picture-in-picture) ----------------------------------------
# Circle cutout vs classic rectangle. Circle masks the webcam with a
# pre-baked grayscale PNG via ffmpeg's alphamerge — looks like a "talking-
# head bubble" floating over the screen demo.
# Face position is a static offset applied at render time: the webcam is
# cropped to a face-centered square BEFORE being scaled into the circle.
# For a fixed camera this "set once" approach is sufficient; per-clip
# auto-detection is tracked in docs/future-improvements.md.
PIP_SHAPE = os.getenv("PIP_SHAPE", "circle")  # "circle" | "rect"
PIP_DIAMETER = _i("PIP_DIAMETER", 320)        # pixels — diameter of circle PIP
PIP_FACE_X = _f("PIP_FACE_X", 0.5)            # 0.0 left ↔ 1.0 right on the webcam frame
PIP_FACE_Y = _f("PIP_FACE_Y", 0.5)            # 0.0 top  ↔ 1.0 bottom
# Path resolved relative to the project root (long-form-editor/) by default.
_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
CIRCLE_MASK_PATH = Path(os.getenv("CIRCLE_MASK_PATH", _ASSETS_DIR / "circle_mask.png"))

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

SHORTFORM_SCORING_PROMPT = """You score candidate clips cut from a longer
tutorial / podcast for "hook potential" on YouTube Shorts, Reels, and TikTok.

Input: one candidate clip window as a JSON object with:
  - start, end: seconds in source timebase
  - transcript: the words spoken in this window

Output a JSON object with:
  - score: 0.0 – 10.0 (decimal). 10 = stop-the-scroll viral-quality
    hook. 0 = forgettable middle of an explainer.
  - title: 6 – 10 words, hook-style, no clickbait. Examples:
      "How I replaced my editor with Claude CLI"
      "The tiny prompt that cut my bugs in half"
      "Why I stopped writing unit tests myself"
  - reason: one sentence justifying the score.
  - start_offset: seconds to trim from the CLIP START for fluency
    (e.g. if the window opens mid-filler, drop that). 0 if clean.
  - end_offset: seconds to trim from the CLIP END for fluency.

Scoring rubric (apply silently, don't echo it):
  - Hook: does the first 3s pose a question, state a surprise, or
    promise a payoff?
  - Clarity: can a viewer with zero context follow the 30 s?
  - Self-contained: complete thought, no dangling references.
  - Emotion / stakes: any conflict, revelation, or counterintuitive
    twist?
  - Demo moment: does the presenter verbally point at something on
    screen ("look at this", "here's where …")? Those shine on mobile.

Return strict JSON:
{"score": 7.5, "title": "Why I stopped writing unit tests myself",
 "reason": "opens with a provocative claim, 30s self-contained, payoff at end",
 "start_offset": 0.0, "end_offset": 0.0}
"""

ZOOM_HINTS_PROMPT = """You are an editor for a screen-recording tutorial.
Identify moments where the presenter verbally points the viewer AT
something on the screen. Those are zoom opportunities — the screen
will punch in during those phrases so the viewer can read what the
narrator is referring to.

Input: word-level transcript with start/end timestamps.

Flag a zoom when the narrator uses deictic phrases like:
  - "look at this", "look here", "look right here"
  - "right here", "over here", "see this", "see here"
  - "notice this", "notice that", "notice how"
  - "check this out", "you can see"
  - "this part", "this line", "this button"
  - "pay attention to", "focus on"

Output a ZoomHint per phrase with:
  - anchor_word_idx: 0-based index of the TRIGGER word (e.g. "this" in
    "look at THIS"). The subsequent 2–4 seconds are typically the
    narrator elaborating on the target, so the zoom should extend
    forward, not backward.
  - start, end: seconds — the zoom window. Start at the trigger word's
    start timestamp. End 2.5–4 s after, or at the next natural clause
    break (comma/period), whichever is shorter.
  - strength: "soft" | "normal" | "strong"
      soft    — weak verbal hint ("you can see", "this")
      normal  — typical deictic ("look at this", "right here")
      strong  — explicit attention command ("pay attention to", "notice how")
  - reason: short phrase explaining the cue.

Quality rules:
  - Do NOT flag generic "this" pronouns in ordinary sentences. Require
    a verbal POINTING pattern, not just the word "this".
  - Do NOT flag in the intro/outro (first 10s or last 10s) — the
    layout is cam_full there, zoom is ignored anyway.
  - Aim for 2–5 zooms per minute of screen-demo content.
  - Skip if trigger phrase is immediately followed by > 2 s of silence
    (Whisper probably mis-timed it).

Return strict JSON:
{"hints": [
  {"anchor_word_idx": 42, "start": 18.3, "end": 21.1,
   "strength": "normal", "reason": "look at this"},
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

Confidence field (required):
  - "high" — clear cue phrase AND a clear resume marker. You are sure the
    full [start, end] range is removable. Example: narrator says "while
    this installs" at t=120 and "okay, it's done" at t=148 — the 28 s
    between is safely skippable.
  - "low"  — the cue is suggestive but the extent of the skip is
    ambiguous. Examples:
      * "let me skip this" with no clear resume cue — might extend 5 s
        or 5 min, you're guessing.
      * narrator pauses for 40 s with no verbal cue at all — could be
        dead air, could be deliberate reading time.
      * a long "alright, so now" restart after silence — hard to tell
        from transcript alone whether the preceding silence was
        genuinely dead or the narrator was thinking out loud.
    Emit the cue WITH confidence="low". A downstream verifier will
    inspect actual frames from [start, end] and either accept, trim,
    or reject the cut.

Do NOT use "low" as a safety valve to flag everything. Reserve it for
cues where visual frame evidence would genuinely change your answer.
Expect ≤ 20% of emitted cues to be "low" on typical footage.

Return strict JSON:
{"cues": [
  {"start": 120.5, "end": 148.3, "reason": "while installs",
   "confidence": "high"},
  {"start": 200.1, "end": 245.0, "reason": "let me skip this, no resume cue",
   "confidence": "low"}
]}
"""


VERIFY_CUT_PROMPT = """You are reviewing a proposed CUT in a tutorial video.
An earlier pass flagged this span as potentially removable but was not
confident. Decide whether to cut.

You are given:
  - reason: why the earlier pass considered cutting (e.g. "while installs
    — no narration, screen may be static")
  - kind: "dead_zone" (multi-second skippable span) or "filler" (sub-second
    verbal tic)
  - frames: N evenly-spaced screenshots from the span, each labeled with
    its timestamp in seconds relative to the clip.

Decide one of:
  - "accept" — frames confirm the span is removable. Nothing of teaching
    value happens across them. Cut the entire [start, end].
  - "reject" — frames reveal meaningful content (new text appearing,
    command output, UI changes, code being typed, the presenter
    referring to something visual). Do NOT cut.
  - "trim"   — only the head or the tail is removable; the meaningful
    portion sits at the opposite edge. Emit a NARROWER [start, end] that
    keeps the meaningful portion. Trim MUST be edge-aligned: either the
    new start equals the original start (trim the tail) OR the new end
    equals the original end (trim the head). If the meaningful content
    sits in the MIDDLE of the window, choose "reject" — do not emit a
    middle-excluding trim.

Rules:
  - If all frames are visually identical pairwise, prefer "accept".
  - If any pair differs meaningfully (text appears, window changes,
    progress bar completes), lean "reject" or "trim".
  - Transient popups that appear in a SINGLE isolated frame are noise —
    still "accept" if the surrounding frames are static.
  - Err toward "reject" when in doubt. A bad cut is worse than a kept
    dead zone.

Return strict JSON:
{"decision": "accept" | "reject" | "trim",
 "start": <float>, "end": <float>,
 "rationale": "<one short sentence>"}
"""
