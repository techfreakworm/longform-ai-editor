"""Shortform pipeline — transcribe → segment → score → reframe → captions → render.

Single entry: `run_all(args)`. Reads inputs (screen + webcam + audio
OR a composited mp4), writes N shortform mp4s to the output directory.

Input modes:
  * **dual-track** — `--screen ext.mov --webcam cam.mov --audio merged.mp4`
    (matches long-form). Each picked clip can use any of the four layouts.
  * **composited** — `--composited long.mp4`. Skips per-clip layout
    picks; all clips render as `screen_full` (the only sensible
    single-source layout on a 16:9 master). face-tracked crop on the
    composite still works when the face is visible in the frame.

Output: `output/shorts/<source-stem>/<rank>_<slug>.mp4` for each of
the top-N clips. Also writes `shortform_plan.json` to the work dir
with the ranked selections and per-clip layout choices.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from src import config
from src.shortform.captions import build_ass
from src.shortform.reframe import (
    CropWindow,
    build_screen_crops,
    build_webcam_crops,
)
from src.shortform.render import ClipSpec, Layout, render_clip
from src.shortform.score import RankedClip, score_candidates
from src.shortform.segment import segment_topics
from src.shortform.transcribe import transcribe

log = logging.getLogger(__name__)


DEFAULT_TOP_N = 3
DEFAULT_MIN_SEC = 30.0
DEFAULT_MAX_SEC = 60.0
DEFAULT_SLUG_MAX = 40


@dataclass
class ShortformPlan:
    rank: int
    start: float
    end: float
    title: str
    layout: Layout
    composite: float
    reason: str


# ----- layout picker -------------------------------------------------

def _save_ranked_cache(ranked: list[RankedClip], path: Path) -> None:
    """Persist ranked clips so re-runs skip the LLM scoring pass."""
    payload = []
    for r in ranked:
        cand = r.candidate
        payload.append({
            "candidate": {
                "start": cand.start, "end": cand.end,
                "sentence_idx_range": list(cand.sentence_idx_range),
            },
            "llm": r.llm.model_dump() if r.llm else None,
            "audio_energy": r.audio_energy,
            "punctuation": r.punctuation,
            "length_score": r.length_score,
            "composite": r.composite,
            "effective_start": r.effective_start,
            "effective_end": r.effective_end,
        })
    path.write_text(json.dumps({"ranked": payload}, indent=2))


def _load_ranked_cache(path: Path) -> list[RankedClip]:
    """Rebuild a list of RankedClip objects from a cached JSON file."""
    from src.shortform.score import ShortformScore
    from src.shortform.segment import Candidate

    data = json.loads(path.read_text())
    out: list[RankedClip] = []
    for r in data.get("ranked", []):
        c = r["candidate"]
        cand = Candidate(
            start=float(c["start"]), end=float(c["end"]),
            sentence_idx_range=tuple(c["sentence_idx_range"]),
        )
        llm = ShortformScore(**r["llm"]) if r.get("llm") else None
        out.append(RankedClip(
            candidate=cand, llm=llm,
            audio_energy=float(r["audio_energy"]),
            punctuation=float(r["punctuation"]),
            length_score=float(r["length_score"]),
            composite=float(r["composite"]),
            effective_start=float(r["effective_start"]),
            effective_end=float(r["effective_end"]),
        ))
    return out


_DEMO_CUES = re.compile(
    r"\b(look at|right here|over here|see this|see how|notice|check this|"
    r"this part|this line|this button|on the screen|in the terminal)\b",
    re.IGNORECASE,
)
_HOOK_CUES = re.compile(
    r"\b(what if|imagine|here's the thing|turns out|but here|let me tell|"
    r"story time|I'll show you)\b",
    re.IGNORECASE,
)


def pick_layout_for_clip(
    transcript: str,
    *,
    dual_track: bool,
    default: Layout = "split_vstack",
) -> Layout:
    """Heuristic layout picker for a clip's transcript.

    The plan had an LLM layout call too, but it's overkill for the
    per-clip decision — shortform already has the LLM scoring pass,
    and layout mostly tracks lexical cues.

    Rules:
      * Single-source input → always `screen_full` (no webcam track).
      * Strong demo-pointing cues → `split_vstack` (user's default),
        with the screen pre-zoomed for phone legibility.
      * Narrative hook with no screen cues → `cam_full`.
      * Everything else → default (usually `split_vstack`).
    """
    if not dual_track:
        return "screen_full"
    demo_hits = len(_DEMO_CUES.findall(transcript or ""))
    hook_hits = len(_HOOK_CUES.findall(transcript or ""))
    if demo_hits >= 2:
        return "split_vstack"
    if hook_hits >= 1 and demo_hits == 0:
        return "cam_full"
    return default


def _slugify(title: str, max_len: int = DEFAULT_SLUG_MAX) -> str:
    t = re.sub(r"[^a-zA-Z0-9]+", "_", title.strip().lower()).strip("_")
    return (t or "clip")[:max_len]


# ----- crop window lookup -------------------------------------------

def _find_crop_at(
    crops: list[CropWindow], t: float, fallback_cx: float, fallback_cy: float,
) -> CropWindow:
    """Return the CropWindow covering time `t`, or a fallback centered window."""
    for c in crops:
        if c.start <= t <= c.end:
            return c
    if crops:
        nearest = min(crops, key=lambda c: abs((c.start + c.end) / 2.0 - t))
        return CropWindow(start=nearest.start, end=nearest.end,
                          cx=nearest.cx, cy=nearest.cy)
    return CropWindow(start=t, end=t + 1.0, cx=fallback_cx, cy=fallback_cy)


# ----- main orchestrator --------------------------------------------

def run_all(args: argparse.Namespace) -> int:
    """Shortform pipeline. Mirrors the long-form run_all shape."""
    work_dir: Path = Path(args.work) if getattr(args, "work", None) else config.WORK_DIR
    work_dir.mkdir(parents=True, exist_ok=True)

    screen: Path | None = getattr(args, "screen", None)
    webcam: Path | None = getattr(args, "webcam", None)
    audio: Path | None = getattr(args, "audio", None)
    composited: Path | None = getattr(args, "composited", None)

    dual_track = screen is not None and webcam is not None and audio is not None
    if not dual_track and composited is None:
        print("[shortform] ERROR: need either --screen/--webcam/--audio OR --composited",
              file=sys.stderr)
        return 1

    output_dir_raw = getattr(args, "output_dir", None)
    output_dir: Path = Path(output_dir_raw) if output_dir_raw else (
        config.OUTPUT_DIR / "shorts"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    top_n: int = int(getattr(args, "top", DEFAULT_TOP_N))
    min_sec: float = float(getattr(args, "min_sec", DEFAULT_MIN_SEC))
    max_sec: float = float(getattr(args, "max_sec", DEFAULT_MAX_SEC))
    cursor: Path | None = getattr(args, "cursor", None)

    primary_audio = audio if dual_track else composited
    primary_screen = screen if dual_track else composited
    primary_webcam = webcam if dual_track else composited
    assert primary_audio is not None and primary_screen is not None

    source_stem = primary_screen.stem
    shorts_dir = output_dir / source_stem
    shorts_dir.mkdir(parents=True, exist_ok=True)
    plan_path = work_dir / "shortform_plan.json"

    try:
        # ---- 1. Transcribe -----------------------------------------
        print(f"[shortform] transcribing {primary_audio.name}", flush=True)
        words_path = work_dir / "shortform_words.json"
        if words_path.exists():
            data = json.loads(words_path.read_text())
            words = data["words"]
            sentences = data["sentences"]
            backend = data.get("backend", "cached")
        else:
            result = transcribe(primary_audio)
            words = result["words"]
            sentences = [
                {"text": s.text, "start": s.start, "end": s.end}
                for s in result["sentences"]
            ]
            backend = result.get("backend", "unknown")
            words_path.write_text(json.dumps({
                "words": words, "sentences": sentences, "backend": backend,
            }, indent=2))
        print(f"[shortform]   backend={backend}, "
              f"{len(words)} words, {len(sentences)} sentences", flush=True)

        # ---- 2. Topical segmentation -------------------------------
        print("[shortform] segmenting into topical candidates", flush=True)
        candidates = segment_topics(sentences)
        # Pre-filter by duration so we don't spend LLM cycles on clips
        # that are obviously outside the viable range. 0.5× sweet-min to
        # 2.5× sweet-max keeps enough margin for the LLM's start/end
        # offset trimming to pull an off-sized window into range.
        filter_min = min_sec * 0.5
        filter_max = max_sec * 2.5
        filtered = [
            c for c in candidates
            if filter_min <= (c.end - c.start) <= filter_max
        ]
        print(f"[shortform]   {len(candidates)} candidates "
              f"({len(filtered)} in [{filter_min:.0f}s, {filter_max:.0f}s])",
              flush=True)
        candidates = filtered

        # ---- 3. Score + rank --------------------------------------
        # Cache the ranked clips so re-runs (e.g. after a late-stage
        # crash) don't repeat the expensive LLM scoring pass.
        ranked_cache = work_dir / "shortform_ranked.json"
        ranked: list[RankedClip]
        if ranked_cache.exists():
            print(f"[shortform] reusing cached rankings from {ranked_cache.name}",
                  flush=True)
            ranked = _load_ranked_cache(ranked_cache)
        else:
            print(f"[shortform] scoring {len(candidates)} candidates (Claude CLI)",
                  flush=True)
            ranked = score_candidates(
                candidates, sentences, primary_audio,
                sweet_min=min_sec, sweet_max=max_sec, top_n=top_n,
            )
            _save_ranked_cache(ranked, ranked_cache)
        print(f"[shortform]   top {len(ranked)} clips selected", flush=True)

        # ---- 4. Precompute crops for whole source ------------------
        duration_s = max((s["end"] for s in sentences), default=0.0)
        if dual_track:
            print("[shortform] building webcam face crops", flush=True)
            webcam_crops = build_webcam_crops(primary_webcam, duration_s=duration_s)
        else:
            webcam_crops = []
        print("[shortform] building screen cursor crops", flush=True)
        screen_crops = build_screen_crops(
            primary_screen, cursor_csv=cursor,
            duration_s=duration_s,
        )

        # ---- 5. Per-clip: captions + render ------------------------
        plan: list[ShortformPlan] = []
        for rank_idx, rc in enumerate(ranked, start=1):
            title = rc.llm.title if rc.llm else f"Clip {rank_idx}"
            reason = rc.llm.reason if rc.llm else "heuristic-only"
            clip_words = [
                w for w in words
                if rc.effective_start <= float(w["start"]) <= rc.effective_end
            ]
            # Shift words so the ASS starts at t=0 inside the clip.
            clip_words_shifted = [
                {**w,
                 "start": float(w["start"]) - rc.effective_start,
                 "end": float(w["end"]) - rc.effective_start}
                for w in clip_words
            ]
            transcript = " ".join(str(w["word"]).strip() for w in clip_words)

            slug = _slugify(title)
            ass_path = shorts_dir / f"{rank_idx:02d}_{slug}.ass"
            build_ass(clip_words_shifted, ass_path)

            layout = pick_layout_for_clip(transcript, dual_track=dual_track)

            mid_t = (rc.effective_start + rc.effective_end) / 2.0
            spec = ClipSpec(
                start_s=rc.effective_start,
                end_s=rc.effective_end,
                layout=layout,
                webcam_crop=_find_crop_at(webcam_crops, mid_t,
                                          config.PIP_FACE_X, config.PIP_FACE_Y),
                screen_crop=_find_crop_at(screen_crops, mid_t, 0.5, 0.5),
                captions_ass=ass_path,
            )
            out_mp4 = shorts_dir / f"{rank_idx:02d}_{slug}.mp4"
            print(f"[shortform] render {rank_idx}: [{rc.effective_start:.1f}s-"
                  f"{rc.effective_end:.1f}s] layout={layout} → {out_mp4.name}",
                  flush=True)
            render_clip(
                spec,
                screen_path=primary_screen,
                webcam_path=primary_webcam,
                audio_path=primary_audio,
                out_path=out_mp4,
            )
            plan.append(ShortformPlan(
                rank=rank_idx,
                start=rc.effective_start, end=rc.effective_end,
                title=title, layout=layout,
                composite=rc.composite, reason=reason,
            ))

        plan_path.write_text(json.dumps(
            {"clips": [asdict(p) for p in plan]}, indent=2,
        ))
        print(f"[shortform] wrote plan to {plan_path}", flush=True)
        print(f"[shortform] ✓ {len(plan)} shortform clip(s) in {shorts_dir}",
              flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001
        log.exception("shortform pipeline failed")
        print(f"[shortform] ERROR: {exc}", file=sys.stderr)
        return 1


__all__ = [
    "ShortformPlan",
    "DEFAULT_TOP_N",
    "DEFAULT_MIN_SEC",
    "DEFAULT_MAX_SEC",
    "pick_layout_for_clip",
    "run_all",
]
