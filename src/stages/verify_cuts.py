"""Frame-based verifier for low-confidence cut decisions.

When the LLM analyze pass emits a DeadZoneCue with confidence="low",
this stage samples N frames uniformly across the proposed cut, sends
them to a multimodal Claude with the original reason, and returns
one of three decisions:

  accept — entire [start, end] confirmed skippable, apply the cut.
  reject — meaningful content found in the frames, do NOT cut.
  trim   — narrower edge-aligned range is skippable; apply the trim.

Design notes:
  * Edge-aligned trims only. If Claude says "cut the middle" (a sub-range
    that touches neither original edge), we reject + log rather than
    emit two separate cuts. Middle trims are rare and complicate the
    downstream subtract_range code; reject is the safer default.
  * Trims can recurse (up to `max_recursion`) so Claude can progressively
    narrow the range. Accept / reject short-circuit the recursion.
  * Frames are cached by (screen basename, mtime, timestamp) so repeat
    runs + recursion passes don't re-extract the same JPG.
  * The runtime writes `work/verify_log.json` with every decision's
    rationale + frame timestamps — crucial for auditing a bad render.

Claude call is injected via `_call_verifier` so tests stay offline.
In production the default goes through `claude -p --model opus` with
image attachments referenced by `@<path>` inline in the prompt body.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

from src import config

log = logging.getLogger(__name__)

CutKind = Literal["dead_zone", "filler"]
Decision = Literal["accept", "reject", "trim"]

_EPS = 1e-3


# --- dataclasses ------------------------------------------------------


@dataclass(frozen=True)
class VerifyInput:
    start: float
    end: float
    reason: str
    kind: CutKind


@dataclass(frozen=True)
class VerifyOutput:
    start: float
    end: float
    decision: Decision
    rationale: str
    original_start: float
    original_end: float
    frame_timestamps: tuple[float, ...]
    recursion_depth: int = 0


# --- frame sampling ---------------------------------------------------


def sample_frame_timestamps(
    start: float, end: float, n: int, min_gap: float,
) -> list[float]:
    """Uniformly sample up to `n` timestamps in [start, end].

    The budget `min_gap` is the minimum spacing between samples. A span
    shorter than `(n - 1) * min_gap` gets fewer than `n` samples:
    `floor((end - start) / min_gap) + 1`, always ≥ 2 so we always get
    the two endpoints. A zero-length span returns a single sample.
    """
    if end < start:
        raise ValueError(f"end ({end}) < start ({start})")
    length = end - start
    if length <= _EPS:
        return [start]
    max_fit = int(length / min_gap) + 1
    count = max(2, min(n, max_fit))
    if count == 2:
        return [start, end]
    step = length / (count - 1)
    return [start + i * step for i in range(count)]


# --- frame extraction -------------------------------------------------


def _run_ffmpeg(cmd: list[str], **kwargs: Any) -> None:
    """Indirection so tests can patch ffmpeg invocations."""
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {result.returncode}): "
            f"{result.stderr[-400:] or result.stdout[-400:]}"
        )


def _frame_cache_key(screen_path: Path, t_s: float) -> str:
    try:
        mtime = int(screen_path.stat().st_mtime)
    except FileNotFoundError:
        mtime = 0
    h = hashlib.sha1(
        f"{screen_path.name}|{mtime}|{t_s:.4f}".encode()
    ).hexdigest()[:16]
    return h


def extract_frames(
    screen_path: Path,
    timestamps: Iterable[float],
    work_dir: Path,
) -> list[tuple[float, Path]]:
    """Extract a JPG per timestamp. Content-hashed filenames so repeat
    calls (including verifier recursion) hit the cache instead of
    re-running ffmpeg.
    """
    cache_dir = work_dir / "verify_frames"
    cache_dir.mkdir(parents=True, exist_ok=True)

    out: list[tuple[float, Path]] = []
    for t in timestamps:
        key = _frame_cache_key(screen_path, t)
        jpg_path = cache_dir / f"{key}.jpg"
        if not jpg_path.exists():
            cmd = [
                "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error",
                "-y",
                "-ss", f"{t:.3f}",
                "-i", str(screen_path),
                "-frames:v", "1",
                "-q:v", "4",
                str(jpg_path),
            ]
            _run_ffmpeg(cmd)
        out.append((t, jpg_path))
    return out


# --- Claude multimodal call ------------------------------------------


def _default_verifier_call(
    inp: VerifyInput,
    frames: list[tuple[float, Path]],
    *,
    model: str,
    effort: str,
    timeout_s: float,
) -> dict[str, Any]:
    """Shell out to `claude -p` with inline image references.

    Requires `claude` on PATH. Images are referenced via `@<path>` inline
    in the prompt body — Claude CLI's non-interactive mode resolves these
    before sending. Failure modes mirror `_call_via_claude_cli` in
    `analyze_llm.py` (non-zero exit → RuntimeError).
    """
    if shutil.which("claude") is None:
        raise RuntimeError("claude CLI not on PATH — cannot run verifier")

    frame_block = "\n".join(
        f"  - t={t:.2f}s @{p}" for t, p in frames
    )
    user_body = (
        f"kind: {inp.kind}\n"
        f"original_start: {inp.start}\n"
        f"original_end: {inp.end}\n"
        f"reason: {inp.reason}\n"
        f"frames:\n{frame_block}\n"
    )
    prompt = (
        config.VERIFY_CUT_PROMPT.rstrip()
        + "\n\nReturn ONLY a valid JSON object. No markdown fences, no preamble, no commentary."
        + "\n\nInput:\n"
        + user_body
    )

    cmd = [
        "claude", "-p",
        "--model", model,
        "--effort", effort,
        "--output-format", "text",
        "--permission-mode", "bypassPermissions",
        "--no-session-persistence",
    ]
    log.debug("calling claude verifier: %s", cmd)
    r = subprocess.run(
        cmd, input=prompt, capture_output=True, text=True,
        timeout=timeout_s, check=False,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"claude verifier exited {r.returncode}: "
            f"{r.stderr[-400:] or r.stdout[-400:]}"
        )
    from src.stages.analyze_llm import _extract_json_body
    body = _extract_json_body(r.stdout)
    return json.loads(body)


# --- core verification ------------------------------------------------


def _classify_trim(
    original: VerifyInput, new_start: float, new_end: float,
) -> tuple[Decision, str]:
    """Figure out whether a proposed trim is edge-aligned or middle-cut.

    Returns (decision, reason). If the trim is valid, decision == "trim"
    and reason is empty. Otherwise decision == "reject" with a short
    explanation for the audit log.
    """
    if new_end <= new_start:
        return "reject", "verifier returned empty or inverted trim range"
    if new_start < original.start - _EPS or new_end > original.end + _EPS:
        return "reject", "verifier trim range is outside the original bounds"
    head_aligned = abs(new_start - original.start) <= _EPS
    tail_aligned = abs(new_end - original.end) <= _EPS
    if not head_aligned and not tail_aligned:
        return "reject", (
            "verifier proposed a middle-excluding trim — rejected "
            "(edge-aligned trims only in v1)"
        )
    if new_end - new_start <= _EPS:
        return "reject", "verifier collapsed the trim to zero length"
    return "trim", ""


def verify_one(
    inp: VerifyInput,
    screen_path: Path,
    work_dir: Path,
    *,
    n_frames: int | None = None,
    min_gap: float | None = None,
    model: str | None = None,
    effort: str | None = None,
    timeout_s: float | None = None,
    max_recursion: int | None = None,
    _call_verifier: Callable[..., dict[str, Any]] | None = None,
    _recursion_depth: int = 0,
) -> VerifyOutput:
    """Ask the verifier whether this cut is real.

    Recurses into "trim" decisions until `max_recursion` is hit, then
    returns the latest trim as-is. Accept / reject / middle-trim all
    short-circuit the recursion.
    """
    n = n_frames if n_frames is not None else config.VERIFY_FRAME_SAMPLES
    gap = min_gap if min_gap is not None else config.VERIFY_MIN_FRAME_GAP_SEC
    mdl = model or config.VERIFY_MODEL
    eff = effort or config.VERIFY_EFFORT
    timeout = timeout_s if timeout_s is not None else config.LLM_TIMEOUT_SEC
    max_rec = max_recursion if max_recursion is not None else config.VERIFY_MAX_RECURSION
    call_fn = _call_verifier or _default_verifier_call

    ts = sample_frame_timestamps(inp.start, inp.end, n=n, min_gap=gap)
    frames = extract_frames(screen_path, ts, work_dir)

    raw = call_fn(inp, frames, model=mdl, effort=eff, timeout_s=timeout)
    decision = raw.get("decision", "reject")
    rationale = str(raw.get("rationale", ""))

    if decision == "accept":
        return VerifyOutput(
            start=inp.start, end=inp.end,
            decision="accept", rationale=rationale,
            original_start=inp.start, original_end=inp.end,
            frame_timestamps=tuple(ts),
            recursion_depth=_recursion_depth,
        )

    if decision == "reject":
        return VerifyOutput(
            start=inp.start, end=inp.end,
            decision="reject", rationale=rationale,
            original_start=inp.start, original_end=inp.end,
            frame_timestamps=tuple(ts),
            recursion_depth=_recursion_depth,
        )

    if decision == "trim":
        new_start = float(raw.get("start", inp.start))
        new_end = float(raw.get("end", inp.end))
        classified, reject_reason = _classify_trim(inp, new_start, new_end)
        if classified == "reject":
            return VerifyOutput(
                start=inp.start, end=inp.end,
                decision="reject", rationale=reject_reason,
                original_start=inp.start, original_end=inp.end,
                frame_timestamps=tuple(ts),
                recursion_depth=_recursion_depth,
            )
        # Edge-aligned trim. Recurse if budget remains.
        if _recursion_depth < max_rec:
            narrower = VerifyInput(
                start=new_start, end=new_end,
                reason=inp.reason, kind=inp.kind,
            )
            inner = verify_one(
                narrower,
                screen_path=screen_path, work_dir=work_dir,
                n_frames=n, min_gap=gap, model=mdl, effort=eff,
                timeout_s=timeout, max_recursion=max_rec,
                _call_verifier=call_fn,
                _recursion_depth=_recursion_depth + 1,
            )
            # Preserve the ORIGINAL bounds in the audit trail.
            return VerifyOutput(
                start=inner.start, end=inner.end,
                decision=inner.decision,
                rationale=inner.rationale or rationale,
                original_start=inp.start, original_end=inp.end,
                frame_timestamps=inner.frame_timestamps,
                recursion_depth=inner.recursion_depth,
            )
        return VerifyOutput(
            start=new_start, end=new_end,
            decision="trim", rationale=rationale,
            original_start=inp.start, original_end=inp.end,
            frame_timestamps=tuple(ts),
            recursion_depth=_recursion_depth,
        )

    # Unknown decision label — treat as reject for safety.
    return VerifyOutput(
        start=inp.start, end=inp.end,
        decision="reject",
        rationale=f"unknown decision label: {decision!r}",
        original_start=inp.start, original_end=inp.end,
        frame_timestamps=tuple(ts),
        recursion_depth=_recursion_depth,
    )


# --- batch runner + audit log -----------------------------------------


def run(
    candidates: list[VerifyInput],
    screen_path: Path,
    work_dir: Path,
    *,
    n_frames: int | None = None,
    min_gap: float | None = None,
    model: str | None = None,
    effort: str | None = None,
    timeout_s: float | None = None,
    max_recursion: int | None = None,
    _call_verifier: Callable[..., dict[str, Any]] | None = None,
) -> list[VerifyOutput]:
    """Run the verifier on a batch and write `work/verify_log.json`."""
    if not candidates:
        return []

    work_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[VerifyOutput] = []
    for inp in candidates:
        try:
            out = verify_one(
                inp, screen_path=screen_path, work_dir=work_dir,
                n_frames=n_frames, min_gap=min_gap,
                model=model, effort=effort, timeout_s=timeout_s,
                max_recursion=max_recursion,
                _call_verifier=_call_verifier,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "verifier failed on [%.2f, %.2f] (%s): %s — treating as reject",
                inp.start, inp.end, inp.reason, exc,
            )
            out = VerifyOutput(
                start=inp.start, end=inp.end,
                decision="reject",
                rationale=f"verifier error: {exc}",
                original_start=inp.start, original_end=inp.end,
                frame_timestamps=(),
                recursion_depth=0,
            )
        outputs.append(out)

    log_path = work_dir / "verify_log.json"
    entries = [
        {
            **asdict(o),
            "frame_timestamps": list(o.frame_timestamps),
        }
        for o in outputs
    ]
    log_path.write_text(json.dumps({"entries": entries}, indent=2))

    return outputs


__all__ = [
    "CutKind",
    "Decision",
    "VerifyInput",
    "VerifyOutput",
    "sample_frame_timestamps",
    "extract_frames",
    "verify_one",
    "run",
]
