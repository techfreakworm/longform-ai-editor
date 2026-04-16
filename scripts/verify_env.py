#!/usr/bin/env python3
"""Sanity-check the long-form-editor environment.

Exits 0 if everything looks ready, 1 with a diagnostic report otherwise.

Checks:
  1. System binaries: ffmpeg, auto-editor, jq
  2. Python deps importable: mlx_whisper, mlx_lm, librosa, cv2, numpy,
     ffmpeg_normalize, httpx, pydantic
  3. LLM server reachable at LLM_SERVER_URL
  4. Required HF models present in ~/.cache/huggingface
  5. obs-source-record plugin present (warning only — not strictly required
     to run the pipeline, only to record new sessions)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET}  {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")


def check_binaries() -> bool:
    print("▶ system binaries")
    needed = ["ffmpeg", "ffprobe", "auto-editor", "jq"]
    missing = []
    for b in needed:
        if shutil.which(b):
            ok(f"{b} ({shutil.which(b)})")
        else:
            fail(f"{b} not found on PATH")
            missing.append(b)
    return not missing


def check_python_deps() -> bool:
    print("▶ python imports")
    imports = [
        "mlx_whisper",
        "mlx_lm",
        "librosa",
        "cv2",
        "numpy",
        "ffmpeg_normalize",
        "httpx",
        "pydantic",
        "tenacity",
        "dotenv",
    ]
    missing = []
    for mod in imports:
        try:
            __import__(mod)
            ok(mod)
        except ImportError as e:
            fail(f"{mod} — {e}")
            missing.append(mod)
    return not missing


def check_llm_server() -> bool:
    print("▶ MLX LLM server")
    url = os.environ.get("LLM_SERVER_URL", "http://127.0.0.1:8080/v1")
    try:
        import httpx

        r = httpx.get(f"{url.rstrip('/')}/models", timeout=3.0)
        if r.status_code == 200:
            models = [m.get("id", "?") for m in r.json().get("data", [])]
            ok(f"reachable at {url}")
            ok(f"models: {', '.join(models)[:80]}")
            return True
        else:
            fail(f"{url} returned HTTP {r.status_code}")
            return False
    except Exception as e:
        warn(f"{url} not reachable — {e}")
        warn("  start it with:")
        warn("  mlx_lm.server --model Qwen/Qwen3-235B-A22B-MLX-4bit --port 8080")
        return False


def check_models() -> bool:
    print("▶ HuggingFace model cache")
    # hf_hub uses ~/.cache/huggingface/hub/models--<org>--<name>
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    needed = [
        "models--mlx-community--whisper-large-v3-turbo",
        "models--Qwen--Qwen3-235B-A22B-MLX-4bit",
    ]
    missing = []
    for m in needed:
        p = cache / m
        if p.is_dir():
            size_gb = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9
            ok(f"{m.split('--',1)[-1]} ({size_gb:.1f} GB)")
        else:
            fail(f"{m} not found in {cache}")
            missing.append(m)
    if missing:
        warn("  pre-pull with:")
        for m in missing:
            name = m.replace("models--", "").replace("--", "/")
            warn(f"  hf download {name}")
    return not missing


def check_obs_plugin() -> bool:
    print("▶ OBS plugins")
    candidates = [
        Path.home() / "Library/Application Support/obs-studio/plugins/source-record.plugin",
        Path("/Library/Application Support/obs-studio/plugins/source-record.plugin"),
    ]
    for p in candidates:
        if p.exists():
            ok(f"obs-source-record ({p})")
            return True
    warn("obs-source-record plugin not found (optional — only needed for recording)")
    warn("  download .pkg from: https://github.com/exeldro/obs-source-record/releases/tag/0.4.8")
    return True  # non-fatal


def main() -> int:
    print("long-form-editor environment check")
    print()
    checks = [
        ("binaries", check_binaries),
        ("python", check_python_deps),
        ("models", check_models),
        ("llm server", check_llm_server),
        ("obs plugin", check_obs_plugin),
    ]
    results = {}
    for name, fn in checks:
        results[name] = fn()
        print()

    all_pass = all(results.values())
    if all_pass:
        print(f"{GREEN}✓ all checks passed{RESET}")
        return 0
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"{RED}✗ failed: {', '.join(failed)}{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
