#!/usr/bin/env bash
# Full long-form-editor installation on macOS arm64.
# Installs system deps (brew), creates venv, installs Python deps, pre-pulls
# Hugging Face models.
#
# Run: ./scripts/install.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$HERE/venv"

echo "▶ long-form-editor installer"
echo "  base: $HERE"
echo

# --- System deps ---------------------------------------------------------
echo "▶ system deps (brew)"
brew_install_if_missing() {
  local pkg="$1"
  if brew list --formula "$pkg" >/dev/null 2>&1; then
    echo "  ✓ $pkg (already installed)"
  else
    echo "  ▶ installing $pkg"
    brew install "$pkg"
  fi
}

brew_install_if_missing ffmpeg
brew_install_if_missing auto-editor
brew_install_if_missing jq
# rubberband is GPL-2.0+ — only install if user wants higher-quality audio stretch.
# Uncomment to enable:
# brew_install_if_missing rubberband

# OBS is a cask; only needed for recording, not for running the pipeline.
if ! brew list --cask obs >/dev/null 2>&1; then
  echo "  ℹ  OBS not found (not required for pipeline, only for recording)"
  echo "     install with: brew install --cask obs"
fi

# obs-source-record plugin: must be installed manually by the user because
# Gatekeeper blocks unsigned .pkg installs from scripts.
echo
echo "  ℹ  obs-source-record plugin (required for dual-track recording)"
echo "     download from: https://github.com/exeldro/obs-source-record/releases/tag/0.4.8"
echo "     file: source-record-0.4.8-macos-arm64.pkg"
echo "     install by double-clicking (Gatekeeper may prompt to allow)"

# --- Python venv ---------------------------------------------------------
echo
echo "▶ Python venv"

PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    version="$("$candidate" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    major="${version%%.*}"
    minor="${version##*.}"
    if [[ "$major" -eq 3 && "$minor" -ge 11 ]]; then
      PYTHON="$candidate"
      break
    fi
  fi
done

if [[ -z "$PYTHON" ]]; then
  echo "error: python 3.11 or newer required" >&2
  echo "  install via: brew install python@3.12" >&2
  exit 1
fi
echo "  ✓ using $PYTHON ($(command -v "$PYTHON"))"

if [[ -d "$VENV" ]]; then
  echo "  ✓ venv exists at $VENV"
else
  echo "  ▶ creating $VENV"
  "$PYTHON" -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --quiet --upgrade pip
echo "  ▶ installing python deps"
pip install -e ".[dev]"
deactivate

# --- HuggingFace login + model pre-pull ---------------------------------
echo
echo "▶ Hugging Face setup"

if ! command -v hf >/dev/null 2>&1; then
  echo "  ▶ installing hf CLI"
  source "$VENV/bin/activate"
  pip install --quiet "huggingface_hub[cli]"
  deactivate
fi

if hf whoami >/dev/null 2>&1; then
  echo "  ✓ already authenticated ($(hf whoami 2>/dev/null | head -1))"
else
  echo "  ▶ hf auth login (interactive — paste token from https://huggingface.co/settings/tokens)"
  hf auth login
fi

echo "  ▶ pre-pulling models to ~/.cache/huggingface (one-time, ~36 GB)"
hf download mlx-community/whisper-large-v3-turbo
hf download Qwen/Qwen3-235B-A22B-MLX-4bit

# --- Done ----------------------------------------------------------------
cat <<'DONE'

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ long-form-editor installed

Next steps:

1. Install obs-source-record .pkg (see link above) if you haven't.
2. Copy .env.example to .env and adjust if needed.
3. Start the LLM server in a separate terminal:

     source venv/bin/activate
     mlx_lm.server --model Qwen/Qwen3-235B-A22B-MLX-4bit --port 8080

4. Verify the environment:

     python scripts/verify_env.py

5. For recording sessions, run the cursor tracker separately:

     ./scripts/install_cursor_tracker.sh   # first time
     ./cursor-tracker/record.sh ~/sessions/ep01.csv

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DONE
