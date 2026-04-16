#!/usr/bin/env bash
# Install the cursor tracker — standalone venv, ready for immediate use in
# recording sessions. Does NOT install the rest of the long-form-editor
# pipeline; for that run ./scripts/install.sh.
#
# Usage: ./scripts/install_cursor_tracker.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRACKER="$HERE/cursor-tracker"
VENV="$TRACKER/venv"

echo "▶ Installing cursor tracker"
echo "  base: $TRACKER"
echo

# 1. Locate Python 3.11+
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
  echo "error: python 3.11 or newer not found" >&2
  echo "  install via: brew install python@3.12" >&2
  exit 1
fi
echo "✓ using $PYTHON ($(command -v "$PYTHON"))"

# 2. Create venv (per user preference: python -m venv, no conda)
if [[ -d "$VENV" ]]; then
  echo "✓ venv already exists at $VENV"
else
  echo "▶ creating venv at $VENV"
  "$PYTHON" -m venv "$VENV"
fi

# 3. Install deps
echo "▶ installing pynput into venv"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet -r "$TRACKER/requirements.txt"
# Quick import smoke test so install failures surface now, not at first clap.
python -c "import pynput; import AppKit" || {
  echo "error: installed packages failed to import" >&2
  exit 1
}
deactivate

# 4. Make scripts executable
chmod +x "$TRACKER/cursor_logger.py" "$TRACKER/screen_flash.py" "$TRACKER/record.sh"

# 5. Print macOS permissions instructions
cat <<'PERMS'

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ⚠  macOS PERMISSIONS REQUIRED (one-time, per terminal app)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pynput hooks global input events. macOS sandbox requires explicit permission
for each app that does this. Without these grants, cursor_logger.py will
receive ZERO events and warn after 3 s.

  1. Open  System Settings  →  Privacy & Security
  2. Click Accessibility    →  "+"  →  add your Terminal (Terminal.app,
                                       iTerm, Ghostty, VS Code, etc.)
  3. Click Input Monitoring →  "+"  →  add the same Terminal
  4. FULLY QUIT the Terminal (Cmd+Q) and re-open it. macOS only re-reads
     grants on process restart.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ cursor tracker installed.

To start logging a session:

  ./cursor-tracker/record.sh ~/sessions/episode_01.csv

In-session:
  · Clap hotkey:  Ctrl + Option + Cmd + K
  · Stop logger:  Ctrl + C

PERMS
