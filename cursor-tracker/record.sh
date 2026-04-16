#!/usr/bin/env bash
# Launch cursor_logger.py with the project venv activated.
# Usage:
#   ./cursor-tracker/record.sh                # writes cursor_<epoch>.csv here
#   ./cursor-tracker/record.sh ~/sessions/s01.csv
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/venv"

if [[ ! -d "$VENV" ]]; then
  echo "error: venv not found at $VENV" >&2
  echo "run: $HERE/../scripts/install_cursor_tracker.sh" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
exec python "$HERE/cursor_logger.py" "$@"
