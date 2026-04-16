#!/usr/bin/env python3
"""Cursor position + click logger for dual-track video recording sessions.

Runs alongside OBS (which records screen + webcam via obs-source-record).
Writes a CSV of every cursor move, click, and scroll event with timestamps
relative to logger start. The downstream long-form-editor pipeline consumes
this CSV to drive cursor-anchored auto-zoom segments.

Also binds a global hotkey (Ctrl+Option+Cmd+K) that:
  1. Flashes a full-screen white window briefly (the "clap cue" for sync)
  2. Writes a row with event="clap" to the CSV

Usage:
  python cursor_logger.py                   # writes cursor_<epoch>.csv in cwd
  python cursor_logger.py session_01.csv    # custom output path

macOS permissions (one-time, in System Settings → Privacy & Security):
  - Accessibility: add Terminal (or your IDE)
  - Input Monitoring: add Terminal
Without these, pynput silently fails to receive events. Script aborts with
a diagnostic if we detect zero events after 3 seconds.
"""
from __future__ import annotations

import argparse
import csv
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    from pynput import keyboard, mouse
except ImportError:
    sys.stderr.write(
        "ERROR: pynput not installed.\n"
        "Run: scripts/install_cursor_tracker.sh (from repo root)\n"
    )
    sys.exit(1)


HERE = Path(__file__).resolve().parent
FLASH_SCRIPT = HERE / "screen_flash.py"


class CursorLogger:
    def __init__(self, out_path: Path) -> None:
        self.out_path = out_path
        self.t0 = time.monotonic()
        self._lock = threading.Lock()
        self._event_count = 0
        self._stop = threading.Event()

        self._fp = out_path.open("w", newline="", buffering=1)  # line-buffered
        self._writer = csv.writer(self._fp)
        self._writer.writerow(["t_s", "x", "y", "event", "button", "down"])
        self._fp.flush()

    def _log(
        self,
        event: str,
        x: float = 0.0,
        y: float = 0.0,
        button: str = "",
        down: int | str = "",
    ) -> None:
        with self._lock:
            t = f"{time.monotonic() - self.t0:.4f}"
            self._writer.writerow([t, x, y, event, button, down])
            self._event_count += 1

    # Mouse event callbacks
    def on_move(self, x: int, y: int) -> None:
        self._log("move", x, y)

    def on_click(self, x: int, y: int, button, pressed: bool) -> None:
        self._log("click", x, y, button.name, int(pressed))

    def on_scroll(self, x: int, y: int, dx: int, dy: int) -> None:
        self._log("scroll", x, y, button=f"{dx},{dy}")

    # Hotkey callback
    def on_clap(self) -> None:
        self._log("clap")
        print(f"  · clap cue at t={time.monotonic()-self.t0:.2f}s — flashing screen")
        # Spawn flash as subprocess so tkinter gets its own interpreter instance
        # (avoids threading issues with pynput's listener).
        subprocess.Popen(
            [sys.executable, str(FLASH_SCRIPT)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )

    def sanity_check(self) -> None:
        """After 3 s, warn if no events received — macOS permissions likely missing."""
        time.sleep(3.0)
        if self._event_count == 0 and not self._stop.is_set():
            sys.stderr.write(
                "\n⚠  No cursor events received after 3s.\n"
                "   macOS requires Accessibility + Input Monitoring permissions.\n"
                "   System Settings → Privacy & Security → add this Terminal/IDE.\n"
                "   Then quit and re-run.\n\n"
            )

    def run(self) -> None:
        print(f"✓ Logging to: {self.out_path}")
        print("  Hotkey: Ctrl+Option+Cmd+K = clap cue (screen flash + CSV marker)")
        print("  Ctrl+C to stop\n")

        mouse_listener = mouse.Listener(
            on_move=self.on_move,
            on_click=self.on_click,
            on_scroll=self.on_scroll,
        )
        hotkey_listener = keyboard.GlobalHotKeys(
            {"<ctrl>+<alt>+<cmd>+k": self.on_clap}
        )

        # Graceful shutdown on SIGINT / SIGTERM
        def _shutdown(_signum=None, _frame=None):
            self._stop.set()
            mouse_listener.stop()
            hotkey_listener.stop()

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        threading.Thread(target=self.sanity_check, daemon=True).start()

        mouse_listener.start()
        hotkey_listener.start()
        try:
            mouse_listener.join()
        finally:
            hotkey_listener.stop()
            self._fp.close()
            print(
                f"\n✓ Wrote {self._event_count} events to {self.out_path} "
                f"(duration {time.monotonic() - self.t0:.1f}s)"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Log cursor events to CSV for long-form-editor pipeline."
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=f"cursor_{int(time.time())}.csv",
        help="CSV output path (default: cursor_<epoch>.csv in cwd)",
    )
    args = parser.parse_args()

    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    CursorLogger(out).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
