#!/usr/bin/env python3
"""Full-screen white flash via native macOS NSWindow (AppKit/Cocoa).

Invoked as a subprocess by cursor_logger.py when the clap hotkey fires.
Appears above every other window including the menu bar, holds for ~80ms,
then closes. The flash is captured by OBS's Display Capture, and the
sync_clap.py stage later detects the luminance jump to compute T=0.

Uses PyObjC instead of tkinter because:
  - Homebrew Python doesn't bundle Tcl/Tk by default
  - Tk's -fullscreen + -topmost combo is unreliable on macOS
  - NSWindow at NSScreenSaverWindowLevel is above the menu bar and
    reliably comes to front via orderFrontRegardless()

Exits 0 on success, 1 on failure (with diagnostic on stderr — unlike the
old version, stderr is NOT redirected to /dev/null so you see the error).
"""
from __future__ import annotations

import argparse
import sys

try:
    from AppKit import (
        NSApplication,
        NSApplicationActivationPolicyAccessory,
        NSBackingStoreBuffered,
        NSColor,
        NSScreen,
        NSScreenSaverWindowLevel,
        NSWindow,
        NSWindowStyleMaskBorderless,
    )
    from Foundation import NSDate, NSRunLoop
except ImportError as e:
    sys.stderr.write(
        "ERROR: PyObjC / AppKit not available.\n"
        "Install with:\n"
        "    cd <long-form-editor>\n"
        "    ./scripts/install_cursor_tracker.sh\n"
        "(which runs `pip install pyobjc-framework-Cocoa` into the venv)\n"
        f"\nOriginal import error: {e}\n"
    )
    sys.exit(1)


def flash(duration_s: float = 0.08) -> None:
    """Show a full-screen white window for duration_s seconds, then close."""
    app = NSApplication.sharedApplication()
    # Accessory = runs without a dock icon and doesn't steal focus.
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

    screen = NSScreen.mainScreen()
    if screen is None:
        raise RuntimeError("no main screen detected")
    frame = screen.frame()

    window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        frame,
        NSWindowStyleMaskBorderless,
        NSBackingStoreBuffered,
        False,
    )
    window.setBackgroundColor_(NSColor.whiteColor())
    window.setOpaque_(True)
    # ScreenSaver level sits above the menu bar + every app window.
    window.setLevel_(NSScreenSaverWindowLevel)
    window.setIgnoresMouseEvents_(True)
    window.orderFrontRegardless()

    # Pump the event loop for duration_s so the window paints and holds,
    # then fall through to close.
    deadline = NSDate.dateWithTimeIntervalSinceNow_(duration_s)
    NSRunLoop.currentRunLoop().runUntilDate_(deadline)

    window.orderOut_(None)


def main() -> int:
    parser = argparse.ArgumentParser(description="macOS full-screen white clap flash.")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.08,
        help="flash duration in seconds (default: 0.08 = ~5 frames @ 60 fps)",
    )
    args = parser.parse_args()
    flash(args.duration)
    return 0


if __name__ == "__main__":
    sys.exit(main())
