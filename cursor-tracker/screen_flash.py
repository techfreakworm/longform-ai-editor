#!/usr/bin/env python3
"""Full-screen white flash for ~80ms — the "clap cue" marker.

Invoked as a subprocess by cursor_logger.py when the user presses the clap
hotkey. The flash is captured by OBS's Display Capture and gets detected by
the sync_clap.py stage later, giving us a frame-accurate T=0 alignment with
the webcam audio clap.

Runs in its own interpreter so tkinter's NSApp doesn't collide with pynput's
listener threads.
"""
from __future__ import annotations

import tkinter as tk


def flash(duration_ms: int = 80) -> None:
    root = tk.Tk()
    # Make sure it's on top of everything including OBS preview.
    root.attributes("-fullscreen", True)
    root.attributes("-topmost", True)
    root.configure(bg="white")
    # overrideredirect removes the window chrome.
    root.overrideredirect(True)
    # Force an initial paint before starting the timer — on macOS the
    # compositor may defer the first frame otherwise.
    root.update_idletasks()
    root.update()
    root.after(duration_ms, root.destroy)
    root.mainloop()


if __name__ == "__main__":
    flash()
