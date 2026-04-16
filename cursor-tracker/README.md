# Cursor Tracker

Records cursor moves, clicks, and a clap-cue marker to a CSV sidecar during
video recording sessions. The long-form-editor pipeline consumes this CSV to
drive cursor-anchored auto-zoom and to correlate with OBS's screen recording
for frame-accurate T=0 sync.

## Install (one-time)

From the repo root (`long-form-editor/`):

```bash
./scripts/install_cursor_tracker.sh
```

This creates `cursor-tracker/venv/`, installs `pynput`, and prints the macOS
permissions you need to grant (Accessibility + Input Monitoring). Follow the
instructions exactly or the logger will receive zero events.

## Record a session

```bash
./cursor-tracker/record.sh ~/sessions/episode_01.csv
```

Output:
```
✓ Logging to: ~/sessions/episode_01.csv
  Hotkey: Ctrl+Option+Cmd+K = clap cue (screen flash + CSV marker)
  Ctrl+C to stop
```

### Recommended workflow

1. Start OBS with the `obs-source-record` filters attached to Display Capture
   and Video Capture Device (see [`../IMPLEMENTATION_PLAN.md`](../IMPLEMENTATION_PLAN.md) §M0b).
2. Hit **Record** in OBS.
3. Run `./cursor-tracker/record.sh <output.csv>` in a terminal.
4. **Within 2–3 s**, press `Ctrl+Option+Cmd+K` — this flashes a full-screen
   white window (captured by OBS as the sync cue) and writes `event=clap` in
   the CSV. Clap audibly at the same moment so the webcam track has a
   matching audio onset.
5. Record your session normally.
6. Stop OBS. Press **Ctrl+C** in the terminal to stop the logger.

The CSV now has:
- Every cursor move as `event=move`
- Every click as `event=click` with `button` and `down` (1=press, 0=release)
- Every scroll as `event=scroll` with `button="dx,dy"`
- One or more `event=clap` rows at the hotkey timestamps

## CSV schema

```
t_s,x,y,event,button,down
0.0523,1024.0,768.0,move,,
0.0814,1025.0,769.0,move,,
2.1403,512.0,384.0,clap,,
4.5612,843.0,221.0,click,left,1
4.6031,843.0,221.0,click,left,0
```

Timestamps are monotonic seconds from logger start. Coordinates are in
screen pixels (not normalized).

## Troubleshooting

**"No cursor events received after 3s" warning**
You haven't granted macOS permissions. Go to **System Settings → Privacy &
Security**:
- **Accessibility** → add your terminal (Terminal.app, iTerm, Ghostty, etc.)
- **Input Monitoring** → same
Then fully quit and re-open the terminal before re-running.

**Flash doesn't appear**
- Verify PyObjC is installed: `python -c "import AppKit"` from the cursor-tracker
  venv should succeed. If it fails, rerun `./scripts/install_cursor_tracker.sh`.
- The flash fires on **every connected display simultaneously**, so it works
  without configuration whether OBS is capturing your built-in screen, an
  external monitor, or any combination.
- Test the flash standalone first: `python cursor-tracker/screen_flash.py --duration 0.5`
  (500 ms — human-visible). You should see every monitor go white for half a second.
  If that works but the hotkey version doesn't, run `./cursor-tracker/record.sh`
  and watch for stderr from the flash subprocess when you press the hotkey.

**Flash appears but OBS didn't capture it**
OBS Display Capture on macOS sometimes clips the top menu bar or dock area.
The flash is true fullscreen so this shouldn't matter, but verify by scrubbing
the resulting screen mkv in QuickTime — you should see a one-frame white flash.

**Permission dialog keeps popping up**
macOS caches per-binary. If you upgrade Python or switch terminal apps, you
need to re-grant perms for the new binary.

## What's NOT logged

- Keyboard input (except the clap hotkey itself). If you want keystroke
  logging for subtitle cues or demo replay, extend `cursor_logger.py` —
  pynput's `keyboard.Listener` is the right primitive.
- Window focus changes or app switching. Would require private macOS APIs
  (CGWindowListCopyWindowInfo) and extra permissions.
- Touchpad gestures beyond scroll (pinch, rotate). Same reason.
