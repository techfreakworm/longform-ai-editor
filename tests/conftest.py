"""Pytest configuration + shared fixtures for long-form-editor.

Fixtures live under tests/fixtures/real/ as symlinks to an actual short
recording the user produced with OBS + cursor-tracker:

    tests/fixtures/real/
      merged.mp4   ← OBS merged recording (video useless, audio authoritative)
      cam.mov      ← webcam source-record (1920x1080 @60, h264 + aac)
      ext.mov      ← external monitor source-record (2560x1440 @60, h264)
      cursor.csv   ← cursor_logger.py output, with one event=clap row

Tests skip with a clear message if these symlinks are missing (e.g., on a
fresh clone without the real recording).
"""
from __future__ import annotations

from pathlib import Path

import pytest

REAL_FIXTURES = Path(__file__).parent / "fixtures" / "real"


def _require_real(*names: str) -> dict[str, Path]:
    """Return paths if all named symlinks resolve; otherwise skip the test."""
    resolved: dict[str, Path] = {}
    missing: list[str] = []
    for name in names:
        p = REAL_FIXTURES / name
        if not p.exists():
            missing.append(str(p))
        else:
            resolved[name] = p
    if missing:
        pytest.skip(
            "real fixtures not present — run the fixture-symlink setup: "
            + ", ".join(missing)
        )
    return resolved


@pytest.fixture(scope="session")
def real_ext() -> Path:
    return _require_real("ext.mov")["ext.mov"]


@pytest.fixture(scope="session")
def real_cam() -> Path:
    return _require_real("cam.mov")["cam.mov"]


@pytest.fixture(scope="session")
def real_merged() -> Path:
    return _require_real("merged.mp4")["merged.mp4"]


@pytest.fixture(scope="session")
def real_cursor() -> Path:
    return _require_real("cursor.csv")["cursor.csv"]


@pytest.fixture(scope="session")
def real_bundle(real_ext: Path, real_cam: Path, real_merged: Path, real_cursor: Path) -> dict[str, Path]:
    return {
        "ext": real_ext,
        "cam": real_cam,
        "merged": real_merged,
        "cursor": real_cursor,
    }
