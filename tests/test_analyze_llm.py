"""Tests for Stage B.2 — LLM-driven filler + layout analysis.

Unit tests mock httpx (local MLX path) or subprocess (Claude CLI path) to
avoid needing a running LLM. @slow integration tests hit the real
backends if reachable.

The `_force_local_llm_path` autouse fixture pins the dispatcher to the
local-MLX path for most tests so legacy httpx-mocking tests don't
accidentally go through Claude CLI (which would be present on the dev's
machine but absent in CI). Tests that want to exercise the Claude path
use `claude_enabled` explicitly.
"""
from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.stages.analyze_llm import (
    FillerCut,
    FillerCutsResponse,
    LayoutPlanResponse,
    LayoutSegment,
    _call_via_claude_cli,
    _extract_json_body,
    _fill_coverage_gaps,
    _have_claude_cli,
    analyze_fillers,
    analyze_layout,
    call_llm_json,
    strip_thinking,
)


@pytest.fixture(autouse=True)
def _force_local_llm_path(monkeypatch):
    """Default: dispatcher goes to local MLX so httpx mocks work.
    Tests that need the Claude path explicitly override this.
    """
    monkeypatch.setattr("src.stages.analyze_llm._have_claude_cli", lambda: False)


@pytest.fixture
def claude_enabled(monkeypatch):
    """Opt-in: enable Claude CLI dispatch for a specific test."""
    monkeypatch.setattr("src.stages.analyze_llm._have_claude_cli", lambda: True)
    # Also ensure FORCE_LOCAL_LLM is not set
    from src import config
    monkeypatch.setattr(config, "FORCE_LOCAL_LLM", False)


# --- strip_thinking -----------------------------------------------------

def test_strip_thinking_removes_tags() -> None:
    assert strip_thinking("<think>internal</think>hello") == "hello"


def test_strip_thinking_multiline() -> None:
    txt = "pre\n<think>\nmulti\nline\n</think>\npost"
    assert strip_thinking(txt) == "pre\n\npost"


def test_strip_thinking_no_tags_passthrough() -> None:
    assert strip_thinking("plain text") == "plain text"


# --- _extract_json_body ------------------------------------------------

def test_extract_json_body_plain() -> None:
    assert _extract_json_body('{"a": 1}') == '{"a": 1}'


def test_extract_json_body_with_markdown_fence() -> None:
    text = '```json\n{"cuts": []}\n```'
    assert _extract_json_body(text) == '{"cuts": []}'


def test_extract_json_body_with_preamble() -> None:
    text = 'Sure, here is the JSON:\n{"x": 42}\nEnd.'
    assert _extract_json_body(text) == '{"x": 42}'


def test_extract_json_body_nested_braces() -> None:
    text = '{"outer": {"inner": 1}}'
    assert _extract_json_body(text) == text


def test_extract_json_body_no_json_raises() -> None:
    with pytest.raises(ValueError, match="no JSON"):
        _extract_json_body("just prose, no JSON here")


def test_extract_json_body_unbalanced_raises() -> None:
    with pytest.raises(ValueError, match="unbalanced"):
        _extract_json_body('{"oops": "never closed"')


# --- _fill_coverage_gaps ------------------------------------------------

def _L(start, end, layout):
    return LayoutSegment(start=start, end=end, layout=layout)


def test_fill_coverage_no_gap_no_change() -> None:
    segs = [_L(0, 10, "cam_full"), _L(10, 30, "pip")]
    out = _fill_coverage_gaps(segs, 30.0)
    assert [(s.start, s.end, s.layout) for s in out] == [
        (0, 10, "cam_full"), (10, 30, "pip"),
    ]


def test_fill_coverage_leading_gap_inserts_default() -> None:
    segs = [_L(5, 15, "pip")]
    out = _fill_coverage_gaps(segs, 15.0, default_layout="pip")
    assert [(s.start, s.end, s.layout) for s in out] == [
        (0, 5, "pip"), (5, 15, "pip"),
    ]


def test_fill_coverage_middle_gap_extends_prev() -> None:
    """Gap between 10 and 15 is filled by extending the first segment."""
    segs = [_L(0, 10, "cam_full"), _L(15, 30, "pip")]
    out = _fill_coverage_gaps(segs, 30.0)
    assert out[0].end == 15  # extended
    assert out[0].layout == "cam_full"
    assert out[1].start == 15


def test_fill_coverage_trailing_gap_extends_last() -> None:
    segs = [_L(0, 20, "pip")]
    out = _fill_coverage_gaps(segs, 30.0)
    assert out[-1].end == 30
    assert out[-1].layout == "pip"


def test_fill_coverage_overlap_truncates_later() -> None:
    segs = [_L(0, 20, "pip"), _L(15, 30, "cam_full")]
    out = _fill_coverage_gaps(segs, 30.0)
    assert out[0].end == 20
    assert out[1].start == 20  # truncated
    assert out[1].end == 30


def test_fill_coverage_empty_input_single_default() -> None:
    out = _fill_coverage_gaps([], 30.0, default_layout="cam_full")
    assert len(out) == 1
    assert (out[0].start, out[0].end, out[0].layout) == (0, 30, "cam_full")


def test_fill_coverage_segment_fully_inside_other_dropped() -> None:
    segs = [_L(0, 30, "pip"), _L(5, 10, "cam_full")]
    out = _fill_coverage_gaps(segs, 30.0)
    # The inner cam_full is fully inside the outer pip — dropped
    assert len(out) == 1
    assert out[0].layout == "pip"


# --- call_llm_json (mocked http) -------------------------------------

def _mock_httpx_response(content_json: dict) -> MagicMock:
    m = MagicMock(spec=httpx.Response)
    m.status_code = 200
    m.raise_for_status = MagicMock()
    m.json.return_value = {
        "choices": [
            {"message": {"content": json.dumps(content_json)}}
        ],
    }
    return m


def test_call_llm_json_returns_parsed_body() -> None:
    with patch("src.stages.analyze_llm.httpx.post",
               return_value=_mock_httpx_response({"result": 42})):
        out = call_llm_json("sys prompt", {"user": "payload"})
    assert out == {"result": 42}


def test_call_llm_json_strips_think_tags() -> None:
    content = "<think>noisy reasoning</think>{\"result\": 1}"
    m = MagicMock(spec=httpx.Response)
    m.status_code = 200
    m.raise_for_status = MagicMock()
    m.json.return_value = {"choices": [{"message": {"content": content}}]}
    with patch("src.stages.analyze_llm.httpx.post", return_value=m):
        out = call_llm_json("sys", {})
    assert out == {"result": 1}


# --- analyze_fillers (mocked) ----------------------------------------

def test_analyze_fillers_parses_valid_response() -> None:
    payload = {
        "cuts": [
            {"start": 5.4, "end": 5.8, "reason": "filler"},
            {"start": 12.1, "end": 12.6, "reason": "false_start"},
        ],
    }
    with patch("src.stages.analyze_llm.httpx.post",
               return_value=_mock_httpx_response(payload)):
        out = analyze_fillers([{"word": "um", "start": 5.4, "end": 5.8}])
    assert isinstance(out, FillerCutsResponse)
    assert len(out.cuts) == 2
    assert out.cuts[0].reason == "filler"


def test_analyze_fillers_empty_cuts_is_valid() -> None:
    with patch("src.stages.analyze_llm.httpx.post",
               return_value=_mock_httpx_response({"cuts": []})):
        out = analyze_fillers([])
    assert out.cuts == []


def test_analyze_fillers_retries_on_bad_schema() -> None:
    """A genuinely invalid response triggers tenacity retry — `cuts` must
    be a list, passing a string fails pydantic and forces a second call.
    """
    bad = _mock_httpx_response({"cuts": "not a list"})
    good = _mock_httpx_response({"cuts": [{"start": 1.0, "end": 2.0}]})
    with patch("src.stages.analyze_llm.httpx.post",
               side_effect=[bad, good]) as m:
        out = analyze_fillers([])
    assert m.call_count == 2
    assert len(out.cuts) == 1


# --- analyze_layout (mocked) -----------------------------------------

def test_analyze_layout_post_fills_gaps() -> None:
    """LLM returns a plan with gaps; post-processor fills them to full coverage."""
    payload = {
        "segments": [
            {"start": 0, "end": 10, "layout": "cam_full"},
            {"start": 15, "end": 30, "layout": "pip"},  # 5 s gap
        ],
    }
    with patch("src.stages.analyze_llm.httpx.post",
               return_value=_mock_httpx_response(payload)):
        out = analyze_layout([], total_duration=30.0)
    # Total coverage must equal 30 s, no gaps
    total = sum(s.end - s.start for s in out.segments)
    assert abs(total - 30.0) < 1e-6
    # No overlaps
    for a, b in zip(out.segments, out.segments[1:]):
        assert b.start >= a.end - 1e-6


def test_analyze_layout_rejects_bad_layout() -> None:
    payload = {"segments": [{"start": 0, "end": 10, "layout": "bogus"}]}
    good = _mock_httpx_response({
        "segments": [{"start": 0, "end": 10, "layout": "pip"}],
    })
    with patch("src.stages.analyze_llm.httpx.post",
               side_effect=[_mock_httpx_response(payload), good]) as m:
        out = analyze_layout([], total_duration=10.0)
    assert m.call_count == 2
    assert out.segments[0].layout == "pip"


# --- Claude CLI path --------------------------------------------------

def _mock_completed_process(stdout: str, returncode: int = 0, stderr: str = "") -> MagicMock:
    m = MagicMock(spec=subprocess.CompletedProcess)
    m.stdout = stdout
    m.returncode = returncode
    m.stderr = stderr
    return m


def test_call_via_claude_cli_parses_clean_output(claude_enabled) -> None:
    """Happy path: Claude returns pristine JSON on stdout."""
    with patch("src.stages.analyze_llm.subprocess.run",
               return_value=_mock_completed_process('{"result": "ok"}')) as m:
        out = _call_via_claude_cli("sys", {"q": 1})
    cmd = m.call_args[0][0]
    assert cmd[0] == "claude"
    assert "-p" in cmd
    assert "--model" in cmd
    assert "--output-format" in cmd
    assert "bypassPermissions" in cmd
    assert out == {"result": "ok"}


def test_call_via_claude_cli_tolerates_preamble(claude_enabled) -> None:
    """Claude sometimes adds preamble. _extract_json_body pulls the {...}."""
    noisy = "I'll analyze this for you.\n\n```json\n{\"cuts\": []}\n```\n"
    with patch("src.stages.analyze_llm.subprocess.run",
               return_value=_mock_completed_process(noisy)):
        out = _call_via_claude_cli("sys", {})
    assert out == {"cuts": []}


def test_call_via_claude_cli_nonzero_exit_raises(claude_enabled) -> None:
    """A failed CLI call surfaces as RuntimeError with the tail of stderr."""
    with patch("src.stages.analyze_llm.subprocess.run",
               return_value=_mock_completed_process("", returncode=1, stderr="auth error")):
        with pytest.raises(RuntimeError, match="auth error"):
            _call_via_claude_cli("sys", {})


def test_call_via_claude_cli_passes_effort_flag(claude_enabled) -> None:
    """The CLI command includes `--effort` set to config.CLAUDE_EFFORT (default 'max')."""
    from src import config
    with patch("src.stages.analyze_llm.subprocess.run",
               return_value=_mock_completed_process('{"ok": 1}')) as m:
        _call_via_claude_cli("sys", {})
    cmd = m.call_args[0][0]
    assert "--effort" in cmd
    i = cmd.index("--effort")
    assert cmd[i + 1] == config.CLAUDE_EFFORT


def test_call_via_claude_cli_passes_mcp_config_when_file_exists(
    claude_enabled, monkeypatch, tmp_path
) -> None:
    """When USE_SEQUENTIAL_THINKING is on and .mcp.json exists, `--mcp-config` is passed."""
    from src import config
    mcp_file = tmp_path / ".mcp.json"
    mcp_file.write_text("{}")
    monkeypatch.setattr(config, "USE_SEQUENTIAL_THINKING", True)
    monkeypatch.setattr(config, "CLAUDE_MCP_CONFIG", mcp_file)
    with patch("src.stages.analyze_llm.subprocess.run",
               return_value=_mock_completed_process('{"ok": 1}')) as m:
        _call_via_claude_cli("sys", {})
    cmd = m.call_args[0][0]
    assert "--mcp-config" in cmd
    i = cmd.index("--mcp-config")
    assert cmd[i + 1] == str(mcp_file)


def test_call_via_claude_cli_omits_mcp_config_when_disabled(
    claude_enabled, monkeypatch, tmp_path
) -> None:
    """When USE_SEQUENTIAL_THINKING is off, `--mcp-config` is NOT passed."""
    from src import config
    mcp_file = tmp_path / ".mcp.json"
    mcp_file.write_text("{}")
    monkeypatch.setattr(config, "USE_SEQUENTIAL_THINKING", False)
    monkeypatch.setattr(config, "CLAUDE_MCP_CONFIG", mcp_file)
    with patch("src.stages.analyze_llm.subprocess.run",
               return_value=_mock_completed_process('{"ok": 1}')) as m:
        _call_via_claude_cli("sys", {})
    cmd = m.call_args[0][0]
    assert "--mcp-config" not in cmd


def test_call_via_claude_cli_prepends_sequential_thinking_prefix(
    claude_enabled, monkeypatch,
) -> None:
    """The stdin payload includes SEQUENTIAL_THINKING_PREFIX when the flag is on."""
    from src import config
    monkeypatch.setattr(config, "USE_SEQUENTIAL_THINKING", True)
    with patch("src.stages.analyze_llm.subprocess.run",
               return_value=_mock_completed_process('{"ok": 1}')) as m:
        _call_via_claude_cli("ORIGINAL_SYS_PROMPT", {})
    stdin = m.call_args.kwargs["input"]
    assert config.SEQUENTIAL_THINKING_PREFIX.strip() in stdin
    assert "ORIGINAL_SYS_PROMPT" in stdin


def test_call_via_claude_cli_omits_prefix_when_disabled(
    claude_enabled, monkeypatch,
) -> None:
    """When USE_SEQUENTIAL_THINKING is off, the prefix is absent from stdin."""
    from src import config
    monkeypatch.setattr(config, "USE_SEQUENTIAL_THINKING", False)
    with patch("src.stages.analyze_llm.subprocess.run",
               return_value=_mock_completed_process('{"ok": 1}')) as m:
        _call_via_claude_cli("ORIGINAL_SYS_PROMPT", {})
    stdin = m.call_args.kwargs["input"]
    assert config.SEQUENTIAL_THINKING_PREFIX.strip() not in stdin
    assert "ORIGINAL_SYS_PROMPT" in stdin


# --- Dispatcher logic -------------------------------------------------

def test_dispatcher_prefers_claude_when_available(claude_enabled) -> None:
    """Claude CLI is reached first; httpx.post is NOT called."""
    with patch("src.stages.analyze_llm.subprocess.run",
               return_value=_mock_completed_process('{"v": 1}')) as sub_m, \
         patch("src.stages.analyze_llm.httpx.post") as http_m:
        out = call_llm_json("sys", {})
    assert sub_m.call_count == 1
    assert http_m.call_count == 0
    assert out == {"v": 1}


def test_dispatcher_falls_back_to_local_when_claude_fails(claude_enabled) -> None:
    """Claude CLI throws; dispatcher logs warning, retries with local MLX."""
    http_resp = MagicMock(spec=httpx.Response)
    http_resp.status_code = 200
    http_resp.raise_for_status = MagicMock()
    http_resp.json.return_value = {
        "choices": [{"message": {"content": '{"v": 2}'}}],
    }
    with patch(
        "src.stages.analyze_llm.subprocess.run",
        return_value=_mock_completed_process("", returncode=2, stderr="boom"),
    ) as sub_m, patch(
        "src.stages.analyze_llm.httpx.post",
        return_value=http_resp,
    ) as http_m:
        out = call_llm_json("sys", {})
    assert sub_m.call_count == 1
    assert http_m.call_count == 1
    assert out == {"v": 2}


def test_dispatcher_uses_local_when_claude_unavailable() -> None:
    """Without the `claude_enabled` fixture, _have_claude_cli returns False
    (per autouse `_force_local_llm_path`) and dispatcher goes straight to MLX.
    """
    http_resp = MagicMock(spec=httpx.Response)
    http_resp.status_code = 200
    http_resp.raise_for_status = MagicMock()
    http_resp.json.return_value = {
        "choices": [{"message": {"content": '{"v": 3}'}}],
    }
    with patch("src.stages.analyze_llm.subprocess.run") as sub_m, \
         patch("src.stages.analyze_llm.httpx.post", return_value=http_resp) as http_m:
        out = call_llm_json("sys", {})
    assert sub_m.call_count == 0  # claude never attempted
    assert http_m.call_count == 1
    assert out == {"v": 3}


def test_dispatcher_respects_force_local_env(claude_enabled, monkeypatch) -> None:
    """FORCE_LOCAL_LLM=1 skips Claude even when the CLI is present."""
    from src import config
    monkeypatch.setattr(config, "FORCE_LOCAL_LLM", True)

    http_resp = MagicMock(spec=httpx.Response)
    http_resp.status_code = 200
    http_resp.raise_for_status = MagicMock()
    http_resp.json.return_value = {
        "choices": [{"message": {"content": '{"v": 4}'}}],
    }
    with patch("src.stages.analyze_llm.subprocess.run") as sub_m, \
         patch("src.stages.analyze_llm.httpx.post", return_value=http_resp) as http_m:
        out = call_llm_json("sys", {})
    assert sub_m.call_count == 0
    assert http_m.call_count == 1
    assert out == {"v": 4}


def test_have_claude_cli_matches_which(monkeypatch) -> None:
    import shutil
    monkeypatch.setattr(shutil, "which",
                        lambda name: "/usr/local/bin/claude" if name == "claude" else None)
    assert _have_claude_cli() is True
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert _have_claude_cli() is False


# --- INTEGRATION (real server) ---------------------------------------

def _server_reachable() -> bool:
    try:
        r = httpx.get("http://127.0.0.1:8080/v1/models", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


@pytest.mark.slow
def test_analyze_fillers_against_real_llm() -> None:
    """End-to-end with the actual running LLM. Skipped if server is down."""
    if not _server_reachable():
        pytest.skip("LLM server not running at http://127.0.0.1:8080/v1")
    # A synthetic transcript with some obvious fillers
    words = [
        {"word": "So", "start": 0.0, "end": 0.3, "probability": 0.99},
        {"word": "um", "start": 0.4, "end": 0.7, "probability": 0.95},
        {"word": "today", "start": 0.8, "end": 1.2, "probability": 0.98},
        {"word": "we're", "start": 1.3, "end": 1.5, "probability": 0.95},
        {"word": "going", "start": 1.6, "end": 1.8, "probability": 0.95},
        {"word": "to", "start": 1.9, "end": 2.0, "probability": 0.95},
        {"word": "build", "start": 2.1, "end": 2.4, "probability": 0.95},
        {"word": "uh", "start": 2.5, "end": 2.8, "probability": 0.90},
        {"word": "a", "start": 2.9, "end": 3.0, "probability": 0.95},
        {"word": "video", "start": 3.1, "end": 3.5, "probability": 0.95},
        {"word": "editor", "start": 3.6, "end": 4.0, "probability": 0.95},
    ]
    out = analyze_fillers(words)
    assert isinstance(out, FillerCutsResponse)
    # Should identify at least one of the "um"/"uh" as a filler
    # (model-dependent — we only assert the response shape is valid)
    for c in out.cuts:
        assert c.end > c.start
        assert c.start >= 0.0


@pytest.mark.slow
def test_analyze_layout_against_real_llm() -> None:
    """Layout plan over a sample transcript; verify full coverage of duration."""
    if not _server_reachable():
        pytest.skip("LLM server not running")
    words = [
        {"word": "Hey", "start": 0.0, "end": 0.3, "probability": 0.99},
        {"word": "everyone", "start": 0.4, "end": 1.0, "probability": 0.99},
        {"word": "let", "start": 2.0, "end": 2.2, "probability": 0.99},
        {"word": "me", "start": 2.3, "end": 2.4, "probability": 0.99},
        {"word": "show", "start": 2.5, "end": 2.8, "probability": 0.99},
        {"word": "you", "start": 2.9, "end": 3.1, "probability": 0.99},
        {"word": "a", "start": 3.2, "end": 3.3, "probability": 0.99},
        {"word": "demo", "start": 3.4, "end": 3.8, "probability": 0.99},
        {"word": "thanks", "start": 27.0, "end": 27.4, "probability": 0.99},
        {"word": "bye", "start": 28.0, "end": 28.3, "probability": 0.99},
    ]
    out = analyze_layout(words, total_duration=30.0)
    assert isinstance(out, LayoutPlanResponse)
    # Full coverage invariant (after post-processing)
    total = sum(s.end - s.start for s in out.segments)
    assert abs(total - 30.0) < 0.1, total
    # Every layout is a valid enum
    for s in out.segments:
        assert s.layout in ("cam_full", "pip", "screen_full")
