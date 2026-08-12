"""Tests for the ``compass import`` CLI subcommand (pi/claude/codex/otlp files)."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from compass.cli.main import _detect_trace_format, cli
from compass.core.transcript import Transcript

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _pi_session() -> str:
    header = {"type": "session", "version": 3, "id": "sess-cli",
              "timestamp": "2026-01-01T00:00:00.000Z", "cwd": "/repo"}
    lines = [
        header,
        {"type": "message", "id": "e1", "parentId": None,
         "timestamp": "2026-01-01T00:00:01.000Z",
         "message": {"role": "user", "content": "Weather?", "timestamp": 1735689600000}},
        {"type": "message", "id": "e2", "parentId": "e1",
         "timestamp": "2026-01-01T00:00:02.000Z",
         "message": {"role": "assistant",
                     "content": [{"type": "toolCall", "id": "c1", "name": "get_weather",
                                  "arguments": {"city": "SF"}}],
                     "api": "a", "provider": "openai", "model": "gpt-4o",
                     "usage": {"input": 10, "output": 5, "cacheRead": 0, "cacheWrite": 0,
                               "totalTokens": 15,
                               "cost": {"input": 0, "output": 0, "cacheRead": 0,
                                        "cacheWrite": 0, "total": 0.001}},
                     "stopReason": "toolUse", "timestamp": 1735689602000}},
        {"type": "message", "id": "e3", "parentId": "e2",
         "timestamp": "2026-01-01T00:00:03.000Z",
         "message": {"role": "toolResult", "toolCallId": "c1", "toolName": "get_weather",
                     "content": [{"type": "text", "text": "sunny"}], "isError": False,
                     "timestamp": 1735689603000}},
        {"type": "message", "id": "e4", "parentId": "e3",
         "timestamp": "2026-01-01T00:00:04.000Z",
         "message": {"role": "assistant", "content": [{"type": "text", "text": "Sunny in SF."}],
                     "api": "a", "provider": "openai", "model": "gpt-4o",
                     "usage": {"input": 5, "output": 8, "totalTokens": 13,
                               "cost": {"total": 0.002}},
                     "stopReason": "stop", "timestamp": 1735689604000}},
    ]
    return "\n".join(json.dumps(x) for x in lines) + "\n"


def _claude_stream_json() -> str:
    events = [
        {"type": "system", "subtype": "init", "session_id": "cc-1", "cwd": "/repo"},
        {"type": "user", "session_id": "cc-1",
         "message": {"role": "user", "content": "use bash"}},
        {"type": "assistant", "session_id": "cc-1",
         "message": {"role": "assistant", "id": "m1", "model": "claude-opus-4-8",
                     "stop_reason": "tool_use",
                     "usage": {"input_tokens": 100, "output_tokens": 5},
                     "content": [{"type": "tool_use", "id": "tu1", "name": "Bash",
                                  "input": {"command": "echo 4"}}]}},
        {"type": "user", "session_id": "cc-1",
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": "tu1", "content": "4", "is_error": False}]}},
        {"type": "assistant", "session_id": "cc-1",
         "message": {"role": "assistant", "id": "m2", "model": "claude-opus-4-8",
                     "stop_reason": "end_turn",
                     "usage": {"input_tokens": 10, "output_tokens": 6},
                     "content": [{"type": "text", "text": "It is 4."}]}},
        {"type": "result", "subtype": "success", "session_id": "cc-1", "is_error": False,
         "num_turns": 2, "duration_ms": 500, "duration_api_ms": 400,
         "total_cost_usd": 0.02, "result": "It is 4."},
    ]
    return "\n".join(json.dumps(e) for e in events) + "\n"


def _codex_stream_json() -> str:
    events = [
        {"type": "thread.started", "thread_id": "cx-1"},
        {"type": "turn.started"},
        {"type": "item.started",
         "item": {"id": "item_0", "type": "command_execution",
                  "command": "/bin/zsh -lc 'echo 4'", "aggregated_output": "",
                  "exit_code": None, "status": "in_progress"}},
        {"type": "item.completed",
         "item": {"id": "item_0", "type": "command_execution",
                  "command": "/bin/zsh -lc 'echo 4'", "aggregated_output": "4\n",
                  "exit_code": 0, "status": "completed"}},
        {"type": "item.completed",
         "item": {"id": "item_1", "type": "agent_message", "text": "It is 4."}},
        {"type": "turn.completed",
         "usage": {"input_tokens": 110, "cached_input_tokens": 0,
                   "cache_write_input_tokens": 0, "output_tokens": 11,
                   "reasoning_output_tokens": 0}},
    ]
    return "\n".join(json.dumps(e) for e in events) + "\n"


def _otlp_two_traces() -> str:
    spans = []
    for tr in ("trA", "trB"):
        spans.append({
            "name": "chain", "context": {"trace_id": tr, "span_id": tr + "-c"},
            "parent_id": None, "start_time": "2026-01-01T00:00:00.000000Z",
            "end_time": "2026-01-01T00:00:01.000000Z", "status": {"status_code": "OK"},
            "attributes": {"openinference.span.kind": "CHAIN",
                           "input.value": "hi " + tr, "output.value": "done " + tr},
        })
        spans.append({
            "name": "tool", "context": {"trace_id": tr, "span_id": tr + "-t"},
            "parent_id": tr + "-c", "start_time": "2026-01-01T00:00:00.200000Z",
            "end_time": "2026-01-01T00:00:00.500000Z", "status": {"status_code": "OK"},
            "attributes": {"openinference.span.kind": "TOOL", "tool.name": "search",
                           "output.value": "ok"},
        })
    return json.dumps(spans)


@pytest.fixture
def pi_file(tmp_path):
    p = tmp_path / "session.jsonl"
    p.write_text(_pi_session(), encoding="utf-8")
    return p


@pytest.fixture
def otlp_file(tmp_path):
    p = tmp_path / "phoenix.json"
    p.write_text(_otlp_two_traces(), encoding="utf-8")
    return p


@pytest.fixture
def claude_file(tmp_path):
    p = tmp_path / "run.stream.jsonl"
    p.write_text(_claude_stream_json(), encoding="utf-8")
    return p


@pytest.fixture
def codex_file(tmp_path):
    p = tmp_path / "codex.stream.jsonl"
    p.write_text(_codex_stream_json(), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


class TestDetect:
    def test_detect_pi(self, pi_file):
        assert _detect_trace_format(pi_file) == "pi"

    def test_detect_otlp(self, otlp_file):
        assert _detect_trace_format(otlp_file) == "otlp"

    def test_detect_claude(self, claude_file):
        assert _detect_trace_format(claude_file) == "claude"

    def test_detect_codex(self, codex_file):
        """codex namespaces its event types (thread./turn./item.); nothing else does."""
        assert _detect_trace_format(codex_file) == "codex"

    def test_detect_directory_is_pi(self, tmp_path):
        assert _detect_trace_format(tmp_path) == "pi"

    def test_detect_non_json_is_none(self, tmp_path):
        p = tmp_path / "junk.txt"
        p.write_text("not json at all\n", encoding="utf-8")
        assert _detect_trace_format(p) is None


# ---------------------------------------------------------------------------
# import command
# ---------------------------------------------------------------------------


class TestImportCommand:
    def test_auto_import_pi_summarizes(self, pi_file):
        result = CliRunner().invoke(cli, ["import", str(pi_file)])
        assert result.exit_code == 0
        assert "format: pi" in result.output
        assert "sess-cli" in result.output
        assert "Sunny in SF." in result.output

    def test_auto_import_otlp_multi_trace(self, otlp_file):
        result = CliRunner().invoke(cli, ["import", str(otlp_file)])
        assert result.exit_code == 0
        assert "format: otlp" in result.output
        assert "2 transcript(s)" in result.output
        assert "trA" in result.output and "trB" in result.output

    def test_explicit_format(self, pi_file):
        result = CliRunner().invoke(cli, ["import", str(pi_file), "-f", "pi"])
        assert result.exit_code == 0
        assert "format: pi" in result.output

    def test_auto_import_claude_stream_json(self, claude_file):
        result = CliRunner().invoke(cli, ["import", str(claude_file)])
        assert result.exit_code == 0
        assert "format: claude" in result.output
        assert "cc-1" in result.output
        assert "It is 4." in result.output

    def test_auto_import_codex_stream_json(self, codex_file):
        result = CliRunner().invoke(cli, ["import", str(codex_file)])
        assert result.exit_code == 0
        assert "format: codex" in result.output
        assert "cx-1" in result.output
        assert "It is 4." in result.output

    def test_codex_save_and_reload_with_a_named_model(self, codex_file, tmp_path):
        """The stream records no model, so `--model` is the only way cost and the
        model column can be filled in at all."""
        out = tmp_path / "t.json"
        result = CliRunner().invoke(
            cli, ["import", str(codex_file), "--model", "gpt-4o", "-o", str(out)]
        )
        assert result.exit_code == 0
        loaded = Transcript.load(out)
        assert loaded.trial_id == "cx-1"
        assert any(tc.tool_name == "shell" for tc in loaded.tool_calls)
        assert loaded.environment.model_version == "gpt-4o"
        assert loaded.sum_cost().total_usd > 0

    def test_claude_save_and_reload(self, claude_file, tmp_path):
        out = tmp_path / "t.json"
        result = CliRunner().invoke(
            cli, ["import", str(claude_file), "-f", "claude", "-o", str(out)]
        )
        assert result.exit_code == 0
        loaded = Transcript.load(out)
        assert loaded.trial_id == "cc-1"
        assert any(tc.tool_name == "Bash" for tc in loaded.tool_calls)

    def test_save_single_file(self, pi_file, tmp_path):
        out = tmp_path / "t.json"
        result = CliRunner().invoke(cli, ["import", str(pi_file), "-o", str(out)])
        assert result.exit_code == 0
        assert out.exists()
        # Saved as a real Compass transcript that loads back.
        loaded = Transcript.load(out)
        assert loaded.trial_id == "sess-cli"
        assert any(tc.tool_name == "get_weather" for tc in loaded.tool_calls)

    def test_save_directory_multi(self, otlp_file, tmp_path):
        out = tmp_path / "out"
        result = CliRunner().invoke(cli, ["import", str(otlp_file), "-o", str(out)])
        assert result.exit_code == 0
        assert (out / "trA.json").exists()
        assert (out / "trB.json").exists()

    def test_save_jsonl_format(self, pi_file, tmp_path):
        out = tmp_path / "t.jsonl"
        result = CliRunner().invoke(
            cli, ["import", str(pi_file), "-o", str(out), "--output-format", "jsonl"]
        )
        assert result.exit_code == 0
        loaded = Transcript.load_jsonl(out)
        assert loaded.trial_id == "sess-cli"

    def test_json_output(self, pi_file):
        result = CliRunner().invoke(cli, ["import", str(pi_file), "--json"])
        assert result.exit_code == 0
        assert '"trial_id"' in result.output or '"task_id"' in result.output

    def test_import_directory_of_pi_sessions(self, tmp_path):
        (tmp_path / "a.jsonl").write_text(_pi_session(), encoding="utf-8")
        result = CliRunner().invoke(cli, ["import", str(tmp_path)])
        assert result.exit_code == 0
        assert "format: pi" in result.output

    def test_unknown_format_errors(self, tmp_path):
        p = tmp_path / "junk.txt"
        p.write_text("hello world\n", encoding="utf-8")
        result = CliRunner().invoke(cli, ["import", str(p)])
        assert result.exit_code == 1
        assert "auto-detect" in result.output
        assert "claude" in result.output

    def test_missing_source_errors(self):
        result = CliRunner().invoke(cli, ["import", "/no/such/file.jsonl"])
        assert result.exit_code != 0
