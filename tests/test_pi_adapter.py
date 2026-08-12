"""Tests for the pi adapter.

The CLI is replaced by a stub script that emits a real ``--mode json`` stream on
stdout and edits files in its cwd — so these exercise the adapter's actual
contract (spawn, stream-parse, isolate, diff) without a network call or an API
key. The stream shape is copied from a real ``pi -p --mode json`` run.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import textwrap
from pathlib import Path

import pytest

from compass.adapters import get_adapter, list_adapters
from compass.adapters.base import AgentInput
from compass.adapters.pi import PiAdapter
from compass.core.transcript import Transcript

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A one-commit git repository to run the agent against."""
    repo = tmp_path / "svc"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "app.py").write_text("def rate_limit():\n    pass\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    return repo


def _usage(input_: int, output: int, total_cost: float) -> dict:
    return {
        "input": input_, "output": output, "cacheRead": 0, "cacheWrite": 0,
        "totalTokens": input_ + output,
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0,
                 "total": total_cost},
    }


def _stream_events() -> list[dict]:
    """A two-turn run: one `edit` tool call, then a final answer."""
    user = {"role": "user", "content": [{"type": "text", "text": "Add rate limiting"}],
            "timestamp": 1735689600000}
    editing = {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "Add the limiter."},
            {"type": "toolCall", "id": "c1", "name": "edit",
             "arguments": {"path": "app.py",
                           "edits": [{"oldText": "pass", "newText": "return True"}]}},
        ],
        "api": "google-generative-ai", "provider": "google",
        "model": "gemini-3.6-flash", "stopReason": "toolUse",
        "usage": _usage(1200, 30, 0.0004), "timestamp": 1735689601000,
    }
    result = {"role": "toolResult", "toolCallId": "c1", "toolName": "edit",
              "content": [{"type": "text", "text": "Successfully replaced 1 block(s)."}],
              "isError": False, "timestamp": 1735689603000}
    final = {
        "role": "assistant",
        "content": [{"type": "text", "text": "Added rate limiting."}],
        "api": "google-generative-ai", "provider": "google",
        "model": "gemini-3.6-flash", "stopReason": "stop",
        "usage": _usage(15, 8, 0.0002), "timestamp": 1735689604000,
    }
    return [
        {"type": "session", "version": 3, "id": "pi-test",
         "timestamp": "2025-01-01T00:00:00.000Z", "cwd": "/WORKSPACE"},
        {"type": "agent_start"},
        {"type": "turn_start"},
        {"type": "message_start", "message": user},
        {"type": "message_end", "message": user},
        {"type": "message_end", "message": editing},
        {"type": "tool_execution_start", "toolCallId": "c1", "toolName": "edit"},
        {"type": "tool_execution_end", "toolCallId": "c1", "toolName": "edit",
         "isError": False},
        {"type": "message_end", "message": result},
        {"type": "turn_end", "message": editing, "toolResults": [result]},
        {"type": "turn_start"},
        {"type": "message_end", "message": final},
        {"type": "turn_end", "message": final, "toolResults": []},
        {"type": "agent_end", "messages": [user, editing, result, final],
         "willRetry": False},
        {"type": "agent_settled"},
    ]


def _make_stub_cli(
    tmp_path: Path,
    *,
    events: list[dict] | None = None,
    edits: dict[str, str] | None = None,
    exit_code: int = 0,
    sleep_s: float = 0.0,
    stderr: str = "",
) -> str:
    """Write an executable stub that impersonates the ``pi`` CLI.

    It records its own argv (so tests can assert on flags), optionally writes
    files into its cwd (standing in for the agent's edits), and emits the
    ``--mode json`` stream line by line.
    """
    argv_log = tmp_path / "argv.json"
    payload = {
        "events": events if events is not None else _stream_events(),
        "edits": edits or {},
        "exit_code": exit_code,
        "sleep_s": sleep_s,
        "stderr": stderr,
        "argv_log": str(argv_log),
    }
    spec = tmp_path / "stub_spec.json"
    spec.write_text(json.dumps(payload), encoding="utf-8")

    script = tmp_path / "fake_pi"
    script.write_text(
        textwrap.dedent(
            f'''\
            #!{os.sys.executable}
            import json, pathlib, sys, time

            spec = json.loads(pathlib.Path({str(spec)!r}).read_text())
            pathlib.Path(spec["argv_log"]).write_text(json.dumps(sys.argv[1:]))

            for path, content in spec["edits"].items():
                p = pathlib.Path(path)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content)

            if spec["sleep_s"]:
                time.sleep(spec["sleep_s"])

            for event in spec["events"]:
                print(json.dumps(event), flush=True)

            if spec["stderr"]:
                sys.stderr.write(spec["stderr"])
            sys.exit(spec["exit_code"])
            '''
        ),
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(script)


def _agent_input(prompt: str = "Add rate limiting") -> tuple[AgentInput, Transcript]:
    transcript = Transcript(task_id="add_rate_limit", trial_id="t1")
    return (
        AgentInput(prompt=prompt, context={"transcript": transcript}),
        transcript,
    )


# ---------------------------------------------------------------------------
# Registration & argv
# ---------------------------------------------------------------------------


def test_adapter_is_registered():
    assert "pi" in list_adapters()
    assert get_adapter("pi") is PiAdapter


def test_argv_asks_for_the_json_stream_and_puts_the_prompt_last():
    """`-p` is pi's non-interactive flag, not the prompt's: the prompt is a
    positional message, so anything after it would be read as another message."""
    argv = PiAdapter({})._build_argv("do it")

    assert argv[:4] == ["pi", "-p", "--mode", "json"]
    assert argv[-1] == "do it"


def test_argv_maps_config_to_flags(tmp_path: Path):
    prompt_file = tmp_path / "style.md"
    prompt_file.write_text("Be terse.", encoding="utf-8")

    argv = PiAdapter(
        {
            "provider": "google",
            "model": "gemini-3.6-flash",
            "thinking": "medium",
            "append_system_prompt_file": str(prompt_file),
            "tools": ["read", "edit"],
            "exclude_tools": ["bash"],
            "no_context_files": True,
            "no_extensions": True,
            "approve": False,
            "extra_args": ["--offline"],
        }
    )._build_argv("task")

    assert argv[argv.index("--provider") + 1] == "google"
    assert argv[argv.index("--model") + 1] == "gemini-3.6-flash"
    assert argv[argv.index("--thinking") + 1] == "medium"
    assert argv[argv.index("--append-system-prompt") + 1] == "Be terse."
    assert argv[argv.index("--tools") + 1] == "read,edit"
    assert argv[argv.index("--exclude-tools") + 1] == "bash"
    assert "--no-context-files" in argv and "--no-extensions" in argv
    assert "--no-approve" in argv and "--approve" not in argv
    assert argv[-2:] == ["--offline", "task"]


def test_sessions_are_ephemeral_unless_a_directory_is_configured(tmp_path: Path):
    """One junk session dir per trial, keyed by a worktree path that no longer
    exists, is not a useful record — Compass already has the run."""
    assert "--no-session" in PiAdapter({})._build_argv("task")

    argv = PiAdapter({"session_dir": str(tmp_path)})._build_argv("task")
    assert "--no-session" not in argv
    assert argv[argv.index("--session-dir") + 1] == str(tmp_path)


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


async def test_run_records_steps_and_diff(repo: Path, tmp_path: Path):
    cli = _make_stub_cli(
        tmp_path,
        edits={
            "app.py": "def rate_limit():\n    return True\n",
            "limits.py": "MAX = 100\n",
        },
    )
    adapter = PiAdapter({"repo": str(repo), "cli_path": cli,
                         "model": "google/gemini-3.6-flash"})
    agent_input, transcript = _agent_input()

    output = await adapter.run(agent_input)

    assert output.error is None
    artifact = output.artifacts[0]

    # The trace, not just the exit code: every step is on the transcript.
    names = [tc.tool_name for tc in transcript.tool_calls]
    assert names == ["llm.generation", "edit", "llm.generation"]
    assert transcript.sum_cost().total_usd == pytest.approx(0.0006)
    assert transcript.sum_tokens().total_tokens == 1253
    assert any("Add the limiter" in s for s in transcript.reasoning_steps)
    assert [tc.turn_index for tc in transcript.tool_calls] == [1, 1, 2]

    edit = next(tc for tc in transcript.tool_calls if tc.tool_name == "edit")
    assert edit.call_id == "c1"                       # pairs the call to its result
    assert "Successfully replaced" in (edit.output or "")
    assert edit.duration_ms == pytest.approx(2000.0)

    # The diff, including the file that did not exist before.
    assert "diff --git a/app.py b/app.py" in artifact.diff
    assert "MAX = 100" in artifact.diff
    assert sorted(f.path for f in artifact.files) == ["app.py", "limits.py"]

    assert artifact.execution.exit_code == 0
    assert artifact.execution.stdout == "Added rate limiting."
    assert artifact.metadata["model"] == "google/gemini-3.6-flash"


async def test_the_model_axis_reaches_the_cli(repo: Path, tmp_path: Path):
    """`compass test s.yaml -m a -m b` writes agent.config['model'] — which is
    the whole point of the adapter: one scenario, two models, one leaderboard."""
    cli = _make_stub_cli(tmp_path)
    adapter = PiAdapter({"repo": str(repo), "cli_path": cli,
                         "model": "google/gemini-2.5-flash", "thinking": "high"})
    agent_input, _ = _agent_input()

    output = await adapter.run(agent_input)

    argv = json.loads((tmp_path / "argv.json").read_text())
    assert argv[argv.index("--model") + 1] == "google/gemini-2.5-flash"
    assert output.metadata["model"] == "google/gemini-2.5-flash"
    assert output.metadata["thinking"] == "high"


async def test_the_final_message_is_reachable_by_text_graders(
    repo: Path, tmp_path: Path
):
    from compass.core.artifacts import CodeArtifact, TextArtifact
    from compass.graders.base import GradeContext

    cli = _make_stub_cli(tmp_path, edits={"app.py": "changed\n"})
    adapter = PiAdapter({"repo": str(repo), "cli_path": cli})
    agent_input, transcript = _agent_input()

    output = await adapter.run(agent_input)

    assert isinstance(output.artifacts[0], CodeArtifact)
    texts = [a for a in output.artifacts if isinstance(a, TextArtifact)]
    assert [t.content for t in texts] == ["Added rate limiting."]

    transcript.set_outcome(artifacts=output.artifacts)
    context = GradeContext(transcript=transcript, outcome=transcript.outcome)
    assert context.answer == "Added rate limiting."


async def test_worktree_isolation_leaves_the_source_repo_clean(
    repo: Path, tmp_path: Path
):
    cli = _make_stub_cli(tmp_path, edits={"app.py": "mutated\n"})
    adapter = PiAdapter({"repo": str(repo), "cli_path": cli})
    agent_input, _ = _agent_input()

    output = await adapter.run(agent_input)

    assert (repo / "app.py").read_text() == "def rate_limit():\n    pass\n"
    assert _git(repo, "status", "--porcelain") == ""

    workspace = Path(output.artifacts[0].metadata["workspace"])
    assert workspace != repo
    assert (workspace / "app.py").read_text() == "mutated\n"


async def test_state_delta_reaches_the_grader(repo: Path, tmp_path: Path):
    """The whole chain: CLI stream -> reconstructor -> live transcript -> grader.

    Without it, an agent that gives up on the implementation and rewrites the
    tests instead scores exactly like one that did the work.
    """
    events = _stream_events()
    events[5]["message"]["content"][1]["arguments"]["path"] = "/WORKSPACE/tests/test_app.py"
    cli = _make_stub_cli(tmp_path, events=events, edits={"app.py": "changed\n"})
    agent_input, transcript = _agent_input()

    await PiAdapter({"repo": str(repo), "cli_path": cli}).run(agent_input)

    changes = transcript.all_state_changes()
    assert [(c.kind, c.op, c.target) for c in changes] == [
        ("file", "update", "tests/test_app.py")
    ]

    from compass.graders import GradeContext, get_grader

    grader = get_grader("state_delta")({"forbid": [{"kind": "file", "target": "tests/*"}]})
    result = await grader.grade(
        GradeContext(transcript=transcript, outcome=transcript.outcome)
    )
    assert result.passed is False


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


async def test_missing_cli_reports_the_binary_not_a_silent_pass(repo: Path):
    agent_input, _ = _agent_input()
    output = await PiAdapter({"repo": str(repo), "cli_path": "definitely-not-pi"}).run(
        agent_input
    )

    assert output.error is not None
    assert "definitely-not-pi" in output.error


async def test_no_events_is_an_error_not_an_empty_diff(repo: Path, tmp_path: Path):
    """A CLI that never ran must not score the same as an agent that made no edits."""
    cli = _make_stub_cli(tmp_path, events=[], exit_code=1, stderr="no API key")
    agent_input, _ = _agent_input()

    output = await PiAdapter({"repo": str(repo), "cli_path": cli}).run(agent_input)

    assert output.error is not None
    assert "--mode json" in output.error
    assert "no API key" in output.error


async def test_a_line_over_asyncios_default_limit_still_parses(
    repo: Path, tmp_path: Path
):
    """asyncio's stdout reader stops at 64 KiB per line, and agent events blow
    past that routinely — a big `read` result, or a reasoning model's thought
    signatures. This showed up as a whole case erroring with "Separator is
    found, but chunk is longer than limit", trace and all, on a live
    gemini-3.6-flash run whose only sin was being verbose."""
    events = _stream_events()
    events[5]["message"]["content"][0]["thinking"] = "x" * 200_000
    cli = _make_stub_cli(tmp_path, events=events, edits={"app.py": "changed\n"})
    agent_input, transcript = _agent_input()

    output = await PiAdapter({"repo": str(repo), "cli_path": cli}).run(agent_input)

    assert output.error is None
    assert [tc.tool_name for tc in transcript.tool_calls] == [
        "llm.generation", "edit", "llm.generation"
    ]
    assert "dropped_stdout_lines" not in transcript.metadata


async def test_setup_command_failure_stops_before_the_agent(repo: Path, tmp_path: Path):
    cli = _make_stub_cli(tmp_path)
    adapter = PiAdapter(
        {"repo": str(repo), "cli_path": cli, "setup_commands": ["exit 3"]}
    )
    agent_input, transcript = _agent_input()

    output = await adapter.run(agent_input)

    assert output.error is not None and "exit 3" in output.error
    setup_calls = [tc for tc in transcript.tool_calls if tc.tool_name == "pi.setup"]
    assert len(setup_calls) == 1 and setup_calls[0].status == "error"
    assert not any(tc.tool_name == "llm.generation" for tc in transcript.tool_calls)


async def test_timeout_keeps_the_partial_trace(repo: Path, tmp_path: Path):
    cli = _make_stub_cli(tmp_path, sleep_s=5.0)
    adapter = PiAdapter({"repo": str(repo), "cli_path": cli, "timeout": 0.5})
    agent_input, transcript = _agent_input()

    output = await adapter.run(agent_input)

    assert output.metadata["timed_out"] is True
    assert output.artifacts[0].execution.exit_code == 124
    assert transcript.metadata.get("timed_out") is True


async def test_stream_can_be_saved_for_replay(repo: Path, tmp_path: Path):
    out_dir = tmp_path / "streams"
    cli = _make_stub_cli(tmp_path, edits={"app.py": "x\n"})
    adapter = PiAdapter(
        {"repo": str(repo), "cli_path": cli, "save_stream_to": str(out_dir)}
    )
    agent_input, _ = _agent_input()

    output = await adapter.run(agent_input)

    saved = Path(output.artifacts[0].metadata["stream_json"])
    assert saved.exists()

    # The saved stream is a valid input to the offline importer — the run can be
    # re-graded later without paying for it again.
    from compass.integrations import import_pi_session

    replayed = import_pi_session(saved)
    assert [tc.tool_name for tc in replayed.tool_calls] == [
        "llm.generation", "edit", "llm.generation"
    ]
