"""Tests for the Claude Code adapter.

The CLI is replaced by a stub script that emits real stream-json on stdout and
edits files in its cwd — so these exercise the adapter's actual contract (spawn,
stream-parse, isolate, diff) without a network call or an API key.
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
from compass.adapters.claude_code import ClaudeCodeAdapter
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


def _stream_events() -> list[dict]:
    """A two-turn run: one Edit tool call, then a final answer."""
    return [
        {
            "type": "assistant",
            "session_id": "cc-test",
            "message": {
                "role": "assistant",
                "id": "msg_1",
                "model": "claude-opus-4-6",
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 1200, "output_tokens": 30},
                "content": [
                    {"type": "thinking", "thinking": "Add the limiter.", "signature": "s"},
                    {
                        "type": "tool_use",
                        "id": "tu1",
                        "name": "Edit",
                        "input": {"file_path": "app.py", "new_string": "..."},
                    },
                ],
            },
        },
        {
            "type": "user",
            "session_id": "cc-test",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu1",
                        "content": "ok",
                        "is_error": False,
                    }
                ],
            },
        },
        {
            "type": "assistant",
            "session_id": "cc-test",
            "message": {
                "role": "assistant",
                "id": "msg_2",
                "model": "claude-opus-4-6",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 15, "output_tokens": 8},
                "content": [{"type": "text", "text": "Added rate limiting."}],
            },
        },
        {
            "type": "result",
            "subtype": "success",
            "session_id": "cc-test",
            "is_error": False,
            "num_turns": 2,
            "duration_ms": 5300,
            "total_cost_usd": 0.0456,
            "result": "Added rate limiting.",
            "permission_denials": [],
        },
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
    """Write an executable stub that impersonates the ``claude`` CLI.

    It records its own argv (so tests can assert on flags), optionally writes
    files into its cwd (standing in for the agent's edits), and emits
    stream-json line by line.
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

    script = tmp_path / "fake_claude"
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
# Registration & config
# ---------------------------------------------------------------------------


def test_adapter_is_registered():
    assert "claude_code" in list_adapters()
    assert get_adapter("claude_code") is ClaudeCodeAdapter


def test_unknown_isolation_is_invalid_config():
    assert not ClaudeCodeAdapter({"isolation": "chroot"}).validate_config()
    assert ClaudeCodeAdapter({"isolation": "worktree"}).validate_config()


def test_argv_carries_stream_json_and_verbose():
    """Without --verbose the CLI emits only the final result — nothing to trace."""
    argv = ClaudeCodeAdapter({})._build_argv("do it")
    assert argv[:2] == ["claude", "-p"]
    assert "--output-format" in argv
    assert argv[argv.index("--output-format") + 1] == "stream-json"
    assert "--verbose" in argv


def test_argv_maps_config_to_flags(tmp_path: Path):
    prompt_file = tmp_path / "style.md"
    prompt_file.write_text("Be terse.", encoding="utf-8")

    argv = ClaudeCodeAdapter(
        {
            "model": "claude-opus-4-6",
            "append_system_prompt_file": str(prompt_file),
            "allowed_tools": ["Edit", "Bash(pytest:*)"],
            "disallowed_tools": ["WebFetch"],
            "permission_mode": "acceptEdits",
            "max_turns": 40,
            "extra_args": ["--no-color"],
        }
    )._build_argv("task")

    assert argv[argv.index("--model") + 1] == "claude-opus-4-6"
    assert argv[argv.index("--append-system-prompt") + 1] == "Be terse."
    assert argv[argv.index("--allowedTools") + 1] == "Edit,Bash(pytest:*)"
    assert argv[argv.index("--disallowedTools") + 1] == "WebFetch"
    assert argv[argv.index("--permission-mode") + 1] == "acceptEdits"
    assert argv[argv.index("--max-turns") + 1] == "40"
    assert argv[-1] == "--no-color"


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
    adapter = ClaudeCodeAdapter({"repo": str(repo), "cli_path": cli})
    agent_input, transcript = _agent_input()

    output = await adapter.run(agent_input)

    assert output.error is None
    artifact = output.artifacts[0]

    # The trace, not just the exit code: every step is on the transcript.
    # Order is the reconstructor's: a message's tool calls land before the
    # llm.generation record that summarises the message itself.
    names = [tc.tool_name for tc in transcript.tool_calls]
    assert names == ["Edit", "llm.generation", "llm.generation"]
    assert transcript.sum_cost().total_usd == pytest.approx(0.0456)
    assert transcript.sum_tokens().total_tokens == 1253
    assert any("Add the limiter" in s for s in transcript.reasoning_steps)

    # The diff, including the file that did not exist before.
    assert "diff --git a/app.py b/app.py" in artifact.diff
    assert "limits.py" in artifact.diff
    assert "MAX = 100" in artifact.diff
    assert sorted(f.path for f in artifact.files) == ["app.py", "limits.py"]

    assert artifact.execution.exit_code == 0
    assert artifact.execution.stdout == "Added rate limiting."


async def test_turn_index_and_call_id_survive_the_merge(repo: Path, tmp_path: Path):
    """turn_count and sub-agent attribution read fields a kwargs rebuild drops."""
    cli = _make_stub_cli(tmp_path, edits={"app.py": "changed\n"})
    agent_input, transcript = _agent_input()

    await ClaudeCodeAdapter({"repo": str(repo), "cli_path": cli}).run(agent_input)

    turns = [tc.turn_index for tc in transcript.tool_calls]
    assert turns == [1, 1, 2]

    edit = next(tc for tc in transcript.tool_calls if tc.tool_name == "Edit")
    assert edit.call_id == "tu1"        # pairs the call with its result
    assert edit.output == "ok"          # ... which is how the result got attached


async def test_worktree_isolation_leaves_the_source_repo_clean(
    repo: Path, tmp_path: Path
):
    cli = _make_stub_cli(tmp_path, edits={"app.py": "mutated\n"})
    adapter = ClaudeCodeAdapter({"repo": str(repo), "cli_path": cli})
    agent_input, _ = _agent_input()

    output = await adapter.run(agent_input)

    assert (repo / "app.py").read_text() == "def rate_limit():\n    pass\n"
    assert _git(repo, "status", "--porcelain") == ""

    workspace = Path(output.artifacts[0].metadata["workspace"])
    assert workspace != repo
    assert (workspace / "app.py").read_text() == "mutated\n"


async def test_workspace_is_removed_when_not_kept(repo: Path, tmp_path: Path):
    cli = _make_stub_cli(tmp_path, edits={"app.py": "mutated\n"})
    adapter = ClaudeCodeAdapter(
        {"repo": str(repo), "cli_path": cli, "keep_workspace": False}
    )
    agent_input, _ = _agent_input()

    output = await adapter.run(agent_input)

    workspace = Path(output.artifacts[0].metadata["workspace"])
    assert not workspace.exists()
    # The diff was captured before teardown, so the evidence outlives the tree.
    assert "mutated" in output.artifacts[0].diff


async def test_copy_isolation_works_without_git(tmp_path: Path):
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "app.py").write_text("old\n", encoding="utf-8")

    cli = _make_stub_cli(tmp_path, edits={"app.py": "new\n"})
    adapter = ClaudeCodeAdapter(
        {"repo": str(plain), "cli_path": cli, "isolation": "copy"}
    )
    agent_input, transcript = _agent_input()

    output = await adapter.run(agent_input)

    assert output.error is None
    assert (plain / "app.py").read_text() == "old\n"     # source untouched
    assert len(transcript.tool_calls) == 3               # trace still recorded
    assert output.artifacts[0].diff == ""                # no git, no diff


async def test_worktree_isolation_rejects_a_non_git_directory(tmp_path: Path):
    plain = tmp_path / "plain"
    plain.mkdir()
    cli = _make_stub_cli(tmp_path)
    agent_input, _ = _agent_input()

    output = await ClaudeCodeAdapter({"repo": str(plain), "cli_path": cli}).run(
        agent_input
    )

    assert output.error is not None
    assert "worktree" in output.error and "isolation='copy'" in output.error


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


async def test_missing_repo_is_an_error():
    agent_input, _ = _agent_input()
    output = await ClaudeCodeAdapter({"repo": "/nope/missing"}).run(agent_input)
    assert output.error is not None and "not found" in output.error


async def test_repo_is_required():
    agent_input, _ = _agent_input()
    output = await ClaudeCodeAdapter({}).run(agent_input)
    assert output.error == "Config 'repo' is required"


async def test_missing_cli_reports_the_binary_not_a_silent_pass(repo: Path):
    agent_input, _ = _agent_input()
    output = await ClaudeCodeAdapter(
        {"repo": str(repo), "cli_path": "definitely-not-claude"}
    ).run(agent_input)

    assert output.error is not None
    assert "definitely-not-claude" in output.error


async def test_no_events_is_an_error_not_an_empty_diff(repo: Path, tmp_path: Path):
    """A CLI that never ran must not score the same as an agent that made no edits."""
    cli = _make_stub_cli(tmp_path, events=[], exit_code=1, stderr="boom")
    agent_input, _ = _agent_input()

    output = await ClaudeCodeAdapter({"repo": str(repo), "cli_path": cli}).run(
        agent_input
    )

    assert output.error is not None
    assert "no stream-json events" in output.error
    assert "boom" in output.error


async def test_setup_command_failure_stops_before_the_agent(repo: Path, tmp_path: Path):
    cli = _make_stub_cli(tmp_path)
    adapter = ClaudeCodeAdapter(
        {"repo": str(repo), "cli_path": cli, "setup_commands": ["exit 3"]}
    )
    agent_input, transcript = _agent_input()

    output = await adapter.run(agent_input)

    assert output.error is not None and "exit 3" in output.error
    setup_calls = [
        tc for tc in transcript.tool_calls if tc.tool_name == "claude_code.setup"
    ]
    assert len(setup_calls) == 1 and setup_calls[0].status == "error"
    # The agent never ran, so there is no llm.generation call.
    assert not any(tc.tool_name == "llm.generation" for tc in transcript.tool_calls)


async def test_timeout_keeps_the_partial_trace(repo: Path, tmp_path: Path):
    cli = _make_stub_cli(tmp_path, sleep_s=5.0)
    adapter = ClaudeCodeAdapter(
        {"repo": str(repo), "cli_path": cli, "timeout": 0.5}
    )
    agent_input, transcript = _agent_input()

    output = await adapter.run(agent_input)

    assert output.metadata["timed_out"] is True
    assert output.artifacts[0].execution.exit_code == 124
    assert transcript.metadata.get("timed_out") is True


async def test_stream_json_can_be_saved_for_replay(repo: Path, tmp_path: Path):
    out_dir = tmp_path / "streams"
    cli = _make_stub_cli(tmp_path, edits={"app.py": "x\n"})
    adapter = ClaudeCodeAdapter(
        {"repo": str(repo), "cli_path": cli, "save_stream_to": str(out_dir)}
    )
    agent_input, _ = _agent_input()

    output = await adapter.run(agent_input)

    saved = Path(output.artifacts[0].metadata["stream_json"])
    assert saved.exists()

    # The saved file is a valid input to the offline importer — the run can be
    # re-graded later without paying for it again.
    from compass.integrations import import_claude_stream_json

    replayed = import_claude_stream_json(saved)
    assert [tc.tool_name for tc in replayed.tool_calls] == [
        "Edit",
        "llm.generation",
        "llm.generation",
    ]


# ---------------------------------------------------------------------------
# Prompt handling
# ---------------------------------------------------------------------------


async def test_prompt_reaches_the_cli(repo: Path, tmp_path: Path):
    cli = _make_stub_cli(tmp_path)
    agent_input, _ = _agent_input("Implement the spec in SPEC.md")

    await ClaudeCodeAdapter({"repo": str(repo), "cli_path": cli}).run(agent_input)

    argv = json.loads((tmp_path / "argv.json").read_text())
    assert argv[argv.index("-p") + 1] == "Implement the spec in SPEC.md"


async def test_task_file_is_appended_to_the_prompt(repo: Path, tmp_path: Path):
    task = tmp_path / "TASK.md"
    task.write_text("## Acceptance\n- returns 429\n", encoding="utf-8")

    cli = _make_stub_cli(tmp_path)
    transcript = Transcript(task_id="c", trial_id="t")
    agent_input = AgentInput(
        prompt="Do the task.",
        params={"task_file": str(task)},
        context={"transcript": transcript},
    )

    await ClaudeCodeAdapter({"repo": str(repo), "cli_path": cli}).run(agent_input)

    argv = json.loads((tmp_path / "argv.json").read_text())
    sent = argv[argv.index("-p") + 1]
    assert sent.startswith("Do the task.")
    assert "returns 429" in sent


def test_unreadable_system_prompt_file_is_reported():
    adapter = ClaudeCodeAdapter({"append_system_prompt_file": "/nope/missing.md"})
    with pytest.raises(Exception, match="append_system_prompt_file"):
        adapter._build_argv("task")
