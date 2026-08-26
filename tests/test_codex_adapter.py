"""Tests for the codex adapter.

The CLI is replaced by a stub script that emits a real ``codex exec --json``
event stream on stdout and edits files in its cwd — so these exercise the
adapter's actual contract (spawn, stream-parse, isolate, diff) without a network
call or a ChatGPT login. The stream shape is copied from a real
``codex exec --json -m gpt-5.4-mini`` run.
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
from compass.adapters.cli_agent import CliAgentError
from compass.adapters.codex import CodexAdapter
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


def _usage(prompt: int, cached: int, output: int) -> dict:
    """codex's turn usage — ``input_tokens`` is the whole prompt, cache included."""
    return {
        "input_tokens": prompt,
        "cached_input_tokens": cached,
        "cache_write_input_tokens": 0,
        "output_tokens": output,
        "reasoning_output_tokens": 40,
    }


def _stream_events(edited: str = "app.py") -> list[dict]:
    """One turn: a shell probe, an apply_patch, a verifying shell, a final answer.

    ``{CWD}`` is substituted by the stub with its own working directory — codex
    reports the paths it changed absolute, which is exactly the thing the
    adapter has to normalize.
    """
    return [
        {"type": "thread.started", "thread_id": "codex-test"},
        {"type": "turn.started"},
        {"type": "item.completed",
         "item": {"id": "item_0", "type": "reasoning", "text": "**Planning the edit**"}},
        {"type": "item.completed",
         "item": {"id": "item_1", "type": "agent_message",
                  "text": "I'm reading app.py first."}},
        {"type": "item.started",
         "item": {"id": "item_2", "type": "command_execution",
                  "command": "/bin/zsh -lc 'sed -n 1,50p app.py'",
                  "aggregated_output": "", "exit_code": None,
                  "status": "in_progress"}},
        {"type": "item.completed",
         "item": {"id": "item_2", "type": "command_execution",
                  "command": "/bin/zsh -lc 'sed -n 1,50p app.py'",
                  "aggregated_output": "def rate_limit():\n    pass\n",
                  "exit_code": 0, "status": "completed"}},
        {"type": "item.started",
         "item": {"id": "item_3", "type": "file_change",
                  "changes": [{"path": "{CWD}/" + edited, "kind": "update"}],
                  "status": "in_progress"}},
        {"type": "item.completed",
         "item": {"id": "item_3", "type": "file_change",
                  "changes": [{"path": "{CWD}/" + edited, "kind": "update"}],
                  "status": "completed"}},
        {"type": "item.started",
         "item": {"id": "item_4", "type": "command_execution",
                  "command": "/bin/zsh -lc 'pytest -q'",
                  "aggregated_output": "", "exit_code": None,
                  "status": "in_progress"}},
        {"type": "item.completed",
         "item": {"id": "item_4", "type": "command_execution",
                  "command": "/bin/zsh -lc 'pytest -q'",
                  "aggregated_output": "1 failed", "exit_code": 1,
                  "status": "completed"}},
        {"type": "item.completed",
         "item": {"id": "item_5", "type": "agent_message",
                  "text": "Added rate limiting."}},
        {"type": "turn.completed", "usage": _usage(50_000, 39_000, 900)},
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
    """Write an executable stub that impersonates the ``codex`` CLI.

    It records its own argv (so tests can assert on flags), optionally writes
    files into its cwd (standing in for the agent's edits), and emits the
    ``exec --json`` stream line by line.
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

    script = tmp_path / "fake_codex"
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

            cwd = str(pathlib.Path.cwd())
            for event in spec["events"]:
                print(json.dumps(event).replace("{{CWD}}", cwd), flush=True)

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
    assert "codex" in list_adapters()
    assert get_adapter("codex") is CodexAdapter


def test_argv_asks_for_the_json_stream_and_puts_the_prompt_last():
    """The prompt is `codex exec`'s positional argument: anything after it would
    be read as a subcommand or another operand."""
    argv = CodexAdapter({})._build_argv("do it")

    assert argv[:3] == ["codex", "exec", "--json"]
    assert argv[-1] == "do it"


def test_argv_maps_config_to_flags_and_c_overrides():
    argv = CodexAdapter(
        {
            "model": "gpt-5.4-mini",
            "sandbox": "workspace-write",
            "reasoning_effort": "high",
            "reasoning_summary": "detailed",
            "verbosity": "low",
            "web_search": False,
            "ignore_user_config": True,
            "ignore_rules": True,
            "add_dirs": ["/data"],
            "extra_args": ["--color", "never"],
        }
    )._build_argv("task")

    assert argv[argv.index("--model") + 1] == "gpt-5.4-mini"
    assert argv[argv.index("--sandbox") + 1] == "workspace-write"
    overrides = {argv[i + 1] for i, a in enumerate(argv) if a == "-c"}
    assert overrides == {
        "model_reasoning_effort=high",
        "model_reasoning_summary=detailed",
        "model_verbosity=low",
        "tools.web_search=false",
    }
    assert "--ignore-user-config" in argv and "--ignore-rules" in argv
    assert argv[argv.index("--add-dir") + 1] == "/data"
    assert argv[-3:] == ["--color", "never", "task"]


def test_config_overrides_are_rendered_as_toml_not_python():
    """`-c key=value` is parsed as TOML, so `True` has to reach codex as `true`
    — otherwise the override silently arrives as the string "True"."""
    argv = CodexAdapter(
        {"config_overrides": {"tools.view_image": True, "shell_environment_policy.inherit": "all",
                              "sandbox_permissions": ["disk-full-read-access"]}}
    )._build_argv("task")

    overrides = {argv[i + 1] for i, a in enumerate(argv) if a == "-c"}
    assert overrides == {
        "tools.view_image=true",
        "shell_environment_policy.inherit=all",
        'sandbox_permissions=["disk-full-read-access"]',
    }


def test_sessions_are_ephemeral_unless_asked_for():
    """One junk rollout per trial, keyed by a worktree that no longer exists, is
    not a useful record — Compass already has the run."""
    assert "--ephemeral" in CodexAdapter({})._build_argv("task")
    assert "--ephemeral" not in CodexAdapter({"ephemeral": False})._build_argv("task")


def test_bypassing_approvals_replaces_the_sandbox_flag():
    """Passing both would be contradictory; codex's bypass flag *is* the policy."""
    argv = CodexAdapter({"bypass_approvals": True})._build_argv("task")

    assert "--dangerously-bypass-approvals-and-sandbox" in argv
    assert "--sandbox" not in argv


def test_an_unknown_sandbox_fails_config_validation():
    assert CodexAdapter({"sandbox": "workspace-write"}).validate_config() is True
    assert CodexAdapter({"sandbox": "yolo"}).validate_config() is False


def test_system_prompt_becomes_an_instructions_file():
    """codex takes replacement instructions as a *path*, so an inline prompt has
    to be written somewhere — and not into the workspace, where it would show up
    in the diff as work the agent did."""
    adapter = CodexAdapter({"system_prompt": "Be terse."})
    argv = adapter._build_argv("task")

    override = next(
        argv[i + 1] for i, a in enumerate(argv)
        if a == "-c" and argv[i + 1].startswith("model_instructions_file=")
    )
    path = Path(override.split("=", 1)[1])
    assert path.read_text(encoding="utf-8") == "Be terse."

    adapter._discard_instructions_file()
    assert not path.exists()


def test_append_system_prompt_fails_loudly_because_codex_has_no_such_flag():
    """A key that is accepted and ignored is worse than one that errors: the run
    costs the same and silently graded a different agent."""
    with pytest.raises(CliAgentError, match="append-system-prompt"):
        CodexAdapter({"append_system_prompt": "Be terse."})._build_argv("task")


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
    adapter = CodexAdapter(
        {"repo": str(repo), "cli_path": cli, "model": "gpt-5.4-mini"}
    )
    agent_input, transcript = _agent_input()

    output = await adapter.run(agent_input)

    assert output.error is None
    artifact = output.artifacts[0]

    # The trace, not just the exit code: every step is on the transcript, with
    # codex's own tool names.
    names = [tc.tool_name for tc in transcript.tool_calls]
    assert names == ["shell", "apply_patch", "shell", "llm.generation"]
    assert [tc.turn_index for tc in transcript.tool_calls] == [1, 1, 1, 1]
    assert any("Planning the edit" in s for s in transcript.reasoning_steps)

    # A shell command that exited non-zero is an error step, the way a failed
    # Bash is on a Claude trace.
    failing = transcript.tool_calls[2]
    assert failing.status == "error" and failing.metadata["exit_code"] == 1

    # The diff, including the file that did not exist before.
    assert "diff --git a/app.py b/app.py" in artifact.diff
    assert "MAX = 100" in artifact.diff
    assert sorted(f.path for f in artifact.files) == ["app.py", "limits.py"]

    assert artifact.execution.exit_code == 0
    assert artifact.execution.stdout == "Added rate limiting."
    assert artifact.metadata["model"] == "gpt-5.4-mini"
    assert artifact.metadata["sandbox"] == "workspace-write"


async def test_the_model_axis_reaches_the_cli(repo: Path, tmp_path: Path):
    """`compass test s.yaml -m a -m b` writes agent.config['model'] — which is
    the whole point of the adapter: one scenario, two models, one leaderboard."""
    cli = _make_stub_cli(tmp_path)
    adapter = CodexAdapter(
        {"repo": str(repo), "cli_path": cli, "model": "gpt-5.4-mini",
         "reasoning_effort": "high"}
    )
    agent_input, _ = _agent_input()

    output = await adapter.run(agent_input)

    argv = json.loads((tmp_path / "argv.json").read_text())
    assert argv[argv.index("--model") + 1] == "gpt-5.4-mini"
    assert "-c" in argv and "model_reasoning_effort=high" in argv
    assert output.metadata["model"] == "gpt-5.4-mini"
    assert output.metadata["reasoning_effort"] == "high"


async def test_tokens_keep_cached_traffic_separate(repo: Path, tmp_path: Path):
    """codex's `input_tokens` is the whole prompt bill; Compass reports the fresh
    part separately, because a cache read is priced at a fraction of it."""
    cli = _make_stub_cli(tmp_path, edits={"app.py": "changed\n"})
    agent_input, transcript = _agent_input()

    await CodexAdapter(
        {"repo": str(repo), "cli_path": cli, "model": "gpt-5.4-mini"}
    ).run(agent_input)

    tokens = transcript.sum_tokens()
    assert tokens.input_tokens == 11_000       # 50k prompt - 39k cache reads
    assert tokens.cache_read_tokens == 39_000
    assert tokens.output_tokens == 900
    assert tokens.total_tokens == 50_900       # everything the model processed


async def test_cost_is_zero_but_says_so_when_the_model_has_no_price(
    repo: Path, tmp_path: Path
):
    """codex never reports dollars. An unpriced model therefore yields $0.00 —
    which a cost_budget grader would pass having measured nothing, so the trace
    has to record that the number is missing rather than small."""
    from compass.llm import MODEL_PRICING, register_pricing

    cli = _make_stub_cli(tmp_path, edits={"app.py": "changed\n"})
    agent_input, transcript = _agent_input()

    await CodexAdapter(
        {"repo": str(repo), "cli_path": cli, "model": "codex-unpriced-test"}
    ).run(agent_input)

    assert transcript.sum_cost().total_usd == 0.0
    assert transcript.metadata["cost_unpriced_model"] == "codex-unpriced-test"

    # With a rate registered, the same tokens do produce a cost.
    register_pricing("codex-unpriced-test", {"input": 1.0, "output": 10.0})
    try:
        agent_input, priced = _agent_input()
        await CodexAdapter(
            {"repo": str(repo), "cli_path": cli, "model": "codex-unpriced-test"}
        ).run(agent_input)
        assert priced.sum_cost().total_usd == pytest.approx(
            11_000 / 1e6 * 1.0 + 900 / 1e6 * 10.0
        )
        assert "cost_unpriced_model" not in priced.metadata
    finally:
        MODEL_PRICING.pop("codex-unpriced-test", None)


async def test_state_delta_reaches_the_grader(repo: Path, tmp_path: Path):
    """The whole chain: CLI stream -> reconstructor -> live transcript -> grader.

    codex reports absolute paths, so this also pins the normalization: without it
    the target is a different random worktree path every trial and no glob a user
    writes could match.
    """
    cli = _make_stub_cli(
        tmp_path,
        events=_stream_events(edited="tests/test_app.py"),
        edits={"tests/test_app.py": "assert True\n"},
    )
    agent_input, transcript = _agent_input()

    await CodexAdapter({"repo": str(repo), "cli_path": cli}).run(agent_input)

    changes = transcript.all_state_changes()
    assert [(c.kind, c.op, c.target) for c in changes] == [
        ("file", "update", "tests/test_app.py")
    ]

    from compass.graders import GradeContext, get_grader

    grader = get_grader("state_delta")(
        {"forbid": [{"kind": "file", "target": "tests/*"}]}
    )
    result = await grader.grade(
        GradeContext(transcript=transcript, outcome=transcript.outcome)
    )
    assert result.passed is False


async def test_the_final_message_is_reachable_by_text_graders(
    repo: Path, tmp_path: Path
):
    from compass.core.artifacts import CodeArtifact, TextArtifact
    from compass.graders.base import GradeContext

    cli = _make_stub_cli(tmp_path, edits={"app.py": "changed\n"})
    agent_input, transcript = _agent_input()

    output = await CodexAdapter({"repo": str(repo), "cli_path": cli}).run(agent_input)

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
    agent_input, _ = _agent_input()

    output = await CodexAdapter({"repo": str(repo), "cli_path": cli}).run(agent_input)

    assert (repo / "app.py").read_text() == "def rate_limit():\n    pass\n"
    assert _git(repo, "status", "--porcelain") == ""

    workspace = Path(output.artifacts[0].metadata["workspace"])
    assert workspace != repo
    assert (workspace / "app.py").read_text() == "mutated\n"


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


async def test_missing_cli_reports_the_binary_not_a_silent_pass(repo: Path):
    agent_input, _ = _agent_input()
    output = await CodexAdapter(
        {"repo": str(repo), "cli_path": "definitely-not-codex"}
    ).run(agent_input)

    assert output.error is not None
    assert "definitely-not-codex" in output.error


async def test_no_events_is_an_error_not_an_empty_diff(repo: Path, tmp_path: Path):
    """A CLI that never ran must not score the same as an agent that made no edits."""
    cli = _make_stub_cli(tmp_path, events=[], exit_code=1, stderr="not logged in")
    agent_input, _ = _agent_input()

    output = await CodexAdapter({"repo": str(repo), "cli_path": cli}).run(agent_input)

    assert output.error is not None
    assert "exec --json" in output.error
    assert "not logged in" in output.error


async def test_a_failed_turn_is_an_error_even_though_events_arrived(
    repo: Path, tmp_path: Path
):
    """codex reports a dead run *in-band*: a bad model id or a rate limit comes
    back as `turn.failed` after a healthy-looking stream. The events check
    passes, the diff is empty, and without this the trial would score as an
    agent that considered the task and declined to edit anything."""
    cli = _make_stub_cli(
        tmp_path,
        events=[
            {"type": "thread.started", "thread_id": "codex-test"},
            {"type": "item.completed",
             "item": {"id": "item_0", "type": "error",
                      "message": "Model metadata for `nope` not found."}},
            {"type": "turn.started"},
            {"type": "turn.failed",
             "error": {"message": "The 'nope' model is not supported."}},
        ],
        exit_code=1,
    )
    agent_input, transcript = _agent_input()

    output = await CodexAdapter(
        {"repo": str(repo), "cli_path": cli, "model": "nope"}
    ).run(agent_input)

    assert output.error is not None
    assert "not supported" in output.error
    assert transcript.metadata["turn_failures"] == [
        "The 'nope' model is not supported."
    ]


async def test_a_timeout_keeps_the_steps_it_completed(repo: Path, tmp_path: Path):
    """A killed run is still evidence: the point of streaming the trace is that
    a partial one survives."""
    events = _stream_events()
    cli = _make_stub_cli(tmp_path, events=events, sleep_s=5.0)
    agent_input, transcript = _agent_input()

    output = await CodexAdapter(
        {"repo": str(repo), "cli_path": cli, "timeout": 0.5}
    ).run(agent_input)

    assert output.metadata["timed_out"] is True
    assert transcript.metadata["timed_out"] is True
    # Nothing was emitted before the sleep, so the failure is loud rather than
    # an empty diff that looks like a considered decision.
    assert output.error is not None


async def test_setup_command_failure_stops_before_the_agent(
    repo: Path, tmp_path: Path
):
    cli = _make_stub_cli(tmp_path)
    adapter = CodexAdapter(
        {"repo": str(repo), "cli_path": cli, "setup_commands": ["exit 3"]}
    )
    agent_input, transcript = _agent_input()

    output = await adapter.run(agent_input)

    assert output.error is not None and "exit 3" in output.error
    setup_calls = [tc for tc in transcript.tool_calls if tc.tool_name == "codex.setup"]
    assert len(setup_calls) == 1 and setup_calls[0].status == "error"
    assert not any(tc.tool_name == "llm.generation" for tc in transcript.tool_calls)
