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


async def test_the_final_message_is_reachable_by_text_graders(
    repo: Path, tmp_path: Path
):
    """The closing message comes back as a TextArtifact of its own.

    It was always on the CodeArtifact's ``execution.stdout``, but nothing looks
    for it there: ``GradeContext.answer`` reads ``output_data['final_output']``
    and then the first TextArtifact, so ``rubric``, ``style_convention`` and
    ``semantic_match`` all saw an empty string on adapter-driven runs and
    silently judged nothing. That is fatal for the class of case where the
    right answer is to change nothing and explain why — with no reachable text
    the only thing left to grade is a (correctly) empty diff.
    """
    from compass.core.artifacts import CodeArtifact, TextArtifact
    from compass.graders.base import GradeContext

    cli = _make_stub_cli(tmp_path, edits={"app.py": "changed\n"})
    adapter = ClaudeCodeAdapter({"repo": str(repo), "cli_path": cli})
    agent_input, transcript = _agent_input()

    output = await adapter.run(agent_input)

    # The code artifact stays first, so graders reading `code_artifact` are
    # unaffected by the addition.
    assert isinstance(output.artifacts[0], CodeArtifact)
    texts = [a for a in output.artifacts if isinstance(a, TextArtifact)]
    assert [t.content for t in texts] == ["Added rate limiting."]

    transcript.set_outcome(artifacts=output.artifacts)
    context = GradeContext(transcript=transcript, outcome=transcript.outcome)
    assert context.answer == "Added rate limiting."
    assert context.text_artifact is not None
    assert context.code_artifact is not None


async def test_a_run_with_no_closing_message_adds_no_empty_artifact(
    repo: Path, tmp_path: Path
):
    """An empty TextArtifact would say 'the agent stayed silent'. A missing one
    says 'there is no text here' — which is what a grader pointed at
    `text_artifact` should report, rather than scoring the empty string."""
    from compass.core.artifacts import TextArtifact

    events = _stream_events()
    events[-2]["message"]["content"] = []  # no closing text block
    events[-1]["result"] = ""
    cli = _make_stub_cli(tmp_path, events=events, edits={"app.py": "changed\n"})
    adapter = ClaudeCodeAdapter({"repo": str(repo), "cli_path": cli})
    agent_input, _ = _agent_input()

    output = await adapter.run(agent_input)

    assert output.error is None
    assert not [a for a in output.artifacts if isinstance(a, TextArtifact)]


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


# ---------------------------------------------------------------------------
# Skill installation (the skill axis)
# ---------------------------------------------------------------------------


def _skill(
    root: Path, dirname: str, *, name: str = "report-writer", body: str = "Write it."
) -> Path:
    """A skill directory whose frontmatter name differs from its own."""
    skill = root / dirname
    (skill / "scripts").mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Writes reports.\n---\n\n# {name}\n\n{body}\n",
        encoding="utf-8",
    )
    (skill / "scripts" / "render.py").write_text("print('rendered')\n", encoding="utf-8")
    return skill


async def test_skill_is_installed_under_its_own_name(repo: Path, tmp_path: Path):
    """v1 and v2 live in sibling directories but must reach the agent identically.

    Installing under the *source* directory name would hand the agent a skill
    called ``report-writer-v2``, changing its name and description-facing
    identity between the arms — which is a different experiment from the one
    the user asked for.
    """
    skill = _skill(tmp_path, "report-writer-v2")
    cli = _make_stub_cli(tmp_path, edits={"app.py": "x\n"})
    adapter = ClaudeCodeAdapter(
        {"repo": str(repo), "cli_path": cli, "skill": str(skill)}
    )
    agent_input, transcript = _agent_input()

    output = await adapter.run(agent_input)

    assert output.error is None
    workspace = Path(output.metadata["workspace"])
    installed = workspace / ".claude" / "skills" / "report-writer"
    assert (installed / "SKILL.md").exists()
    assert (installed / "scripts" / "render.py").exists()
    assert not (workspace / ".claude" / "skills" / "report-writer-v2").exists()

    record = output.metadata["skills"][0]
    assert record["name"] == "report-writer"
    assert record["source"] == str(skill)
    assert record["files"] == 2
    # The grader reads it from here when the scenario does not name the skill.
    assert transcript.metadata["skills"] == output.metadata["skills"]


async def test_two_versions_are_told_apart_by_their_digest(repo: Path, tmp_path: Path):
    """"v2 scored higher" is only a fact if *which* v2 is recorded."""
    v1 = _skill(tmp_path, "rw-v1", body="Write it.")
    v2 = _skill(tmp_path, "rw-v2", body="Write it, then check the numbers.")
    cli = _make_stub_cli(tmp_path)

    digests = []
    for skill in (v1, v2):
        agent_input, _ = _agent_input()
        output = await ClaudeCodeAdapter(
            {"repo": str(repo), "cli_path": cli, "skill": str(skill)}
        ).run(agent_input)
        digests.append(output.metadata["skills"][0]["digest"])

    assert digests[0] != digests[1]


async def test_a_missing_skill_md_is_an_error_not_a_baseline_run(
    repo: Path, tmp_path: Path
):
    """The one failure that would invalidate a comparison without leaving a mark."""
    empty = tmp_path / "not-a-skill"
    empty.mkdir()
    cli = _make_stub_cli(tmp_path)
    agent_input, _ = _agent_input()

    output = await ClaudeCodeAdapter(
        {"repo": str(repo), "cli_path": cli, "skill": str(empty)}
    ).run(agent_input)

    assert output.error is not None and "No SKILL.md" in output.error
    assert output.metadata["phase"] == "skills"


async def test_an_empty_skill_is_the_baseline_arm(repo: Path, tmp_path: Path):
    """``-m ""`` is how a sweep asks for "no skill at all", not a config error."""
    cli = _make_stub_cli(tmp_path, edits={"app.py": "x\n"})
    agent_input, transcript = _agent_input()

    output = await ClaudeCodeAdapter(
        {"repo": str(repo), "cli_path": cli, "skill": ""}
    ).run(agent_input)

    assert output.error is None
    assert "skills" not in output.metadata
    assert "skills" not in transcript.metadata
    workspace = Path(output.metadata["workspace"])
    assert not (workspace / ".claude" / "skills").exists()


async def test_the_installed_skill_is_not_counted_as_the_agents_work(
    repo: Path, tmp_path: Path
):
    """Otherwise every with-skill arm shows a few hundred extra changed lines
    and ``diff_size`` ends up scoring the harness."""
    skill = _skill(tmp_path, "report-writer")
    cli = _make_stub_cli(tmp_path, edits={"app.py": "changed\n"})
    agent_input, _ = _agent_input()

    output = await ClaudeCodeAdapter(
        {"repo": str(repo), "cli_path": cli, "skill": str(skill)}
    ).run(agent_input)

    artifact = output.artifacts[0]
    assert [f.path for f in artifact.files] == ["app.py"]
    assert ".claude" not in artifact.diff
    assert output.metadata["changed_files"] == ["app.py"]


async def test_settings_are_scoped_to_the_workspace_when_a_skill_is_installed(
    repo: Path, tmp_path: Path
):
    """The operator's own ``~/.claude/skills`` must not reach the run.

    A skill named there shadows — or silently supplements — the one under
    test, and the baseline arm quietly stops being a baseline. That failure is
    invisible in the results, so the default has to be the safe one.
    """
    skill = _skill(tmp_path, "report-writer")
    cli = _make_stub_cli(tmp_path)
    agent_input, _ = _agent_input()

    output = await ClaudeCodeAdapter(
        {"repo": str(repo), "cli_path": cli, "skill": str(skill)}
    ).run(agent_input)

    argv = json.loads((tmp_path / "argv.json").read_text())
    assert argv[argv.index("--setting-sources") + 1] == "project"
    assert output.metadata["setting_sources"] == "project"


def test_setting_sources_stays_off_without_a_skill():
    """Existing scenarios keep the CLI's own default; nothing changes for them."""
    assert "--setting-sources" not in ClaudeCodeAdapter({})._build_argv("task")


def test_an_explicit_setting_sources_wins(tmp_path: Path):
    skill = _skill(tmp_path, "report-writer")
    argv = ClaudeCodeAdapter(
        {"skill": str(skill), "setting_sources": "user,project"}
    )._build_argv("task")
    assert argv[argv.index("--setting-sources") + 1] == "user,project"

    # And "" is a real answer — "leave the flag off, I know what I am doing".
    assert "--setting-sources" not in ClaudeCodeAdapter(
        {"skill": str(skill), "setting_sources": ""}
    )._build_argv("task")


async def test_installing_into_the_source_repo_is_refused(repo: Path, tmp_path: Path):
    """isolation='none' would write the skill into the user's own checkout."""
    skill = _skill(tmp_path, "report-writer")
    cli = _make_stub_cli(tmp_path)
    agent_input, _ = _agent_input()

    output = await ClaudeCodeAdapter(
        {"repo": str(repo), "cli_path": cli, "skill": str(skill), "isolation": "none"}
    ).run(agent_input)

    assert output.error is not None and "isolation='none'" in output.error
    assert not (repo / ".claude").exists()


async def test_companion_skills_install_alongside_the_one_under_test(
    repo: Path, tmp_path: Path
):
    under_test = _skill(tmp_path, "rw", name="report-writer")
    companion = _skill(tmp_path, "ch", name="chart-helper")
    cli = _make_stub_cli(tmp_path)
    agent_input, _ = _agent_input()

    output = await ClaudeCodeAdapter(
        {
            "repo": str(repo),
            "cli_path": cli,
            "skill": str(under_test),
            "skills": [str(companion)],
        }
    ).run(agent_input)

    names = [s["name"] for s in output.metadata["skills"]]
    # The one under test leads, so `skill_trigger` inherits the right name.
    assert names == ["report-writer", "chart-helper"]


# ---------------------------------------------------------------------------
# State delta
# ---------------------------------------------------------------------------


def _editing_events(workspace_marker: str) -> list[dict]:
    """A run that edits one in-remit file and one out-of-remit test file."""
    return [
        {"type": "system", "subtype": "init", "session_id": "cc",
         "cwd": workspace_marker},
        {"type": "assistant", "session_id": "cc", "message": {
            "role": "assistant", "id": "m1", "model": "fake", "usage": {},
            "stop_reason": "tool_use", "content": [
                {"type": "tool_use", "id": "e1", "name": "Edit",
                 "input": {"file_path": f"{workspace_marker}/app.py"}},
                {"type": "tool_use", "id": "e2", "name": "Edit",
                 "input": {"file_path": f"{workspace_marker}/tests/test_app.py"}},
            ]}},
        {"type": "user", "session_id": "cc", "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "e1", "content": "ok",
             "is_error": False}]}},
        {"type": "user", "session_id": "cc", "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "e2", "content": "ok",
             "is_error": False}]}},
        {"type": "result", "subtype": "success", "session_id": "cc", "is_error": False,
         "num_turns": 1, "duration_ms": 100, "total_cost_usd": 0.01,
         "result": "Done.", "permission_denials": []},
    ]


async def test_state_delta_reaches_the_live_transcript(repo: Path, tmp_path: Path):
    """The whole chain: CLI stream -> reconstructor -> live transcript -> grader."""
    marker = "/WORKSPACE"
    cli = _make_stub_cli(
        tmp_path, events=_editing_events(marker), edits={"app.py": "changed\n"}
    )
    agent_input, transcript = _agent_input()

    await ClaudeCodeAdapter({"repo": str(repo), "cli_path": cli}).run(agent_input)

    changes = transcript.all_state_changes()
    assert [(c.kind, c.op, c.target) for c in changes] == [
        ("file", "update", "app.py"),
        ("file", "update", "tests/test_app.py"),
    ]

    from compass.graders import GradeContext, get_grader

    grader = get_grader("state_delta")({
        "forbid": [{"kind": "file", "target": "tests/*"}],
    })
    result = await grader.grade(
        GradeContext(transcript=transcript, outcome=transcript.outcome)
    )
    assert result.passed is False
    assert result.details["violations"][0]["call_id"] == "e2"


async def test_state_delta_survives_the_object_merge(repo: Path, tmp_path: Path):
    """state_delta is one of the fields a kwargs rebuild would have dropped."""
    cli = _make_stub_cli(
        tmp_path, events=_editing_events("/WORKSPACE"), edits={"app.py": "x\n"}
    )
    agent_input, transcript = _agent_input()

    await ClaudeCodeAdapter({"repo": str(repo), "cli_path": cli}).run(agent_input)

    edits = [tc for tc in transcript.tool_calls if tc.tool_name == "Edit"]
    assert all(len(tc.state_delta) == 1 for tc in edits)
