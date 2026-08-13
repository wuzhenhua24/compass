"""Claude Code adapter — drive the ``claude`` CLI over a real git repository.

This is the adapter for the "does the agent implement the business requirement"
evaluation: point it at a repo, give it a task prompt, and it returns both the
**diff the agent produced** and the **full step-by-step trace of how it got
there** — so Compass's OUTCOME graders and its TRANSCRIPT graders both have
something to grade.

Why an adapter, when ``environment`` can already shell out to ``claude``:
``environment`` records the run as a *single* ``environment.agent`` tool call
(exit code, stdout length). Everything Compass exists to measure — the 30 Edit
/ Bash / Read steps, per-turn tokens, the CLI's cost total, retries, loops,
sub-agent fan-out — is inside that opaque stdout. This adapter parses the CLI's
``--output-format stream-json`` as it streams, through the same mapping
:mod:`compass.integrations.claude_agent` already uses for offline imports, and
merges the result into the live transcript. ``cost_budget``, ``turn_count``,
``loop_detection``, ``tool_usage``, ``efficiency`` and ``trajectory_judge`` all
start working as a result.

Everything that is not Claude-specific — worktree isolation, setup commands,
the streaming spawn, the transcript merge, diff capture — lives on
:class:`~compass.adapters.cli_agent.CliAgentAdapter`, which
:mod:`compass.adapters.pi` shares.

YAML::

    agent:
      adapter: claude_code
      config:
        repo: "./fixtures/billing-service"
        base_ref: "main"
        model: "claude-opus-4-6"                  # the -m axis
        append_system_prompt_file: "prompts/a.md" # the prompt axis
        permission_mode: "acceptEdits"
        allowed_tools: ["Edit", "Write", "Bash(pytest:*)"]
        max_turns: 40
        timeout: 900
        setup_commands: ["uv sync"]

Sweeping either axis is the ordinary multi-variant run — the axis is just a key
in this config::

    compass test coding.yaml -m claude-opus-4-6 -m claude-sonnet-5
    compass test coding.yaml --model-key append_system_prompt_file \\
        -m prompts/terse.md -m prompts/detailed.md

**The skill axis works the same way.** ``skill:`` installs a skill directory
into the workspace under ``.claude/skills/<its own name>``, so "v1 vs v2 vs no
skill at all" is one more sweep — and the run records the skill's content
digest, so which version produced a number is a fact rather than a claim::

    compass test skill.yaml --model-key skill \\
        -m ./skills/report-writer-v1 -m ./skills/report-writer-v2 -m ""
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from compass.adapters.cli_agent import (
    CliAgentAdapter,
    CliAgentError,
    StreamReconstructor,
)
from compass.adapters.registry import register_adapter
from compass.integrations.claude_agent import WireReconstructor

logger = logging.getLogger(__name__)

# Kept under its old name: the workspace failures it named are raised by the
# shared CLI-agent base now, and both adapters use the one type.
ClaudeCodeError = CliAgentError


@register_adapter("claude_code")
class ClaudeCodeAdapter(CliAgentAdapter):
    """Run the Claude Code CLI against a git repo and record the whole run.

    Config — the shared keys (``repo``, ``base_ref``, ``isolation``,
    ``keep_workspace``, ``cli_path``, ``model``, ``system_prompt*``,
    ``append_system_prompt*``, ``extra_args``, ``timeout``, ``setup_commands``,
    ``setup_timeout``, ``env``, ``save_stream_to``) are documented on
    :class:`~compass.adapters.cli_agent.CliAgentAdapter`. Claude-specific:

        allowed_tools:    list[str] — ``--allowedTools``.
        disallowed_tools: list[str] — ``--disallowedTools``.
        permission_mode:  str  — ``--permission-mode`` (e.g. ``acceptEdits``).
        max_turns:        int  — ``--max-turns``.
        setting_sources:  str  — ``--setting-sources`` (``user,project,local``).
                                Set it to ``project`` for a skill comparison:
                                see :attr:`skills_dir` below.
    """

    name = "claude_code"
    default_cli_path = "claude"
    cli_label = "claude"
    stream_label = "stream-json"
    #: Project-scoped skills live here, so this is where ``skill:`` installs.
    #:
    #: Installing is only half of a hermetic A/B. Claude Code also loads the
    #: operator's *own* skills from ``~/.claude/skills``, and a skill named
    #: there shadows — or silently supplements — the one under test, which
    #: makes the baseline arm quietly not a baseline. ``--setting-sources
    #: project`` cuts that off; when ``skill``/``skills`` is configured and
    #: ``setting_sources`` is not, Compass supplies ``project`` rather than
    #: let the comparison be wrong by default. Set it explicitly (on the shared
    #: ``agent.config``, so every arm of the sweep gets it) to override.
    skills_dir = ".claude/skills"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        c = self.config
        self.allowed_tools: list[str] = c.get("allowed_tools", [])
        self.disallowed_tools: list[str] = c.get("disallowed_tools", [])
        self.permission_mode: str = c.get("permission_mode", "")
        self.max_turns: int | None = c.get("max_turns")
        # None (unset) and "" (explicitly "leave the CLI's default alone") are
        # different answers; only the first one gets the hermetic default.
        self.setting_sources: str | None = c.get("setting_sources")

    # ------------------------------------------------------------------
    # CLI invocation
    # ------------------------------------------------------------------

    def _build_argv(self, prompt: str) -> list[str]:
        argv = [
            self.cli_path,
            "-p",
            prompt,
            "--output-format",
            "stream-json",
            # stream-json only emits the per-step events (rather than just the
            # final result) with --verbose; without it there is nothing to
            # reconstruct a trace from.
            "--verbose",
        ]
        if self.model:
            argv += ["--model", self.model]

        appended = self._read_prompt_option(
            self.append_system_prompt,
            self.append_system_prompt_file,
            "append_system_prompt_file",
        )
        if appended:
            argv += ["--append-system-prompt", appended]

        replaced = self._read_prompt_option(
            self.system_prompt, self.system_prompt_file, "system_prompt_file"
        )
        if replaced:
            argv += ["--system-prompt", replaced]

        if self.allowed_tools:
            argv += ["--allowedTools", ",".join(self.allowed_tools)]
        if self.disallowed_tools:
            argv += ["--disallowedTools", ",".join(self.disallowed_tools)]
        if self.permission_mode:
            argv += ["--permission-mode", self.permission_mode]
        if self.max_turns is not None:
            argv += ["--max-turns", str(self.max_turns)]
        sources = self._setting_sources()
        if sources:
            argv += ["--setting-sources", sources]
        argv += [str(a) for a in self.extra_args]
        return argv

    def _setting_sources(self) -> str:
        """The value for ``--setting-sources``; "" leaves the flag off."""
        if self.setting_sources is not None:
            return str(self.setting_sources)
        if self._skill_sources():
            logger.info(
                "%s: a skill is installed, so settings are scoped to the "
                "workspace (--setting-sources project). Set "
                "agent.config.setting_sources to override.",
                self.name,
            )
            return "project"
        return ""

    def _run_metadata(self) -> dict[str, Any]:
        sources = self._setting_sources()
        return {"setting_sources": sources} if sources else {}

    def _new_reconstructor(
        self, task_id: str, workspace: Path
    ) -> StreamReconstructor:
        # Claude Code reports its own cwd in the init event; workspace is unused.
        return WireReconstructor(task_id=task_id)
