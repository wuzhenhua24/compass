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
    """

    name = "claude_code"
    default_cli_path = "claude"
    cli_label = "claude"
    stream_label = "stream-json"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        c = self.config
        self.allowed_tools: list[str] = c.get("allowed_tools", [])
        self.disallowed_tools: list[str] = c.get("disallowed_tools", [])
        self.permission_mode: str = c.get("permission_mode", "")
        self.max_turns: int | None = c.get("max_turns")

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
        argv += [str(a) for a in self.extra_args]
        return argv

    def _new_reconstructor(
        self, task_id: str, workspace: Path
    ) -> StreamReconstructor:
        # Claude Code reports its own cwd in the init event; workspace is unused.
        return WireReconstructor(task_id=task_id)
