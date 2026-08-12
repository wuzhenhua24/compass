"""pi adapter — drive the ``pi`` coding-agent CLI over a real git repository.

Same evaluation as :mod:`compass.adapters.claude_code`, different executable:
put the agent in a throwaway checkout, hand it a task, and come back with the
**diff it produced** and the **trace of how it got there**. What makes pi worth
its own adapter is the axis it opens up — pi is provider-agnostic, so *the model
is a flag*::

    compass test coding.yaml -m google/gemini-2.5-flash -m google/gemini-3.6-flash

That is one scenario, one prompt, one repo, two models, and a leaderboard that
answers both halves of the question: which one gets it *right* (OUTCOME graders
over the diff) and which one gets there *cheaply and predictably* (TRANSCRIPT
graders over cost, turns, tools, loops). ``compass compare`` will then tell you
whether the gap survived the noise.

The trace comes from ``--mode json``, pi's event stream, parsed by
:class:`~compass.integrations.pi_sessions.PiStreamReconstructor` — the same
mapping the offline session importer uses, so a run Compass drove and a session
a user recorded by hand grade identically.

YAML::

    agent:
      adapter: pi
      config:
        repo: "./fixtures/billing-service"
        base_ref: "main"
        model: "google/gemini-3.6-flash"   # the -m axis ("provider/id", or a pattern)
        thinking: "medium"                 # off|minimal|low|medium|high|xhigh|max
        exclude_tools: ["bash"]
        timeout: 900
        setup_commands: ["uv sync"]

**Sessions are off by default.** pi normally persists every run under
``~/.pi/agent/sessions/<cwd>/``; with a fresh worktree per trial that is one
junk directory per trial, keyed by a path that no longer exists. Compass already
records the run (and ``save_stream_to`` keeps the raw stream for replay), so the
adapter passes ``--no-session``. Set ``session_dir`` to opt back in when you want
``pi --resume`` to be able to open a run.
"""

from __future__ import annotations

import logging
from typing import Any

from compass.adapters.cli_agent import CliAgentAdapter, StreamReconstructor
from compass.adapters.registry import register_adapter
from compass.integrations.pi_sessions import PiStreamReconstructor

logger = logging.getLogger(__name__)


@register_adapter("pi")
class PiAdapter(CliAgentAdapter):
    """Run the pi CLI against a git repo and record the whole run.

    Config — the shared keys (``repo``, ``base_ref``, ``isolation``,
    ``keep_workspace``, ``cli_path``, ``model``, ``system_prompt*``,
    ``append_system_prompt*``, ``extra_args``, ``timeout``, ``setup_commands``,
    ``setup_timeout``, ``env``, ``save_stream_to``) are documented on
    :class:`~compass.adapters.cli_agent.CliAgentAdapter`. pi-specific:

        provider:      str  — ``--provider`` (default: pi's own, ``google``).
                              Unnecessary when ``model`` is ``"provider/id"``.
        thinking:      str  — ``--thinking`` (``off`` … ``max``). Its own axis:
                              sweep it with ``--model-key thinking``.
        tools:         list[str] — ``--tools``, an allowlist. Everything else off.
        exclude_tools: list[str] — ``--exclude-tools``, a denylist.
        no_tools:         bool — ``--no-tools`` (all tools off).
        no_builtin_tools: bool — ``--no-builtin-tools`` (extensions still on).
        session_dir:   str  — ``--session-dir``; unset means ``--no-session``.
        no_extensions:    bool — ``--no-extensions``.
        no_skills:        bool — ``--no-skills``.
        no_context_files: bool — ``--no-context-files`` (skip AGENTS.md/CLAUDE.md).
        approve:       bool | None — ``--approve`` / ``--no-approve``: whether to
                              trust project-local files (extensions, skills) in
                              the workspace. ``None`` (default) leaves the
                              decision to pi.

    The last four exist for one reason: an eval is only comparable if both runs
    got the same context. Extensions, skills and ``AGENTS.md`` are picked up from
    whatever machine the suite happens to run on, so turn them off when the
    numbers are meant to travel.
    """

    name = "pi"
    default_cli_path = "pi"
    cli_label = "pi"
    stream_label = "--mode json"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        c = self.config
        self.provider: str = c.get("provider", "")
        self.thinking: str = c.get("thinking", "")
        self.tools: list[str] = c.get("tools", [])
        self.exclude_tools: list[str] = c.get("exclude_tools", [])
        self.no_tools: bool = c.get("no_tools", False)
        self.no_builtin_tools: bool = c.get("no_builtin_tools", False)
        self.session_dir: str = c.get("session_dir", "")
        self.no_extensions: bool = c.get("no_extensions", False)
        self.no_skills: bool = c.get("no_skills", False)
        self.no_context_files: bool = c.get("no_context_files", False)
        self.approve: bool | None = c.get("approve")

    # ------------------------------------------------------------------
    # CLI invocation
    # ------------------------------------------------------------------

    def _build_argv(self, prompt: str) -> list[str]:
        # `-p` is a boolean here (pi's non-interactive mode), not the prompt's
        # flag: the prompt is a positional message, so it goes last, after every
        # option — otherwise the flags that follow it are read as more messages.
        argv = [self.cli_path, "-p", "--mode", "json"]

        if self.provider:
            argv += ["--provider", self.provider]
        if self.model:
            argv += ["--model", self.model]
        if self.thinking:
            argv += ["--thinking", self.thinking]

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

        if self.no_tools:
            argv += ["--no-tools"]
        if self.no_builtin_tools:
            argv += ["--no-builtin-tools"]
        if self.tools:
            argv += ["--tools", ",".join(self.tools)]
        if self.exclude_tools:
            argv += ["--exclude-tools", ",".join(self.exclude_tools)]

        if self.session_dir:
            argv += ["--session-dir", self.session_dir]
        else:
            argv += ["--no-session"]

        if self.no_extensions:
            argv += ["--no-extensions"]
        if self.no_skills:
            argv += ["--no-skills"]
        if self.no_context_files:
            argv += ["--no-context-files"]
        if self.approve is True:
            argv += ["--approve"]
        elif self.approve is False:
            argv += ["--no-approve"]

        argv += [str(a) for a in self.extra_args]
        argv.append(prompt)
        return argv

    def _new_reconstructor(self, task_id: str) -> StreamReconstructor:
        return PiStreamReconstructor(task_id=task_id)

    def _run_metadata(self) -> dict[str, Any]:
        """The pi-side axes, so a result file says which run it was."""
        metadata: dict[str, Any] = {}
        if self.provider:
            metadata["provider"] = self.provider
        if self.thinking:
            metadata["thinking"] = self.thinking
        return metadata
