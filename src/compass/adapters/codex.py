"""codex adapter — drive the OpenAI ``codex`` CLI over a real git repository.

Same evaluation as :mod:`compass.adapters.claude_code` and
:mod:`compass.adapters.pi`, a third executable: put the agent in a throwaway
checkout, hand it a task, and come back with the **diff it produced** (for
OUTCOME graders) and the **trace of how it got there** (for TRANSCRIPT graders).
With three CLI adapters the interesting comparison is no longer only "which
model", it is **which stack**::

    uv run python examples/coding_agent/run.py --suite crossstack.codex.yaml \\
        -m gpt-5.4-mini --trials 3 --out ./crossstack-run
    uv run python examples/coding_agent/run.py --suite crossstack.pi.yaml \\
        -m google/gemini-2.5-flash --trials 3 --out ./crossstack-run

    compass compare ./crossstack-run/results.gpt-5.4-mini.json \\
                    ./crossstack-run/results.google-gemini-2.5-flash.json --on correctness

The trace comes from ``codex exec --json``, codex's JSONL event stream, parsed by
:class:`~compass.integrations.codex_exec.CodexStreamReconstructor` — the same
mapping the offline stream importer uses, so a run Compass drove and a stream a
user captured by hand grade identically.

YAML::

    agent:
      adapter: codex
      config:
        repo: "./fixtures/billing-service"
        base_ref: "main"
        model: "gpt-5.4-mini"        # the -m axis
        reasoning_effort: "medium"   # low|medium|high|xhigh — its own axis
        sandbox: "workspace-write"
        timeout: 900
        setup_commands: ["uv sync"]

**Three things to know before reading codex numbers next to another stack's.**

- **No cost.** Codex reports tokens but never dollars — a ChatGPT-plan run is
  not billed per token at all. Compass computes cost from its pricing table, so
  a model with no rate registered produces ``$0.00`` and a ``cost_budget``
  grader that passes having measured nothing. Register a rate (see
  :func:`~compass.adapters.llm.register_pricing` or ``COMPASS_PRICING_FILE``)
  before that column means anything; the adapter puts
  ``cost_unpriced_model`` in the transcript metadata when it could not price a
  run, so the gap is visible rather than silent.
- **One LLM turn.** ``codex exec`` is a single turn by construction and codex
  aggregates usage over the turn, so ``turn_count`` with
  ``count_filter: "llm"`` always reads 1 here. Compare *tool-call* totals
  instead.
- **Different tool names.** Codex's tools are ``shell`` / ``apply_patch`` /
  ``update_plan`` / ``web_search``, not Claude's ``Bash`` / ``Edit`` or pi's
  ``bash`` / ``edit``. Name-keyed grader dimensions (``forbidden_tools``) are
  per-stack; only totals are cross-stack.

**Sessions are off by default.** Codex normally persists every run under
``~/.codex/sessions/``; with a fresh worktree per trial that is one junk rollout
per trial. Compass already records the run (and ``save_stream_to`` keeps the raw
stream for replay), so the adapter passes ``--ephemeral``. Set
``ephemeral: false`` to opt back in when you want ``codex resume`` to be able to
open a run.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from compass.adapters.base import AgentInput, AgentOutput
from compass.adapters.cli_agent import (
    CliAgentAdapter,
    CliAgentError,
    StreamReconstructor,
)
from compass.adapters.registry import register_adapter
from compass.integrations.codex_exec import CodexStreamReconstructor

if TYPE_CHECKING:  # the run record is internal to cli_agent
    from compass.adapters.cli_agent import _CliRun

logger = logging.getLogger(__name__)

# Sandbox policies `codex exec -s` accepts.
_SANDBOX_MODES = ("read-only", "workspace-write", "danger-full-access")


@register_adapter("codex")
class CodexAdapter(CliAgentAdapter):
    """Run the codex CLI against a git repo and record the whole run.

    Config — the shared keys (``repo``, ``base_ref``, ``isolation``,
    ``keep_workspace``, ``cli_path``, ``model``, ``system_prompt*``,
    ``extra_args``, ``timeout``, ``setup_commands``, ``setup_timeout``, ``env``,
    ``save_stream_to``) are documented on
    :class:`~compass.adapters.cli_agent.CliAgentAdapter`. codex-specific:

        sandbox:          str  — ``-s``: ``read-only`` | ``workspace-write``
                                 (default) | ``danger-full-access``. An eval
                                 that expects edits needs at least
                                 ``workspace-write``.
        reasoning_effort: str  — ``low`` | ``medium`` | ``high`` | ``xhigh``
                                 (``-c model_reasoning_effort``). Its own axis:
                                 sweep it with ``--model-key reasoning_effort``.
        reasoning_summary: str — ``none`` | ``auto`` | ``concise`` | ``detailed``
                                 (``-c model_reasoning_summary``). Controls
                                 whether ``reasoning`` items reach the trace at
                                 all — with ``none`` there are no reasoning
                                 steps to grade.
        verbosity:        str  — ``low`` | ``medium`` | ``high``
                                 (``-c model_verbosity``).
        web_search:       bool | None — ``-c tools.web_search``. ``None``
                                 (default) leaves codex's own setting alone.
        config_overrides: dict — extra ``-c key=value`` pairs, verbatim. Values
                                 are parsed by codex as TOML, falling back to a
                                 literal string.
        profile:          str  — ``-p``, a config profile from ``$CODEX_HOME``.
        ephemeral:        bool — ``--ephemeral``, do not persist a session
                                 (default ``True``).
        ignore_user_config: bool — ``--ignore-user-config``: skip
                                 ``$CODEX_HOME/config.toml`` (auth still works).
        ignore_rules:     bool — ``--ignore-rules``: skip user/project
                                 execpolicy ``.rules`` files.
        skip_git_repo_check: bool — ``--skip-git-repo-check``, for
                                 ``isolation: copy`` over a plain directory.
        add_dirs:    list[str] — ``--add-dir``, extra writable roots.
        bypass_approvals: bool — ``--dangerously-bypass-approvals-and-sandbox``.
                                 No sandbox *and* no prompts; only for an
                                 environment that is already isolated.
        approve_for_me:   bool — ``--approve-for-me``, route approvals through
                                 codex's automatic review instead of failing.

    ``ignore_user_config`` / ``ignore_rules`` exist for one reason: an eval is
    only comparable if both runs got the same context. A ``config.toml`` and a
    ``.rules`` file are picked up from whatever machine the suite happens to run
    on, so turn them off when the numbers are meant to travel. The repo's own
    ``AGENTS.md`` is deliberately *not* suppressed — that is part of the project
    under test, and every stack should read it.

    **No ``max_turns`` and no ``append_system_prompt``.** Codex has neither flag.
    A turn budget would be a key that quietly does nothing (use ``timeout`` and
    the ``turn_count`` / ``loop_detection`` diagnostics instead), and appending to
    the CLI's own instructions is not something codex exposes — ``system_prompt``
    *replaces* them via ``-c model_instructions_file``, which is the closest
    equivalent. Setting ``append_system_prompt`` fails loudly rather than being
    accepted and ignored.
    """

    name = "codex"
    default_cli_path = "codex"
    cli_label = "codex"
    stream_label = "exec --json"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        c = self.config
        self.sandbox: str = c.get("sandbox", "workspace-write")
        self.reasoning_effort: str = c.get("reasoning_effort", "")
        self.reasoning_summary: str = c.get("reasoning_summary", "")
        self.verbosity: str = c.get("verbosity", "")
        self.web_search: bool | None = c.get("web_search")
        self.config_overrides: dict[str, Any] = c.get("config_overrides", {})
        self.profile: str = c.get("profile", "")
        self.ephemeral: bool = c.get("ephemeral", True)
        self.ignore_user_config: bool = c.get("ignore_user_config", False)
        self.ignore_rules: bool = c.get("ignore_rules", False)
        self.skip_git_repo_check: bool = c.get("skip_git_repo_check", False)
        self.add_dirs: list[str] = c.get("add_dirs", [])
        self.bypass_approvals: bool = c.get("bypass_approvals", False)
        self.approve_for_me: bool = c.get("approve_for_me", False)
        # Instructions written to a temp file for -c model_instructions_file,
        # kept alive for the run (codex reads the path, not our handle).
        self._instructions_file: Path | None = None

    # ------------------------------------------------------------------
    # Adapter contract
    # ------------------------------------------------------------------

    def validate_config(self) -> bool:
        """The shared checks, plus a sandbox codex actually accepts."""
        return super().validate_config() and self.sandbox in _SANDBOX_MODES

    async def run(self, input: AgentInput) -> AgentOutput:
        """The shared run, plus cleanup of the instructions file it may write."""
        try:
            return await super().run(input)
        finally:
            self._discard_instructions_file()

    # ------------------------------------------------------------------
    # CLI invocation
    # ------------------------------------------------------------------

    def _build_argv(self, prompt: str) -> list[str]:
        # `exec` is codex's non-interactive subcommand; the prompt is a
        # positional argument, so it goes last, after every option.
        argv = [self.cli_path, "exec", "--json"]

        if self.model:
            argv += ["--model", self.model]
        if self.profile:
            argv += ["--profile", self.profile]

        if self.bypass_approvals:
            # This flag *replaces* --sandbox; passing both is contradictory.
            argv += ["--dangerously-bypass-approvals-and-sandbox"]
        else:
            argv += ["--sandbox", self.sandbox]
            if self.approve_for_me:
                argv += ["--approve-for-me"]

        for key, value in self._config_pairs().items():
            argv += ["-c", f"{key}={value}"]

        if self.ephemeral:
            argv += ["--ephemeral"]
        if self.ignore_user_config:
            argv += ["--ignore-user-config"]
        if self.ignore_rules:
            argv += ["--ignore-rules"]
        if self.skip_git_repo_check:
            argv += ["--skip-git-repo-check"]
        for directory in self.add_dirs:
            argv += ["--add-dir", str(directory)]

        argv += [str(a) for a in self.extra_args]
        argv.append(prompt)
        return argv

    def _config_pairs(self) -> dict[str, str]:
        """The ``-c key=value`` overrides this run's config asks for.

        ``config_overrides`` is applied last so an escape hatch can always win
        over a key the adapter also sets.
        """
        pairs: dict[str, str] = {}
        if self.reasoning_effort:
            pairs["model_reasoning_effort"] = self.reasoning_effort
        if self.reasoning_summary:
            pairs["model_reasoning_summary"] = self.reasoning_summary
        if self.verbosity:
            pairs["model_verbosity"] = self.verbosity
        if self.web_search is not None:
            pairs["tools.web_search"] = "true" if self.web_search else "false"

        instructions = self._instructions_path()
        if instructions:
            pairs["model_instructions_file"] = str(instructions)

        pairs.update({str(k): _toml_value(v) for k, v in self.config_overrides.items()})
        return pairs

    def _instructions_path(self) -> Path | None:
        """The file codex should read its base instructions from, if replaced.

        Codex takes replacement instructions as a *path*, not a string, so an
        inline ``system_prompt`` is written to a temp file. Appending is not
        something codex supports at all — fail rather than accept a key that
        would silently do nothing.
        """
        if self.append_system_prompt or self.append_system_prompt_file:
            raise CliAgentError(
                "codex has no --append-system-prompt: it can only *replace* its "
                "instructions. Use system_prompt / system_prompt_file (which map "
                "to -c model_instructions_file), or put project guidance in the "
                "repo's AGENTS.md so every stack reads it."
            )

        if self.system_prompt_file and not self.system_prompt:
            path = Path(self.system_prompt_file).expanduser()
            if not path.is_file():
                raise CliAgentError(
                    f"Could not read system_prompt_file '{self.system_prompt_file}'"
                )
            return path

        if not self.system_prompt:
            return None
        if self._instructions_file is None:
            handle = tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix="compass_codex_instructions_",
                suffix=".md",
                delete=False,
            )
            with handle:
                handle.write(self.system_prompt)
            self._instructions_file = Path(handle.name)
        return self._instructions_file

    def _discard_instructions_file(self) -> None:
        """Remove the temp instructions file, if this run wrote one.

        Deliberately *not* written into the workspace: a file the agent did not
        create would show up in ``git diff`` as work it did, inflating
        ``diff_size`` and the changed-file list.
        """
        if self._instructions_file is None:
            return
        self._instructions_file.unlink(missing_ok=True)
        self._instructions_file = None

    def _new_reconstructor(
        self, task_id: str, workspace: Path
    ) -> StreamReconstructor:
        # The workspace is not optional here: codex reports the paths it edited
        # absolute and never echoes its own cwd, so without it `state_delta`
        # targets would be a different random worktree path every trial.
        return CodexStreamReconstructor(
            task_id=task_id, model=self.model, cwd=str(workspace)
        )

    def _run_metadata(self) -> dict[str, Any]:
        """The codex-side axes, so a result file says which run it was."""
        metadata: dict[str, Any] = {"sandbox": self.sandbox}
        if self.bypass_approvals:
            metadata["sandbox"] = "bypassed"
        if self.reasoning_effort:
            metadata["reasoning_effort"] = self.reasoning_effort
        if self.reasoning_summary:
            metadata["reasoning_summary"] = self.reasoning_summary
        if self.verbosity:
            metadata["verbosity"] = self.verbosity
        return metadata

    def _run_error(self, run: _CliRun) -> str | None:
        """The shared no-events check, plus the failure codex reports in-band.

        ``turn.failed`` is how codex reports a run that never completed — a bad
        model id, a rate limit, a dropped stream. It arrives *as an event*, so
        the run looks healthy by the shared check: events were seen, the
        transcript has steps, and the empty diff that follows would grade as an
        agent that considered the task and declined to edit anything. It is not
        that, and it must not score like it.
        """
        shared = super()._run_error(run)
        if shared:
            return shared
        failures = run.transcript.metadata.get("turn_failures")
        if failures:
            return f"codex turn failed: {failures[-1]}"
        return None


def _toml_value(value: Any) -> str:
    """Render a YAML config value the way ``codex -c key=value`` expects it.

    Codex parses the right-hand side as TOML and falls back to a literal string
    when that fails — so ``True`` has to become ``true`` and a list has to become
    ``["a", "b"]``, or the override silently arrives as the string ``"True"``.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value))  # JSON arrays are valid TOML arrays
    return str(value)
