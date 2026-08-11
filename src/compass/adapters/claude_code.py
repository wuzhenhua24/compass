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

**No sandbox, on purpose.** :class:`~compass.sandbox.local.LocalSandbox` is a
scrubbed temp dir; the Claude Code CLI needs real network, real credentials, a
real git repo and the project's own toolchain, and fights the sandbox on all
four. Isolation here comes from **git worktree** instead: each trial gets its
own checkout of the same base ref, so parallel trials never collide and the
diff is computed against a known-clean baseline. The repo you point at is never
written to (with ``isolation: worktree`` or ``copy``).

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

import asyncio
import logging
import os
import shlex
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from compass.adapters.base import Adapter, AgentInput, AgentOutput
from compass.adapters.registry import register_adapter
from compass.core.artifacts import CodeArtifact, ExecutionResult, GeneratedFile
from compass.core.transcript import Transcript
from compass.integrations.claude_agent import WireReconstructor

logger = logging.getLogger(__name__)

# Extension -> language, for the changed-file listing on the CodeArtifact.
_EXT_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cs": "csharp",
    ".kt": "kotlin",
    ".swift": "swift",
    ".scala": "scala",
    ".sh": "bash",
    ".sql": "sql",
    ".html": "html",
    ".css": "css",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".md": "markdown",
}

# A changed file this big is a lockfile or a vendored blob, not something a
# grader will read. The diff still records it; we just don't inline the body.
_MAX_INLINE_FILE_BYTES = 256 * 1024


class ClaudeCodeError(RuntimeError):
    """Raised when the workspace cannot be prepared for a run."""


@register_adapter("claude_code")
class ClaudeCodeAdapter(Adapter):
    """Run the Claude Code CLI against a git repo and record the whole run.

    Config:
        repo:            str  — path to the git repository (required; may also
                                be given per-case via ``params["repo"]``).
        base_ref:        str  — ref to branch the worktree from (default: the
                                repo's current HEAD).
        isolation:       str  — ``"worktree"`` (default) | ``"copy"`` | ``"none"``.
                                ``none`` runs in the repo itself and *will*
                                modify it; only use it for one-off manual runs.
        keep_workspace:  bool — keep the workspace after the run (default
                                ``True``, because graders run after the adapter
                                and need the files; see below).
        cli_path:        str  — the executable (default ``"claude"``).
        model:           str  — ``--model``.
        append_system_prompt:      str — ``--append-system-prompt`` (inline).
        append_system_prompt_file: str — same, read from a file.
        system_prompt / system_prompt_file: str — ``--system-prompt``, which
                                *replaces* the CLI's own prompt rather than
                                appending to it.
        allowed_tools:    list[str] — ``--allowedTools``.
        disallowed_tools: list[str] — ``--disallowedTools``.
        permission_mode:  str  — ``--permission-mode`` (e.g. ``acceptEdits``).
        max_turns:        int  — ``--max-turns``.
        extra_args:       list[str] — appended verbatim to the command line.
        timeout:          float — seconds before the CLI is killed
                                (default ``900``). A timed-out run still
                                returns the steps completed so far.
        setup_commands:  list[str] — shell commands run in the workspace before
                                the agent (install deps, seed a DB).
        setup_timeout:    float — per setup command (default ``600``).
        env:             dict — extra environment variables for the CLI.
        save_stream_to:  str  — directory to write the raw stream-json to, for
                                replaying the run without paying for it again.

    **keep_workspace defaults to True.** Graders run *after* the adapter
    returns, and ``integration_test`` scores the agent by running hidden tests
    against the tree it produced — deleting that tree first would destroy the
    evidence before anything reads it. The path is on
    ``CodeArtifact.metadata["workspace"]``, and ``integration_test`` resolves
    ``workdir: "{workspace}"`` to it. Reclaim the disk afterwards with
    ``git worktree prune`` in the source repo.
    """

    name = "claude_code"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        c = self.config
        self.repo: str = c.get("repo", "")
        self.base_ref: str = c.get("base_ref", "")
        self.isolation: str = c.get("isolation", "worktree")
        self.keep_workspace: bool = c.get("keep_workspace", True)
        self.cli_path: str = c.get("cli_path", "claude")
        self.model: str = c.get("model", "")
        self.append_system_prompt: str = c.get("append_system_prompt", "")
        self.append_system_prompt_file: str = c.get("append_system_prompt_file", "")
        self.system_prompt: str = c.get("system_prompt", "")
        self.system_prompt_file: str = c.get("system_prompt_file", "")
        self.allowed_tools: list[str] = c.get("allowed_tools", [])
        self.disallowed_tools: list[str] = c.get("disallowed_tools", [])
        self.permission_mode: str = c.get("permission_mode", "")
        self.max_turns: int | None = c.get("max_turns")
        self.extra_args: list[str] = c.get("extra_args", [])
        self.timeout: float = c.get("timeout", 900)
        self.setup_commands: list[str] = c.get("setup_commands", [])
        self.setup_timeout: float = c.get("setup_timeout", 600)
        self.env: dict[str, str] = c.get("env", {})
        self.save_stream_to: str = c.get("save_stream_to", "")

    def validate_config(self) -> bool:
        """A repo is required (unless supplied per-case) and isolation must be known."""
        return self.isolation in ("worktree", "copy", "none")

    async def health_check(self) -> bool:
        """Whether the CLI is on PATH and answers ``--version``."""
        try:
            proc = await asyncio.create_subprocess_exec(
                self.cli_path,
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=30)
            return proc.returncode == 0
        except (OSError, asyncio.TimeoutError):
            return False

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self, input: AgentInput) -> AgentOutput:
        repo = str(input.params.get("repo") or self.repo)
        if not repo:
            return AgentOutput(error="Config 'repo' is required")
        repo_path = Path(repo).expanduser().resolve()
        if not repo_path.is_dir():
            return AgentOutput(error=f"Repository not found: {repo_path}")

        base_ref = str(input.params.get("base_ref") or self.base_ref)

        try:
            workspace, cleanup = await self._prepare_workspace(repo_path, base_ref)
        except ClaudeCodeError as e:
            return AgentOutput(error=str(e))

        try:
            return await self._run_in_workspace(input, repo_path, workspace)
        finally:
            if not self.keep_workspace:
                await cleanup()

    async def _run_in_workspace(
        self,
        input: AgentInput,
        repo_path: Path,
        workspace: Path,
    ) -> AgentOutput:
        # --- Setup commands (deps, fixtures) ---
        for cmd in self.setup_commands:
            started = time.monotonic()
            result = await self._exec(cmd, workspace, self.setup_timeout)
            elapsed = (time.monotonic() - started) * 1000
            self._record_tool_call(
                input,
                tool_name="claude_code.setup",
                tool_type="environment",
                input={"command": cmd},
                output={
                    "exit_code": result.exit_code,
                    "stdout_len": len(result.stdout or ""),
                    "stderr_len": len(result.stderr or ""),
                },
                status="ok" if result.exit_code == 0 else "error",
                duration_ms=elapsed,
            )
            if result.exit_code != 0:
                return AgentOutput(
                    error=f"Setup command failed: {cmd} (exit {result.exit_code})",
                    metadata={
                        "phase": "setup",
                        "workspace": str(workspace),
                        "stderr": (result.stderr or "")[-2000:],
                    },
                )

        # --- Run the agent ---
        prompt = self._build_prompt(input)
        argv = self._build_argv(prompt)
        logger.info("claude_code: running in %s", workspace)

        started = time.monotonic()
        run = await self._run_cli(argv, workspace, input)
        duration_ms = (time.monotonic() - started) * 1000

        # --- Capture what changed ---
        diff, changed = await self._capture_changes(workspace)

        # Merge the reconstructed run into the transcript the runner is
        # recording, so transcript-scope graders see the real steps.
        self._merge_transcript(input, run.transcript, workspace)

        if run.timed_out:
            logger.warning(
                "claude_code: timed out after %.0fs — grading the partial run",
                self.timeout,
            )

        metadata: dict[str, Any] = {
            "workspace": str(workspace),
            "repo": str(repo_path),
            "exit_code": run.exit_code,
            "timed_out": run.timed_out,
            "duration_ms": duration_ms,
            "changed_files": [f.path for f in changed],
            "command": " ".join(shlex.quote(a) for a in argv),
        }
        if self.model:
            metadata["model"] = self.model
        if run.stream_path:
            metadata["stream_json"] = run.stream_path

        artifact = CodeArtifact(
            files=changed,
            diff=diff,
            execution=ExecutionResult(
                exit_code=run.exit_code,
                stdout=run.final_text,
                stderr=run.stderr[-8000:],
                duration_ms=duration_ms,
            ),
            metadata=metadata,
        )

        # A CLI that never produced a parseable event did not run — an empty
        # diff from *that* is a harness failure, not an agent that declined to
        # edit anything, and the two must not score the same.
        if not run.events_seen:
            return AgentOutput(
                artifacts=[artifact],
                metadata=metadata,
                error=(
                    f"claude CLI produced no stream-json events "
                    f"(exit {run.exit_code}): {(run.stderr or '')[-500:] or 'no stderr'}"
                ),
            )

        return AgentOutput(artifacts=[artifact], metadata=metadata)

    # ------------------------------------------------------------------
    # Workspace preparation
    # ------------------------------------------------------------------

    async def _prepare_workspace(
        self, repo_path: Path, base_ref: str
    ) -> tuple[Path, Any]:
        """Return (workspace, async cleanup callable) for one trial."""

        async def _noop() -> None:
            return None

        if self.isolation == "none":
            return repo_path, _noop

        if self.isolation == "copy":
            dest = Path(tempfile.mkdtemp(prefix="compass_claude_"))
            target = dest / repo_path.name
            await asyncio.to_thread(
                shutil.copytree, repo_path, target, symlinks=True
            )

            async def _rm() -> None:
                await asyncio.to_thread(shutil.rmtree, dest, True)

            return target, _rm

        # worktree (default)
        if not (repo_path / ".git").exists():
            raise ClaudeCodeError(
                f"isolation='worktree' needs a git repository, but {repo_path} "
                "has no .git. Use isolation='copy' for a plain directory."
            )
        ref = base_ref or await self._git_head(repo_path)
        dest = Path(tempfile.mkdtemp(prefix="compass_claude_")) / (
            f"wt_{uuid.uuid4().hex[:8]}"
        )
        result = await self._exec(
            f"git worktree add --detach {shlex.quote(str(dest))} {shlex.quote(ref)}",
            repo_path,
            self.setup_timeout,
        )
        if result.exit_code != 0:
            raise ClaudeCodeError(
                f"git worktree add failed (exit {result.exit_code}): "
                f"{(result.stderr or '').strip()[-500:]}"
            )

        async def _remove() -> None:
            await self._exec(
                f"git worktree remove --force {shlex.quote(str(dest))}",
                repo_path,
                self.setup_timeout,
            )
            await asyncio.to_thread(shutil.rmtree, dest.parent, True)

        return dest, _remove

    async def _git_head(self, repo_path: Path) -> str:
        """The repo's current commit, so every trial branches from one baseline."""
        result = await self._exec("git rev-parse HEAD", repo_path, 60)
        if result.exit_code != 0:
            raise ClaudeCodeError(
                f"git rev-parse HEAD failed in {repo_path}: "
                f"{(result.stderr or '').strip()[-500:]}"
            )
        return (result.stdout or "").strip()

    # ------------------------------------------------------------------
    # CLI invocation
    # ------------------------------------------------------------------

    def _build_prompt(self, input: AgentInput) -> str:
        prompt = input.prompt
        extra = input.params.get("task_file")
        if extra:
            try:
                text = Path(str(extra)).expanduser().read_text(encoding="utf-8")
            except OSError as e:
                logger.warning("claude_code: could not read task_file %s: %s", extra, e)
            else:
                prompt = f"{prompt}\n\n{text}" if prompt else text
        return prompt

    def _read_prompt_option(self, inline: str, path: str, label: str) -> str:
        """Inline value, or the contents of the file it points at."""
        if inline:
            return inline
        if not path:
            return ""
        try:
            return Path(path).expanduser().read_text(encoding="utf-8")
        except OSError as e:
            raise ClaudeCodeError(f"Could not read {label} '{path}': {e}") from e

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

    def _cli_env(self) -> dict[str, str]:
        """The CLI's environment: the host's, plus configured extras.

        Deliberately *not* the scrubbed sandbox environment — the CLI
        authenticates with ``ANTHROPIC_API_KEY`` or with the credentials under
        ``$HOME``, and needs the project's toolchain on PATH to run the repo's
        tests.
        """
        env = dict(os.environ)
        env.update({str(k): str(v) for k, v in self.env.items()})
        return env

    async def _run_cli(
        self, argv: list[str], workspace: Path, input: AgentInput
    ) -> _CliRun:
        """Run the CLI, reconstructing the transcript from stdout as it streams."""
        task_id = self._task_id(input)
        reconstructor = WireReconstructor(task_id=task_id)
        raw_lines: list[str] = []
        events_seen = 0

        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(workspace),
                env=self._cli_env(),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return _CliRun(
                transcript=reconstructor.finish(),
                exit_code=127,
                stderr=(
                    f"claude CLI not found: '{self.cli_path}'. Install it, or set "
                    "agent.config.cli_path to its location."
                ),
                events_seen=0,
            )

        assert proc.stdout is not None and proc.stderr is not None
        stderr_chunks: list[str] = []

        async def drain_stderr() -> None:
            async for chunk in proc.stderr:  # type: ignore[union-attr]
                stderr_chunks.append(chunk.decode("utf-8", errors="replace"))

        stderr_task = asyncio.create_task(drain_stderr())

        async def read_stdout() -> int:
            nonlocal events_seen
            async for raw in proc.stdout:  # type: ignore[union-attr]
                line = raw.decode("utf-8", errors="replace")
                raw_lines.append(line)
                if reconstructor.feed_line(line):
                    events_seen += 1
            return await proc.wait()

        timed_out = False
        try:
            exit_code = await asyncio.wait_for(read_stdout(), timeout=self.timeout)
        except asyncio.TimeoutError:
            timed_out = True
            exit_code = 124  # conventional timeout code
            proc.kill()
            await proc.wait()
        finally:
            stderr_task.cancel()
            try:
                await stderr_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

        transcript = reconstructor.finish()
        if timed_out:
            transcript.metadata["timed_out"] = True
            transcript.metadata["timeout_s"] = self.timeout

        stream_path = self._save_stream(raw_lines, task_id)

        return _CliRun(
            transcript=transcript,
            exit_code=exit_code,
            stderr="".join(stderr_chunks),
            events_seen=events_seen,
            timed_out=timed_out,
            stream_path=stream_path,
        )

    def _save_stream(self, raw_lines: list[str], task_id: str) -> str:
        """Persist the raw stream-json, so the run can be replayed for free."""
        if not self.save_stream_to or not raw_lines:
            return ""
        out_dir = Path(self.save_stream_to).expanduser()
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in task_id)
            path = out_dir / f"{safe}_{uuid.uuid4().hex[:8]}.stream.jsonl"
            path.write_text("".join(raw_lines), encoding="utf-8")
        except OSError as e:
            logger.warning("claude_code: could not save stream-json: %s", e)
            return ""
        return str(path)

    @staticmethod
    def _task_id(input: AgentInput) -> str:
        transcript = input.context.get("transcript")
        task_id = getattr(transcript, "task_id", None)
        return str(task_id) if task_id else "claude_code"

    # ------------------------------------------------------------------
    # Transcript merge
    # ------------------------------------------------------------------

    def _merge_transcript(
        self, input: AgentInput, run: Transcript, workspace: Path
    ) -> None:
        """Fold the reconstructed run into the transcript the runner owns.

        The runner opened a live ``Transcript`` and passed it in ``context``;
        it already holds the case's prompt, run id and config hash. So we move
        the *evidence* across (tool calls, reasoning, the CLI's own totals) and
        leave the runner's own bookkeeping alone.

        The calls move as objects, not as kwargs: the reconstructor already
        resolved ``call_id``, ``turn_index`` and ``agent_name``, and rebuilding
        them would throw away exactly the fields ``turn_count`` and the
        sub-agent attribution depend on.
        """
        for call in run.tool_calls:
            self._emit_tool_call(input, call)

        live = input.context.get("transcript")
        if live is None:
            return

        for step in run.reasoning_steps:
            live.add_reasoning_step(step)

        live.metadata.update(run.metadata)
        live.metadata["workspace"] = str(workspace)
        if run.environment.model_version:
            live.environment.model_version = run.environment.model_version

    # ------------------------------------------------------------------
    # Change capture
    # ------------------------------------------------------------------

    async def _capture_changes(
        self, workspace: Path
    ) -> tuple[str, list[GeneratedFile]]:
        """The agent's work as a unified diff plus the changed files' contents.

        ``git add -N`` registers new files as intent-to-add so ``git diff`` shows
        them; without it a run whose entire contribution is new files would look
        like it changed nothing.
        """
        if not (workspace / ".git").exists():
            return "", []

        await self._exec("git add -A -N", workspace, self.setup_timeout)
        diff_result = await self._exec("git diff", workspace, self.setup_timeout)
        diff = diff_result.stdout or "" if diff_result.exit_code == 0 else ""

        names = await self._exec(
            "git diff --name-only", workspace, self.setup_timeout
        )
        files: list[GeneratedFile] = []
        if names.exit_code == 0:
            for rel in (names.stdout or "").splitlines():
                rel = rel.strip()
                if not rel:
                    continue
                path = workspace / rel
                content = ""
                if path.is_file() and path.stat().st_size <= _MAX_INLINE_FILE_BYTES:
                    try:
                        content = path.read_text(encoding="utf-8")
                    except (OSError, UnicodeDecodeError):
                        content = ""
                files.append(
                    GeneratedFile(
                        path=rel,
                        content=content,
                        language=_EXT_TO_LANGUAGE.get(path.suffix, ""),
                    )
                )
        return diff, files

    # ------------------------------------------------------------------
    # Shell helper
    # ------------------------------------------------------------------

    async def _exec(
        self, command: str, cwd: Path, timeout: float
    ) -> ExecutionResult:
        """Run a shell command in ``cwd`` with the host environment."""
        started = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=str(cwd),
                env=self._cli_env(),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as e:
            return ExecutionResult(exit_code=127, stdout="", stderr=str(e))

        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return ExecutionResult(
                exit_code=124,
                stdout="",
                stderr=f"Command timed out after {timeout}s: {command}",
                duration_ms=(time.monotonic() - started) * 1000,
            )

        return ExecutionResult(
            exit_code=proc.returncode if proc.returncode is not None else -1,
            stdout=out.decode("utf-8", errors="replace"),
            stderr=err.decode("utf-8", errors="replace"),
            duration_ms=(time.monotonic() - started) * 1000,
        )


class _CliRun:
    """One CLI invocation's results."""

    def __init__(
        self,
        transcript: Transcript,
        exit_code: int,
        stderr: str,
        events_seen: int,
        timed_out: bool = False,
        stream_path: str = "",
    ) -> None:
        self.transcript = transcript
        self.exit_code = exit_code
        self.stderr = stderr
        self.events_seen = events_seen
        self.timed_out = timed_out
        self.stream_path = stream_path

    @property
    def final_text(self) -> str:
        """The agent's final message, as the CLI reported it."""
        outcome = self.transcript.outcome
        if outcome is not None and outcome.output_data:
            return str(outcome.output_data.get("final_output", ""))
        return ""
