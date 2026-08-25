"""Shared plumbing for adapters that drive a coding-agent CLI over a real repo.

``claude_code``, ``pi`` and ``codex`` are the same evaluation shape with three
different executables: put an agent CLI in a throwaway checkout of a git repo,
hand it a task, and come back with **the diff it produced** (for OUTCOME
graders) and **the step-by-step trace of how it got there** (for TRANSCRIPT
graders). Only three things actually differ between them — the command line, the
parser for the CLI's machine-readable output, and what the run is called in an
error message — so everything else lives here:

- **workspace isolation** (git worktree / copy / none) and teardown
- **skill installation** — the version of a skill under test is copied into the
  workspace and fingerprinted, so "which skill was actually in play" is a fact
  in the trace rather than a claim in a commit message
- **setup commands**, recorded as tool calls so a failed ``uv sync`` is visible
- **spawning the CLI** and streaming its stdout through a reconstructor, with a
  timeout that keeps the partial trace instead of discarding the run
- **merging** the reconstructed transcript into the runner's live one
- **change capture** (``git add -A -N`` + ``git diff``) and artifact assembly

A subclass supplies :meth:`~CliAgentAdapter._build_argv` and
:meth:`~CliAgentAdapter._new_reconstructor`, plus the two labels used in error
messages; :meth:`~CliAgentAdapter._run_error` is there for a CLI that reports a
dead run *in-band* rather than by falling silent. See
:mod:`compass.adapters.claude_code`, :mod:`compass.adapters.pi` and
:mod:`compass.adapters.codex`.

**No sandbox, on purpose.** :class:`~compass.sandbox.local.LocalSandbox` is a
scrubbed temp dir; an agent CLI needs real network, real credentials, a real
git repo and the project's own toolchain, and fights the sandbox on all four.
Isolation comes from **git worktree** instead: each trial gets its own checkout
of the same base ref, so parallel trials never collide and the diff is computed
against a known-clean baseline.
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
from typing import Any, Protocol

from compass.adapters.base import Adapter, AgentInput, AgentOutput
from compass.core.artifacts import (
    CodeArtifact,
    ExecutionResult,
    GeneratedFile,
    TextArtifact,
)
from compass.core.skills import read_skill_name, tree_digest
from compass.core.transcript import Transcript

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

# Max bytes in one line of the CLI's stdout.
#
# asyncio's default is 64 KiB, and one line of an agent's event stream blows
# past that routinely: a `read` of a large file comes back as one JSON object,
# and reasoning models attach kilobytes of opaque thought signatures to every
# turn. Over the limit, ``readline()`` raises ``ValueError`` — which surfaced as
# a whole *case* erroring out with "Separator is found, but chunk is longer than
# limit" and no trace at all, on a model whose only sin was being verbose.
_STDOUT_LINE_LIMIT = 16 * 1024 * 1024


class CliAgentError(RuntimeError):
    """Raised when the workspace cannot be prepared for a run."""


class StreamReconstructor(Protocol):
    """The parser contract: fed raw stdout lines, yields a Transcript."""

    def feed_line(self, line: str) -> bool:
        """Consume one raw line; False if it was not a usable JSON event."""
        ...

    def finish(self) -> Transcript:
        """The Transcript reconstructed so far. Valid at any point."""
        ...


class CliAgentAdapter(Adapter):
    """Base for adapters that run an agent CLI against a git repository.

    Shared config (a subclass adds its own CLI's flags on top):
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
        cli_path:        str  — the executable (default: the subclass's).
        model:           str  — the model axis; how it reaches the CLI is the
                                subclass's business.
        append_system_prompt:      str — appended to the CLI's own prompt.
        append_system_prompt_file: str — same, read from a file.
        system_prompt / system_prompt_file: str — *replaces* the CLI's prompt.
        extra_args:       list[str] — appended verbatim to the command line.
        timeout:          float — seconds before the CLI is killed
                                (default ``900``). A timed-out run still
                                returns the steps completed so far.
        skill:           str  — directory of one skill to install into the
                                workspace before the run. This is the *skill
                                axis*: an empty string is the no-skill baseline,
                                so ``--model-key skill -m ./v1 -m ./v2 -m ""``
                                is a v1 / v2 / baseline sweep.
        skills:          list[str] — further skill directories to install
                                alongside it (companions the one under test
                                needs). Merged with ``skill``.
        setup_commands:  list[str] — shell commands run in the workspace before
                                the agent (install deps, seed a DB).
        setup_timeout:    float — per setup command (default ``600``).
        env:             dict — extra environment variables for the CLI.
        save_stream_to:  str  — directory to write the raw event stream to, for
                                replaying the run without paying for it again.

    **keep_workspace defaults to True.** Graders run *after* the adapter
    returns, and ``integration_test`` scores the agent by running hidden tests
    against the tree it produced — deleting that tree first would destroy the
    evidence before anything reads it. The path is on
    ``CodeArtifact.metadata["workspace"]``, and ``integration_test`` resolves
    ``workdir: "{workspace}"`` to it. Reclaim the disk afterwards with
    ``git worktree prune`` in the source repo.
    """

    #: Executable used when the config does not name one.
    default_cli_path: str = ""
    #: How the executable is named in error messages ("claude", "pi").
    cli_label: str = "agent"
    #: How its machine-readable output is named in error messages.
    stream_label: str = "stream"
    #: Argument that makes the CLI print its version, for the health check.
    version_arg: str = "--version"
    #: Where this CLI discovers project-scoped skills, relative to the
    #: workspace (``.claude/skills`` for Claude Code). Empty means Compass has
    #: no way to install a skill for this CLI, and configuring one is an error
    #: rather than a run that silently has no skill.
    skills_dir: str = ""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        c = self.config
        self.repo: str = c.get("repo", "")
        self.base_ref: str = c.get("base_ref", "")
        self.isolation: str = c.get("isolation", "worktree")
        self.keep_workspace: bool = c.get("keep_workspace", True)
        self.cli_path: str = c.get("cli_path", self.default_cli_path)
        self.model: str = c.get("model", "")
        self.append_system_prompt: str = c.get("append_system_prompt", "")
        self.append_system_prompt_file: str = c.get("append_system_prompt_file", "")
        self.system_prompt: str = c.get("system_prompt", "")
        self.system_prompt_file: str = c.get("system_prompt_file", "")
        self.extra_args: list[str] = c.get("extra_args", [])
        self.skill: str = c.get("skill", "")
        self.skills: list[str] = list(c.get("skills", []))
        self.timeout: float = c.get("timeout", 900)
        self.setup_commands: list[str] = c.get("setup_commands", [])
        self.setup_timeout: float = c.get("setup_timeout", 600)
        self.env: dict[str, str] = c.get("env", {})
        self.save_stream_to: str = c.get("save_stream_to", "")

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def _build_argv(self, prompt: str) -> list[str]:
        """The full command line, starting with ``self.cli_path``."""
        raise NotImplementedError

    def _new_reconstructor(
        self, task_id: str, workspace: Path
    ) -> StreamReconstructor:
        """A parser for this CLI's machine-readable output.

        ``workspace`` is the directory the CLI ran in. Some CLIs report the paths
        they edited absolute and never echo their own cwd (codex is one), so the
        only way to turn those into portable, globbable targets is to be told.
        """
        raise NotImplementedError

    def _run_metadata(self) -> dict[str, Any]:
        """Extra keys for the run's metadata (the CLI's own axes)."""
        return {}

    def _run_error(self, run: _CliRun) -> str | None:
        """Why this run should be reported as an adapter error, or None.

        A CLI that never produced a parseable event did not run — an empty diff
        from *that* is a harness failure, not an agent that declined to edit
        anything, and the two must not score the same. Subclasses extend this
        with the failures their own CLI reports in-band (see
        :class:`~compass.adapters.codex.CodexAdapter`).
        """
        if not run.events_seen:
            return (
                f"{self.cli_label} CLI produced no {self.stream_label} events "
                f"(exit {run.exit_code}): {(run.stderr or '')[-500:] or 'no stderr'}"
            )
        return None

    # ------------------------------------------------------------------
    # Adapter contract
    # ------------------------------------------------------------------

    def validate_config(self) -> bool:
        """A repo is required (unless supplied per-case) and isolation must be known."""
        return self.isolation in ("worktree", "copy", "none")

    async def health_check(self) -> bool:
        """Whether the CLI is on PATH and answers ``--version``."""
        try:
            proc = await asyncio.create_subprocess_exec(
                self.cli_path,
                self.version_arg,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=30)
            return proc.returncode == 0
        except (OSError, asyncio.TimeoutError):
            return False

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
        except CliAgentError as e:
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
        # --- Install the skill(s) under test ---
        try:
            installed = await asyncio.to_thread(self._install_skills, workspace)
        except CliAgentError as e:
            return AgentOutput(
                error=str(e),
                metadata={"phase": "skills", "workspace": str(workspace)},
            )

        # --- Setup commands (deps, fixtures) ---
        for cmd in self.setup_commands:
            started = time.monotonic()
            result = await self._exec(cmd, workspace, self.setup_timeout)
            elapsed = (time.monotonic() - started) * 1000
            self._record_tool_call(
                input,
                tool_name=f"{self.name}.setup",
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
        logger.info("%s: running in %s", self.name, workspace)

        started = time.monotonic()
        run = await self._run_cli(argv, workspace, input)
        duration_ms = (time.monotonic() - started) * 1000

        # --- Capture what changed ---
        diff, changed = await self._capture_changes(workspace)

        # Merge the reconstructed run into the transcript the runner is
        # recording, so transcript-scope graders see the real steps.
        self._merge_transcript(input, run.transcript, workspace, skills=installed)

        if run.timed_out:
            logger.warning(
                "%s: timed out after %.0fs — grading the partial run",
                self.name,
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
        if installed:
            metadata["skills"] = installed
        metadata.update(self._run_metadata())
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

        # The agent's closing message, as a text artifact of its own.
        #
        # It was already on the CodeArtifact's ``execution.stdout``, but nothing
        # reads it there: ``GradeContext.answer`` looks at
        # ``output_data["final_output"]`` and then at the first *TextArtifact*,
        # so for adapter-driven runs every text grader — ``rubric``,
        # ``style_convention``, ``semantic_match`` — saw an empty string and
        # silently judged nothing. That matters for the class of case where the
        # right move is to *not* edit and say why: with no reachable text there
        # is nothing left to grade but the (correctly empty) diff.
        artifacts: list[Any] = [artifact]
        if run.final_text:
            artifacts.append(
                TextArtifact(
                    content=run.final_text,
                    format="markdown",
                    word_count=len(run.final_text.split()),
                    metadata={"source": f"{self.name}.final_message"},
                )
            )

        run_error = self._run_error(run)
        if run_error:
            return AgentOutput(
                artifacts=artifacts, metadata=metadata, error=run_error
            )

        return AgentOutput(artifacts=artifacts, metadata=metadata)

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
            dest = Path(tempfile.mkdtemp(prefix=f"compass_{self.name}_"))
            target = dest / repo_path.name
            await asyncio.to_thread(
                shutil.copytree, repo_path, target, symlinks=True
            )

            async def _rm() -> None:
                await asyncio.to_thread(shutil.rmtree, dest, True)

            return target, _rm

        # worktree (default)
        if not (repo_path / ".git").exists():
            raise CliAgentError(
                f"isolation='worktree' needs a git repository, but {repo_path} "
                "has no .git. Use isolation='copy' for a plain directory."
            )
        ref = base_ref or await self._git_head(repo_path)
        dest = Path(tempfile.mkdtemp(prefix=f"compass_{self.name}_")) / (
            f"wt_{uuid.uuid4().hex[:8]}"
        )
        result = await self._exec(
            f"git worktree add --detach {shlex.quote(str(dest))} {shlex.quote(ref)}",
            repo_path,
            self.setup_timeout,
        )
        if result.exit_code != 0:
            raise CliAgentError(
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
            raise CliAgentError(
                f"git rev-parse HEAD failed in {repo_path}: "
                f"{(result.stderr or '').strip()[-500:]}"
            )
        return (result.stdout or "").strip()

    # ------------------------------------------------------------------
    # Skill installation
    # ------------------------------------------------------------------

    def _skill_sources(self) -> list[str]:
        """The configured skill directories, in order, without the blanks.

        ``skill: ""`` is the *no-skill baseline* of a sweep, not a mistake —
        which is why an empty entry drops out here instead of raising.
        """
        seen: dict[str, None] = {}
        for raw in [self.skill, *self.skills]:
            path = str(raw).strip()
            if path:
                seen.setdefault(path, None)
        return list(seen)

    def _install_skills(self, workspace: Path) -> list[dict[str, Any]]:
        """Copy the configured skills into the workspace; describe what landed.

        Two things make this worth doing in the framework rather than in a
        ``setup_commands`` one-liner.

        **The install name is the skill's own name, not the source directory's.**
        A v1/v2 comparison keeps the versions in sibling directories
        (``report-writer-v1`` / ``report-writer-v2``), but installing under
        those names would hand the agent two *differently named* skills and
        quietly change the thing being measured. Both land at
        ``<skills_dir>/report-writer`` instead, so the only difference between
        the arms is the content.

        **The content is fingerprinted.** ``digest`` is a hash of every file in
        the skill, recorded on the transcript. Six months later that is the
        difference between "v2 scored higher" and "*this* v2 scored higher".
        """
        sources = self._skill_sources()
        if not sources:
            return []
        if not self.skills_dir:
            raise CliAgentError(
                f"{self.name} has no skill directory convention, so "
                f"'skill'/'skills' cannot be installed for it. Remove the key, "
                f"or use an adapter that supports skills (claude_code)."
            )
        if self.isolation == "none":
            raise CliAgentError(
                "isolation='none' runs in the repository itself, and installing "
                f"a skill there would write into {workspace}/{self.skills_dir} "
                "and leave it behind. Use isolation='worktree' or 'copy'."
            )

        installed: list[dict[str, Any]] = []
        for raw in sources:
            src = Path(raw).expanduser().resolve()
            if not src.is_dir():
                raise CliAgentError(f"Skill directory not found: {src}")
            skill_md = src / "SKILL.md"
            if not skill_md.is_file():
                # A path typo would otherwise install nothing and score as a
                # perfectly ordinary baseline run — the one failure mode that
                # invalidates a comparison without leaving a mark.
                raise CliAgentError(f"No SKILL.md in skill directory: {src}")

            name = read_skill_name(src) or src.name
            dest = workspace / self.skills_dir / name
            if dest.exists():
                shutil.rmtree(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(
                src,
                dest,
                symlinks=True,
                ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
            )

            digest, file_count = tree_digest(dest)
            installed.append(
                {
                    "name": name,
                    "source": str(src),
                    "path": f"{self.skills_dir}/{name}",
                    "digest": digest,
                    "files": file_count,
                }
            )
            logger.info(
                "%s: installed skill %s (%s, %d files) from %s",
                self.name,
                name,
                digest,
                file_count,
                src,
            )
        return installed

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
                logger.warning("%s: could not read task_file %s: %s", self.name, extra, e)
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
            raise CliAgentError(f"Could not read {label} '{path}': {e}") from e

    def _cli_env(self) -> dict[str, str]:
        """The CLI's environment: the host's, plus configured extras.

        Deliberately *not* the scrubbed sandbox environment — the CLI
        authenticates with a provider key or with the credentials under
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
        reconstructor = self._new_reconstructor(task_id, workspace)
        raw_lines: list[str] = []
        events_seen = 0
        dropped_lines = 0

        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(workspace),
                env=self._cli_env(),
                # The prompt goes on the command line; nothing here ever wants
                # the parent's stdin. Inheriting it lets a CLI that reads stdin
                # when it is not a TTY (codex says "Reading additional input
                # from stdin...") block on a pipe nobody is going to close —
                # a hang with no output, in a subprocess, under a timeout.
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=_STDOUT_LINE_LIMIT,
            )
        except FileNotFoundError:
            return _CliRun(
                transcript=reconstructor.finish(),
                exit_code=127,
                stderr=(
                    f"{self.cli_label} CLI not found: '{self.cli_path}'. Install it, "
                    "or set agent.config.cli_path to its location."
                ),
                events_seen=0,
            )

        assert proc.stdout is not None and proc.stderr is not None
        stderr_chunks: list[str] = []

        async def drain_stderr() -> None:
            stderr = proc.stderr
            assert stderr is not None
            while True:
                try:
                    chunk = await stderr.readline()
                except ValueError:
                    continue  # over-long stderr line; same deal as stdout
                if not chunk:
                    return
                stderr_chunks.append(chunk.decode("utf-8", errors="replace"))

        stderr_task = asyncio.create_task(drain_stderr())

        async def read_stdout() -> int:
            nonlocal events_seen, dropped_lines
            stdout = proc.stdout
            assert stdout is not None
            while True:
                try:
                    raw = await stdout.readline()
                except ValueError:
                    # Still over the (already generous) limit. readline has
                    # dropped that line's bytes; skip it and keep reading —
                    # losing one event beats losing the whole run. It is counted
                    # so the trace never quietly claims to be complete.
                    dropped_lines += 1
                    logger.warning(
                        "%s: dropped a stdout line over %d bytes",
                        self.name,
                        _STDOUT_LINE_LIMIT,
                    )
                    continue
                if not raw:
                    break
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
        if dropped_lines:
            transcript.metadata["dropped_stdout_lines"] = dropped_lines

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
        """Persist the raw event stream, so the run can be replayed for free."""
        if not self.save_stream_to or not raw_lines:
            return ""
        out_dir = Path(self.save_stream_to).expanduser()
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in task_id)
            path = out_dir / f"{safe}_{uuid.uuid4().hex[:8]}.stream.jsonl"
            path.write_text("".join(raw_lines), encoding="utf-8")
        except OSError as e:
            logger.warning("%s: could not save the event stream: %s", self.name, e)
            return ""
        return str(path)

    def _task_id(self, input: AgentInput) -> str:
        transcript = input.context.get("transcript")
        task_id = getattr(transcript, "task_id", None)
        return str(task_id) if task_id else self.name

    # ------------------------------------------------------------------
    # Transcript merge
    # ------------------------------------------------------------------

    def _merge_transcript(
        self,
        input: AgentInput,
        run: Transcript,
        workspace: Path,
        skills: list[dict[str, Any]] | None = None,
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
        if skills:
            # ``skill_trigger`` reads this to know which skill it is asking
            # about, so a scenario does not have to repeat the name in every
            # grader config — and an offline regrade still knows.
            live.metadata["skills"] = skills
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

        An installed skill is *our* file, not the agent's work, so it is
        excluded from the pathspec — otherwise every with-skill arm of a
        comparison would show a few hundred extra changed lines and
        ``diff_size`` would score the harness instead of the agent.
        """
        if not (workspace / ".git").exists():
            return "", []

        spec = self._diff_pathspec()
        await self._exec(f"git add -A -N{spec}", workspace, self.setup_timeout)
        diff_result = await self._exec(f"git diff{spec}", workspace, self.setup_timeout)
        diff = diff_result.stdout or "" if diff_result.exit_code == 0 else ""

        names = await self._exec(
            f"git diff --name-only{spec}", workspace, self.setup_timeout
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

    def _diff_pathspec(self) -> str:
        """Pathspec suffix that hides the installed skills from ``git diff``.

        Empty — leaving the git commands byte-identical to what they were —
        unless a skill was actually installed.
        """
        if not (self._skill_sources() and self.skills_dir):
            return ""
        return f" -- . {shlex.quote(f':(exclude,glob){self.skills_dir}/**')}"

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
