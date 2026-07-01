"""Local subprocess-based sandbox implementation."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import signal
import tempfile
import time
from pathlib import Path
from typing import Any

from compass.core.artifacts import ExecutionResult, GeneratedFile
from compass.sandbox.base import Sandbox
from compass.sandbox.registry import register_sandbox

logger = logging.getLogger(__name__)

# Environment variables that are safe to pass through to the sandbox.
_SAFE_ENV_VARS = frozenset({
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TERM",
    "SHELL",
    "USER",
    "LOGNAME",
    "TMPDIR",
})

# Provider-/cloud-specific prefixes that indicate secrets.
_BLOCKED_ENV_PREFIXES = (
    "AWS_",
    "OPENAI_",
    "ANTHROPIC_",
    "AZURE_",
    "GCP_",
    "GITHUB_",
)

# Secret-indicator tokens matched anywhere in the name. Real-world secrets are
# usually *suffixed* (FOO_API_KEY, MY_SERVICE_SECRET, USER_TOKEN), so a
# prefix-only check would miss them. Over-blocking here is intentional: this is
# a fail-safe, and refusing to pass through a benign var is far cheaper than
# leaking a credential into the sandbox.
_BLOCKED_ENV_SUBSTRINGS = (
    "API_KEY",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "PRIVATE_KEY",
    "TOKEN",
)

# File extension to language mapping for collect_files.
_EXT_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".r": "r",
    ".R": "r",
    ".sql": "sql",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "zsh",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".xml": "xml",
    ".md": "markdown",
    ".txt": "text",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".csv": "csv",
}


def _is_blocked(name: str) -> bool:
    """Return True if the environment variable name should be blocked."""
    upper = name.upper()
    if any(upper.startswith(prefix) for prefix in _BLOCKED_ENV_PREFIXES):
        return True
    return any(token in upper for token in _BLOCKED_ENV_SUBSTRINGS)


class PathTraversalError(ValueError):
    """Raised when a path attempts to escape the sandbox directory."""

    pass


@register_sandbox("local")
class LocalSandbox(Sandbox):
    """Sandbox backed by a local temporary directory and subprocess execution.

    WARNING: This sandbox is NOT a strong security boundary. It provides basic
    isolation for trusted code but should not be used with untrusted input.
    For untrusted code, use a containerized sandbox (e.g., Docker).

    Config options (passed as ``config`` dict):
        base_dir: Parent directory for the sandbox temp dir (default: system temp).
        default_timeout: Default command timeout in seconds (default: 30).
        env_passthrough: Extra env var names to pass through from the host.
        env_overrides: Dict of env vars to forcibly set in the sandbox.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self._workdir: str | None = None

    # -- properties -----------------------------------------------------------

    @property
    def workdir(self) -> str:
        if self._workdir is None:
            raise RuntimeError("Sandbox not set up yet. Call setup() first.")
        return self._workdir

    # -- path safety ----------------------------------------------------------

    def _safe_path(self, relative_path: str, must_exist: bool = False) -> Path:
        """Resolve a relative path safely within the sandbox.

        Args:
            relative_path: Path relative to workdir (may contain ..).
            must_exist: If True, raise FileNotFoundError if path doesn't exist.

        Returns:
            Resolved absolute Path guaranteed to be within workdir.

        Raises:
            PathTraversalError: If the resolved path escapes the sandbox.
            FileNotFoundError: If must_exist=True and path doesn't exist.
        """
        workdir = Path(self.workdir).resolve()
        # Join and resolve to handle any ".." components
        target = (workdir / relative_path).resolve()

        # Check that the resolved path is within workdir
        try:
            target.relative_to(workdir)
        except ValueError:
            raise PathTraversalError(
                f"Path traversal detected: '{relative_path}' resolves to "
                f"'{target}' which is outside sandbox '{workdir}'"
            )

        if must_exist and not target.exists():
            raise FileNotFoundError(f"File not found in sandbox: {relative_path}")

        return target

    # -- lifecycle ------------------------------------------------------------

    async def setup(self) -> None:
        base_dir = self.config.get("base_dir")
        self._workdir = tempfile.mkdtemp(
            prefix="compass_sandbox_",
            dir=base_dir,
        )
        logger.debug("Sandbox created at %s", self._workdir)

    async def cleanup(self) -> None:
        if self._workdir is not None and os.path.exists(self._workdir):
            shutil.rmtree(self._workdir)
            logger.debug("Sandbox cleaned up: %s", self._workdir)
        self._workdir = None

    # -- command execution ----------------------------------------------------

    def _build_env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        """Build an isolated environment dict for subprocess execution."""
        env: dict[str, str] = {}

        # Passthrough safe vars from host
        passthrough = set(_SAFE_ENV_VARS)
        passthrough.update(self.config.get("env_passthrough", []))
        for key in passthrough:
            val = os.environ.get(key)
            if val is not None and not _is_blocked(key):
                env[key] = val

        # Override HOME to sandbox
        env["HOME"] = self.workdir

        # Apply config-level overrides
        for k, v in self.config.get("env_overrides", {}).items():
            if not _is_blocked(k):
                env[k] = v

        # Apply per-call overrides (also filtered)
        if extra:
            for k, v in extra.items():
                if not _is_blocked(k):
                    env[k] = v

        return env

    async def exec(
        self,
        command: str,
        *,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
        workdir: str | None = None,
    ) -> ExecutionResult:
        if timeout is None:
            timeout = self.config.get("default_timeout", 30)

        cwd = self.workdir
        if workdir is not None:
            # Validate workdir doesn't escape sandbox
            resolved = self._safe_path(workdir)
            cwd = str(resolved)

        sandbox_env = self._build_env(env)

        t0 = time.monotonic()
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=sandbox_env,
            start_new_session=True,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            # Kill the entire process group
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            # Wait for process to be reaped
            try:
                await proc.wait()
            except Exception:
                pass
            duration_ms = (time.monotonic() - t0) * 1000
            return ExecutionResult(
                exit_code=-1,
                stdout="",
                stderr=f"Command timed out after {timeout}s",
                duration_ms=duration_ms,
            )

        duration_ms = (time.monotonic() - t0) * 1000
        return ExecutionResult(
            exit_code=proc.returncode or 0,
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
            duration_ms=duration_ms,
        )

    # -- file operations ------------------------------------------------------

    async def write_file(self, path: str, content: str) -> None:
        target = self._safe_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    async def read_file(self, path: str) -> str:
        target = self._safe_path(path, must_exist=True)
        return target.read_text(encoding="utf-8")

    async def copy_in(self, src: str, dest: str) -> None:
        src_path = Path(src)
        dest_path = self._safe_path(dest)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if src_path.is_dir():
            shutil.copytree(str(src_path), str(dest_path))
        else:
            shutil.copy2(str(src_path), str(dest_path))

    async def collect_files(self, pattern: str = "**/*") -> list[GeneratedFile]:
        root = Path(self.workdir)
        results: list[GeneratedFile] = []
        for p in sorted(root.glob(pattern)):
            if not p.is_file():
                continue
            try:
                content = p.read_text(encoding="utf-8")
            except (UnicodeDecodeError, ValueError):
                continue
            rel = str(p.relative_to(root))
            lang = _EXT_TO_LANGUAGE.get(p.suffix, "")
            results.append(GeneratedFile(path=rel, content=content, language=lang))
        return results
