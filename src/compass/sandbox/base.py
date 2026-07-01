"""Abstract base class for sandboxed execution environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from compass.core.artifacts import ExecutionResult, GeneratedFile


class Sandbox(ABC):
    """Abstract base for all sandbox implementations.

    Provides an isolated environment for executing commands and managing files.
    Supports ``async with`` for automatic setup/cleanup.
    """

    @property
    @abstractmethod
    def workdir(self) -> str:
        """Return the sandbox working directory path.

        Raises:
            RuntimeError: If called before ``setup()``.
        """
        ...

    @abstractmethod
    async def setup(self) -> None:
        """Create the isolated environment (temp directory, etc.)."""
        ...

    @abstractmethod
    async def exec(
        self,
        command: str,
        *,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
        workdir: str | None = None,
    ) -> ExecutionResult:
        """Execute a shell command inside the sandbox.

        Args:
            command: Shell command string to execute.
            timeout: Per-call timeout in seconds (``None`` uses default).
            env: Extra environment variables for this call.
            workdir: Working directory override (relative to sandbox root).

        Returns:
            An :class:`ExecutionResult` with exit_code, stdout, stderr, duration_ms.
        """
        ...

    @abstractmethod
    async def write_file(self, path: str, content: str) -> None:
        """Write *content* to *path* inside the sandbox.

        Parent directories are created automatically.
        """
        ...

    @abstractmethod
    async def read_file(self, path: str) -> str:
        """Read and return the contents of *path* inside the sandbox.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        ...

    @abstractmethod
    async def copy_in(self, src: str, dest: str) -> None:
        """Copy a host file or directory into the sandbox.

        Args:
            src: Absolute path on the host.
            dest: Relative path inside the sandbox.
        """
        ...

    @abstractmethod
    async def collect_files(self, pattern: str = "**/*") -> list[GeneratedFile]:
        """Collect files matching *pattern* from the sandbox.

        Args:
            pattern: Glob pattern relative to the sandbox working directory.

        Returns:
            List of :class:`GeneratedFile` instances.
        """
        ...

    @abstractmethod
    async def cleanup(self) -> None:
        """Remove the sandbox directory and all its contents."""
        ...

    # -- async context manager ------------------------------------------------

    async def __aenter__(self) -> Sandbox:
        await self.setup()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        await self.cleanup()
