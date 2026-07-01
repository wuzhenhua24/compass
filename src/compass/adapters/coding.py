"""Coding agent adapter for code execution workflows."""

from __future__ import annotations

import time
from typing import Any

from compass.adapters.base import Adapter, AgentInput, AgentOutput
from compass.adapters.registry import register_adapter
from compass.core.artifacts import CodeArtifact, ExecutionResult, GeneratedFile
from compass.sandbox import get_sandbox_class


@register_adapter("coding")
class CodingAdapter(Adapter):
    """Adapter for coding agent workflows.

    Writes code files into a sandbox, executes a command, collects
    generated files, and returns a :class:`CodeArtifact`.

    Config options:
        run_command: Shell command to execute (default ``"python main.py"``).
        timeout: Command timeout in seconds (default ``30``).
        sandbox_type: Sandbox implementation name (default ``"local"``).
        sandbox_config: Extra config dict passed to the sandbox constructor.
        collect_pattern: Glob pattern for collecting output files
            (default ``"**/*"``).
    """

    name = "coding"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.run_command: str = self.config.get("run_command", "python main.py")
        self.timeout: float = self.config.get("timeout", 30)
        self.sandbox_type: str = self.config.get("sandbox_type", "local")
        self.sandbox_config: dict[str, Any] = self.config.get("sandbox_config", {})
        self.collect_pattern: str = self.config.get("collect_pattern", "**/*")

    async def run(self, input: AgentInput) -> AgentOutput:
        """Run a coding workflow inside a sandbox.

        Expects ``input.params`` to contain:
            files: list[dict] | dict — either a list where each dict has
                ``path`` and ``content`` keys, or a dict mapping path→content.
            run_command: str (optional) — overrides the adapter-level default.

        Returns:
            AgentOutput with a single :class:`CodeArtifact`.
        """
        files_raw = input.params.get("files", [])
        # Normalize files to list[{path, content}] format
        # Accepts both dict[path→content] and list[{path, content}]
        if isinstance(files_raw, dict):
            files: list[dict[str, str]] = [
                {"path": path, "content": content}
                for path, content in files_raw.items()
            ]
        else:
            files = files_raw
        run_command = input.params.get("run_command", self.run_command)

        sandbox_cls = get_sandbox_class(self.sandbox_type)
        sandbox = sandbox_cls(self.sandbox_config)

        try:
            async with sandbox:
                # Write all source files into the sandbox
                for file_spec in files:
                    await sandbox.write_file(
                        file_spec["path"],
                        file_spec["content"],
                    )

                # Execute the command
                result = await sandbox.exec(run_command, timeout=self.timeout)

                # Collect output files
                collected = await sandbox.collect_files(self.collect_pattern)

            self._record_tool_call(
                input,
                tool_name="sandbox.exec",
                tool_type="code",
                input={
                    "command": run_command,
                    "timeout": self.timeout,
                    "sandbox_type": self.sandbox_type,
                },
                output={
                    "exit_code": result.exit_code,
                    "stdout_len": len(result.stdout or ""),
                    "stderr_len": len(result.stderr or ""),
                },
                status="ok",
                duration_ms=result.duration_ms,
                metadata={"collect_pattern": self.collect_pattern},
            )

            code_artifact = CodeArtifact(
                files=collected,
                execution=result,
            )

            return AgentOutput(artifacts=[code_artifact])

        except Exception as e:
            self._record_tool_call(
                input,
                tool_name="sandbox.exec",
                tool_type="code",
                input={
                    "command": run_command,
                    "timeout": self.timeout,
                    "sandbox_type": self.sandbox_type,
                },
                status="error",
                duration_ms=0.0,
                error={"type": type(e).__name__, "message": str(e)},
                metadata={"collect_pattern": self.collect_pattern},
            )
            return AgentOutput(error=str(e))
