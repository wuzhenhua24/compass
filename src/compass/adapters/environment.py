"""Environment adapter — run agents inside reproducible software environments.

Inspired by Stripe's Agent Benchmark pattern: each evaluation is a standalone
software project with an ``environment/`` directory (initial state for the
agent), a ``grader/`` directory (hidden tests), and optionally a ``solution/``
directory.

The EnvironmentAdapter manages the lifecycle of such environments:

1. **Setup** — copy the environment directory into a sandbox, run setup commands
   (install dependencies, seed databases, start services).
2. **Agent execution** — invoke an external agent command (e.g. ``goose``,
   ``aider``, ``claude``) inside the sandbox, passing the task prompt.
3. **Teardown** — collect output artifacts, record tool calls, stop services.

The adapter returns a :class:`CodeArtifact` with the agent's execution result
and any files produced, ready for grading by ``integration_test`` or other
graders.

YAML example::

    agent:
      adapter: environment
      config:
        environment_dir: "./benchmarks/galtee-basic/environment"
        agent_command: "goose run"
        problem_file: "PROBLEM.md"
        setup_commands:
          - "npm --prefix server install"
        teardown_commands:
          - "pkill -f 'node server' || true"
        timeout: 600
        env:
          STRIPE_SECRET_KEY: "sk_test_..."
"""

from __future__ import annotations

import os
import time
from typing import Any

from compass.adapters.base import Adapter, AgentInput, AgentOutput
from compass.adapters.registry import register_adapter
from compass.core.artifacts import CodeArtifact, ExecutionResult
from compass.sandbox import get_sandbox_class


@register_adapter("environment")
class EnvironmentAdapter(Adapter):
    """Run an agent inside a reproducible software environment.

    The adapter copies an environment directory into a sandbox, runs optional
    setup commands, invokes the agent, and returns the result as a
    :class:`CodeArtifact`.

    Config:
        environment_dir:  str        — path to the environment directory
                                       (required, or per-case via params).
        agent_command:    str        — shell command to invoke the agent
                                       (default: ``""``; required).
        problem_file:     str        — filename inside environment_dir that
                                       contains the task prompt. If set, the
                                       file content is appended to the prompt
                                       (default: ``"PROBLEM.md"``).
        setup_commands:   list[str]  — commands to run before the agent
                                       (e.g. install deps, start services).
        teardown_commands:list[str]  — commands to run after the agent
                                       (e.g. stop services, collect logs).
        timeout:          float      — agent command timeout in seconds
                                       (default: ``300``).
        setup_timeout:    float      — per-command timeout for setup/teardown
                                       (default: ``120``).
        env:              dict       — extra environment variables.
        sandbox_type:     str        — sandbox backend (default: ``"local"``).
        sandbox_config:   dict       — extra sandbox constructor config.
        collect_pattern:  str        — glob for collecting output files
                                       (default: ``"**/*"``).
        workdir:          str        — working directory inside sandbox
                                       (default: root of copied environment).
    """

    name = "environment"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.environment_dir: str = self.config.get("environment_dir", "")
        self.agent_command: str = self.config.get("agent_command", "")
        self.problem_file: str = self.config.get("problem_file", "PROBLEM.md")
        self.setup_commands: list[str] = self.config.get("setup_commands", [])
        self.teardown_commands: list[str] = self.config.get("teardown_commands", [])
        self.timeout: float = self.config.get("timeout", 300)
        self.setup_timeout: float = self.config.get("setup_timeout", 120)
        self.env: dict[str, str] = self.config.get("env", {})
        self.sandbox_type: str = self.config.get("sandbox_type", "local")
        self.sandbox_config: dict[str, Any] = self.config.get("sandbox_config", {})
        self.collect_pattern: str = self.config.get("collect_pattern", "**/*")
        self.workdir: str | None = self.config.get("workdir")

    def validate_config(self) -> bool:
        """Both environment_dir and agent_command are required."""
        return bool(self.environment_dir) and bool(self.agent_command)

    async def run(self, input: AgentInput) -> AgentOutput:
        # Allow per-case overrides via params
        env_dir = input.params.get("environment_dir", self.environment_dir)
        agent_cmd = input.params.get("agent_command", self.agent_command)

        if not env_dir:
            return AgentOutput(error="Config 'environment_dir' is required")
        if not agent_cmd:
            return AgentOutput(error="Config 'agent_command' is required")

        env_dir = os.path.abspath(env_dir)
        if not os.path.isdir(env_dir):
            return AgentOutput(error=f"Environment directory not found: {env_dir}")

        sandbox_cls = get_sandbox_class(self.sandbox_type)
        sandbox = sandbox_cls(self.sandbox_config)

        total_start = time.monotonic()

        try:
            async with sandbox:
                # --- Phase 1: Copy environment into sandbox ---
                # Copy each top-level item individually to avoid
                # shutil.copytree failing when dest already exists.
                for item in os.listdir(env_dir):
                    await sandbox.copy_in(
                        os.path.join(env_dir, item), item,
                    )

                # Read problem file and build full prompt
                full_prompt = input.prompt
                if self.problem_file:
                    try:
                        problem_content = await sandbox.read_file(self.problem_file)
                        if problem_content.strip():
                            full_prompt = (
                                f"{input.prompt}\n\n{problem_content}"
                                if input.prompt
                                else problem_content
                            )
                    except FileNotFoundError:
                        pass  # problem_file is optional

                # --- Phase 2: Run setup commands ---
                setup_results: list[dict[str, Any]] = []
                for cmd in self.setup_commands:
                    setup_start = time.monotonic()
                    result = await sandbox.exec(
                        cmd,
                        timeout=self.setup_timeout,
                        env=self.env or None,
                        workdir=self.workdir,
                    )
                    setup_ms = (time.monotonic() - setup_start) * 1000

                    setup_results.append({
                        "command": cmd,
                        "exit_code": result.exit_code,
                        "duration_ms": setup_ms,
                    })

                    self._record_tool_call(
                        input,
                        tool_name="environment.setup",
                        tool_type="environment",
                        input={"command": cmd},
                        output={
                            "exit_code": result.exit_code,
                            "stdout_len": len(result.stdout or ""),
                            "stderr_len": len(result.stderr or ""),
                        },
                        status="ok" if result.exit_code == 0 else "error",
                        duration_ms=setup_ms,
                    )

                    if result.exit_code != 0:
                        return AgentOutput(
                            error=(
                                f"Setup command failed: {cmd} "
                                f"(exit code {result.exit_code})"
                            ),
                            metadata={
                                "phase": "setup",
                                "setup_results": setup_results,
                                "stderr": result.stderr,
                            },
                        )

                # --- Phase 3: Run agent command ---
                # Substitute {prompt} placeholder in agent command
                resolved_cmd = agent_cmd.replace("{prompt}", full_prompt)

                # Write prompt to a file so the agent can read it
                await sandbox.write_file(
                    ".compass_prompt.txt", full_prompt,
                )

                # Also substitute {prompt_file} placeholder
                prompt_file_path = os.path.join(
                    sandbox.workdir, ".compass_prompt.txt",
                )
                resolved_cmd = resolved_cmd.replace(
                    "{prompt_file}", prompt_file_path,
                )

                agent_start = time.monotonic()
                agent_result = await sandbox.exec(
                    resolved_cmd,
                    timeout=self.timeout,
                    env=self.env or None,
                    workdir=self.workdir,
                )
                agent_ms = (time.monotonic() - agent_start) * 1000

                self._record_tool_call(
                    input,
                    tool_name="environment.agent",
                    tool_type="environment",
                    input={
                        "command": resolved_cmd,
                        "timeout": self.timeout,
                        "environment_dir": env_dir,
                    },
                    output={
                        "exit_code": agent_result.exit_code,
                        "stdout_len": len(agent_result.stdout or ""),
                        "stderr_len": len(agent_result.stderr or ""),
                    },
                    status="ok" if agent_result.exit_code == 0 else "error",
                    duration_ms=agent_ms,
                    metadata={
                        "prompt_length": len(full_prompt),
                        "setup_results": setup_results,
                    },
                )

                # --- Phase 4: Teardown ---
                for cmd in self.teardown_commands:
                    await sandbox.exec(
                        cmd,
                        timeout=self.setup_timeout,
                        env=self.env or None,
                        workdir=self.workdir,
                    )

                # --- Phase 5: Collect output files ---
                collected = await sandbox.collect_files(self.collect_pattern)

            total_ms = (time.monotonic() - total_start) * 1000

            code_artifact = CodeArtifact(
                files=collected,
                execution=ExecutionResult(
                    exit_code=agent_result.exit_code,
                    stdout=agent_result.stdout,
                    stderr=agent_result.stderr,
                    duration_ms=agent_ms,
                ),
            )

            return AgentOutput(
                artifacts=[code_artifact],
                metadata={
                    "environment_dir": env_dir,
                    "agent_command": resolved_cmd,
                    "total_duration_ms": total_ms,
                    "setup_results": setup_results,
                },
            )

        except Exception as e:
            self._record_tool_call(
                input,
                tool_name="environment.agent",
                tool_type="environment",
                input={
                    "command": agent_cmd,
                    "environment_dir": env_dir,
                },
                status="error",
                duration_ms=0.0,
                error={"type": type(e).__name__, "message": str(e)},
            )
            return AgentOutput(error=str(e))
