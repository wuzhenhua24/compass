"""Tests for the EnvironmentAdapter."""

from __future__ import annotations

import os
import tempfile

import pytest

from compass.adapters.base import AgentInput
from compass.adapters.environment import EnvironmentAdapter
from compass.adapters.registry import _adapter_registry, get_adapter
from compass.core.artifacts import CodeArtifact, ExecutionResult

# ===================================================================
# Helpers
# ===================================================================


def _make_env_dir(files: dict[str, str] | None = None) -> str:
    """Create a temporary environment directory with given files."""
    tmpdir = tempfile.mkdtemp(prefix="compass_env_test_")
    if files:
        for name, content in files.items():
            path = os.path.join(tmpdir, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
    return tmpdir


# ===================================================================
# Registry tests
# ===================================================================


class TestEnvironmentAdapterRegistry:
    def test_registered(self):
        assert "environment" in _adapter_registry

    def test_get_adapter_returns_class(self):
        assert get_adapter("environment") is EnvironmentAdapter

    def test_name_attribute(self):
        assert EnvironmentAdapter.name == "environment"


# ===================================================================
# Config validation tests
# ===================================================================


class TestEnvironmentAdapterValidation:
    def test_validate_config_missing_both(self):
        adapter = EnvironmentAdapter({})
        assert adapter.validate_config() is False

    def test_validate_config_missing_agent_command(self):
        adapter = EnvironmentAdapter({"environment_dir": "/tmp"})
        assert adapter.validate_config() is False

    def test_validate_config_missing_environment_dir(self):
        adapter = EnvironmentAdapter({"agent_command": "echo hi"})
        assert adapter.validate_config() is False

    def test_validate_config_valid(self):
        adapter = EnvironmentAdapter({
            "environment_dir": "/tmp",
            "agent_command": "echo hi",
        })
        assert adapter.validate_config() is True


# ===================================================================
# Run tests
# ===================================================================


class TestEnvironmentAdapterRun:
    @pytest.mark.asyncio
    async def test_missing_env_dir_config(self):
        adapter = EnvironmentAdapter({"agent_command": "echo hi"})
        output = await adapter.run(AgentInput(prompt="test"))
        assert output.error is not None
        assert "environment_dir" in output.error.lower()

    @pytest.mark.asyncio
    async def test_missing_agent_command_config(self):
        env_dir = _make_env_dir()
        adapter = EnvironmentAdapter({"environment_dir": env_dir})
        output = await adapter.run(AgentInput(prompt="test"))
        assert output.error is not None
        assert "agent_command" in output.error.lower()

    @pytest.mark.asyncio
    async def test_nonexistent_env_dir(self):
        adapter = EnvironmentAdapter({
            "environment_dir": "/nonexistent/path",
            "agent_command": "echo hi",
        })
        output = await adapter.run(AgentInput(prompt="test"))
        assert output.error is not None
        assert "not found" in output.error.lower()

    @pytest.mark.asyncio
    async def test_basic_agent_execution(self):
        env_dir = _make_env_dir({"hello.txt": "world"})
        adapter = EnvironmentAdapter({
            "environment_dir": env_dir,
            "agent_command": "echo 'agent done'",
            "problem_file": "",  # no problem file
        })
        output = await adapter.run(AgentInput(prompt="do something"))
        assert output.error is None
        assert len(output.artifacts) == 1
        artifact = output.artifacts[0]
        assert isinstance(artifact, CodeArtifact)
        assert artifact.execution is not None
        assert artifact.execution.exit_code == 0
        assert "agent done" in artifact.execution.stdout

    @pytest.mark.asyncio
    async def test_environment_files_copied(self):
        env_dir = _make_env_dir({
            "server.js": "console.log('server')",
            "package.json": '{"name": "test"}',
        })
        adapter = EnvironmentAdapter({
            "environment_dir": env_dir,
            "agent_command": "cat server.js",
            "problem_file": "",
        })
        output = await adapter.run(AgentInput(prompt=""))
        artifact = output.artifacts[0]
        assert isinstance(artifact, CodeArtifact)
        assert "console.log" in artifact.execution.stdout

    @pytest.mark.asyncio
    async def test_problem_file_read(self):
        env_dir = _make_env_dir({
            "PROBLEM.md": "# Task\nBuild a server",
        })
        adapter = EnvironmentAdapter({
            "environment_dir": env_dir,
            "agent_command": "cat .compass_prompt.txt",
        })
        output = await adapter.run(AgentInput(prompt="Extra instructions"))
        artifact = output.artifacts[0]
        assert isinstance(artifact, CodeArtifact)
        # Prompt file should contain both the input prompt and problem file
        assert "Extra instructions" in artifact.execution.stdout
        assert "Build a server" in artifact.execution.stdout

    @pytest.mark.asyncio
    async def test_problem_file_missing_is_ok(self):
        env_dir = _make_env_dir({})  # no PROBLEM.md
        adapter = EnvironmentAdapter({
            "environment_dir": env_dir,
            "agent_command": "echo ok",
        })
        output = await adapter.run(AgentInput(prompt="test"))
        assert output.error is None

    @pytest.mark.asyncio
    async def test_setup_commands_run(self):
        env_dir = _make_env_dir({})
        adapter = EnvironmentAdapter({
            "environment_dir": env_dir,
            "agent_command": "cat setup_marker.txt",
            "problem_file": "",
            "setup_commands": [
                "echo 'setup-done' > setup_marker.txt",
            ],
        })
        output = await adapter.run(AgentInput(prompt=""))
        artifact = output.artifacts[0]
        assert isinstance(artifact, CodeArtifact)
        assert "setup-done" in artifact.execution.stdout

    @pytest.mark.asyncio
    async def test_setup_command_failure_stops_execution(self):
        env_dir = _make_env_dir({})
        adapter = EnvironmentAdapter({
            "environment_dir": env_dir,
            "agent_command": "echo should-not-run",
            "problem_file": "",
            "setup_commands": ["exit 1"],
        })
        output = await adapter.run(AgentInput(prompt=""))
        assert output.error is not None
        assert "setup command failed" in output.error.lower()
        assert output.metadata.get("phase") == "setup"

    @pytest.mark.asyncio
    async def test_agent_nonzero_exit_code(self):
        env_dir = _make_env_dir({})
        adapter = EnvironmentAdapter({
            "environment_dir": env_dir,
            "agent_command": "exit 42",
            "problem_file": "",
        })
        output = await adapter.run(AgentInput(prompt=""))
        assert output.error is None  # non-zero is not an error, just a result
        artifact = output.artifacts[0]
        assert isinstance(artifact, CodeArtifact)
        assert artifact.execution.exit_code == 42

    @pytest.mark.asyncio
    async def test_prompt_placeholder_substitution(self):
        env_dir = _make_env_dir({})
        adapter = EnvironmentAdapter({
            "environment_dir": env_dir,
            "agent_command": "echo '{prompt}'",
            "problem_file": "",
        })
        output = await adapter.run(AgentInput(prompt="hello world"))
        artifact = output.artifacts[0]
        assert isinstance(artifact, CodeArtifact)
        assert "hello world" in artifact.execution.stdout

    @pytest.mark.asyncio
    async def test_prompt_file_placeholder_substitution(self):
        env_dir = _make_env_dir({})
        adapter = EnvironmentAdapter({
            "environment_dir": env_dir,
            "agent_command": "cat {prompt_file}",
            "problem_file": "",
        })
        output = await adapter.run(AgentInput(prompt="file content test"))
        artifact = output.artifacts[0]
        assert isinstance(artifact, CodeArtifact)
        assert "file content test" in artifact.execution.stdout

    @pytest.mark.asyncio
    async def test_env_vars_passed(self):
        env_dir = _make_env_dir({})
        adapter = EnvironmentAdapter({
            "environment_dir": env_dir,
            "agent_command": "echo $MY_VAR",
            "problem_file": "",
            "env": {"MY_VAR": "secret_value"},
        })
        output = await adapter.run(AgentInput(prompt=""))
        artifact = output.artifacts[0]
        assert isinstance(artifact, CodeArtifact)
        assert "secret_value" in artifact.execution.stdout

    @pytest.mark.asyncio
    async def test_teardown_commands_run(self):
        env_dir = _make_env_dir({})
        adapter = EnvironmentAdapter({
            "environment_dir": env_dir,
            "agent_command": "echo agent",
            "problem_file": "",
            "teardown_commands": [
                "echo 'teardown' > teardown_marker.txt",
            ],
        })
        output = await adapter.run(AgentInput(prompt=""))
        # Teardown should run after agent but we can verify no error
        assert output.error is None

    @pytest.mark.asyncio
    async def test_per_case_override_env_dir(self):
        env_dir = _make_env_dir({"case_file.txt": "case content"})
        adapter = EnvironmentAdapter({
            "environment_dir": "/nonexistent",  # default is wrong
            "agent_command": "cat case_file.txt",
            "problem_file": "",
        })
        output = await adapter.run(AgentInput(
            prompt="",
            params={"environment_dir": env_dir},  # override
        ))
        assert output.error is None
        artifact = output.artifacts[0]
        assert "case content" in artifact.execution.stdout

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        env_dir = _make_env_dir({})
        adapter = EnvironmentAdapter({
            "environment_dir": env_dir,
            "agent_command": "sleep 60",
            "problem_file": "",
            "timeout": 0.5,
        })
        output = await adapter.run(AgentInput(prompt=""))
        assert output.error is None
        artifact = output.artifacts[0]
        assert isinstance(artifact, CodeArtifact)
        assert artifact.execution.exit_code == -1

    @pytest.mark.asyncio
    async def test_metadata_populated(self):
        env_dir = _make_env_dir({})
        adapter = EnvironmentAdapter({
            "environment_dir": env_dir,
            "agent_command": "echo ok",
            "problem_file": "",
        })
        output = await adapter.run(AgentInput(prompt=""))
        assert output.metadata.get("environment_dir") == env_dir
        assert "total_duration_ms" in output.metadata

    @pytest.mark.asyncio
    async def test_tool_call_recording(self):
        env_dir = _make_env_dir({})
        tool_calls: list = []
        adapter = EnvironmentAdapter({
            "environment_dir": env_dir,
            "agent_command": "echo recorded",
            "problem_file": "",
            "setup_commands": ["echo setup"],
        })
        output = await adapter.run(AgentInput(
            prompt="",
            context={"tool_calls": tool_calls},
        ))
        assert output.error is None
        # Should have at least 2 tool calls: setup + agent
        assert len(tool_calls) >= 2
        tool_names = [tc.tool_name for tc in tool_calls]
        assert "environment.setup" in tool_names
        assert "environment.agent" in tool_names


# ===================================================================
# Artifact tests
# ===================================================================


class TestEnvironmentAdapterArtifacts:
    @pytest.mark.asyncio
    async def test_returns_code_artifact(self):
        env_dir = _make_env_dir({"data.txt": "hello"})
        adapter = EnvironmentAdapter({
            "environment_dir": env_dir,
            "agent_command": "echo done",
            "problem_file": "",
        })
        output = await adapter.run(AgentInput(prompt=""))
        assert len(output.artifacts) == 1
        assert isinstance(output.artifacts[0], CodeArtifact)

    @pytest.mark.asyncio
    async def test_execution_result_populated(self):
        env_dir = _make_env_dir({})
        adapter = EnvironmentAdapter({
            "environment_dir": env_dir,
            "agent_command": "echo stdout_data",
            "problem_file": "",
        })
        output = await adapter.run(AgentInput(prompt=""))
        artifact = output.artifacts[0]
        assert isinstance(artifact.execution, ExecutionResult)
        assert artifact.execution.duration_ms > 0
        assert "stdout_data" in artifact.execution.stdout

    @pytest.mark.asyncio
    async def test_collected_files(self):
        env_dir = _make_env_dir({
            "server/app.js": "// app",
            "server/package.json": "{}",
        })
        adapter = EnvironmentAdapter({
            "environment_dir": env_dir,
            "agent_command": "echo done",
            "problem_file": "",
        })
        output = await adapter.run(AgentInput(prompt=""))
        artifact = output.artifacts[0]
        paths = {f.path for f in artifact.files}
        assert "server/app.js" in paths
        assert "server/package.json" in paths
