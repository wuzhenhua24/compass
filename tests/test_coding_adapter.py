"""Tests for the CodingAdapter."""

from __future__ import annotations

import pytest

from compass.adapters.base import AgentInput, AgentOutput
from compass.adapters.coding import CodingAdapter
from compass.adapters.registry import get_adapter, _adapter_registry
from compass.core.artifacts import CodeArtifact, ExecutionResult, GeneratedFile


# ===================================================================
# Registry tests
# ===================================================================


class TestCodingAdapterRegistry:
    def test_registered_as_coding(self):
        assert "coding" in _adapter_registry

    def test_get_adapter_returns_class(self):
        assert get_adapter("coding") is CodingAdapter

    def test_name_attribute(self):
        assert CodingAdapter.name == "coding"


# ===================================================================
# Run tests
# ===================================================================


class TestCodingAdapterRun:
    @pytest.mark.asyncio
    async def test_write_files_and_execute(self):
        adapter = CodingAdapter({"run_command": "python main.py"})
        input = AgentInput(
            prompt="run code",
            params={
                "files": [
                    {"path": "main.py", "content": "print('hello')"},
                ],
            },
        )
        output = await adapter.run(input)

        assert output.error is None
        assert len(output.artifacts) == 1
        artifact = output.artifacts[0]
        assert isinstance(artifact, CodeArtifact)
        assert artifact.execution is not None
        assert artifact.execution.exit_code == 0
        assert "hello" in artifact.execution.stdout

    @pytest.mark.asyncio
    async def test_stdout_and_stderr_captured(self):
        adapter = CodingAdapter()
        input = AgentInput(
            prompt="",
            params={
                "files": [
                    {
                        "path": "main.py",
                        "content": (
                            "import sys\n"
                            "print('out')\n"
                            "print('err', file=sys.stderr)\n"
                        ),
                    },
                ],
            },
        )
        output = await adapter.run(input)
        artifact = output.artifacts[0]
        assert isinstance(artifact, CodeArtifact)
        assert "out" in artifact.execution.stdout
        assert "err" in artifact.execution.stderr

    @pytest.mark.asyncio
    async def test_nonzero_exit_code(self):
        adapter = CodingAdapter({"run_command": "python main.py"})
        input = AgentInput(
            prompt="",
            params={
                "files": [
                    {"path": "main.py", "content": "raise SystemExit(42)"},
                ],
            },
        )
        output = await adapter.run(input)
        artifact = output.artifacts[0]
        assert isinstance(artifact, CodeArtifact)
        assert artifact.execution.exit_code == 42

    @pytest.mark.asyncio
    async def test_custom_run_command_in_params(self):
        adapter = CodingAdapter()
        input = AgentInput(
            prompt="",
            params={
                "files": [
                    {"path": "script.sh", "content": "echo custom"},
                ],
                "run_command": "bash script.sh",
            },
        )
        output = await adapter.run(input)
        artifact = output.artifacts[0]
        assert isinstance(artifact, CodeArtifact)
        assert artifact.execution.exit_code == 0
        assert "custom" in artifact.execution.stdout

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        adapter = CodingAdapter({"timeout": 0.5, "run_command": "sleep 60"})
        input = AgentInput(
            prompt="",
            params={"files": [{"path": "main.py", "content": ""}]},
        )
        output = await adapter.run(input)
        artifact = output.artifacts[0]
        assert isinstance(artifact, CodeArtifact)
        assert artifact.execution.exit_code == -1
        assert "timed out" in artifact.execution.stderr.lower()

    @pytest.mark.asyncio
    async def test_no_files_runs_gracefully(self):
        adapter = CodingAdapter({"run_command": "echo nofiles"})
        input = AgentInput(prompt="", params={})
        output = await adapter.run(input)
        assert output.error is None
        artifact = output.artifacts[0]
        assert isinstance(artifact, CodeArtifact)
        assert artifact.execution.exit_code == 0


# ===================================================================
# Artifact tests
# ===================================================================


class TestCodingAdapterArtifacts:
    @pytest.mark.asyncio
    async def test_returns_code_artifact(self):
        adapter = CodingAdapter({"run_command": "echo done"})
        input = AgentInput(
            prompt="",
            params={
                "files": [{"path": "a.py", "content": "x = 1"}],
            },
        )
        output = await adapter.run(input)
        assert len(output.artifacts) == 1
        assert isinstance(output.artifacts[0], CodeArtifact)

    @pytest.mark.asyncio
    async def test_execution_field_correct(self):
        adapter = CodingAdapter({"run_command": "echo ok"})
        input = AgentInput(
            prompt="",
            params={"files": [{"path": "f.txt", "content": "data"}]},
        )
        output = await adapter.run(input)
        artifact = output.artifacts[0]
        assert isinstance(artifact, CodeArtifact)
        assert isinstance(artifact.execution, ExecutionResult)
        assert artifact.execution.duration_ms > 0

    @pytest.mark.asyncio
    async def test_files_field_correct(self):
        adapter = CodingAdapter({"run_command": "echo done"})
        input = AgentInput(
            prompt="",
            params={
                "files": [
                    {"path": "a.py", "content": "x = 1"},
                    {"path": "sub/b.py", "content": "y = 2"},
                ],
            },
        )
        output = await adapter.run(input)
        artifact = output.artifacts[0]
        assert isinstance(artifact, CodeArtifact)
        paths = {f.path for f in artifact.files}
        assert "a.py" in paths
        assert "sub/b.py" in paths
        assert all(isinstance(f, GeneratedFile) for f in artifact.files)

    @pytest.mark.asyncio
    async def test_has_output_property(self):
        adapter = CodingAdapter({"run_command": "echo x"})
        input = AgentInput(
            prompt="",
            params={"files": [{"path": "f.py", "content": "pass"}]},
        )
        output = await adapter.run(input)
        assert output.has_output
