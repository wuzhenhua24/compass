"""Tests for the sandbox module."""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

from compass.core.artifacts import ExecutionResult, GeneratedFile
from compass.sandbox.base import Sandbox
from compass.sandbox.local import LocalSandbox
from compass.sandbox.registry import (
    _sandbox_registry,
    get_sandbox_class,
    list_sandboxes,
    register_sandbox,
    unregister_sandbox,
)


# ===================================================================
# Registry tests
# ===================================================================


class TestSandboxRegistry:
    def test_local_sandbox_registered(self):
        """LocalSandbox is registered at import time."""
        assert "local" in _sandbox_registry

    def test_get_sandbox_class(self):
        assert get_sandbox_class("local") is LocalSandbox

    def test_unknown_type_raises(self):
        with pytest.raises(KeyError, match="not found"):
            get_sandbox_class("nonexistent_sandbox")

    def test_register_custom_type(self):
        @register_sandbox("test_custom")
        class CustomSandbox(Sandbox):
            @property
            def workdir(self) -> str:
                return ""

            async def setup(self) -> None: ...
            async def exec(self, command: str, **kw: Any) -> ExecutionResult:
                return ExecutionResult()
            async def write_file(self, path: str, content: str) -> None: ...
            async def read_file(self, path: str) -> str:
                return ""
            async def copy_in(self, src: str, dest: str) -> None: ...
            async def collect_files(self, pattern: str = "**/*") -> list[GeneratedFile]:
                return []
            async def cleanup(self) -> None: ...

        try:
            assert get_sandbox_class("test_custom") is CustomSandbox
            assert "test_custom" in list_sandboxes()
        finally:
            unregister_sandbox("test_custom")

    def test_unregister(self):
        _sandbox_registry["_to_remove"] = LocalSandbox  # type: ignore[assignment]
        unregister_sandbox("_to_remove")
        assert "_to_remove" not in _sandbox_registry

    def test_list_sandboxes(self):
        names = list_sandboxes()
        assert "local" in names


# ===================================================================
# Lifecycle tests
# ===================================================================


class TestLocalSandboxLifecycle:
    @pytest.mark.asyncio
    async def test_setup_creates_directory(self):
        sb = LocalSandbox()
        await sb.setup()
        try:
            assert os.path.isdir(sb.workdir)
            assert "compass_sandbox_" in sb.workdir
        finally:
            await sb.cleanup()

    @pytest.mark.asyncio
    async def test_cleanup_removes_directory(self):
        sb = LocalSandbox()
        await sb.setup()
        wd = sb.workdir
        await sb.cleanup()
        assert not os.path.exists(wd)

    @pytest.mark.asyncio
    async def test_cleanup_idempotent(self):
        sb = LocalSandbox()
        await sb.setup()
        await sb.cleanup()
        # Second cleanup should not raise
        await sb.cleanup()

    @pytest.mark.asyncio
    async def test_custom_base_dir(self, tmp_path):
        sb = LocalSandbox(config={"base_dir": str(tmp_path)})
        await sb.setup()
        try:
            assert sb.workdir.startswith(str(tmp_path))
        finally:
            await sb.cleanup()

    def test_workdir_before_setup_raises(self):
        sb = LocalSandbox()
        with pytest.raises(RuntimeError, match="not set up"):
            _ = sb.workdir


# ===================================================================
# Context manager tests
# ===================================================================


class TestLocalSandboxContextManager:
    @pytest.mark.asyncio
    async def test_context_manager_normal(self):
        async with LocalSandbox() as sb:
            wd = sb.workdir
            assert os.path.isdir(wd)
        assert not os.path.exists(wd)

    @pytest.mark.asyncio
    async def test_context_manager_cleans_up_on_error(self):
        wd = None
        with pytest.raises(ValueError, match="boom"):
            async with LocalSandbox() as sb:
                wd = sb.workdir
                raise ValueError("boom")
        assert wd is not None
        assert not os.path.exists(wd)


# ===================================================================
# Exec tests
# ===================================================================


class TestLocalSandboxExec:
    @pytest.mark.asyncio
    async def test_simple_command(self):
        async with LocalSandbox() as sb:
            result = await sb.exec("echo hello")
            assert result.stdout.strip() == "hello"
            assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_stderr_captured(self):
        async with LocalSandbox() as sb:
            result = await sb.exec("echo error >&2")
            assert "error" in result.stderr

    @pytest.mark.asyncio
    async def test_nonzero_exit_code(self):
        async with LocalSandbox() as sb:
            result = await sb.exec("exit 42")
            assert result.exit_code == 42

    @pytest.mark.asyncio
    async def test_timeout_kills_process(self):
        async with LocalSandbox() as sb:
            result = await sb.exec("sleep 60", timeout=0.5)
            assert result.exit_code == -1
            assert "timed out" in result.stderr.lower()

    @pytest.mark.asyncio
    async def test_returns_execution_result(self):
        async with LocalSandbox() as sb:
            result = await sb.exec("echo test")
            assert isinstance(result, ExecutionResult)
            assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_per_call_env(self):
        async with LocalSandbox() as sb:
            result = await sb.exec("echo $MY_VAR", env={"MY_VAR": "custom_value"})
            assert result.stdout.strip() == "custom_value"

    @pytest.mark.asyncio
    async def test_config_default_timeout(self):
        async with LocalSandbox(config={"default_timeout": 0.5}) as sb:
            result = await sb.exec("sleep 60")
            assert result.exit_code == -1

    @pytest.mark.asyncio
    async def test_exec_in_subdirectory(self):
        async with LocalSandbox() as sb:
            subdir = "sub"
            os.makedirs(os.path.join(sb.workdir, subdir))
            result = await sb.exec("pwd", workdir=subdir)
            assert result.stdout.strip().endswith(subdir)


# ===================================================================
# File operation tests
# ===================================================================


class TestLocalSandboxFileOps:
    @pytest.mark.asyncio
    async def test_write_read_roundtrip(self):
        async with LocalSandbox() as sb:
            await sb.write_file("test.txt", "hello world")
            content = await sb.read_file("test.txt")
            assert content == "hello world"

    @pytest.mark.asyncio
    async def test_auto_create_parent_dirs(self):
        async with LocalSandbox() as sb:
            await sb.write_file("a/b/c/deep.txt", "deep content")
            content = await sb.read_file("a/b/c/deep.txt")
            assert content == "deep content"

    @pytest.mark.asyncio
    async def test_read_nonexistent_raises(self):
        async with LocalSandbox() as sb:
            with pytest.raises(FileNotFoundError):
                await sb.read_file("no_such_file.txt")

    @pytest.mark.asyncio
    async def test_copy_in_file(self, tmp_path):
        src = tmp_path / "source.txt"
        src.write_text("copied content")
        async with LocalSandbox() as sb:
            await sb.copy_in(str(src), "dest.txt")
            content = await sb.read_file("dest.txt")
            assert content == "copied content"

    @pytest.mark.asyncio
    async def test_copy_in_directory(self, tmp_path):
        src_dir = tmp_path / "src_dir"
        src_dir.mkdir()
        (src_dir / "a.txt").write_text("aaa")
        (src_dir / "b.txt").write_text("bbb")
        async with LocalSandbox() as sb:
            await sb.copy_in(str(src_dir), "imported")
            assert await sb.read_file("imported/a.txt") == "aaa"
            assert await sb.read_file("imported/b.txt") == "bbb"


# ===================================================================
# Collect files tests
# ===================================================================


class TestLocalSandboxCollectFiles:
    @pytest.mark.asyncio
    async def test_collect_all(self):
        async with LocalSandbox() as sb:
            await sb.write_file("a.py", "x=1")
            await sb.write_file("b.js", "var y=2")
            files = await sb.collect_files()
            paths = {f.path for f in files}
            assert "a.py" in paths
            assert "b.js" in paths

    @pytest.mark.asyncio
    async def test_glob_filter(self):
        async with LocalSandbox() as sb:
            await sb.write_file("code.py", "pass")
            await sb.write_file("data.json", "{}")
            files = await sb.collect_files("*.py")
            assert len(files) == 1
            assert files[0].path == "code.py"

    @pytest.mark.asyncio
    async def test_language_detection(self):
        async with LocalSandbox() as sb:
            await sb.write_file("main.py", "print(1)")
            await sb.write_file("app.js", "console.log(1)")
            await sb.write_file("style.css", "body {}")
            await sb.write_file("unknown.xyz", "stuff")
            files = await sb.collect_files()
            lang_map = {f.path: f.language for f in files}
            assert lang_map["main.py"] == "python"
            assert lang_map["app.js"] == "javascript"
            assert lang_map["style.css"] == "css"
            assert lang_map["unknown.xyz"] == ""

    @pytest.mark.asyncio
    async def test_returns_generated_file_type(self):
        async with LocalSandbox() as sb:
            await sb.write_file("f.py", "pass")
            files = await sb.collect_files()
            assert all(isinstance(f, GeneratedFile) for f in files)

    @pytest.mark.asyncio
    async def test_recursive_pattern(self):
        async with LocalSandbox() as sb:
            await sb.write_file("top.py", "1")
            await sb.write_file("sub/deep.py", "2")
            files = await sb.collect_files("**/*.py")
            paths = {f.path for f in files}
            assert "top.py" in paths
            assert "sub/deep.py" in paths


# ===================================================================
# Environment isolation tests
# ===================================================================


class TestLocalSandboxEnvironmentIsolation:
    @pytest.mark.asyncio
    async def test_sensitive_vars_not_leaked(self):
        os.environ["AWS_SECRET_ACCESS_KEY"] = "supersecret"
        try:
            async with LocalSandbox() as sb:
                result = await sb.exec("env")
                assert "supersecret" not in result.stdout
                assert "AWS_SECRET_ACCESS_KEY" not in result.stdout
        finally:
            del os.environ["AWS_SECRET_ACCESS_KEY"]

    @pytest.mark.asyncio
    async def test_path_available(self):
        async with LocalSandbox() as sb:
            result = await sb.exec("echo $PATH")
            assert result.stdout.strip() != ""

    @pytest.mark.asyncio
    async def test_home_points_to_sandbox(self):
        async with LocalSandbox() as sb:
            result = await sb.exec("echo $HOME")
            assert result.stdout.strip() == sb.workdir

    @pytest.mark.asyncio
    async def test_env_overrides(self):
        async with LocalSandbox(config={"env_overrides": {"CUSTOM": "val"}}) as sb:
            result = await sb.exec("echo $CUSTOM")
            assert result.stdout.strip() == "val"

    @pytest.mark.asyncio
    async def test_exec_env_sensitive_blocked(self):
        async with LocalSandbox() as sb:
            result = await sb.exec(
                "echo $AWS_SECRET_KEY",
                env={"AWS_SECRET_KEY": "should_be_blocked"},
            )
            assert "should_be_blocked" not in result.stdout


# ===================================================================
# Concurrency tests
# ===================================================================


class TestLocalSandboxConcurrency:
    @pytest.mark.asyncio
    async def test_multiple_sandboxes_independent(self):
        async with LocalSandbox() as sb1, LocalSandbox() as sb2:
            assert sb1.workdir != sb2.workdir
            await sb1.write_file("only_in_1.txt", "one")
            await sb2.write_file("only_in_2.txt", "two")
            with pytest.raises(FileNotFoundError):
                await sb1.read_file("only_in_2.txt")
            with pytest.raises(FileNotFoundError):
                await sb2.read_file("only_in_1.txt")

    @pytest.mark.asyncio
    async def test_concurrent_exec(self):
        async with LocalSandbox() as sb:
            results = await asyncio.gather(
                sb.exec("echo a"),
                sb.exec("echo b"),
                sb.exec("echo c"),
            )
            outputs = {r.stdout.strip() for r in results}
            assert outputs == {"a", "b", "c"}
            assert all(r.exit_code == 0 for r in results)
