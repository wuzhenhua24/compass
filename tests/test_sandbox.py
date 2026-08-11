"""Tests for the sandbox module."""

from __future__ import annotations

import asyncio
import logging
import os
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


# ===================================================================
# Environment variable blocking
# ===================================================================


class TestEnvBlocking:
    """The secret-blocking safety net must catch suffix-style names too."""

    def test_prefix_style_secrets_blocked(self):
        from compass.sandbox.local import _is_blocked

        for name in ["AWS_SECRET_ACCESS_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
            assert _is_blocked(name) is True

    def test_suffix_style_secrets_blocked(self):
        """Regression: prefix-only matching missed the common *_API_KEY /
        *_SECRET / *_PASSWORD / *_TOKEN convention."""
        from compass.sandbox.local import _is_blocked

        for name in [
            "MYSVC_API_KEY",
            "DB_PASSWORD",
            "GCP_SERVICE_SECRET",
            "USER_TOKEN",
            "SERVICE_ACCESS_TOKEN",
        ]:
            assert _is_blocked(name) is True, name

    def test_benign_vars_not_blocked(self):
        from compass.sandbox.local import _is_blocked

        for name in ["PATH", "LANG", "HOME", "TERM", "MY_VAR"]:
            assert _is_blocked(name) is False, name

    @pytest.mark.asyncio
    async def test_passthrough_secret_is_filtered(self):
        """A secret named in env_passthrough must NOT reach the sandbox env."""
        os.environ["MYSVC_API_KEY"] = "leaked-value"
        try:
            sb = LocalSandbox(config={"env_passthrough": ["MYSVC_API_KEY"]})
            async with sb:
                env = sb._build_env()
                assert "MYSVC_API_KEY" not in env
        finally:
            os.environ.pop("MYSVC_API_KEY", None)


class TestEnvAllowlist:
    """env_allow is the escape hatch for running a *credentialed agent* under test.

    Without it, an agent like ``claude`` cannot authenticate at all:
    ANTHROPIC_API_KEY trips both the prefix rule and the substring rule, on
    every route into the sandbox.
    """

    def test_allowlisted_name_is_not_blocked(self):
        from compass.sandbox.local import _is_blocked

        allow = frozenset({"ANTHROPIC_API_KEY"})
        assert _is_blocked("ANTHROPIC_API_KEY", allow) is False
        # Still blocked: allowing one credential must not allow its neighbours.
        assert _is_blocked("ANTHROPIC_AUTH_TOKEN", allow) is True
        assert _is_blocked("AWS_SECRET_ACCESS_KEY", allow) is True

    def test_allowlist_is_case_insensitive_on_names_only(self):
        from compass.sandbox.local import _is_blocked

        allow = frozenset({"MYSVC_API_KEY"})
        assert _is_blocked("mysvc_api_key", allow) is False
        # No globbing — a prefix of an allowed name is not itself allowed.
        assert _is_blocked("MYSVC_API_KEY_2", allow) is True

    @pytest.mark.asyncio
    async def test_allowlisted_secret_reaches_the_sandbox(self):
        os.environ["ANTHROPIC_API_KEY"] = "sk-test-value"
        try:
            sb = LocalSandbox(config={"env_allow": ["ANTHROPIC_API_KEY"]})
            async with sb:
                env = sb._build_env()
                assert env["ANTHROPIC_API_KEY"] == "sk-test-value"
                result = await sb.exec("echo $ANTHROPIC_API_KEY")
                assert "sk-test-value" in result.stdout
        finally:
            os.environ.pop("ANTHROPIC_API_KEY", None)

    @pytest.mark.asyncio
    async def test_allowlist_does_not_widen_to_other_secrets(self):
        os.environ["ANTHROPIC_API_KEY"] = "sk-test-value"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "should-not-leak"
        try:
            sb = LocalSandbox(
                config={
                    "env_allow": ["ANTHROPIC_API_KEY"],
                    "env_passthrough": ["AWS_SECRET_ACCESS_KEY"],
                    "env_overrides": {"DB_PASSWORD": "nope"},
                }
            )
            async with sb:
                env = sb._build_env(extra={"OPENAI_API_KEY": "nope"})
                assert env["ANTHROPIC_API_KEY"] == "sk-test-value"
                assert "AWS_SECRET_ACCESS_KEY" not in env
                assert "DB_PASSWORD" not in env
                assert "OPENAI_API_KEY" not in env
        finally:
            os.environ.pop("ANTHROPIC_API_KEY", None)
            os.environ.pop("AWS_SECRET_ACCESS_KEY", None)

    @pytest.mark.asyncio
    async def test_allowlist_applies_to_overrides_and_per_call_env(self):
        sb = LocalSandbox(
            config={
                "env_allow": ["ANTHROPIC_API_KEY", "MYSVC_TOKEN"],
                "env_overrides": {"ANTHROPIC_API_KEY": "from-config"},
            }
        )
        async with sb:
            env = sb._build_env(extra={"MYSVC_TOKEN": "from-call"})
            assert env["ANTHROPIC_API_KEY"] == "from-config"
            assert env["MYSVC_TOKEN"] == "from-call"

    @pytest.mark.asyncio
    async def test_home_is_the_sandbox_by_default(self):
        sb = LocalSandbox(config={})
        async with sb:
            assert sb._build_env()["HOME"] == sb.workdir

    @pytest.mark.asyncio
    async def test_preserve_home_keeps_the_host_home(self):
        """Subscription-auth agents read credentials from the real ~."""
        sb = LocalSandbox(config={"preserve_home": True})
        async with sb:
            assert sb._build_env()["HOME"] == os.environ["HOME"]

    @pytest.mark.asyncio
    async def test_allowlist_is_logged(self, caplog):
        with caplog.at_level(logging.WARNING, logger="compass.sandbox.local"):
            async with LocalSandbox(config={"env_allow": ["ANTHROPIC_API_KEY"]}):
                pass
        assert "ANTHROPIC_API_KEY" in caplog.text


class TestCopyInDirectory:
    @pytest.mark.asyncio
    async def test_copy_dir_into_sandbox_root(self, tmp_path):
        """Regression: copy_in(dir, ".") raised FileExistsError, which made
        IntegrationGrader's documented `workdir` config unusable."""
        src = tmp_path / "tree"
        (src / "pkg").mkdir(parents=True)
        (src / "pkg" / "mod.py").write_text("x = 1\n", encoding="utf-8")
        (src / "README.md").write_text("hi\n", encoding="utf-8")

        async with LocalSandbox() as sb:
            await sb.copy_in(str(src), ".")
            assert await sb.read_file("README.md") == "hi\n"
            assert await sb.read_file("pkg/mod.py") == "x = 1\n"

    @pytest.mark.asyncio
    async def test_copy_dir_into_subdirectory(self, tmp_path):
        src = tmp_path / "tree"
        src.mkdir()
        (src / "mod.py").write_text("x = 1\n", encoding="utf-8")

        async with LocalSandbox() as sb:
            await sb.copy_in(str(src), "vendor")
            assert await sb.read_file("vendor/mod.py") == "x = 1\n"
