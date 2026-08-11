"""The SWE-bench correctness protocol, as a Compass grader.

SWE-bench does not score a patch by looking at it. It scores it by running two
disjoint sets of tests against the tree the agent produced:

    FAIL_TO_PASS   the tests the issue is about — red before, must be green after
    PASS_TO_PASS   everything else that already worked — must still be green

An instance counts as **resolved** only if every test in both sets passes. That
binary is the number the leaderboards report, and it is what this grader puts in
``passed``. ``score`` carries the proportion instead, which is useless as a
headline and genuinely useful when you are staring at a failure: "23 of 24" and
"0 of 24" are the same red light and very different problems.

Three details of the protocol are easy to get wrong, and getting them wrong
inflates your numbers:

1. **The agent must never see the tests.** ``test_patch`` is applied *after* the
   agent has finished. It is carried in case metadata, never in the prompt.
2. **Test files are reset first.** An agent that edited a test file would
   otherwise have its edits merged with the official tests. SWE-bench checks the
   test paths back out at the base commit before applying the patch; so does
   this grader.
3. **A test that did not run is a failure.** If a node id is missing from the
   report — collection error, import failure, typo — it counts as failed, not
   as absent. Anything else quietly turns a broken environment into a pass.

This lives in ``examples/`` on purpose. Compass's core carries the reusable
spine (transcript model, process graders, metrics); *correctness* is
domain-specific and plugs in as user code. A benchmark's scoring protocol is
about as domain-specific as it gets.
"""

from __future__ import annotations

import asyncio
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from compass.graders import (
    CodeGrader,
    GradeContext,
    GradeResult,
    GraderScope,
    register_grader,
)

# `pytest -rA` short-summary lines: "PASSED tests/test_x.py::test_y".
_PYTEST_LINE = re.compile(
    r"^(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\s+(\S+)", re.MULTILINE
)
_PASSING = frozenset({"PASSED", "XFAIL"})

# ``{python}`` resolves to the interpreter running Compass for a local run, and
# to plain ``python`` inside a container. Bare ``python`` is not safe as a
# default: on plenty of systems (and inside a uv/venv-managed checkout) only
# ``python3`` exists, and the command would fail with no tests run — which this
# grader correctly, and unhelpfully, scores as zero.
DEFAULT_TEST_COMMAND = "{python} -m pytest -rA --tb=no -p no:cacheprovider {tests}"


@register_grader("swebench_tests")
class SweBenchTestsGrader(CodeGrader):
    """Run an instance's FAIL_TO_PASS / PASS_TO_PASS sets against the agent's tree.

    Scope: OUTCOME — it grades the workspace the agent produced.

    Reads from ``context.code_artifact``:
        ``metadata["workspace"]``  the per-trial worktree (the ``claude_code``
                                   adapter records it; that is what makes this
                                   grader adapter-agnostic).

    Reads from the case metadata (``context.metadata``):
        ``test_patch``   unified diff adding the official tests.
        ``fail_to_pass`` list of test node ids that must go red -> green.
        ``pass_to_pass`` list of test node ids that must stay green.
        ``base_commit``  commit to reset test files to before patching.

    Config:
        test_command:  str  — ``{tests}`` is replaced with the space-joined node
                       ids (default: pytest with a short report).
        docker_image:  str  — run the command inside this image with the
                       workspace mounted, instead of on the host. Use the
                       instance's official image when the repo needs system
                       dependencies you do not have.
        docker_workdir: str — mount point inside the image (default
                       ``/testbed``, which is where SWE-bench images keep the
                       repo, so the mount replaces it with the agent's version).
        timeout:       float — seconds per test command (default ``1800``).
        run_together:  bool — run both sets in one command (default ``True``);
                       set False if a shared-state test suite makes that unsafe.
    """

    grader_scope = GraderScope.OUTCOME
    version = "1.0"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.test_command: str = self.config.get("test_command", DEFAULT_TEST_COMMAND)
        self.docker_image: str = self.config.get("docker_image", "")
        self.docker_workdir: str = self.config.get("docker_workdir", "/testbed")
        self.timeout: float = self.config.get("timeout", 1800)
        self.run_together: bool = self.config.get("run_together", True)

    # ------------------------------------------------------------------

    async def grade(self, context: GradeContext) -> GradeResult:
        meta = context.metadata or {}
        fail_to_pass = _as_list(meta.get("fail_to_pass"))
        pass_to_pass = _as_list(meta.get("pass_to_pass"))
        if not fail_to_pass:
            return self._error("case metadata has no fail_to_pass tests")

        workspace = self._workspace(context)
        if workspace is None:
            return self._error(
                "no agent workspace on the outcome — this grader needs an "
                "adapter that records CodeArtifact.metadata['workspace'] "
                "(claude_code does)"
            )

        try:
            self._install_official_tests(workspace, meta)
        except _ProtocolError as e:
            return self._error(str(e))

        started = time.monotonic()
        if self.run_together:
            report, output = await self._run_tests(workspace, fail_to_pass + pass_to_pass)
        else:
            report, output = await self._run_tests(workspace, fail_to_pass)
            p2p_report, p2p_output = await self._run_tests(workspace, pass_to_pass)
            report.update(p2p_report)
            output = f"{output}\n{p2p_output}"
        elapsed_ms = (time.monotonic() - started) * 1000

        f2p_ok = [t for t in fail_to_pass if report.get(t) in _PASSING]
        p2p_ok = [t for t in pass_to_pass if report.get(t) in _PASSING]
        resolved = len(f2p_ok) == len(fail_to_pass) and len(p2p_ok) == len(pass_to_pass)

        total = len(fail_to_pass) + len(pass_to_pass)
        score = (len(f2p_ok) + len(p2p_ok)) / total if total else 0.0

        failure_tags = []
        if not f2p_ok or len(f2p_ok) < len(fail_to_pass):
            failure_tags.append("fail_to_pass_incomplete")
        if len(p2p_ok) < len(pass_to_pass):
            # Broke something that already worked. Worth its own label: it is a
            # different kind of bad from "did not finish the job".
            failure_tags.append("regression")

        # Nothing parsed at all: a collection error, an import failure, a
        # missing interpreter, a non-pytest runner. Scoring it zero is right,
        # but a bare 0/24 is indistinguishable from an agent that did nothing —
        # so say so, and carry the output that explains it.
        diagnostics: dict[str, Any] = {}
        if not report:
            failure_tags.append("no_tests_ran")
            diagnostics["no_tests_ran"] = True
            diagnostics["output_tail"] = output.strip()[-1500:]

        return GradeResult(
            name=self.name,
            grader_type=self.grader_type,
            grader_scope=self.grader_scope,
            passed=resolved,
            score=score,
            details={
                "resolved": resolved,
                "fail_to_pass": {
                    "passed": len(f2p_ok), "total": len(fail_to_pass),
                    "failing": [t for t in fail_to_pass if t not in f2p_ok][:20],
                },
                "pass_to_pass": {
                    "passed": len(p2p_ok), "total": len(pass_to_pass),
                    "failing": [t for t in pass_to_pass if t not in p2p_ok][:20],
                },
                "duration_ms": elapsed_ms,
                "runner": "docker" if self.docker_image else "local",
                **diagnostics,
            },
            metrics={
                "resolved": resolved,
                "f2p_rate": len(f2p_ok) / len(fail_to_pass),
                "p2p_rate": (len(p2p_ok) / len(pass_to_pass)) if pass_to_pass else 1.0,
            },
            failure_tags=failure_tags,
            reasoning=(
                f"{'resolved' if resolved else 'not resolved'} — "
                f"FAIL_TO_PASS {len(f2p_ok)}/{len(fail_to_pass)}, "
                f"PASS_TO_PASS {len(p2p_ok)}/{len(pass_to_pass)}"
            ),
        )

    # ------------------------------------------------------------------
    # Protocol steps
    # ------------------------------------------------------------------

    def _install_official_tests(self, workspace: Path, meta: dict[str, Any]) -> None:
        """Reset the test files, then apply the official test patch."""
        test_patch = meta.get("test_patch") or ""
        if not test_patch.strip():
            raise _ProtocolError("case metadata has no test_patch")

        base_commit = meta.get("base_commit") or ""
        touched = _paths_in_patch(test_patch)

        # Step 1 — undo anything the agent did to these files. Without this an
        # agent that edited a test would keep its edit wherever the official
        # patch does not overwrite it, and `git apply` would likely reject.
        if base_commit and touched:
            existing = [p for p in touched if (workspace / p).exists()]
            if existing:
                _git(workspace, "checkout", base_commit, "--", *existing)

        # Step 2 — apply the tests the agent was never allowed to see.
        result = subprocess.run(
            ["git", "apply", "-v", "-"],
            cwd=workspace,
            input=test_patch,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise _ProtocolError(
                f"git apply of test_patch failed: {result.stderr.strip()[-400:]}"
            )

    async def _run_tests(
        self, workspace: Path, tests: list[str]
    ) -> tuple[dict[str, str], str]:
        """Run node ids; return ({test_id: status}, raw output).

        The raw output comes back too because an empty report is the ambiguous
        case — it needs the runner's own words to be diagnosable.
        """
        if not tests:
            return {}, ""
        command = self.test_command.replace(
            "{tests}", " ".join(shlex.quote(t) for t in tests)
        ).replace(
            # Host interpreter for a local run; inside a container that path
            # does not exist, so fall back to whatever the image calls python.
            "{python}", "python" if self.docker_image else shlex.quote(sys.executable)
        )
        if self.docker_image:
            command = (
                f"docker run --rm "
                f"-v {shlex.quote(str(workspace))}:{shlex.quote(self.docker_workdir)} "
                f"-w {shlex.quote(self.docker_workdir)} "
                f"{shlex.quote(self.docker_image)} "
                f"bash -lc {shlex.quote(command)}"
            )

        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(workspace),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=self.timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            # Every requested test counts as failed — see the module docstring.
            return {}, f"timed out after {self.timeout}s"

        output = out.decode("utf-8", errors="replace")
        return _parse_pytest_report(output), output

    # ------------------------------------------------------------------

    @staticmethod
    def _workspace(context: GradeContext) -> Path | None:
        artifact = context.code_artifact
        raw = (artifact.metadata or {}).get("workspace") if artifact else None
        if not raw:
            return None
        path = Path(str(raw))
        return path if path.is_dir() else None

    def _error(self, message: str) -> GradeResult:
        return GradeResult(
            name=self.name,
            grader_type=self.grader_type,
            grader_scope=self.grader_scope,
            passed=False,
            score=None,  # not measured, so it stays out of the score denominator
            error=message,
            failure_tags=["harness_error"],
        )


class _ProtocolError(RuntimeError):
    """The instance could not be set up for scoring."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_list(value: Any) -> list[str]:
    """SWE-bench ships these as JSON *strings* in some exports, lists in others."""
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except ValueError:
            return [value]
        return [str(v) for v in parsed] if isinstance(parsed, list) else [str(parsed)]
    return []


def _paths_in_patch(patch: str) -> list[str]:
    """Repo-relative paths a unified diff touches, from its ``+++ b/…`` lines."""
    paths = []
    for line in patch.splitlines():
        if line.startswith("+++ ") and not line.startswith("+++ /dev/null"):
            path = line[4:].strip()
            if path.startswith("b/"):
                path = path[2:]
            if path and path != "/dev/null":
                paths.append(path)
    return paths


def _git(workspace: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=workspace, capture_output=True, text=True
    )


def _parse_pytest_report(stdout: str) -> dict[str, str]:
    """{node_id: STATUS} from a ``pytest -rA`` short summary.

    Only pytest is understood. Repos with their own runner (Django's, tox
    wrappers) report differently and will come back empty — which this grader
    treats as "everything failed" rather than guessing. Point ``test_command``
    at something that emits a pytest-shaped report, or run inside the
    instance's official image.
    """
    return {match.group(2): match.group(1) for match in _PYTEST_LINE.finditer(stdout)}
