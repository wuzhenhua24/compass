"""Run any executable as a grader.

Compass graders are Python classes, which is right for the built-ins: they get
the `Transcript` object directly, run concurrently, and are type-checked. But
it puts a floor under who can extend the framework — a domain check has to be
written in Python and import Compass.

This grader removes that floor. Anything executable becomes a grader:
``rsvg-convert``, ``pytest``, ``npm test``, a Go binary, a shell one-liner that
greps a log. The contract is a subprocess boundary, so the checker needs no
knowledge of Compass beyond a few environment variables.

The checker contract
--------------------

The program is executed with **no arguments**, with the shared grade workspace
as its working directory — so it composes with the rest of the grade pipeline:
it can read what earlier graders produced and leave files for later ones.

Environment:

- ``COMPASS_WORKSPACE``     absolute path to the shared grade workspace (== cwd)
- ``COMPASS_TRANSCRIPT``    path to a JSON file holding the execution trace
- ``COMPASS_OUTCOME``       path to a JSON file holding the final outcome
- ``COMPASS_PROMPT``        the case's prompt
- ``COMPASS_ANSWER``        the agent's final text answer, when there is one
- ``COMPASS_REFERENCE_ANSWER``  the golden answer, when the case supplies one
- ``COMPASS_CONFIG``        the grader's full config as JSON, for structured values
- ``COMPASS_CONFIG_<KEY>``  every scalar config key, uppercased — for shell scripts

Output:

- **exit code** decides pass/fail (0 = pass)
- **stdout** may be a JSON object with ``score``, ``tags``, ``metrics``,
  ``notes`` and ``details``; anything else is folded into ``details`` rather
  than trusted at the top level
- **stderr** is kept as the error message when the checker fails

One deliberate deviation from smevals, whose contract this follows: there, a
check that fails without emitting a score leaves the grade *unscored*. Compass
gives "unscored" a stronger meaning — it is excluded from the score denominator
**and** fails the case, because it means "we could not measure this". A checker
exiting non-zero is a measurement ("this check failed"), so it scores 0.0.
``score=None`` is reserved for the cases where Compass genuinely learned
nothing: the program is missing or not executable, it timed out, or a signal
killed it.

Security: this executes whatever the scenario names, exactly like a Python
grader module does. Treat scenario files as trusted input.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from compass.graders.base import (
    CodeGrader,
    GradeContext,
    GradeResult,
    GraderScope,
    GraderType,
)
from compass.graders.registry import register_grader

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 60.0

#: Keys the checker owns at the top level of its JSON output. Everything else
#: is folded into ``details`` so a checker can never clobber a core field.
_RESULT_KEYS = {"score", "passed", "tags", "metrics", "notes", "details"}


@register_grader("external_checker")
class ExternalCheckerGrader(CodeGrader):
    """Grade by running an external program.

    Example::

        graders:
          - name: external_checker
            type: code
            required: true
            creates: render.png          # the `creates:` contract still applies
            config:
              command: ./checkers/render-svg
              input: extracted.svg       # -> COMPASS_CONFIG_INPUT
              timeout: 30
    """

    name = "external_checker"
    grader_type = GraderType.CODE
    # BOTH is the conservative choice: the checker is handed the transcript and
    # the outcome, and Compass cannot know which it actually reads. It also
    # means a lossy JSONL trace is refused rather than scored against an empty
    # outcome.
    grader_scope = GraderScope.BOTH
    version = "1.0"

    async def grade(self, context: GradeContext) -> GradeResult:
        command = self._resolve_command()
        if isinstance(command, GradeResult):
            return command  # configuration error, already shaped as a result

        if context.workspace is None:
            return self._fail(
                "external_checker needs a grade workspace; it is only usable "
                "from a runner that provides one",
                unscored=True,
                tag="checker_unavailable",
            )
        context.workspace.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="compass-checker-") as tmp:
            env = self._build_env(context, Path(tmp))
            try:
                code, stdout, stderr = await self._execute(
                    command, cwd=context.workspace, env=env
                )
            except FileNotFoundError:
                return self._fail(
                    f"checker not found: {command[0]}",
                    unscored=True,
                    tag="checker_unavailable",
                )
            except asyncio.TimeoutError:
                return self._fail(
                    f"checker timed out after {self.timeout:g}s",
                    unscored=True,
                    tag="checker_timeout",
                )

        return self._build_result(code, stdout, stderr)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @property
    def timeout(self) -> float:
        return float(self.config.get("timeout", DEFAULT_TIMEOUT))

    def _resolve_command(self) -> list[str] | GradeResult:
        """Normalize ``command`` to argv, or return a configuration failure.

        A string is a single program (no shell — a scenario is config, not a
        place to write shell); a list is argv as given.
        """
        raw = self.config.get("command")
        if not raw:
            return self._fail(
                "external_checker requires config.command",
                unscored=True,
                tag="checker_misconfigured",
            )

        argv = [str(raw)] if isinstance(raw, str) else [str(a) for a in raw]
        program = Path(argv[0])

        if program.is_absolute() or os.sep in argv[0] or argv[0].startswith("."):
            # A path: resolve against the working directory Compass was
            # started from, which is where scenario-relative paths make sense.
            resolved = program.expanduser().resolve()
            if not resolved.is_file():
                return self._fail(
                    f"checker not found: {resolved}",
                    unscored=True,
                    tag="checker_unavailable",
                )
            if not os.access(resolved, os.X_OK):
                return self._fail(
                    f"checker is not executable (chmod +x): {resolved}",
                    unscored=True,
                    tag="checker_unavailable",
                )
            argv[0] = str(resolved)
        else:
            # A bare name: look it up on PATH
            found = shutil.which(argv[0])
            if found is None:
                return self._fail(
                    f"checker not found on PATH: {argv[0]}",
                    unscored=True,
                    tag="checker_unavailable",
                )
            argv[0] = found

        return argv

    def _build_env(self, context: GradeContext, scratch: Path) -> dict[str, str]:
        """The environment the checker sees.

        Transcript and outcome go to files in a scratch directory rather than
        into the workspace: the workspace is kept as scoring *evidence*, and
        dumping Compass's own inputs there would pollute it.
        """
        env = dict(os.environ)

        if context.transcript is not None:
            transcript_file = scratch / "transcript.json"
            transcript_file.write_text(
                json.dumps(context.transcript.to_dict(), ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            env["COMPASS_TRANSCRIPT"] = str(transcript_file)

        if context.outcome is not None:
            outcome_file = scratch / "outcome.json"
            outcome_file.write_text(
                json.dumps(context.outcome.to_dict(), ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            env["COMPASS_OUTCOME"] = str(outcome_file)

        env["COMPASS_WORKSPACE"] = str(context.workspace)
        if context.prompt:
            env["COMPASS_PROMPT"] = context.prompt
        if context.answer:
            env["COMPASS_ANSWER"] = context.answer
        if context.reference_answer:
            env["COMPASS_REFERENCE_ANSWER"] = context.reference_answer

        env["COMPASS_CONFIG"] = json.dumps(self.config, ensure_ascii=False, default=str)
        env.update(_scalar_env_vars("COMPASS_CONFIG_", self.config))
        return env

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def _execute(
        self, argv: list[str], cwd: Path, env: dict[str, str]
    ) -> tuple[int, str, str]:
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(
                process.communicate(), timeout=self.timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise

        return (
            process.returncode or 0,
            out.decode("utf-8", errors="replace"),
            err.decode("utf-8", errors="replace"),
        )

    # ------------------------------------------------------------------
    # Result mapping
    # ------------------------------------------------------------------

    def _build_result(self, code: int, stdout: str, stderr: str) -> GradeResult:
        info = _parse_output(stdout)
        passed = code == 0

        score = info.get("score")
        if score is None:
            # No explicit measurement: the exit code is the measurement.
            score = 1.0 if passed else 0.0
        else:
            score = float(score)

        details: dict[str, Any] = dict(info.get("details") or {})
        details["exit_code"] = code

        error = None
        if not passed:
            error = (stderr.strip() or info.get("notes") or f"checker exited {code}")

        # A signal (negative return code on POSIX) is the harness dying, not a
        # verdict about the agent.
        if code < 0:
            return self._fail(
                f"checker killed by signal {-code}",
                unscored=True,
                tag="checker_crashed",
                details=details,
            )

        return GradeResult(
            name=self.name,
            grader_type=self.grader_type,
            grader_scope=self.grader_scope,
            grader_version=self.version,
            passed=passed,
            score=score,
            tags=list(info.get("tags") or []),
            # First-class, not folded into details: the aggregate reads these,
            # and anything the aggregate reads must not live in the checker's
            # own namespace.
            metrics=dict(info.get("metrics") or {}),
            failure_tags=[] if passed else ["checker_failed"],
            reasoning=str(info.get("notes") or ""),
            details=details,
            error=error,
        )

    def _fail(
        self,
        message: str,
        *,
        unscored: bool = False,
        tag: str = "checker_failed",
        details: dict[str, Any] | None = None,
    ) -> GradeResult:
        """A failure Compass itself detected, as opposed to a checker verdict."""
        return GradeResult(
            name=self.name,
            grader_type=self.grader_type,
            grader_scope=self.grader_scope,
            grader_version=self.version,
            passed=False,
            # `unscored` means Compass learned nothing — kept out of the score
            # denominator, and it fails the case.
            score=None if unscored else 0.0,
            failure_tags=[tag],
            details=details or {},
            error=message,
        )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _scalar_env_vars(prefix: str, mapping: dict[str, Any]) -> dict[str, str]:
    """Scalar config values as env vars: ``input`` -> ``COMPASS_CONFIG_INPUT``.

    Only scalars — a shell script cannot do anything useful with a nested
    object, and those are available in full via ``COMPASS_CONFIG``.
    """
    out: dict[str, str] = {}
    for key, value in mapping.items():
        if isinstance(value, bool):
            out[prefix + _env_key(key)] = "1" if value else "0"
        elif isinstance(value, (str, int, float)):
            out[prefix + _env_key(key)] = str(value)
    return out


def _env_key(key: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(key)).upper()


def _parse_output(stdout: str) -> dict[str, Any]:
    """Apply the result contract to whatever the checker printed.

    The checker owns five keys; anything else is folded into ``details``
    instead of being trusted at the top level, so a checker cannot reach in
    and set a field Compass reserves for itself. Non-JSON output is not an
    error — plenty of useful checkers just exit 0 or 1 — it is kept as a note.
    """
    text = stdout.strip()
    if not text:
        return {}

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"notes": text[:2000]}

    if not isinstance(data, dict):
        return {"details": {"output": data}}

    result: dict[str, Any] = {}
    if isinstance(data.get("score"), (int, float)) and not isinstance(
        data.get("score"), bool
    ):
        result["score"] = float(data["score"])
    if isinstance(data.get("metrics"), dict):
        result["metrics"] = data["metrics"]
    if isinstance(data.get("tags"), list):
        result["tags"] = [str(t) for t in data["tags"]]
    if data.get("notes"):
        result["notes"] = str(data["notes"])

    raw_details = data.get("details")
    details: dict[str, Any] = raw_details if isinstance(raw_details, dict) else {}
    extras = {k: v for k, v in data.items() if k not in _RESULT_KEYS}
    if details or extras:
        result["details"] = {**details, **extras}

    return result
