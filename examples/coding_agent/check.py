"""Validate the case set before you spend money running agents against it.

    uv run python examples/coding_agent/check.py            # every case
    uv run python examples/coding_agent/check.py bug_locate_promo_boundary

This is the step that separates a dataset from a pile of files, and it is the
one people skip. Four checks per case, each catching a way a task can be
quietly worthless:

    RED         hidden tests fail on the untouched project
                -> passing here means the task is already done. A free point
                   for every agent, and a silent inflation of your numbers.
    GREEN       hidden tests pass with the reference solution applied
                -> failing here means the task is not achievable as written:
                   a broken environment, or a test asserting something the
                   requirement never said.
    NO-REGRESS  the project's own suite still passes with the solution applied
                -> failing here means your own reference breaks the repo, so
                   no agent can satisfy both gates either.
    STABLE      GREEN repeated; any test that flips is flaky
                -> a flaky test does not cost you one case, it corrupts
                   pass^k across the whole run.

A case with no ``solutions/<id>/`` directory is reported as UNVERIFIED rather
than skipped silently: it can still be run, but nobody has shown it is
solvable, and that is worth seeing in the summary.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_HERE = Path(__file__).parent
_PROJECT = _HERE / "project"
_HIDDEN = _HERE / "grader_tests"
_SOLUTIONS = _HERE / "solutions"
_SUITE = _HERE / "suite.yaml"

_STABILITY_RUNS = 3


@dataclass
class CaseReport:
    case_id: str
    red: bool | None = None
    green: bool | None = None
    no_regress: bool | None = None
    stable: bool | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def verified(self) -> bool:
        return bool(self.red and self.green and self.no_regress and self.stable)

    @property
    def unverified(self) -> bool:
        """No reference solution, so only RED could be checked."""
        return self.green is None


# ---------------------------------------------------------------------------


def find_placeholder_cases(cases: list[dict]) -> list[str]:
    """Case ids that look unfilled.

    An unfilled slot must not be a case at all. ``check.py`` can be taught to
    skip one, but ``compass test`` cannot — it just sees a case with no
    correctness graders, runs the process guards, and passes. Three such stubs
    once handed both models a free 0.997 apiece and inflated the pass rate
    while costing real money. So this is an error, not a skip: comment the slot
    out until it is ready.
    """
    suspicious = []
    for case in cases:
        prompt = (case.get("input") or {}).get("prompt", "")
        if (
            case["id"].upper().startswith("TODO")
            or case.get("metadata", {}).get("todo")
            or prompt.strip().upper() in ("", "TODO")
        ):
            suspicious.append(case["id"])
    return suspicious


def load_cases(only: list[str]) -> list[dict]:
    suite = yaml.safe_load(_SUITE.read_text(encoding="utf-8"))
    cases = suite["cases"]

    placeholders = find_placeholder_cases(cases)
    if placeholders:
        raise SystemExit(
            f"unfilled slot(s) present as runnable cases: {', '.join(placeholders)}\n"
            "  `compass test` will run them, and they pass trivially — every "
            "model gets a free point.\n"
            "  Comment the slot out in suite.yaml until it has a prompt, "
            "hidden tests and a reference solution."
        )

    if only:
        by_id = {c["id"]: c for c in cases}
        missing = [c for c in only if c not in by_id]
        if missing:
            raise SystemExit(f"no such case(s): {', '.join(missing)}")
        cases = [by_id[c] for c in only]
    return cases


def hidden_tests_for(case: dict) -> Path:
    path = _HIDDEN / f"test_{case['id']}.py"
    if not path.exists():
        raise SystemExit(
            f"{case['id']}: no hidden tests at {path.relative_to(_HERE)}"
        )
    return path


def _run_pytest(workdir: Path, target: str) -> tuple[bool, str]:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", target, "-q", "-p", "no:cacheprovider"],
        cwd=workdir,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0, result.stdout + result.stderr


def _workspace(dest: Path, solution: Path | None) -> Path:
    """A fresh copy of the project, optionally with a solution applied."""
    work = dest / "work"
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(_PROJECT, work)
    if solution is not None:
        for src in solution.rglob("*"):
            if src.is_file():
                target = work / src.relative_to(solution)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, target)
    return work


def check_case(case: dict, tmp: Path) -> CaseReport:
    report = CaseReport(case_id=case["id"])
    hidden = hidden_tests_for(case)
    solution = _SOLUTIONS / case["id"]

    # RED — on the untouched project the hidden tests must fail.
    work = _workspace(tmp, None)
    passed, output = _run_pytest(work, str(hidden))
    report.red = not passed
    if passed:
        report.notes.append(
            "hidden tests already pass on the base project — the task is done"
        )
    elif "error" in output.lower() and "collected 0 items" in output:
        report.notes.append("hidden tests failed to collect; check imports")

    if not solution.is_dir():
        report.notes.append(f"no solutions/{case['id']}/ — solvability unproven")
        return report

    # GREEN — with the reference solution they must pass.
    work = _workspace(tmp, solution)
    report.green, output = _run_pytest(work, str(hidden))
    if not report.green:
        report.notes.append(f"reference solution does not satisfy the tests: {_tail(output)}")

    # NO-REGRESS — and the project's own suite must survive it.
    report.no_regress, output = _run_pytest(work, "tests/")
    if not report.no_regress:
        report.notes.append(f"reference solution breaks the repo suite: {_tail(output)}")

    # STABLE — repeat GREEN; anything that flips is flaky.
    report.stable = True
    if report.green:
        for _ in range(_STABILITY_RUNS - 1):
            again, _ = _run_pytest(work, str(hidden))
            if not again:
                report.stable = False
                report.notes.append("hidden tests are flaky — they flipped on a rerun")
                break

    return report


def _tail(output: str, lines: int = 2) -> str:
    interesting = [
        line for line in output.splitlines()
        if line.strip() and not line.startswith("=")
    ]
    return " | ".join(interesting[-lines:])[:160]


# ---------------------------------------------------------------------------


def _mark(value: bool | None) -> str:
    return {True: "  ✓ ", False: "  ✗ ", None: "  – "}[value]


def print_report(reports: list[CaseReport]) -> None:
    print(f"\n  {'case':<38}{'RED':<6}{'GREEN':<7}{'NO-REG':<8}{'STABLE':<8}")
    print(f"  {'-' * 66}")
    for r in reports:
        print(
            f"  {r.case_id:<38}{_mark(r.red):<6}{_mark(r.green):<7}"
            f"{_mark(r.no_regress):<8}{_mark(r.stable):<8}"
        )
    for r in reports:
        for note in r.notes:
            print(f"    {r.case_id}: {note}")

    verified = sum(1 for r in reports if r.verified)
    unverified = sum(1 for r in reports if r.unverified)
    broken = len(reports) - verified - unverified
    print(f"\n  {verified} verified", end="")
    if unverified:
        print(f", {unverified} unverified (no reference solution)", end="")
    if broken:
        print(f", {broken} BROKEN", end="")
    print(f"  —  of {len(reports)} case(s)\n")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="check.py", description=__doc__.splitlines()[0])
    parser.add_argument("cases", nargs="*", help="case ids to check (default: all)")
    args = parser.parse_args(argv)

    cases = load_cases(args.cases)
    print(f"\nValidating {len(cases)} case(s) against {_PROJECT.name}/ …")

    with tempfile.TemporaryDirectory(prefix="compass_case_check_") as tmp:
        reports = [check_case(case, Path(tmp)) for case in cases]

    print_report(reports)
    # Unverified is a warning, not a failure — a case can be legitimately
    # awaiting its reference solution. Broken is a failure.
    return 1 if any(not r.verified and not r.unverified for r in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
