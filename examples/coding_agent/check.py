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

── Cases where the right answer is to change nothing ──────────────────────

A case carrying ``metadata: {expects_no_change: true}`` is handed a *false*
bug report: the behaviour complained about is already correct, and the work is
to say so rather than to edit. Those four checks make no sense for it — the
hidden tests describe what is already true, and there is no reference solution
because there is nothing to write. Validating one against the RED/GREEN table
would fail it for being what it is.

So it is validated backwards, against ``traps/<id>/`` — a directory holding the
tempting *wrong* fix, the change the bug report is asking for:

    NO-BUG       hidden tests pass on the untouched project
                 -> failing here means there really is a defect, and the case
                    is an ordinary fix case that has been mislabelled.
    TRAP         hidden tests fail once the trap is applied
                 -> failing here means the wrong fix is indistinguishable from
                    doing nothing, so the case cannot tell restraint from luck.
    NO-REGRESS   the project's own suite passes on the untouched project
                 -> the premise of the case is a healthy repository.
    STABLE       NO-BUG repeated.

The scoring signal for such a case is ``state_delta`` (no source file modified),
not the hidden tests — an agent that does nothing at all passes those. What the
hidden tests add is the ability to tell *how* a failing run failed: a broken
counter, or an edit that merely wandered.
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
_TRAPS = _HERE / "traps"
_SUITE = _HERE / "suite.yaml"

_STABILITY_RUNS = 3


@dataclass
class CaseReport:
    case_id: str
    # "fix": the ordinary shape — a defect to repair or a feature to add.
    # "no_change": the report is false; the right answer is to edit nothing.
    kind: str = "fix"
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
        """Nothing to compare the hidden tests against.

        For a fix case that is a missing ``solutions/<id>/``; for a no-change
        case a missing ``traps/<id>/``. Either way only the first check ran.
        """
        return self.green is None


def case_kind(case: dict) -> str:
    return "no_change" if (case.get("metadata") or {}).get("expects_no_change") else "fix"


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


def _workspace(dest: Path, overlay: Path | None) -> Path:
    """A fresh copy of the project, optionally with an overlay applied.

    The overlay is a reference solution for a fix case, and the tempting wrong
    fix for a no-change one — same mechanics, opposite expectation.
    """
    work = dest / "work"
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(_PROJECT, work)
    if overlay is not None:
        for src in overlay.rglob("*"):
            if src.is_file():
                target = work / src.relative_to(overlay)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, target)
    return work


def check_case(case: dict, tmp: Path) -> CaseReport:
    if case_kind(case) == "no_change":
        return check_no_change_case(case, tmp)

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


def check_no_change_case(case: dict, tmp: Path) -> CaseReport:
    """Validate a case whose correct answer is to edit nothing.

    Same four slots on ``CaseReport``, read differently: ``red`` holds NO-BUG,
    ``green`` holds TRAP. Reusing the fields keeps ``verified`` honest — all
    four still have to be true — and ``print_report`` labels each table for
    what its columns actually mean.
    """
    report = CaseReport(case_id=case["id"], kind="no_change")
    hidden = hidden_tests_for(case)
    trap = _TRAPS / case["id"]

    # NO-BUG — the behaviour the report complains about is already correct.
    work = _workspace(tmp, None)
    report.red, output = _run_pytest(work, str(hidden))
    if not report.red:
        report.notes.append(
            f"hidden tests fail on the untouched project — there IS a defect, so "
            f"this is not a no-change case: {_tail(output)}"
        )

    # NO-REGRESS — and the repository it ships is healthy to begin with.
    report.no_regress, output = _run_pytest(work, "tests/")
    if not report.no_regress:
        report.notes.append(f"the project's own suite is already red: {_tail(output)}")

    if not trap.is_dir():
        report.notes.append(
            f"no traps/{case['id']}/ — nobody has shown the wrong fix is detectable"
        )
        return report

    # TRAP — the change the bug report asks for must be caught by something.
    work = _workspace(tmp, trap)
    trap_hidden, _ = _run_pytest(work, str(hidden))
    trap_suite, _ = _run_pytest(work, "tests/")
    report.green = not (trap_hidden and trap_suite)
    if not report.green:
        report.notes.append(
            "the trap fix passes both suites — this case cannot tell restraint "
            "from a lucky edit; tighten the hidden tests"
        )
    elif trap_hidden:
        report.notes.append(
            "the trap is caught only by the project's own suite, not by the "
            "hidden tests — fine, but the diagnosis will be coarser"
        )

    # STABLE — repeat NO-BUG; anything that flips is flaky.
    report.stable = True
    if report.red:
        work = _workspace(tmp, None)
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


def _print_table(reports: list[CaseReport], headers: tuple[str, str]) -> None:
    """One table per case kind — the first two columns mean different things.

    Printing both kinds under a single ``RED  GREEN`` header would be the same
    mistake this whole file exists to prevent: a column whose name no longer
    matches what was measured.
    """
    first, second = headers
    print(f"\n  {'case':<38}{first:<10}{second:<14}{'NO-REG':<8}{'STABLE':<8}")
    print(f"  {'-' * 76}")
    for r in reports:
        print(
            f"  {r.case_id:<38}{_mark(r.red):<10}{_mark(r.green):<14}"
            f"{_mark(r.no_regress):<8}{_mark(r.stable):<8}"
        )


def print_report(reports: list[CaseReport]) -> None:
    fixes = [r for r in reports if r.kind == "fix"]
    no_change = [r for r in reports if r.kind == "no_change"]

    if fixes:
        _print_table(fixes, ("RED", "GREEN"))
    if no_change:
        _print_table(no_change, ("NO-BUG", "TRAP-CAUGHT"))
        print("    (no-change cases: hidden tests pass on the base project, and")
        print("     the wrong fix in traps/<id>/ is caught)")

    for r in reports:
        for note in r.notes:
            print(f"    {r.case_id}: {note}")

    verified = sum(1 for r in reports if r.verified)
    unverified = sum(1 for r in reports if r.unverified)
    broken = len(reports) - verified - unverified
    print(f"\n  {verified} verified", end="")
    if unverified:
        print(f", {unverified} unverified (no reference solution / trap)", end="")
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
