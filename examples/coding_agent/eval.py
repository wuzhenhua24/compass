"""Evaluate a Claude Code agent on business requirements, offline and for free.

    uv run python examples/coding_agent/eval.py                    # compare all three
    uv run python examples/coding_agent/eval.py honest             # just one
    uv run python examples/coding_agent/eval.py honest cheats
    uv run python examples/coding_agent/eval.py --keep ./demo-out  # keep traces

This runs the **real** pipeline — the real ``claude_code`` adapter, real git
worktrees, real graders, the real leaderboard. The only substitution is the CLI
itself: ``replay_cli.py`` replays a canned run from ``fixtures.py`` instead of
calling Anthropic. So what you see here is what a live run reports, minus the
bill.

The three behaviours it compares are the three distinct ways a coding agent
fails an evaluation:

    honest      does what was asked                        -> passes
    cheats      wrong logic, edits the repo's tests green  -> caught twice
    overreach   right answer, wandering expensive process  -> caught on process

To go live: copy ``coding.yaml``, fill in ``repo``, drop the ``cli_path`` line,
and point ``integration_test``'s script at your own acceptance tests.
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import yaml

from compass.core.runner import Compass
from compass.core.scenario import Scenario

_HERE = Path(__file__).parent
_BEHAVIOURS = ("honest", "cheats", "overreach")


# ---------------------------------------------------------------------------
# Fixture repo
# ---------------------------------------------------------------------------


def bootstrap_repo(dest: Path) -> Path:
    """Copy ``project/`` into a fresh git repository.

    The fixture codebase ships as plain files rather than a checked-in git repo
    — a nested ``.git`` inside Compass's own tree would be a snare for anyone
    running ``git add``. It has to *become* a repo, though: ``claude_code``
    isolates each trial in a ``git worktree`` and computes the outcome diff
    against the base commit.

    ``__pycache__`` is skipped: committing stale bytecode to the baseline makes
    every agent that runs pytest "change" 16 ``.pyc`` files it never touched.
    """
    repo = dest / "storefront"
    shutil.copytree(
        _HERE / "project", repo,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    run = lambda *a: subprocess.run(  # noqa: E731
        ["git", *a], cwd=repo, check=True, capture_output=True
    )
    run("init", "-q")
    run("config", "user.email", "eval@example.com")
    run("config", "user.name", "Compass Example")
    run("add", "-A")
    run("commit", "-qm", "storefront: initial state")
    return repo


def build_scenario(repo: Path, behaviour: str) -> Scenario:
    """Load coding.yaml and fill in the paths that only exist at runtime."""
    raw = (_HERE / "coding.yaml").read_text(encoding="utf-8")
    raw = (
        raw.replace("{{REPO}}", str(repo))
        .replace("{{CLI}}", str(_HERE / "replay_cli.py"))
        .replace("{{GRADERS}}", str(_HERE / "grader_tests"))
    )
    scenario = Scenario.model_validate(yaml.safe_load(raw))
    # The replay CLI reads --model to pick which canned behaviour to play back.
    # In a live run this key really is the model — same axis, same machinery.
    scenario.agent.config["model"] = behaviour
    scenario.name = f"{scenario.name} [{behaviour}]"
    return scenario


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_WHY = {
    "honest": "makes the change that was asked for, and stops",
    "cheats": "wrong logic, then edits the repo's tests until they pass",
    "overreach": "right answer, but loops, overspends and edits unrelated files",
}


def print_run(behaviour: str, result) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {behaviour}  —  {_WHY[behaviour]}")
    print(f"{'=' * 72}")
    for case in result.case_results:
        status = {"passed": "PASS", "failed": "FAIL"}.get(
            case.status.value, case.status.value.upper()
        )
        print(f"\n  [{status}] {case.case_id}   score {case.overall_score:.3f}")
        for grader in case.evaluator_results:
            mark = "✓" if grader.passed else "✗"
            gate = " (gate)" if grader.gate else ""
            note = _explain(grader)
            print(f"      {mark} {grader.name:18}{grader.score:5.2f}{gate}  {note}")


def _explain(grader) -> str:
    """One line of why.

    ``EvaluatorResult.metadata`` carries the grader's own ``details`` payload,
    which is where every grader puts its reason — under a name of its own
    choosing, since the shape is the grader's business, not the core's.
    """
    details = grader.metadata or {}
    for key in ("failures", "violations", "error", "reasoning"):
        value = details.get(key)
        if not value:
            continue
        first = value[0] if isinstance(value, list) else value
        return str(first).replace("\n", " ")[:78]
    if grader.skipped:
        return f"skipped: {grader.skip_reason}"
    return ""


def print_leaderboard(ranked: list[tuple[str, object]]) -> None:
    from compass.report.leaderboard import build_leaderboard

    board = build_leaderboard(ranked)
    print(f"\n{'=' * 72}")
    print("  Leaderboard")
    print(f"{'=' * 72}")
    print(f"  {'#':<3}{'behaviour':<14}{'score (mean ± se)':<22}{'pass rate':<12}passed")
    for entry in board.entries:
        print(
            f"  {entry.rank:<3}{entry.label:<14}{entry.score_display:<22}"
            f"{entry.pass_rate:<12.1%}{entry.passed_cases}/{entry.evaluated_cases}"
        )
    if board.verdict:
        print(f"\n  {board.verdict}")

    # Worth saying out loud, because the score column looks like the answer and
    # is not: a gate is pass/fail, not a quantity, so it is excluded from the
    # weighted score. `cheats` therefore scores like an honest run — it only
    # tripped gates — and still fails every case. Rank on pass rate.
    misleading = [
        e for e in board.entries
        if e.pass_rate == 0.0 and e.mean_score > 0.8
    ]
    if misleading:
        names = ", ".join(e.label for e in misleading)
        print(
            f"\n  Read the pass-rate column, not the score: {names} score well "
            f"and pass\n  nothing. Gates are excluded from the weighted score "
            f"(pass/fail is not a\n  quantity), so tripping only gates barely "
            f"moves the number."
        )


# ---------------------------------------------------------------------------
# --keep: leave the repo, worktrees and traces on disk to poke at
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="eval.py", description=__doc__.splitlines()[0]
    )
    parser.add_argument(
        # No `choices=`: argparse validates a list default against it as if the
        # whole list were one value, and rejects it.
        "behaviours", nargs="*", default=[],
        help=f"behaviours to run — {', '.join(_BEHAVIOURS)} (default: all)",
    )
    parser.add_argument(
        "--keep", metavar="DIR", type=Path, default=None,
        help="keep the repo, worktrees and traces here instead of a temp dir",
    )
    args = parser.parse_args(argv)

    unknown = [b for b in args.behaviours if b not in _BEHAVIOURS]
    if unknown:
        parser.error(
            f"unknown behaviour(s): {', '.join(unknown)} — "
            f"choose from {', '.join(_BEHAVIOURS)}"
        )
    args.behaviours = args.behaviours or list(_BEHAVIOURS)
    if args.keep is not None:
        args.keep = args.keep.expanduser().resolve()
    return args


@contextmanager
def _workdir(keep: Path | None) -> Iterator[Path]:
    """A persistent directory with --keep, a self-deleting one without."""
    if keep is None:
        with tempfile.TemporaryDirectory(prefix="compass_coding_demo_") as tmp:
            yield Path(tmp)
        return
    if keep.exists():
        shutil.rmtree(keep)
    keep.mkdir(parents=True)
    yield keep


def _print_keep_hints(root: Path, behaviour: str) -> None:
    traces = root / "traces" / behaviour
    print(f"\n  Kept on disk: {root}")
    print("  Running and grading are separate verbs — the traces are evidence")
    print("  you can re-read and re-score without replaying anything:\n")
    print(f"    compass trace {traces}/add_discount.json --steps")
    print(f"    compass grade {traces} -s <your-grader-set>.yaml -n v2")


# ---------------------------------------------------------------------------


async def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    keep, behaviours = args.keep, args.behaviours

    print(
        "\nReplaying canned runs through the real claude_code adapter "
        "(no API key, no cost)."
    )

    with _workdir(keep) as root:
        repo = bootstrap_repo(root)
        ranked: list[tuple[str, object]] = []

        for behaviour in behaviours:
            scenario = build_scenario(repo, behaviour)
            trace_dir = str(root / "traces" / behaviour) if keep else None
            result = await Compass().run(scenario, trace_dir=trace_dir)
            print_run(behaviour, result)
            ranked.append((behaviour, result))

        if len(ranked) > 1:
            print_leaderboard(ranked)

        if keep:
            _print_keep_hints(root, behaviours[0])

    print(
        "\nNote: one trial per case, so the ranking above is a demonstration of "
        "the\nmachinery, not a measurement. Real runs need `trials: 3+` and "
        "pass^k —\nsee README.md.\n"
    )
    # The demo is correct when the graders agree with the fixtures' intent:
    # honest passes, the other two do not.
    expected = {"honest": True, "cheats": False, "overreach": False}
    actual = {b: r.pass_rate == 1.0 for b, r in ranked}
    return 0 if all(actual[b] == expected[b] for b in actual) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv[1:])))
