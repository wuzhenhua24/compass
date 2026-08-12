"""Resolve a suite into a runnable scenario and launch a live comparison.

    # first, a smoke run: one model, one case, one trial — proves the wiring
    uv run python examples/coding_agent/run.py -m haiku --case fix_rounding

    # then the real thing
    uv run python examples/coding_agent/run.py -m haiku -m sonnet --trials 3

    # the same cases through pi, where the model is just a flag
    uv run python examples/coding_agent/run.py --suite pi.yaml \\
        -m google/gemini-2.5-flash -m google/gemini-3.6-flash --trials 3

    # one stack against another, under one grading contract (same --out both times)
    uv run python examples/coding_agent/run.py --suite crossstack.codex.yaml \\
        -m gpt-5.4-mini --trials 3 --out ./crossstack-run

    # your own project instead of the bundled one
    uv run python examples/coding_agent/run.py --repo ~/work/my-service -m sonnet

The suites ship with ``{{REPO}}``, ``{{GRADERS}}`` and ``{{STREAMS}}``
placeholders because those are absolute paths that only exist on your machine.
This fills them in, writes the resolved scenario where you can read it, and
runs ``compass test``.

It prints the command before running it. Copy that line and you never need this
script again — it is a convenience, not a layer.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).parent
_DEFAULT_OUT = Path("./coding-eval-run")


def bootstrap_repo(dest: Path) -> Path:
    """Turn the bundled ``project/`` into a git repository.

    Only for the bundled project — ``--repo`` skips this entirely. It ships as
    plain files because a nested ``.git`` inside Compass's tree is a trap for
    anyone running ``git add``, but ``claude_code`` needs a real repo: each
    trial is a ``git worktree`` off the base commit.

    Re-running reuses an existing checkout, so traces from earlier runs stay
    comparable against the same baseline.

    ``__pycache__`` is left behind on purpose. It is gitignored in Compass's own
    tree but sits on disk after anyone runs the project's tests, and copying it
    in would commit stale bytecode to the baseline — after which every agent
    that runs pytest "changes" 16 ``.pyc`` files, inflating ``diff_size`` and the
    changed-file list with work it did not do.
    """
    repo = dest / "repo"
    if (repo / ".git").exists():
        return repo
    shutil.copytree(
        _HERE / "project",
        repo,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    run = lambda *a: subprocess.run(  # noqa: E731
        ["git", *a], cwd=repo, check=True, capture_output=True, text=True
    )
    run("init", "-q")
    run("config", "user.email", "eval@example.com")
    run("config", "user.name", "Compass Eval")
    run("add", "-A")
    run("commit", "-qm", "project: initial state")
    return repo


def resolve_suite(
    repo: Path, out: Path, *, trials: int | None, suite: Path | None = None
) -> Path:
    suite = suite or _HERE / "suite.yaml"
    raw = suite.read_text(encoding="utf-8")
    raw = (
        raw.replace("{{REPO}}", str(repo.resolve()))
        .replace("{{GRADERS}}", str((_HERE / "grader_tests").resolve()))
        .replace("{{STREAMS}}", str((out / "streams").resolve()))
        .replace("{{JUDGE}}", str((_HERE / "judge_cli.py").resolve()))
        .replace("{{CHECKER}}", str((_HERE / "tests_intact.py").resolve()))
    )
    if trials is not None:
        raw = raw.replace("  trials: 1\n", f"  trials: {trials}\n", 1)
    resolved = out / f"{suite.stem}.resolved.yaml"
    resolved.write_text(raw, encoding="utf-8")
    return resolved


def build_command(
    scenario: Path, out: Path, models: list[str], cases: list[str], extra: list[str]
) -> list[str]:
    command = ["uv", "run", "compass", "test", str(scenario)]
    for model in models:
        command += ["-m", model]
    for case in cases:
        command += ["-c", case]
    # Traces are the point of a paid run: they let you re-grade later without
    # paying again, and `compass compare` needs the per-model JSON.
    command += [
        "--trace-dir", str(out / "traces"),
        "-r", "json",
        "-o", str(out / "results.json"),
    ]
    return command + extra


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="run.py", description=__doc__.splitlines()[0]
    )
    parser.add_argument(
        "--model", "-m", dest="models", action="append", default=[],
        help="model to compare, repeatable (e.g. -m haiku -m sonnet)",
    )
    parser.add_argument(
        "--case", "-c", dest="cases", action="append", default=[],
        help="run only these case ids, repeatable",
    )
    parser.add_argument(
        "--trials", type=int, default=None,
        help="override trials per case (use 3+ for a real comparison)",
    )
    parser.add_argument(
        "--repo", type=Path, default=None,
        help="your own git repository instead of the bundled project/",
    )
    parser.add_argument(
        "--suite", type=Path, default=Path("suite.yaml"),
        help="which suite to resolve: suite.yaml (claude_code), pi.yaml (pi), "
             "or one of the crossstack.*.yaml sides (claude / pi / codex)",
    )
    parser.add_argument(
        "--out", type=Path, default=_DEFAULT_OUT,
        help=f"where the repo, traces and results go (default: {_DEFAULT_OUT})",
    )
    parser.add_argument(
        "--print-only", action="store_true",
        help="resolve the scenario and print the command, do not run it",
    )
    args, passthrough = parser.parse_known_args(argv)

    if not args.models:
        parser.error("give at least one -m/--model (e.g. -m haiku -m sonnet)")

    # A bare name means one of the bundled suites; a path means yours.
    suite = args.suite if args.suite.parent != Path(".") else _HERE / args.suite.name
    if not suite.is_file():
        parser.error(f"no such suite: {suite}")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    if args.repo:
        repo = args.repo.expanduser().resolve()
        if not (repo / ".git").exists():
            parser.error(f"{repo} is not a git repository")
        print(f"repo      {repo}  (yours — grader_tests/ still points at the bundled ones)")
    else:
        repo = bootstrap_repo(out)
        print(f"repo      {repo}")

    scenario = resolve_suite(repo, out, trials=args.trials, suite=suite)
    print(f"suite     {suite}")
    print(f"scenario  {scenario}")

    command = build_command(scenario, out, args.models, args.cases, passthrough)
    print(f"\n  {' '.join(command)}\n")

    if args.print_only:
        return 0

    if args.trials in (None, 1) and len(args.models) > 1:
        # Not a hard stop — a single-trial run is the right smoke test. But a
        # leaderboard built on it is noise, and it looks exactly like a result.
        print(
            "  note: trials=1. Fine for checking the wiring; the ranking it\n"
            "        produces is noise. Use --trials 3 before believing it.\n"
        )

    return subprocess.run(command).returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
