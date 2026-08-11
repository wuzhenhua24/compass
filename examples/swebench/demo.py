"""Run the SWE-bench loader + scoring protocol offline, for free.

    uv run python examples/swebench/demo.py                  # all three behaviours
    uv run python examples/swebench/demo.py solves
    uv run python examples/swebench/demo.py solves regresses

Everything here is real except the agent: the real ``claude_code`` adapter, real
git worktrees per trial, the real ``swebench_tests`` grader applying a real test
patch and running real pytest. ``replay_cli.py`` stands in for the CLI so no
credentials, no Docker image and no clone of Django are needed.

**The instances are synthetic.** ``fixtures/instances.jsonl`` is two made-up
instances in SWE-bench's schema over a ten-line local repo. They exist to prove
the plumbing, and they say nothing about any model. For real numbers see the
README — the same loader takes SWE-bench Verified unchanged.

The three canned behaviours are chosen to exercise the parts of the protocol
that are easy to get wrong:

    solves       the fix the issue asked for            -> resolved
    regresses    new test green, an old one broken      -> PASS_TO_PASS catches it
    edits_tests  guts the test file to win              -> the reset step undoes it
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from compass.core.runner import Compass

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

import graders as _graders  # noqa: E402, F401  importing registers swebench_tests
from loader import build_scenario, load_instances  # noqa: E402

_BEHAVIOURS = ("solves", "regresses", "edits_tests")
_WHY = {
    "solves": "makes the fix the issue asked for",
    "regresses": "satisfies the new test by breaking an old one",
    "edits_tests": "tries to win by weakening the test file",
}


def bootstrap_repo(dest: Path) -> tuple[Path, str]:
    """Create the checkout the demo instances refer to. Returns (root, commit).

    Real instances name a GitHub repo and a commit that already exists; the
    demo has to manufacture both, which is the one thing here that a live run
    does differently.
    """
    root = dest / "repos"
    repo = root / "demo__slugkit"
    shutil.copytree(_HERE / "fixtures" / "repo", repo)
    run = lambda *a: subprocess.run(  # noqa: E731
        ["git", *a], cwd=repo, check=True, capture_output=True, text=True
    )
    run("init", "-q")
    run("config", "user.email", "demo@example.com")
    run("config", "user.name", "Compass Demo")
    run("add", "-A")
    run("commit", "-qm", "slugkit: initial state")
    commit = run("rev-parse", "HEAD").stdout.strip()
    return root, commit


def print_run(behaviour: str, result) -> None:
    print(f"\n{'=' * 74}")
    print(f"  {behaviour}  —  {_WHY[behaviour]}")
    print(f"{'=' * 74}")
    for case in result.case_results:
        verdict = "RESOLVED" if case.passed else "NOT RESOLVED"
        print(f"\n  [{verdict}] {case.case_id}")
        for grader in case.evaluator_results:
            if grader.name != "swebench_tests":
                continue
            detail = grader.metadata or {}
            f2p, p2p = detail.get("fail_to_pass", {}), detail.get("pass_to_pass", {})
            print(
                f"      FAIL_TO_PASS {f2p.get('passed', 0)}/{f2p.get('total', 0)}"
                f"   PASS_TO_PASS {p2p.get('passed', 0)}/{p2p.get('total', 0)}"
            )
            for label, bucket in (("F2P", f2p), ("P2P", p2p)):
                for test in bucket.get("failing", []):
                    print(f"        ✗ {label} {test}")
            if grader.error:
                print(f"        ! {grader.error}")
            if grader.failure_tags:
                print(f"        tags: {', '.join(grader.failure_tags)}")


def print_summary(ranked: list[tuple[str, object]]) -> None:
    print(f"\n{'=' * 74}")
    print("  Resolution rate")
    print(f"{'=' * 74}")
    for behaviour, result in ranked:
        n = len(result.case_results)
        resolved = sum(1 for c in result.case_results if c.passed)
        print(f"  {behaviour:<14}{resolved}/{n}   {resolved / n:.0%}")
    print(
        "\n  Two synthetic instances, one trial each. That is a plumbing check,\n"
        "  not a measurement — see README.md for what a real run needs."
    )


async def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="demo.py")
    parser.add_argument("behaviours", nargs="*", default=[])
    args = parser.parse_args(argv)

    behaviours = args.behaviours or list(_BEHAVIOURS)
    unknown = [b for b in behaviours if b not in _BEHAVIOURS]
    if unknown:
        parser.error(
            f"unknown behaviour(s): {', '.join(unknown)} — "
            f"choose from {', '.join(_BEHAVIOURS)}"
        )

    instances = load_instances(_HERE / "fixtures" / "instances.jsonl")
    print(
        f"\n{len(instances)} synthetic instances, replayed through the real "
        f"claude_code adapter and the real scoring protocol."
    )

    with tempfile.TemporaryDirectory(prefix="compass_swebench_demo_") as tmp:
        repo_root, commit = bootstrap_repo(Path(tmp))
        # Real instances carry their own base_commit; these are manufactured.
        for instance in instances:
            instance["base_commit"] = commit

        ranked: list[tuple[str, object]] = []
        for behaviour in behaviours:
            scenario = build_scenario(
                instances,
                repo_root=repo_root,
                name=f"SWE-bench demo [{behaviour}]",
                agent_config={
                    "cli_path": str(_HERE / "replay_cli.py"),
                    # In a live run this is the model. Here it picks the canned
                    # attempt — same axis, same machinery.
                    "model": behaviour,
                },
                # The demo's canned runs are fixed, so process budgets would
                # only be measuring the fixture. A live run wants them.
                process_graders=[],
            )
            result = await Compass().run(scenario)
            print_run(behaviour, result)
            ranked.append((behaviour, result))

        print_summary(ranked)

    # The demo is correct when the protocol reaches the verdict each fixture was
    # built to earn — including the two that must NOT be resolved.
    expected = {"solves": 1.0, "regresses": 0.0, "edits_tests": 0.0}
    actual = {b: r.pass_rate for b, r in ranked}
    wrong = {b: v for b, v in actual.items() if v != expected[b]}
    if wrong:
        print(f"\n  UNEXPECTED: {json.dumps(wrong)} (expected {expected})")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv[1:])))
