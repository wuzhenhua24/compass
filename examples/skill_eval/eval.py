"""Prove a skill rewrite is an improvement — offline and for free.

    uv run python examples/skill_eval/eval.py               # both suites
    uv run python examples/skill_eval/eval.py --suite ab    # just the A/B
    uv run python examples/skill_eval/eval.py --keep ./out  # keep results + traces

This runs the **real** pipeline — the real ``claude_code`` adapter installing a
real skill into a real git worktree, the real graders, the real paired
comparison. The only substitution is the CLI itself: ``replay_cli.py`` acts out
a run instead of calling Anthropic, and it decides what to act out by *reading
the skill the adapter installed*. Break the install and the numbers move.

To go live: drop the ``cli_path`` line from ``ab.yaml`` and point ``repo`` at
your own project.
"""

from __future__ import annotations

import argparse
import asyncio
import json
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
from compass.report.compare import compare_metric, compare_paths

_HERE = Path(__file__).parent
_SKILLS = _HERE / "skills"

# (label, skill directory) — "" is the no-skill baseline arm.
ARMS: list[tuple[str, str]] = [
    ("baseline", ""),
    ("v1", str(_SKILLS / "report-writer-v1")),
    ("v2", str(_SKILLS / "report-writer-v2")),
]

# What the report is actually about. Each is a metric some grader emitted.
METRICS = [
    ("skill_triggered", "触发率", "higher"),
    ("skill_resources_used", "bundled 脚本使用", "higher"),
    ("cost_usd", "每条 case 花费", "lower"),
    ("tool_calls", "工具调用数", "lower"),
]


def bootstrap_repo(dest: Path) -> Path:
    """Copy ``project/`` into a fresh git repository.

    The fixture ships as plain files — a nested ``.git`` inside Compass's own
    tree would be a snare for anyone running ``git add``. It has to *become* a
    repo, though: ``claude_code`` isolates each trial in a ``git worktree``.
    """
    repo = dest / "sales-data"
    shutil.copytree(
        _HERE / "project", repo, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    run = lambda *a: subprocess.run(  # noqa: E731
        ["git", *a], cwd=repo, check=True, capture_output=True
    )
    run("init", "-q")
    run("config", "user.email", "eval@example.com")
    run("config", "user.name", "Compass Example")
    run("add", "-A")
    run("commit", "-qm", "sales-data: initial state")
    return repo


def build_scenario(suite: str, repo: Path, skill: str, trials: int) -> Scenario:
    """Load the suite and fill in what only exists at runtime."""
    raw = (_HERE / f"{suite}.yaml").read_text(encoding="utf-8")
    raw = raw.replace("{{REPO}}", str(repo)).replace(
        "{{CLI}}", str(_HERE / "replay_cli.py")
    )
    scenario = Scenario.model_validate(yaml.safe_load(raw))
    # Exactly what `compass test --model-key skill -m <path>` does. The axis is
    # a key in the agent config either way.
    scenario.agent.config["skill"] = skill
    scenario.defaults.trials = trials
    return scenario


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _case_metric(case, label: str) -> float | None:
    """One case's value for a metric — the cross-trial mean where there is one.

    ``grader_summary`` is the mean over trials and the right thing to read;
    ``evaluator_results`` is one trial's detail and only stands in on a
    single-trial run, where the two are the same number anyway. This is the
    same order ``compass compare`` uses.
    """
    for entry in (case.grader_summary or {}).values():
        if label in (entry.get("metrics") or {}):
            return float(entry["metrics"][label])
    for grader in case.evaluator_results:
        if label in (grader.metrics or {}):
            return float(grader.metrics[label])
    return None


def _rate(result, label: str) -> float:
    """Mean of one metric across the run's cases, or nan when unmeasured."""
    values = [
        value
        for case in result.case_results
        if (value := _case_metric(case, label)) is not None
    ]
    return sum(values) / len(values) if values else float("nan")


def _pct(value: float) -> str:
    return "    —" if value != value else f"{value:>5.0%}"  # nan check


def print_arm(label: str, skill: str, result) -> None:
    where = Path(skill).name if skill else "— 不装 skill —"
    cost = _rate(result, "cost_usd")
    print(f"\n  {label:<9} {where:<22} "
          f"触发 {_pct(_rate(result, 'skill_triggered'))}   "
          f"脚本 {_pct(_rate(result, 'skill_resources_used'))}   "
          f"{'      —' if cost != cost else f'${cost:.3f}'}/case   "
          f"pass {result.pass_rate:>5.0%}")


def print_case_detail(result) -> None:
    for case in result.case_results:
        triggered = _case_metric(case, "skill_triggered")
        expected = "该触发" if case.case_id.startswith("pos") else "不该触发"
        mark = "·" if triggered is None else f"{triggered:.0%}"
        flag = "" if case.passed else "   ← 不符合预期"
        print(f"      {case.case_id:<20} {expected:<8} 实测触发 {mark:>5}{flag}")


def print_comparison(path_a: Path, path_b: Path, label_a: str, label_b: str) -> float:
    """Print the paired comparison; return how far the trigger rate moved."""
    print(f"\n{'=' * 76}")
    print(f"  {label_a} → {label_b}：真的更好了吗")
    print(f"{'=' * 76}")

    report = compare_paths(path_a, path_b, on="triggered")
    report.label_a, report.label_b = label_a, label_b
    print(f"\n  按 `triggered` 判分：{report.verdict}")
    for flip in report.improved:
        print(f"      ↑ {flip.case_id}")
    for flip in report.regressed:
        print(f"      ↓ {flip.case_id}   ← 回退，即使总分涨了也要看这一行")

    print()
    moved = 0.0
    for metric, zh, direction in METRICS:
        comparison = compare_metric(path_a, path_b, metric)
        if comparison.n_paired == 0:
            continue
        if metric == "skill_triggered":
            moved = comparison.mean_b - comparison.mean_a
        arrow = "越高越好" if direction == "higher" else "越低越好"
        print(
            f"  {zh}（{metric}，{arrow}）"
            f"\n      {label_a} {comparison.mean_a:.4g} → {label_b} "
            f"{comparison.mean_b:.4g}"
            f"\n      {comparison.verdict}"
        )
    return moved


# ---------------------------------------------------------------------------


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="eval.py", description=__doc__.splitlines()[0]
    )
    parser.add_argument(
        "--suite", choices=("ab", "trigger", "both"), default="both",
        help="ab = 完整 A/B；trigger = 只测触发率（便宜）",
    )
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument(
        "--keep", metavar="DIR", type=Path, default=None,
        help="把结果和轨迹留在这里，而不是临时目录",
    )
    args = parser.parse_args(argv)
    if args.keep is not None:
        args.keep = args.keep.expanduser().resolve()
    return args


@contextmanager
def _workdir(keep: Path | None) -> Iterator[Path]:
    if keep is None:
        with tempfile.TemporaryDirectory(prefix="compass_skill_demo_") as tmp:
            yield Path(tmp)
        return
    if keep.exists():
        shutil.rmtree(keep)
    keep.mkdir(parents=True)
    yield keep


async def run_suite(suite: str, repo: Path, out: Path, trials: int) -> dict[str, Path]:
    """Run every arm of one suite; return label -> results file."""
    print(f"\n{'=' * 76}")
    print(f"  {suite}.yaml")
    print(f"{'=' * 76}")

    written: dict[str, Path] = {}
    for label, skill in ARMS:
        if suite == "trigger" and label == "baseline":
            continue  # nothing installed means nothing to trigger
        scenario = build_scenario(suite, repo, skill, trials)
        result = await Compass().run(scenario)
        print_arm(label, skill, result)
        if suite == "trigger":
            print_case_detail(result)

        path = out / f"{suite}.{label}.json"
        path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        written[label] = path
    return written


async def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    suites = ["ab", "trigger"] if args.suite == "both" else [args.suite]

    print(
        "\n回放预置运行，跑的是真的 claude_code adapter + 真的 grader"
        "（不需要 API key，不花钱）。"
    )

    ok = True
    with _workdir(args.keep) as root:
        repo = bootstrap_repo(root)
        out = root / "results"
        out.mkdir()

        for suite in suites:
            written = await run_suite(suite, repo, out, args.trials)
            moved = print_comparison(written["v1"], written["v2"], "v1", "v2")
            # The demo is correct when the graders see what the fixtures act
            # out: v2 is reached more often than v1, on both suites.
            ok = ok and moved > 0

        if args.keep:
            print(f"\n  留在磁盘上：{root}")
            print(f"    compass site compare {out}/ab.v1.json {out}/ab.v2.json \\")
            print("        --label-a v1 --label-b v2 -o site/")

    print(
        "\n注意 v2 在 trigger.yaml 的 neg_chart 上误触发了——这不是 bug，是这个"
        "\n例子想让你看见的东西：更「主动」的 description 提高了召回，代价是它"
        "\n开始抢别人的活。只有正向用例的触发率评测看不见这一行。\n"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv[1:])))
