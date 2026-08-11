"""Turn SWE-bench instances into a Compass scenario.

The mapping is almost a rename, because a SWE-bench instance already carries
exactly what an agent evaluation needs:

    repo + base_commit  ->  the worktree each trial branches from
    problem_statement   ->  the case prompt
    FAIL_TO_PASS        ->  must go red -> green
    PASS_TO_PASS        ->  must stay green

so the loader's real job is the small set of things that are easy to get wrong:

- **Per-case repo and commit.** Every instance sits on a different commit, so
  they go in ``case.input.params`` (which the ``claude_code`` adapter reads
  per-case) rather than in the scenario-level agent config.
- **Keeping the answer away from the agent.** ``patch`` (the gold fix),
  ``test_patch`` (the tests) and ``hints_text`` (often the maintainer talking
  through the fix) never enter the prompt. The two the grader needs travel in
  case metadata; ``hints_text`` is dropped unless you ask for it.
- **Contamination.** Every SWE-bench Verified instance has been in training
  data for a long time. Comparing *prompts* is fine — both arms are equally
  contaminated. Comparing *models* on it is not measuring what you think.

::

    from loader import load_instances, build_scenario

    instances = load_instances("swebench_verified.jsonl")[:20]
    scenario  = build_scenario(instances, repo_root="./repos")

Getting the repos: this loader expects the checkouts to already exist under
``repo_root`` as ``<owner>__<name>`` (SWE-bench's own naming). ``clone_command``
in the README prints the git commands for a set of instances.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from compass.core.scenario import (
    AgentConfig,
    DefaultsConfig,
    GraderConfig,
    GraderType,
    InputConfig,
    Scenario,
    TestCase,
)

# The agent is told the task and the ground rules, and nothing about the tests.
# Kept short deliberately: this is the control variable in a prompt comparison,
# so anything clever belongs in the prompt being tested, not baked in here.
DEFAULT_PROMPT_TEMPLATE = """\
Resolve the following issue in the repository you are working in.

<issue>
{problem_statement}
</issue>

Make the minimal source change that fixes it. Do not modify existing tests —
your change is validated against the project's own test suite.
"""


def load_instances(
    path: str | Path,
    *,
    instance_ids: list[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Read instances from a JSONL or JSON file.

    Accepts what ``datasets.load_dataset(...).to_json()`` writes (JSONL) as well
    as a plain JSON array.

    Args:
        path: File of instances.
        instance_ids: Keep only these, in the order given.
        limit: Keep at most this many (applied after ``instance_ids``).
    """
    text = Path(path).expanduser().read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        return []

    if stripped.startswith("["):
        rows = json.loads(stripped)
    else:
        rows = [json.loads(line) for line in stripped.splitlines() if line.strip()]

    instances = [r for r in rows if isinstance(r, dict)]

    if instance_ids:
        by_id = {i["instance_id"]: i for i in instances}
        missing = [i for i in instance_ids if i not in by_id]
        if missing:
            raise KeyError(f"instance_id(s) not in {path}: {', '.join(missing)}")
        instances = [by_id[i] for i in instance_ids]

    return instances[:limit] if limit else instances


def repo_dir(instance: dict[str, Any], repo_root: str | Path) -> Path:
    """Where an instance's checkout lives: ``<repo_root>/<owner>__<name>``."""
    return Path(repo_root).expanduser() / instance["repo"].replace("/", "__")


def instance_to_case(
    instance: dict[str, Any],
    *,
    repo_root: str | Path,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    include_hints: bool = False,
    test_command: str | None = None,
    docker_image: str | None = None,
    timeout: float = 1800,
) -> TestCase:
    """One instance -> one Compass test case.

    Args:
        instance: A SWE-bench instance dict.
        repo_root: Directory holding the repository checkouts.
        prompt_template: Formatted with ``**instance``; must not reference
            ``patch`` or ``test_patch``.
        include_hints: Append ``hints_text`` to the prompt. Off by default —
            the hints are frequently the maintainer describing the fix, so
            turning this on measures reading comprehension, not engineering.
        test_command: Override the grader's test command (``{tests}``
            placeholder).
        docker_image: Run the tests in this image instead of on the host.
            ``{instance_id}`` and ``{repo_underscored}`` are substituted, so the
            official image naming can be given once for a whole run.
        timeout: Seconds allowed for the test command.
    """
    prompt = prompt_template.format(**instance)
    if include_hints and instance.get("hints_text"):
        prompt = f"{prompt}\n\n<hints>\n{instance['hints_text']}\n</hints>\n"

    grader_config: dict[str, Any] = {"timeout": timeout}
    if test_command:
        grader_config["test_command"] = test_command
    if docker_image:
        grader_config["docker_image"] = _format_image(docker_image, instance)

    return TestCase(
        id=instance["instance_id"],
        description=_first_line(instance.get("problem_statement", "")),
        input=InputConfig(
            prompt=prompt,
            # Per-case, because every instance sits on a different commit.
            params={
                "repo": str(repo_dir(instance, repo_root)),
                "base_ref": instance["base_commit"],
            },
        ),
        graders=[
            GraderConfig(
                type=GraderType.CODE,
                name="swebench_tests",
                gate=True,  # resolved / not resolved is pass/fail, not a quantity
                config=grader_config,
            )
        ],
        # What the grader needs and the agent must never see.
        metadata={
            "test_patch": instance.get("test_patch", ""),
            "fail_to_pass": instance.get("FAIL_TO_PASS", []),
            "pass_to_pass": instance.get("PASS_TO_PASS", []),
            "base_commit": instance["base_commit"],
            "swebench_repo": instance["repo"],
            "swebench_version": instance.get("version", ""),
        },
        tags=["swebench", instance["repo"].replace("/", "__")],
    )


def build_scenario(
    instances: list[dict[str, Any]],
    *,
    repo_root: str | Path,
    name: str = "SWE-bench",
    trials: int = 1,
    agent_config: dict[str, Any] | None = None,
    process_graders: list[GraderConfig] | None = None,
    **case_kwargs: Any,
) -> Scenario:
    """A ready-to-run scenario over a set of instances.

    Args:
        instances: Loaded instances.
        repo_root: Directory holding the checkouts.
        name: Scenario name (the leaderboard row label).
        trials: Attempts per instance. Leave at 1 only for a smoke run —
            resolution rates move several points between attempts, so a
            single-trial comparison of two configurations is noise.
        agent_config: Merged into the ``claude_code`` adapter config. This is
            where the comparison axis lives (``model``,
            ``append_system_prompt_file``, …).
        process_graders: Cost/turn/loop guards to attach to every case. Pass
            ``[]`` for none; omitted means :func:`default_process_graders`.
        **case_kwargs: Forwarded to :func:`instance_to_case`.
    """
    graders = (
        default_process_graders() if process_graders is None else process_graders
    )
    config: dict[str, Any] = {
        "isolation": "worktree",
        "permission_mode": "acceptEdits",
        "max_turns": 60,
        "timeout": 1800,
        # The grader runs after the adapter returns and needs the tree.
        "keep_workspace": True,
    }
    config.update(agent_config or {})

    return Scenario(
        name=name,
        description=f"{len(instances)} SWE-bench instance(s)",
        agent=AgentConfig(adapter="claude_code", config=config),
        defaults=DefaultsConfig(trials=trials, timeout=2400),
        default_graders=graders,
        cases=[
            instance_to_case(i, repo_root=repo_root, **case_kwargs)
            for i in instances
        ],
        tags=["swebench"],
    )


def default_process_graders() -> list[GraderConfig]:
    """Process guards worth having on every instance.

    None of these decide whether the instance was resolved — that is
    ``swebench_tests``, and it is the only gate. These answer the question the
    resolution rate cannot: *what did it cost to get there*. Two configurations
    that both resolve 45% are not equivalent if one spends triple.

    The thresholds are placeholders. Set them from your own observed
    distribution — thresholds copied from someone else's repo produce confident
    numbers about nothing.
    """
    return [
        GraderConfig(
            type=GraderType.CODE, name="cost_budget", config={"max_cost_usd": 4.0}
        ),
        GraderConfig(
            type=GraderType.CODE, name="turn_count", config={"max_turns": 60}
        ),
        GraderConfig(
            type=GraderType.CODE,
            name="loop_detection",
            # Re-running a test command is normal work for a coding agent;
            # only flag the same call repeating with no progress.
            config={"max_exact_repetitions": 3, "check_output_repetition": False},
        ),
    ]


def clone_commands(
    instances: list[dict[str, Any]], repo_root: str | Path
) -> list[str]:
    """Shell commands that materialise the checkouts these instances need.

    One clone per repository, not per instance — a repo with forty instances is
    cloned once and each trial takes a worktree off it.
    """
    root = Path(repo_root).expanduser()
    seen: dict[str, str] = {}
    for instance in instances:
        seen.setdefault(instance["repo"], instance["repo"].replace("/", "__"))
    return [
        f"git clone https://github.com/{repo} {root / directory}"
        for repo, directory in sorted(seen.items())
    ]


# ---------------------------------------------------------------------------


def _format_image(template: str, instance: dict[str, Any]) -> str:
    return template.format(
        instance_id=instance["instance_id"],
        repo_underscored=instance["repo"].replace("/", "_1776_"),
    )


def _first_line(text: str, limit: int = 100) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:limit]
    return ""
