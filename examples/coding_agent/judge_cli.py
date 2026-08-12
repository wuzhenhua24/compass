#!/usr/bin/env python3
"""Judge the agent's closing message with the ``claude`` CLI.

    - name: external_checker
      type: code
      label: surfaces_assumption
      config:
        command: ["<abs>/judge_cli.py"]
        judge_model: sonnet
        criteria: [{name: ..., weight: 2.0, description: ...}]

Why a subprocess and not the built-in ``rubric`` grader: every model grader in
Compass talks to a provider SDK, and an SDK call needs an API key. A Claude
subscription authenticates the **CLI**, not the SDK — so on a subscription the
only judge you actually have is ``claude -p``. (The first version of this suite
used ``rubric`` without setting ``provider``, which defaults to OpenAI; all four
judgements in the first paid run died with "openai package required" and took
two cases down with them.)

The contract is ``external_checker``'s: read ``COMPASS_ANSWER`` and
``COMPASS_CONFIG`` from the environment, print a JSON verdict, exit 0 for pass.

Three things this is careful about, all of them the same mistake in different
clothes — reporting something about the agent when the truth is about us:

  - **A judge that could not run is not an agent that scored zero.** Any
    failure to reach the CLI, parse its answer, or read a sane score prints
    ``{"unscored": true}``, which Compass keeps out of the score denominator.
  - **The judge model is pinned in config, never taken from the run.** Judging
    each arm of a comparison with its own model compares judges, not agents.
  - **An empty answer is a measurement, not an error.** An agent that finished
    without saying anything genuinely failed to surface its assumptions.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

DEFAULT_JUDGE_MODEL = "sonnet"
DEFAULT_TIMEOUT = 120

_INSTRUCTIONS = """You are grading one criterion at a time, strictly.

TASK GIVEN TO THE AGENT
-----------------------
{prompt}

THE AGENT'S CLOSING MESSAGE
---------------------------
{answer}

CRITERIA
--------
{criteria}

Each criterion describes what a message that MEETS it says. Score 1.0 when the
message says that, 0.5 when it says it only partially or vaguely, and 0.0 when
it does not.

Read this next part carefully, because it is the mistake graders make here:

  - A criterion naming a specific conclusion is met only by **that**
    conclusion. A message that argues the opposite scores 0.0 on it, no matter
    how clearly, confidently or thoroughly it argues. Confident and wrong is
    the failure this grading exists to catch — never award it partial credit
    for being well reasoned.
  - Grade only what the message actually says. Do not credit what the agent
    probably knew, meant, or did in its edits.
  - Ignore length, formatting, tone and confidence entirely. A short blunt
    message that meets a criterion beats a polished one that does not.

Reply with ONE JSON object and nothing else:

{{"scores": {{{score_keys}}}, "reasoning": "<one sentence>"}}"""


def fail_unscored(note: str) -> None:
    """Tell Compass we learned nothing, rather than blaming the agent."""
    print(json.dumps({"unscored": True, "notes": note}, ensure_ascii=False))
    sys.exit(1)


def main() -> int:
    try:
        config = json.loads(os.environ.get("COMPASS_CONFIG") or "{}")
    except json.JSONDecodeError as e:
        fail_unscored(f"COMPASS_CONFIG is not valid JSON: {e}")

    criteria = config.get("criteria") or []
    if not criteria:
        fail_unscored("no criteria configured")

    answer = os.environ.get("COMPASS_ANSWER", "")
    if not answer.strip():
        # A real verdict: the agent ended the run without saying anything, so
        # it surfaced nothing. Distinct from the judge being unable to look.
        print(json.dumps({
            "score": 0.0,
            "notes": "the agent produced no closing message",
            "details": {"answer_chars": 0},
        }, ensure_ascii=False))
        return 1

    prompt = _INSTRUCTIONS.format(
        prompt=os.environ.get("COMPASS_PROMPT", "(not recorded)"),
        answer=answer[:12000],
        criteria="\n".join(
            f"- {c['name']} (weight {c.get('weight', 1.0)}): {c['description'].strip()}"
            for c in criteria
        ),
        score_keys=", ".join(f'"{c["name"]}": <0.0-1.0>' for c in criteria),
    )

    argv = [
        "claude", "-p", prompt,
        "--output-format", "json",
        "--model", str(config.get("judge_model", DEFAULT_JUDGE_MODEL)),
        # No tools at all. Judging a block of text needs none, and a judge with
        # Read available goes and looks at the repository — which is judging
        # the diff, not the message it was asked about.
        #
        # `--tools ""` is the flag that does this. `--allowed-tools ""` only
        # filters an already-enabled set and leaves them on, and pairing that
        # with `--max-turns 1` made the CLI exit 1 with an empty stderr the
        # moment the judge reached for a tool — one lost judgement per run,
        # intermittent, and invisible except as an unscored case.
        "--tools", "",
    ]
    timeout = float(config.get("judge_timeout", DEFAULT_TIMEOUT))

    # One retry, because the whole judgement is lost to a single malformed
    # reply and a rerun is far cheaper than an unscored case.
    verdict, last = None, ""
    for _ in range(2):
        try:
            run = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        except FileNotFoundError:
            fail_unscored("the `claude` CLI is not on PATH")
        except subprocess.TimeoutExpired:
            fail_unscored("the judge model timed out")

        if run.returncode != 0:
            fail_unscored(
                f"claude CLI exited {run.returncode}: {run.stderr.strip()[:300]}"
            )

        verdict = _extract_verdict(run.stdout)
        if verdict is not None:
            break
        last = run.stdout

    if verdict is None:
        fail_unscored(f"could not parse a verdict from the judge: {last[:300]}")

    scored, missing = {}, []
    for criterion in criteria:
        raw = verdict.get("scores", {}).get(criterion["name"])
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            missing.append(criterion["name"])
            continue
        scored[criterion["name"]] = max(0.0, min(1.0, float(raw)))

    if missing:
        # A partial verdict is not a low score — it is a judge that did not do
        # the job it was asked to do.
        fail_unscored(f"the judge did not score: {', '.join(missing)}")

    total_weight = sum(float(c.get("weight", 1.0)) for c in criteria)
    score = sum(
        scored[c["name"]] * float(c.get("weight", 1.0)) for c in criteria
    ) / total_weight

    print(json.dumps({
        "score": round(score, 4),
        "notes": str(verdict.get("reasoning", ""))[:500],
        "details": {"criteria": scored, "judge_model": argv[-1]},
    }, ensure_ascii=False))
    # Pass/fail is the exit code; the threshold belongs to the scenario, not
    # here, so anything short of every criterion fully met is reported as a
    # non-pass and the weighted score carries the nuance.
    return 0 if score >= float(config.get("pass_threshold", 0.75)) else 1


def _extract_verdict(stdout: str) -> dict | None:
    """The judge's JSON, out of the CLI's JSON envelope.

    Tolerant on purpose. A model told to emit bare JSON mostly does, but
    occasionally fences it or writes a sentence first, and throwing away a
    whole judgement over that is a self-inflicted unscored result.

    Every ``{`` is tried as the start of an object rather than regex-matching
    the outermost braces: the CLI's own envelope is an object too, so a greedy
    match finds it, parses it happily, and reports no verdict — which is how
    this failed the first time it met a multi-criterion reply.
    """
    text = stdout
    try:
        envelope = json.loads(stdout)
        if isinstance(envelope, dict) and isinstance(envelope.get("result"), str):
            text = envelope["result"]
    except json.JSONDecodeError:
        pass

    text = re.sub(r"```(?:json)?|```", "", text)

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            parsed, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("scores"), dict):
            return parsed
    return None


if __name__ == "__main__":
    raise SystemExit(main())
