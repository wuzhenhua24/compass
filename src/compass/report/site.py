"""Report data layer: one run in, one plain document out.

``collect_run`` turns a run's ``EvalResult`` objects into a JSON-serializable
document. The HTML report renders that document and nothing else — no viewer
reaches into result objects. Keeping the extraction in one place is what lets a
page rendered now, a page written to disk, and a page aggregated across
repositories all show the same numbers.

Conventions the document commits to:

- Scores and rates are fractions in ``0..1``, never percentages. Formatting is
  a viewer's job.
- Harness errors are excluded from every denominator, matching
  ``EvalResult.pass_rate``: a crashed adapter is not the agent getting it wrong,
  and counting it as one lets infrastructure flakiness masquerade as a
  regression.
- A grader that was skipped, or that ran without producing a number, is left
  out of averages rather than counted as zero. "Not measured" is not
  "measured as zero".

Case rows are a near-copy of ``EvalResult.to_dict()``'s case records, so
anything that reads a Compass results file (``analyzer.iter_case_dicts`` and
everything downstream of it) reads these too. The differences are deliberate:
the ``grade_results`` alias is dropped (it duplicates ``evaluator_results``
verbatim, and a published document pays for every byte twice), and each row
gains its scenario name plus the two scope axes.
"""

from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from compass.core.result import EvalResult, TestStatus

#: Bumped when the document shape changes incompatibly. A viewer reads this
#: before anything else, so an old page can refuse a new document instead of
#: rendering it wrong.
SCHEMA = "compass.run/1"

_OUTCOME_SCOPES = ("outcome", "both")
_TRANSCRIPT_SCOPES = ("transcript", "both")
_ERROR = TestStatus.ERROR.value


def now_iso() -> str:
    """UTC timestamp, second precision — unambiguous once a page is shared."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _measured(grader: Mapping[str, Any]) -> bool:
    """Whether a grader record carries a usable measurement.

    Mirrors ``EvaluatorResult.scored`` but reads the serialized record, so the
    same rule applies to results loaded back from disk.
    """
    return not grader.get("skipped", False) and grader.get("score") is not None


def scope_scores(graders: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Project one case's grader scores onto the Outcome and Transcript axes.

    A ``both``-scope grader contributes to both axes. Axis presence is reported
    separately from the average: a run where every outcome grader legitimately
    scored 0.0 still *has* an outcome axis, and a viewer must be able to tell
    that apart from a run with no outcome graders at all.
    """
    outcome: list[float] = []
    transcript: list[float] = []

    for grader in graders:
        if not _measured(grader):
            continue
        scope = grader.get("grader_scope", "outcome")
        score = float(grader["score"])
        if scope in _OUTCOME_SCOPES:
            outcome.append(score)
        if scope in _TRANSCRIPT_SCOPES:
            transcript.append(score)

    return {
        "outcome_score": _mean(outcome),
        "transcript_score": _mean(transcript),
        "has_outcome": bool(outcome),
        "has_transcript": bool(transcript),
    }


def iter_cases(doc: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    """Every case row in the document, in scenario order.

    Rows live under their scenario so the document has exactly one home for
    each; this is the flat view a chart or a filter wants.
    """
    for scenario in doc.get("scenarios", []):
        yield from scenario.get("cases", [])


def category_rows(cases: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Per-category breakdown, sorted by category name.

    Uncategorized cases are omitted rather than bucketed under a placeholder —
    an empty result means "nobody labelled anything", which is a viewer's cue
    to hide the table entirely.
    """
    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for case in cases:
        category = case.get("category")
        if category:
            buckets[category].append(case)

    rows = []
    for category in sorted(buckets):
        group = buckets[category]
        evaluated = [c for c in group if c.get("status") != _ERROR]
        passed = sum(1 for c in evaluated if c.get("passed"))
        rows.append(
            {
                "category": category,
                "total": len(group),
                "errors": len(group) - len(evaluated),
                "evaluated": len(evaluated),
                "passed": passed,
                "failed": len(evaluated) - passed,
                "pass_rate": _rate(passed, len(evaluated)),
                "average_score": _mean(
                    [float(c.get("overall_score") or 0.0) for c in evaluated]
                ),
            }
        )
    return rows


def _scenario_block(result: EvalResult) -> dict[str, Any]:
    """One scenario's summary and its case rows."""
    payload = result.to_dict()

    cases = []
    for record in payload["case_results"]:
        row = dict(record)
        row.pop("grade_results", None)  # verbatim alias of evaluator_results
        row["scenario"] = result.scenario_name
        row.update(scope_scores(row.get("evaluator_results") or []))
        cases.append(row)

    return {
        "name": result.scenario_name,
        "run_id": result.run_id,
        "config_hash": result.config_hash,
        "total_cases": result.total_cases,
        "passed_cases": result.passed_cases,
        "failed_cases": result.failed_cases,
        "error_cases": result.error_cases,
        "evaluated_cases": result.evaluated_cases,
        "pass_rate": result.pass_rate,
        "average_score": result.average_score,
        "best_of_k_score": result.best_of_k_score,
        "duration_ms": result.duration_ms,
        "timestamp": payload["timestamp"],
        "cases": cases,
    }


def collect_run(
    results: Sequence[EvalResult],
    *,
    name: str = "",
    generated: str | None = None,
) -> dict[str, Any]:
    """Everything a report needs about one run, as one plain document.

    Args:
        results: The run's scenario results.
        name: Optional label for the run as a whole (a viewer's heading, and
            later the entry a static site indexes it under).
        generated: Override the timestamp — for reproducible output.

    Returns:
        A JSON-serializable dict; see the module docstring for the conventions
        its numbers follow.
    """
    scenarios = [_scenario_block(result) for result in results]
    cases = [case for scenario in scenarios for case in scenario["cases"]]
    evaluated = [case for case in cases if case.get("status") != _ERROR]

    passed_cases = sum(s["passed_cases"] for s in scenarios)
    evaluated_cases = sum(s["evaluated_cases"] for s in scenarios)

    return {
        "schema": SCHEMA,
        "generated": generated or now_iso(),
        "run": {
            "name": name,
            "scenarios": len(scenarios),
            "total_cases": sum(s["total_cases"] for s in scenarios),
            "passed_cases": passed_cases,
            "failed_cases": sum(s["failed_cases"] for s in scenarios),
            "error_cases": sum(s["error_cases"] for s in scenarios),
            "evaluated_cases": evaluated_cases,
            "pass_rate": _rate(passed_cases, evaluated_cases),
            # Means over cases, not over scenario means: scenarios differ in
            # size, so averaging their averages would over-weight small ones.
            "average_score": _mean(
                [float(c.get("overall_score") or 0.0) for c in evaluated]
            ),
            "best_of_k_score": _mean(
                [float(c.get("best_score") or 0.0) for c in evaluated]
            ),
            "duration_ms": sum(s["duration_ms"] for s in scenarios),
        },
        "scenarios": scenarios,
        "categories": category_rows(cases),
        "scopes": {
            "outcome": any(c["has_outcome"] for c in cases),
            "transcript": any(c["has_transcript"] for c in cases),
        },
    }
