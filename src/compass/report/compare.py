"""Paired comparison between two evaluation runs.

Treats an eval comparison as a *measurement*, not a readout: a 2.4-point move
in an average can be a real improvement or pure noise. This module pairs the
two runs case-by-case and reports:

- which cases flipped (pass -> fail is a regression even if the average went up)
- the paired mean difference with a confidence interval, so a "gain" that sits
  inside the noise band is discounted
- the minimum detectable effect (MDE) at the current sample size, so "not
  significant" is distinguishable from "not enough cases to tell"

Cases present on only one side are reported, never silently dropped.

By default the per-case number is ``overall_score``. That is the right default
and the wrong one for a suite that gates on correctness: gate graders are
excluded from the weighted average by definition (see ``GraderConfig``), so in
such a suite ``overall_score`` measures process cost and the correctness signal
never reaches the comparison at all. ``on=`` scopes the whole comparison to one
grader instead — its score, its pass/fail, its flips.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Two-sided 95% confidence
Z_ALPHA = 1.96
# 80% power for the minimum detectable effect
Z_BETA = 0.84


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@dataclass
class CaseRecord:
    """Normalized per-case record extracted from a results file."""

    case_id: str
    passed: bool
    score: float
    # Per-case pass fraction: passed_trials/evaluated_trials when multiple
    # trials were run, else 1.0/0.0 from `passed`. Finer than the boolean.
    pass_fraction: float
    # Content hash of the grading contract this case was scored under; ""
    # for results written before fingerprints existed.
    grader_fingerprint: str = ""


class AmbiguousGraderError(ValueError):
    """``on=`` matched more than one grader in the same case.

    Not resolvable by picking one: the two are different measurements, and
    guessing would produce a comparison that looks fine and answers a question
    nobody asked. The fix belongs in the scenario — give the grader instances
    distinct ``label``s — so this is raised rather than worked around.
    """


def select_grader_score(
    item: dict[str, Any], on: str
) -> tuple[float, bool] | None:
    """``(score, passed)`` for the grader *on* names, or None if unmeasured.

    Matches ``label`` first, then falls back to ``name`` — so a scenario that
    labels its grader instances can select them precisely, and one that does
    not can still select by grader name when the name is unique in the case.

    None means "this case never measured that", which is deliberately distinct
    from a measured zero: the grader was not configured for this case, was
    skipped by a short-circuit, or crashed. Scoring those as 0.0 would report
    an agent failure where the truth is a missing sample.
    """
    graders = item.get("evaluator_results") or item.get("grade_results") or []
    matches = [
        g for g in graders
        if (g.get("label") or "") == on or (not g.get("label") and g.get("name") == on)
    ]
    if not matches:
        # A labelled grader is still findable by its registry name, as long as
        # that name is unambiguous within the case.
        matches = [g for g in graders if g.get("name") == on]
    if not matches:
        return None
    if len(matches) > 1:
        raise AmbiguousGraderError(
            f"{item.get('case_id') or item.get('task_id')!r}: {len(matches)} graders "
            f"match {on!r}. Give them distinct `label:` values in the scenario and "
            f"select on those."
        )

    match = matches[0]
    if match.get("skipped") or match.get("score") is None:
        return None
    return float(match["score"]), bool(match.get("passed"))


def select_metric(item: dict[str, Any], metric: str) -> float | None:
    """The value of *metric* for one case, or None if it was not measured.

    Metrics live in ``EvaluatorResult.metrics`` — the core-owned channel, as
    opposed to ``details``, which is the grader's own payload and may hold
    anything. Booleans are read as 1.0/0.0 so a flag compares as a rate.

    A metric emitted by two graders in the same case is an error rather than a
    guess, for the same reason an ambiguous ``on`` is: silently picking one
    produces a comparison that looks fine and answers a question nobody asked.
    """
    graders = item.get("evaluator_results") or item.get("grade_results") or []
    found: list[tuple[str, float]] = []
    for g in graders:
        if g.get("skipped"):
            continue
        value = (g.get("metrics") or {}).get(metric)
        if isinstance(value, bool):
            found.append((g.get("label") or g.get("name") or "?", float(value)))
        elif isinstance(value, (int, float)):
            found.append((g.get("label") or g.get("name") or "?", float(value)))

    if not found:
        return None
    if len({v for _, v in found}) > 1:
        raise AmbiguousGraderError(
            f"{item.get('case_id') or item.get('task_id')!r}: metric {metric!r} was "
            f"emitted with different values by "
            f"{', '.join(sorted(n for n, _ in found))}. Rename one of them."
        )
    return found[0][1]


def _extract_case_records(
    item: dict[str, Any], on: str | None = None
) -> CaseRecord | None:
    """Normalize one case-level dict, or None if it doesn't look like one.

    With *on* set, the record describes that one grader rather than the case:
    its score and its pass/fail. Returns None when the grader did not measure
    this case, which drops the pair out of the statistics — see
    :func:`select_grader_score`.
    """
    case_id = item.get("case_id") or item.get("task_id")
    if not case_id or "passed" not in item:
        return None

    if on:
        selected = select_grader_score(item, on)
        if selected is None:
            return None
        score, passed = selected
        # Trial counts are recorded per case, not per grader, so the finer
        # pass fraction is not available here — a 3-of-5 case cannot be split
        # into which trials this particular grader passed.
        pass_fraction = 1.0 if passed else 0.0
    else:
        passed = bool(item["passed"])
        score = float(
            item.get("overall_score", item.get("score", 1.0 if passed else 0.0))
        )

        total_trials = item.get("total_trials", 1) or 1
        # Trials lost to harness errors are missing samples, not failed
        # attempts — leaving them in the denominator would read as a drop.
        evaluated_trials = total_trials - (item.get("error_trials") or 0)
        if total_trials > 1 and evaluated_trials > 0:
            pass_fraction = item.get("passed_trials", 0) / evaluated_trials
        else:
            pass_fraction = 1.0 if passed else 0.0

    return CaseRecord(
        case_id=str(case_id),
        passed=passed,
        score=float(score),
        pass_fraction=pass_fraction,
        grader_fingerprint=str(item.get("grader_fingerprint") or ""),
    )


def _load(
    path: str | Path, on: str | None = None
) -> tuple[dict[str, CaseRecord], list[str]]:
    """``(records, unmeasured)`` — see :func:`load_case_records`.

    *unmeasured* holds case ids present in the file that produced no
    measurement for *on*. Kept rather than dropped so a caller can say how much
    of the run the selected grader actually covers: a comparison over 3 of 20
    cases and one over 20 of 20 look identical otherwise.
    """
    from compass.report.analyzer import iter_case_dicts

    path = Path(path)
    json_files = [path] if path.is_file() else sorted(path.glob("**/*.json"))

    records: dict[str, CaseRecord] = {}
    unmeasured: list[str] = []
    for jf in json_files:
        try:
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Skipping %s: %s", jf, e)
            continue

        for item in iter_case_dicts(data):
            record = _extract_case_records(item, on)
            if record is not None:
                records[record.case_id] = record
            elif on and (item.get("case_id") or item.get("task_id")):
                unmeasured.append(str(item.get("case_id") or item.get("task_id")))

    return records, sorted(set(unmeasured) - records.keys())


def load_case_records(
    path: str | Path, on: str | None = None
) -> dict[str, CaseRecord]:
    """Load case records from a results file or directory.

    Accepts the same shapes as ``compass analyze`` — see
    :func:`compass.report.analyzer.iter_case_dicts`, the shared reader — plus a
    directory of such JSON files. Unrecognized files/items are skipped with a
    warning. On duplicate case_ids the last record wins.

    With *on* set, each record describes that one grader rather than the whole
    case, and cases the grader did not measure are absent. Use :func:`_load`
    when you need to know which those were.
    """
    return _load(path, on)[0]


# ---------------------------------------------------------------------------
# Paired statistics
# ---------------------------------------------------------------------------


@dataclass
class PairedStats:
    """Paired mean difference (B - A) with uncertainty."""

    n: int
    mean_diff: float
    ci_low: float
    ci_high: float
    significant: bool  # 95% CI excludes 0
    mde: float  # minimum detectable effect at this n/variance (80% power)
    #: Whether the sample showed any spread at all. When every case moved by
    #: the same amount the variance estimate is 0, which drives the CI and the
    #: MDE to 0 — an artifact, not a precise measurement. Reporting "detectable
    #: at n=5: ~0.0%" would claim infinite resolution from five identical
    #: numbers, so callers must say "not estimable" instead.
    variance_observed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "mean_diff": self.mean_diff,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "significant": self.significant,
            "mde": self.mde,
            "variance_observed": self.variance_observed,
        }


def paired_stats(diffs: list[float]) -> PairedStats:
    """Compute the paired mean difference with a 95% CI and the MDE.

    Normal approximation on the per-case differences. With n < 2 (or zero
    variance) the CI collapses to the point estimate; significance then just
    means the mean is nonzero — treat small-n results with suspicion, which
    is exactly what the reported MDE is for.
    """
    n = len(diffs)
    if n == 0:
        return PairedStats(0, 0.0, 0.0, 0.0, False, math.inf, variance_observed=False)

    mean = sum(diffs) / n
    if n > 1:
        var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
        se = math.sqrt(var / n)
    else:
        se = 0.0

    ci_low = mean - Z_ALPHA * se
    ci_high = mean + Z_ALPHA * se
    significant = ci_low > 0 or ci_high < 0
    mde = (Z_ALPHA + Z_BETA) * se

    return PairedStats(
        n=n,
        mean_diff=mean,
        ci_low=ci_low,
        ci_high=ci_high,
        significant=significant,
        mde=mde,
        # se == 0 means either n == 1 or every case moved identically. Either
        # way the spread was not measured, so the MDE is not a resolution.
        variance_observed=se > 0,
    )


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


@dataclass
class CaseFlip:
    """A case whose pass/fail verdict differs between the two runs."""

    case_id: str
    direction: str  # "improved" (A fail -> B pass) | "regressed" (A pass -> B fail)
    score_a: float
    score_b: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "direction": self.direction,
            "score_a": self.score_a,
            "score_b": self.score_b,
        }


@dataclass
class ComparisonReport:
    """Result of a paired comparison of run A (baseline) vs run B (candidate)."""

    label_a: str
    label_b: str
    n_paired: int
    only_in_a: list[str] = field(default_factory=list)
    only_in_b: list[str] = field(default_factory=list)

    #: The grader this comparison is scoped to, or "" for ``overall_score``.
    on: str = ""
    #: Cases the selected grader did not measure, so they carry no sample.
    #: Distinct from ``only_in_*``: the case ran on both sides, but this
    #: particular grader produced no number for it.
    unmeasured_a: list[str] = field(default_factory=list)
    unmeasured_b: list[str] = field(default_factory=list)

    pass_rate_a: float = 0.0
    pass_rate_b: float = 0.0
    mean_score_a: float = 0.0
    mean_score_b: float = 0.0

    improved: list[CaseFlip] = field(default_factory=list)
    regressed: list[CaseFlip] = field(default_factory=list)
    both_pass: int = 0
    both_fail: int = 0

    pass_stats: PairedStats | None = None
    score_stats: PairedStats | None = None

    #: Paired cases whose grading contract differs between A and B. Their
    #: score delta is at least partly the grader moving, not the agent — the
    #: comparison is not a clean attribution for these cases.
    regraded: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        """One-line reading of the measurement.

        Pass rate is the primary axis; the score CI refines the "no verdict"
        band. "not significant" explicitly carries the MDE so a null result
        at n=5 is not mistaken for evidence of no change.
        """
        if self.n_paired == 0:
            return "no paired cases — nothing to compare"
        assert self.pass_stats is not None and self.score_stats is not None

        if self.pass_stats.significant:
            direction = "improvement" if self.pass_stats.mean_diff > 0 else "regression"
            return (
                f"significant pass-rate {direction}: "
                f"{self.pass_stats.mean_diff:+.1%} "
                f"(95% CI [{self.pass_stats.ci_low:+.1%}, {self.pass_stats.ci_high:+.1%}])"
            )
        if self.score_stats.significant:
            direction = "improvement" if self.score_stats.mean_diff > 0 else "regression"
            return (
                f"pass rate within noise band, but significant score {direction}: "
                f"{self.score_stats.mean_diff:+.3f} "
                f"(95% CI [{self.score_stats.ci_low:+.3f}, {self.score_stats.ci_high:+.3f}])"
            )
        if not self.pass_stats.variance_observed:
            # Every case agreed. The CI and MDE both collapse to 0, which would
            # otherwise read as "this run resolves arbitrarily small effects".
            return (
                f"no disagreement on {self.n_paired} paired case(s): pass-rate "
                f"diff {self.pass_stats.mean_diff:+.1%}, and with zero spread the "
                f"resolution cannot be estimated from this sample — add cases that "
                f"the two runs might answer differently"
            )
        return (
            f"within noise band: pass-rate diff {self.pass_stats.mean_diff:+.1%} "
            f"(95% CI [{self.pass_stats.ci_low:+.1%}, {self.pass_stats.ci_high:+.1%}]); "
            f"detectable at n={self.n_paired}: ~{self.pass_stats.mde:.1%}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "label_a": self.label_a,
            "label_b": self.label_b,
            "n_paired": self.n_paired,
            "on": self.on,
            "only_in_a": self.only_in_a,
            "only_in_b": self.only_in_b,
            "unmeasured_a": self.unmeasured_a,
            "unmeasured_b": self.unmeasured_b,
            "pass_rate_a": self.pass_rate_a,
            "pass_rate_b": self.pass_rate_b,
            "mean_score_a": self.mean_score_a,
            "mean_score_b": self.mean_score_b,
            "improved": [f.to_dict() for f in self.improved],
            "regressed": [f.to_dict() for f in self.regressed],
            "both_pass": self.both_pass,
            "both_fail": self.both_fail,
            "pass_stats": self.pass_stats.to_dict() if self.pass_stats else None,
            "score_stats": self.score_stats.to_dict() if self.score_stats else None,
            "regraded": self.regraded,
            "verdict": self.verdict,
        }


def compare_results(
    records_a: dict[str, CaseRecord],
    records_b: dict[str, CaseRecord],
    label_a: str = "A",
    label_b: str = "B",
    on: str = "",
) -> ComparisonReport:
    """Pair two runs by case_id and compute flips + paired statistics.

    Only cases present in both runs enter the statistics; the rest are listed
    in ``only_in_a`` / ``only_in_b`` so coverage changes are visible.

    *on* is recorded for reporting only — the records must already have been
    built for that grader by :func:`load_case_records`.
    """
    paired_ids = sorted(records_a.keys() & records_b.keys())
    report = ComparisonReport(
        label_a=label_a,
        label_b=label_b,
        n_paired=len(paired_ids),
        on=on,
        only_in_a=sorted(records_a.keys() - records_b.keys()),
        only_in_b=sorted(records_b.keys() - records_a.keys()),
    )
    if not paired_ids:
        return report

    pairs = [(records_a[cid], records_b[cid]) for cid in paired_ids]

    report.pass_rate_a = sum(a.pass_fraction for a, _ in pairs) / len(pairs)
    report.pass_rate_b = sum(b.pass_fraction for _, b in pairs) / len(pairs)
    report.mean_score_a = sum(a.score for a, _ in pairs) / len(pairs)
    report.mean_score_b = sum(b.score for _, b in pairs) / len(pairs)

    for a, b in pairs:
        if a.passed and not b.passed:
            report.regressed.append(
                CaseFlip(a.case_id, "regressed", a.score, b.score)
            )
        elif not a.passed and b.passed:
            report.improved.append(
                CaseFlip(a.case_id, "improved", a.score, b.score)
            )
        elif a.passed:
            report.both_pass += 1
        else:
            report.both_fail += 1

    # Attribution check: a case scored under two different grading contracts
    # is not a clean A/B on the agent. Only flag when both sides recorded a
    # fingerprint — an older results file simply doesn't know.
    report.regraded = [
        a.case_id
        for a, b in pairs
        if a.grader_fingerprint
        and b.grader_fingerprint
        and a.grader_fingerprint != b.grader_fingerprint
    ]

    report.pass_stats = paired_stats(
        [b.pass_fraction - a.pass_fraction for a, b in pairs]
    )
    report.score_stats = paired_stats([b.score - a.score for a, b in pairs])
    return report


@dataclass
class MetricComparison:
    """Paired comparison of one numeric metric across two runs.

    Deliberately not a ``ComparisonReport``. A metric has no pass/fail, so
    there are no flips to list and no pass rate to report, and for most process
    metrics — turns, cost, tool calls — *lower* is better, which inverts what a
    positive difference means. Reusing the score report's shape would invite
    every one of those misreadings.
    """

    metric: str
    label_a: str
    label_b: str
    n_paired: int
    mean_a: float
    mean_b: float
    stats: PairedStats
    unmeasured_a: list[str] = field(default_factory=list)
    unmeasured_b: list[str] = field(default_factory=list)
    only_in_a: list[str] = field(default_factory=list)
    only_in_b: list[str] = field(default_factory=list)
    #: Paired cases that ran more than one trial. A case's grader results are
    #: one trial's, not an average over them, so on a multi-trial run this
    #: comparison silently samples one attempt per case — which is a different
    #: and noisier measurement than the mean the reader will assume.
    multi_trial_cases: int = 0

    @property
    def verdict(self) -> str:
        if self.n_paired == 0:
            return f"no case measured {self.metric!r} in both runs"
        if not self.stats.variance_observed:
            return (
                f"{self.metric}: every paired case moved identically "
                f"({self.stats.mean_diff:+.4g}) — with zero spread the "
                f"resolution cannot be estimated from this sample"
            )
        if self.stats.significant:
            direction = "higher" if self.stats.mean_diff > 0 else "lower"
            return (
                f"{self.metric}: B is significantly {direction} — "
                f"{self.stats.mean_diff:+.4g} "
                f"(95% CI [{self.stats.ci_low:+.4g}, {self.stats.ci_high:+.4g}])"
            )
        return (
            f"{self.metric}: within noise band — {self.stats.mean_diff:+.4g} "
            f"(95% CI [{self.stats.ci_low:+.4g}, {self.stats.ci_high:+.4g}]); "
            f"detectable at n={self.n_paired}: ~{self.stats.mde:.4g}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "label_a": self.label_a,
            "label_b": self.label_b,
            "n_paired": self.n_paired,
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "stats": self.stats.to_dict(),
            "unmeasured_a": self.unmeasured_a,
            "unmeasured_b": self.unmeasured_b,
            "only_in_a": self.only_in_a,
            "only_in_b": self.only_in_b,
            "multi_trial_cases": self.multi_trial_cases,
            "verdict": self.verdict,
        }


def _load_metric(
    path: str | Path, metric: str, missing_as_zero: bool = False
) -> tuple[dict[str, float], list[str], set[str]]:
    """``(values, unmeasured, multi_trial)`` for one metric.

    *missing_as_zero* resolves a genuine ambiguity the caller has to settle,
    because both readings are right for different metrics. ``tool_usage``
    emits ``calls_<tool>`` only for tools that were actually called, so an
    absent ``calls_grep`` means "never grepped" — a measured zero. But an
    absent ``cost_usd`` means the cost grader never ran, which is not a run
    that cost nothing. Defaulting either way silently invents evidence, so the
    flag is explicit and off by default.
    """
    from compass.report.analyzer import iter_case_dicts

    path = Path(path)
    json_files = [path] if path.is_file() else sorted(path.glob("**/*.json"))

    values: dict[str, float] = {}
    unmeasured: list[str] = []
    multi_trial: set[str] = set()
    for jf in json_files:
        try:
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Skipping %s: %s", jf, e)
            continue
        for item in iter_case_dicts(data):
            case_id = item.get("case_id") or item.get("task_id")
            if not case_id:
                continue
            value = select_metric(item, metric)
            if value is None and missing_as_zero and _measured_anything(item):
                value = 0.0
            if value is None:
                unmeasured.append(str(case_id))
            else:
                values[str(case_id)] = value
                if (item.get("total_trials") or 1) > 1:
                    multi_trial.add(str(case_id))
    return values, sorted(set(unmeasured) - values.keys()), multi_trial


def _measured_anything(item: dict[str, Any]) -> bool:
    """Whether any grader on this case emitted metrics at all.

    The guard on ``missing_as_zero``: a case where nothing was measured must
    stay unmeasured, or a run that never graded would read as a run of zeros.
    """
    graders = item.get("evaluator_results") or item.get("grade_results") or []
    return any(g.get("metrics") for g in graders if not g.get("skipped"))


def compare_metric(
    path_a: str | Path,
    path_b: str | Path,
    metric: str,
    missing_as_zero: bool = False,
) -> MetricComparison:
    """Pair two runs by case_id and compare one metric.

    Raises:
        AmbiguousGraderError: two graders emitted *metric* with different
            values in the same case.
    """
    a, unmeasured_a, multi_a = _load_metric(path_a, metric, missing_as_zero)
    b, unmeasured_b, multi_b = _load_metric(path_b, metric, missing_as_zero)
    paired = sorted(a.keys() & b.keys())
    diffs = [b[c] - a[c] for c in paired]

    return MetricComparison(
        metric=metric,
        label_a=str(path_a),
        label_b=str(path_b),
        n_paired=len(paired),
        mean_a=sum(a[c] for c in paired) / len(paired) if paired else 0.0,
        mean_b=sum(b[c] for c in paired) / len(paired) if paired else 0.0,
        stats=paired_stats(diffs),
        unmeasured_a=unmeasured_a,
        unmeasured_b=unmeasured_b,
        only_in_a=sorted(a.keys() - b.keys()),
        only_in_b=sorted(b.keys() - a.keys()),
        multi_trial_cases=len((multi_a | multi_b) & set(paired)),
    )


def compare_paths(
    path_a: str | Path,
    path_b: str | Path,
    on: str | None = None,
) -> ComparisonReport:
    """Load two results paths and compare them (A = baseline, B = candidate).

    With *on*, the comparison is scoped to one grader — its score, its
    pass/fail, its flips — instead of the case-level ``overall_score``.

    Raises:
        AmbiguousGraderError: *on* matches more than one grader in some case.
    """
    records_a, unmeasured_a = _load(path_a, on)
    records_b, unmeasured_b = _load(path_b, on)
    report = compare_results(
        records_a,
        records_b,
        label_a=str(path_a),
        label_b=str(path_b),
        on=on or "",
    )
    report.unmeasured_a = unmeasured_a
    report.unmeasured_b = unmeasured_b
    return report
