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
    # Per-case pass fraction: passed_trials/total_trials when multiple trials
    # were run, else 1.0/0.0 from `passed`. Finer-grained than the boolean.
    pass_fraction: float


def _extract_case_records(item: dict[str, Any]) -> CaseRecord | None:
    """Normalize one case-level dict, or None if it doesn't look like one."""
    case_id = item.get("case_id") or item.get("task_id")
    if not case_id or "passed" not in item:
        return None

    passed = bool(item["passed"])
    score = item.get("overall_score", item.get("score", 1.0 if passed else 0.0))

    total_trials = item.get("total_trials", 1) or 1
    if total_trials > 1:
        pass_fraction = item.get("passed_trials", 0) / total_trials
    else:
        pass_fraction = 1.0 if passed else 0.0

    return CaseRecord(
        case_id=str(case_id),
        passed=passed,
        score=float(score),
        pass_fraction=pass_fraction,
    )


def load_case_records(path: str | Path) -> dict[str, CaseRecord]:
    """Load case records from a results file or directory.

    Accepts the same shapes as ``compass analyze``: an EvalResult dict (with
    ``case_results``), a list of case dicts, a single case dict, or a
    directory of such JSON files. Unrecognized files/items are skipped with a
    warning. On duplicate case_ids the last record wins.
    """
    path = Path(path)
    json_files = [path] if path.is_file() else sorted(path.glob("**/*.json"))

    records: dict[str, CaseRecord] = {}
    for jf in json_files:
        try:
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Skipping %s: %s", jf, e)
            continue

        if isinstance(data, dict) and "case_results" in data:
            items = data["case_results"]  # EvalResult.to_dict() shape
        elif isinstance(data, list):
            items = data
        else:
            items = [data]

        for item in items:
            if not isinstance(item, dict):
                continue
            record = _extract_case_records(item)
            if record is not None:
                records[record.case_id] = record

    return records


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "mean_diff": self.mean_diff,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "significant": self.significant,
            "mde": self.mde,
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
        return PairedStats(0, 0.0, 0.0, 0.0, False, math.inf)

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
            "only_in_a": self.only_in_a,
            "only_in_b": self.only_in_b,
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
            "verdict": self.verdict,
        }


def compare_results(
    records_a: dict[str, CaseRecord],
    records_b: dict[str, CaseRecord],
    label_a: str = "A",
    label_b: str = "B",
) -> ComparisonReport:
    """Pair two runs by case_id and compute flips + paired statistics.

    Only cases present in both runs enter the statistics; the rest are listed
    in ``only_in_a`` / ``only_in_b`` so coverage changes are visible.
    """
    paired_ids = sorted(records_a.keys() & records_b.keys())
    report = ComparisonReport(
        label_a=label_a,
        label_b=label_b,
        n_paired=len(paired_ids),
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

    report.pass_stats = paired_stats(
        [b.pass_fraction - a.pass_fraction for a, b in pairs]
    )
    report.score_stats = paired_stats([b.score - a.score for a, b in pairs])
    return report


def compare_paths(
    path_a: str | Path,
    path_b: str | Path,
) -> ComparisonReport:
    """Load two results paths and compare them (A = baseline, B = candidate)."""
    return compare_results(
        load_case_records(path_a),
        load_case_records(path_b),
        label_a=str(path_a),
        label_b=str(path_b),
    )
