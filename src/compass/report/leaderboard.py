"""Rank several runs of one scenario against each other.

A leaderboard answers "which configuration scored highest". That is a
**reading**, not a measurement: with 20 cases, two models a couple of points
apart are usually indistinguishable, and a ranked table invites people to read
a winner out of noise. So every leaderboard here carries the uncertainty with
it — mean ± standard error per row, and an explicit verdict on whether the gap
between the top two rows survives a paired test.

That pairing is the point of building this on top of ``compass compare``
rather than beside it: the table tells you the order, the paired statistics
tell you whether the order means anything.
"""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass, field
from typing import Any

from compass.core.result import CaseResult, EvalResult, TestStatus
from compass.report.compare import CaseRecord, PairedStats, compare_results

#: Rows whose score renders identically share a rank (standard competition
#: ranking) — printing "1." and "2." for two identical 0.830s is a lie the
#: display itself contradicts.
_SCORE_PRECISION = 3


def slugify(text: str) -> str:
    """Filesystem-safe label, for per-model trace directories."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", text).strip("-") or "run"


@dataclass
class LeaderboardEntry:
    """One row: a labelled run, with the uncertainty around its score."""

    label: str
    rank: int
    mean_score: float
    stderr: float
    pass_rate: float
    passed_cases: int
    evaluated_cases: int
    error_cases: int
    #: Mean pass^k across cases, when every case reported the same k.
    reliability: float | None = None
    reliability_k: int | None = None

    @property
    def score_display(self) -> str:
        if self.evaluated_cases > 1 and self.stderr > 0:
            return f"{self.mean_score:.{_SCORE_PRECISION}f} ±{self.stderr:.{_SCORE_PRECISION}f}"
        return f"{self.mean_score:.{_SCORE_PRECISION}f}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "rank": self.rank,
            "mean_score": self.mean_score,
            "stderr": self.stderr,
            "pass_rate": self.pass_rate,
            "passed_cases": self.passed_cases,
            "evaluated_cases": self.evaluated_cases,
            "error_cases": self.error_cases,
            "reliability": self.reliability,
            "reliability_k": self.reliability_k,
        }


@dataclass
class Leaderboard:
    """A ranked table plus the statistics that qualify it."""

    entries: list[LeaderboardEntry] = field(default_factory=list)
    #: Paired difference between the top two rows, when there are two to pair.
    top_gap: PairedStats | None = None
    #: Cases only one of the top two runs covered — excluded from the pairing.
    unpaired: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        """One line on whether the ranking's top gap is real."""
        if len(self.entries) < 2:
            return "single run — nothing to rank against"
        first, second = self.entries[0], self.entries[1]
        if first.rank == second.rank:
            return f"{first.label} and {second.label} are tied"
        if self.top_gap is None:
            return f"{first.label} leads, but no paired cases to test the gap"
        if self.top_gap.significant:
            return (
                f"{first.label} beats {second.label} by "
                f"{self.top_gap.mean_diff:+.3f} "
                f"(95% CI [{self.top_gap.ci_low:+.3f}, {self.top_gap.ci_high:+.3f}])"
            )
        return (
            f"{first.label} leads {second.label} by "
            f"{self.top_gap.mean_diff:+.3f}, but the 95% CI "
            f"[{self.top_gap.ci_low:+.3f}, {self.top_gap.ci_high:+.3f}] spans 0 — "
            f"the ranking is within noise (detectable at this n: "
            f"~{self.top_gap.mde:.3f})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": [e.to_dict() for e in self.entries],
            "top_gap": self.top_gap.to_dict() if self.top_gap else None,
            "unpaired": self.unpaired,
            "verdict": self.verdict,
        }


def case_records(result: EvalResult) -> dict[str, CaseRecord]:
    """CaseRecords for an in-memory EvalResult, for paired comparison.

    Mirrors what ``compare.load_case_records`` builds from a file, so a
    leaderboard row and a ``compass compare`` run agree on the numbers.
    """
    records: dict[str, CaseRecord] = {}
    for case in result.case_results:
        if case.status == TestStatus.ERROR:
            # Harness failures are missing samples, not evidence
            continue
        records[case.case_id] = CaseRecord(
            case_id=case.case_id,
            passed=case.passed,
            score=case.overall_score,
            pass_fraction=_pass_fraction(case),
            grader_fingerprint=case.grader_fingerprint,
        )
    return records


def build_leaderboard(runs: list[tuple[str, EvalResult]]) -> Leaderboard:
    """Rank labelled runs by mean case score, highest first.

    Args:
        runs: ``(label, result)`` pairs — typically one per model.
    """
    entries = [_entry(label, result) for label, result in runs]
    entries.sort(key=lambda e: (-e.mean_score, e.label))

    # Standard competition ranking on the *displayed* score
    previous_display: str | None = None
    rank = 0
    for position, entry in enumerate(entries, 1):
        if entry.score_display != previous_display:
            rank = position
            previous_display = entry.score_display
        entry.rank = rank

    board = Leaderboard(entries=entries)
    if len(entries) >= 2:
        by_label = dict(runs)
        board.top_gap, board.unpaired = _pair_top_two(
            by_label[entries[0].label], by_label[entries[1].label]
        )
    return board


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------


def _entry(label: str, result: EvalResult) -> LeaderboardEntry:
    scores = [
        c.overall_score
        for c in result.case_results
        if c.status != TestStatus.ERROR
    ]
    mean = statistics.mean(scores) if scores else 0.0
    stderr = (
        statistics.stdev(scores) / math.sqrt(len(scores)) if len(scores) > 1 else 0.0
    )
    reliability, k = _reliability(result)

    return LeaderboardEntry(
        label=label,
        rank=0,
        mean_score=mean,
        stderr=stderr,
        pass_rate=result.pass_rate,
        passed_cases=result.passed_cases,
        evaluated_cases=result.evaluated_cases,
        error_cases=result.error_cases,
        reliability=reliability,
        reliability_k=k,
    )


def _pass_fraction(case: CaseResult) -> float:
    evaluated = case.total_trials - case.error_trials
    if case.total_trials > 1 and evaluated > 0:
        return case.passed_trials / evaluated
    return 1.0 if case.passed else 0.0


def _reliability(result: EvalResult) -> tuple[float | None, int | None]:
    """Mean pass^k, when every evaluated case reported the same k.

    Reported only when it is comparable across the whole run — averaging
    pass^3 for some cases with pass^5 for others would be a meaningless number.
    """
    evaluated = [c for c in result.case_results if c.status != TestStatus.ERROR]
    if not evaluated or any(c.total_trials <= 1 for c in evaluated):
        return None, None

    per_case: list[dict[str, Any]] = []
    for case in evaluated:
        metrics = case.trial_metrics or {}
        ks = {
            int(key.removeprefix("pass_all_"))
            for key in metrics
            if key.startswith("pass_all_")
        }
        if not ks:
            return None, None
        per_case.append({"metrics": metrics, "ks": ks})

    shared = set.intersection(*(c["ks"] for c in per_case))
    if not shared:
        return None, None

    k = max(shared)
    values = [float(c["metrics"][f"pass_all_{k}"]) for c in per_case]
    return statistics.mean(values), k


def _pair_top_two(
    first: EvalResult, second: EvalResult
) -> tuple[PairedStats | None, list[str]]:
    """Paired score difference between the two leading runs."""
    a, b = case_records(second), case_records(first)  # B = the leader
    report = compare_results(a, b)
    if report.n_paired == 0:
        return None, sorted(set(a) ^ set(b))
    return report.score_stats, sorted(report.only_in_a + report.only_in_b)
