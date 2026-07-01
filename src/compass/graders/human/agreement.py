"""Inter-rater agreement metrics for human evaluation.

Pure Python implementation (no external dependencies beyond stdlib).

Provides:
- Cohen's Kappa: agreement between exactly 2 raters
- Krippendorff's Alpha: agreement among N raters (handles missing data)
- RaterAgreement: convenience class for collecting scores and computing reports
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any


def compute_cohens_kappa(
    rater1_scores: list[float],
    rater2_scores: list[float],
    tolerance: float = 0.0,
) -> float | None:
    """Compute Cohen's Kappa for two raters.

    Args:
        rater1_scores: Scores from rater 1, ordered by item.
        rater2_scores: Scores from rater 2, ordered by item (same length).
        tolerance: Maximum difference to consider as agreement (for interval data).

    Returns:
        Kappa value in [-1, 1], or None if insufficient data.
        1.0 = perfect agreement, 0.0 = chance agreement, <0 = worse than chance.
    """
    if len(rater1_scores) != len(rater2_scores):
        return None
    n = len(rater1_scores)
    if n == 0:
        return None

    # Observed agreement
    agreed = sum(
        1 for a, b in zip(rater1_scores, rater2_scores, strict=True)
        if abs(a - b) <= tolerance
    )
    p_observed = agreed / n

    # Expected agreement by chance
    # Count category frequencies for each rater
    if tolerance == 0.0:
        categories1 = Counter(rater1_scores)
        categories2 = Counter(rater2_scores)
        all_categories = set(categories1.keys()) | set(categories2.keys())
        p_expected = sum(
            (categories1.get(c, 0) / n) * (categories2.get(c, 0) / n)
            for c in all_categories
        )
    else:
        # For tolerance-based agreement, estimate expected agreement
        # by computing pairwise chance agreement through random permutation logic
        agree_count = 0
        for a in rater1_scores:
            for b in rater2_scores:
                if abs(a - b) <= tolerance:
                    agree_count += 1
        p_expected = agree_count / (n * n)

    if p_expected >= 1.0:
        return 1.0 if p_observed >= 1.0 else 0.0

    kappa = (p_observed - p_expected) / (1.0 - p_expected)
    return kappa


def compute_krippendorffs_alpha(
    data: dict[str, dict[str, float]],
) -> float | None:
    """Compute Krippendorff's Alpha for N raters with interval data.

    Handles missing data naturally (each item needs at least 2 raters).

    Args:
        data: Mapping of rater_id -> {item_id: score}.
              Not all raters need to rate all items.

    Returns:
        Alpha value, or None if insufficient data.
        1.0 = perfect agreement, 0.0 = chance-level, <0 = systematic disagreement.
    """
    if len(data) < 2:
        return None

    # Collect all items and their scores
    # item_id -> list of scores (from different raters)
    items: dict[str, list[float]] = {}
    for rater_scores in data.values():
        for item_id, score in rater_scores.items():
            if item_id not in items:
                items[item_id] = []
            items[item_id].append(score)

    # Filter to items with at least 2 ratings
    pairable_items = {
        item_id: scores
        for item_id, scores in items.items()
        if len(scores) >= 2
    }

    if not pairable_items:
        return None

    # D_observed: within-unit disagreement
    # For interval data: sum of squared differences within each item,
    # weighted by 1/(m_u - 1) where m_u is number of raters for item u
    d_observed = 0.0
    total_pairs = 0

    for scores in pairable_items.values():
        m = len(scores)
        # All pairs within this item
        item_disagreement = 0.0
        pair_count = 0
        for i in range(m):
            for j in range(i + 1, m):
                item_disagreement += (scores[i] - scores[j]) ** 2
                pair_count += 1
        # Weight by 1/(m-1)
        d_observed += item_disagreement / (m - 1)
        total_pairs += pair_count

    # D_expected: expected disagreement from the pool of all values
    all_values: list[float] = []
    for scores in pairable_items.values():
        all_values.extend(scores)

    n_total = len(all_values)
    if n_total < 2:
        return None

    # Expected disagreement = mean squared difference across all value pairs
    d_expected = 0.0
    for i in range(n_total):
        for j in range(i + 1, n_total):
            d_expected += (all_values[i] - all_values[j]) ** 2

    # Number of possible pairs in the pool
    n_pairs = n_total * (n_total - 1) / 2
    d_expected = d_expected / n_pairs

    # Normalize D_observed
    n_items = len(pairable_items)
    d_observed = d_observed / n_items

    if d_expected == 0.0:
        return 1.0 if d_observed == 0.0 else 0.0

    alpha = 1.0 - (d_observed / d_expected)
    return alpha


@dataclass
class AgreementReport:
    """Report of inter-rater agreement metrics."""

    num_raters: int
    num_items: int
    cohens_kappa: float | None = None
    krippendorffs_alpha: float | None = None
    pairwise_agreement: float = 0.0
    per_criterion: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def interpretation(self) -> str:
        """Interpret agreement level using Landis & Koch scale.

        Uses Krippendorff's Alpha if available, else Cohen's Kappa.
        """
        value = (
            self.krippendorffs_alpha
            if self.krippendorffs_alpha is not None
            else self.cohens_kappa
        )
        if value is None:
            return "insufficient_data"
        if value < 0.0:
            return "poor"
        if value < 0.21:
            return "slight"
        if value < 0.41:
            return "fair"
        if value < 0.61:
            return "moderate"
        if value < 0.81:
            return "substantial"
        return "almost_perfect"

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_raters": self.num_raters,
            "num_items": self.num_items,
            "cohens_kappa": self.cohens_kappa,
            "krippendorffs_alpha": self.krippendorffs_alpha,
            "pairwise_agreement": self.pairwise_agreement,
            "per_criterion": self.per_criterion,
            "interpretation": self.interpretation,
        }


class RaterAgreement:
    """Collects multi-rater scores and computes agreement metrics.

    Usage:
        ra = RaterAgreement()
        ra.add_score("rater1", "item1", 4.0)
        ra.add_score("rater2", "item1", 3.5)
        report = ra.compute()
    """

    def __init__(self, tolerance: float = 0.0) -> None:
        self.tolerance = tolerance
        # criterion -> rater_id -> {item_id: score}
        self._scores: dict[str, dict[str, dict[str, float]]] = {}

    def add_score(
        self,
        rater_id: str,
        item_id: str,
        score: float,
        criterion: str = "",
    ) -> None:
        """Add a single score."""
        if criterion not in self._scores:
            self._scores[criterion] = {}
        if rater_id not in self._scores[criterion]:
            self._scores[criterion][rater_id] = {}
        self._scores[criterion][rater_id][item_id] = score

    def compute(self) -> AgreementReport:
        """Compute agreement metrics across all criteria."""
        all_raters: set[str] = set()
        all_items: set[str] = set()
        for criterion_data in self._scores.values():
            for rater_id, items in criterion_data.items():
                all_raters.add(rater_id)
                all_items.update(items.keys())

        num_raters = len(all_raters)
        num_items = len(all_items)

        if num_raters < 2 or num_items == 0:
            return AgreementReport(
                num_raters=num_raters,
                num_items=num_items,
            )

        # Compute per-criterion metrics
        per_criterion: dict[str, dict[str, Any]] = {}
        for criterion, criterion_data in self._scores.items():
            if criterion == "":
                continue
            metrics = self._compute_for_data(criterion_data)
            per_criterion[criterion] = metrics

        # Compute overall (default criterion or merged)
        if "" in self._scores:
            overall_data = self._scores[""]
        else:
            # Merge all criteria
            overall_data = self._merge_all_criteria()

        overall_metrics = self._compute_for_data(overall_data)

        return AgreementReport(
            num_raters=num_raters,
            num_items=num_items,
            cohens_kappa=overall_metrics.get("cohens_kappa"),
            krippendorffs_alpha=overall_metrics.get("krippendorffs_alpha"),
            pairwise_agreement=overall_metrics.get("pairwise_agreement", 0.0),
            per_criterion=per_criterion,
        )

    def _compute_for_data(
        self, data: dict[str, dict[str, float]]
    ) -> dict[str, Any]:
        """Compute metrics for a single criterion's data."""
        rater_ids = sorted(data.keys())
        num_raters = len(rater_ids)

        # Krippendorff's Alpha (works for N raters)
        alpha = compute_krippendorffs_alpha(data)

        # Cohen's Kappa (only for exactly 2 raters)
        kappa = None
        if num_raters == 2:
            r1_id, r2_id = rater_ids
            # Get common items
            common_items = sorted(
                set(data[r1_id].keys()) & set(data[r2_id].keys())
            )
            if common_items:
                r1_scores = [data[r1_id][item] for item in common_items]
                r2_scores = [data[r2_id][item] for item in common_items]
                kappa = compute_cohens_kappa(r1_scores, r2_scores, self.tolerance)

        # Pairwise agreement rate
        pairwise_agreement = self._compute_pairwise_agreement(data)

        return {
            "cohens_kappa": kappa,
            "krippendorffs_alpha": alpha,
            "pairwise_agreement": pairwise_agreement,
        }

    def _compute_pairwise_agreement(
        self, data: dict[str, dict[str, float]]
    ) -> float:
        """Compute simple pairwise agreement rate."""
        rater_ids = sorted(data.keys())
        if len(rater_ids) < 2:
            return 0.0

        total_pairs = 0
        agreed_pairs = 0

        for i in range(len(rater_ids)):
            for j in range(i + 1, len(rater_ids)):
                r1_id = rater_ids[i]
                r2_id = rater_ids[j]
                common_items = set(data[r1_id].keys()) & set(data[r2_id].keys())
                for item in common_items:
                    total_pairs += 1
                    if abs(data[r1_id][item] - data[r2_id][item]) <= self.tolerance:
                        agreed_pairs += 1

        return agreed_pairs / total_pairs if total_pairs > 0 else 0.0

    def _merge_all_criteria(self) -> dict[str, dict[str, float]]:
        """Merge scores across criteria into composite item IDs."""
        merged: dict[str, dict[str, float]] = {}
        for criterion, criterion_data in self._scores.items():
            for rater_id, items in criterion_data.items():
                if rater_id not in merged:
                    merged[rater_id] = {}
                for item_id, score in items.items():
                    composite_id = f"{criterion}:{item_id}" if criterion else item_id
                    merged[rater_id][composite_id] = score
        return merged
