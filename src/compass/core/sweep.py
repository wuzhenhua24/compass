"""Parameter sweep support for systematic parameter exploration.

Provides grid sweep functionality to systematically explore parameter
combinations (e.g., quality=low/medium/high x size=1024/1536), running
the full evaluation pipeline for each combination and generating
comparison reports.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from compass.core.result import CaseResult
    from compass.core.scenario import TestCase

SWEEP_SEPARATOR = "__sweep__"


class SweepConfig(BaseModel):
    """Configuration for parameter sweep.

    Defines parameter names mapped to lists of values to explore.
    The cartesian product of all parameter values is generated.

    Example:
        sweep:
          params:
            steps: [20, 30, 50]
            cfg_scale: [5.0, 7.5]
        -> 6 combinations: (20, 5.0), (20, 7.5), (30, 5.0), ...
    """

    params: dict[str, list[Any]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_params(self) -> SweepConfig:
        """Ensure all parameter value lists are non-empty."""
        for key, values in self.params.items():
            if not values:
                raise ValueError(f"Sweep parameter '{key}' has an empty value list")
        return self

    @property
    def total_combinations(self) -> int:
        """Total number of parameter combinations."""
        if not self.params:
            return 0
        result = 1
        for values in self.params.values():
            result *= len(values)
        return result

    def combinations(self) -> list[dict[str, Any]]:
        """Generate cartesian product of all parameter values.

        Keys are sorted alphabetically for deterministic ordering.
        """
        if not self.params:
            return []
        sorted_keys = sorted(self.params.keys())
        sorted_values = [self.params[k] for k in sorted_keys]
        return [
            dict(zip(sorted_keys, combo, strict=True))
            for combo in itertools.product(*sorted_values)
        ]


def _format_sweep_id(base_id: str, combo: dict[str, Any]) -> str:
    """Format a sweep case ID from base ID and parameter combination.

    Keys are sorted alphabetically. Values are converted to strings.
    Example: "cat__sweep__cfg_scale=5.0__steps=20"
    """
    parts = [f"{k}={v}" for k, v in sorted(combo.items())]
    return f"{base_id}{SWEEP_SEPARATOR}{'__'.join(parts)}"


def expand_sweeps(
    cases: list[TestCase],
    default_sweep: SweepConfig | None = None,
) -> list[TestCase]:
    """Expand test cases with sweep configurations into concrete cases.

    For each case that has a sweep config (case-level takes priority over
    default_sweep), generates one TestCase per parameter combination.
    Cases without sweep are passed through unchanged.

    Args:
        cases: Original list of test cases.
        default_sweep: Scenario-level default sweep config.

    Returns:
        Expanded list of test cases.
    """
    expanded: list[TestCase] = []

    for case in cases:
        sweep = case.sweep or default_sweep
        if sweep is None or not sweep.params:
            expanded.append(case)
            continue

        combos = sweep.combinations()
        for idx, combo in enumerate(combos):
            # Merge base params with sweep combo (sweep overrides)
            merged_params = {**case.input.params, **combo}

            # Build sweep metadata
            sweep_meta = {
                "base_case_id": case.id,
                "sweep_index": idx,
                "sweep_params": combo,
            }

            # Merge with existing metadata
            merged_metadata = {**case.metadata, "sweep": sweep_meta}

            # Create new InputConfig with merged params
            new_input = case.input.model_copy(update={"params": merged_params})

            # Create expanded case
            new_case = case.model_copy(
                update={
                    "id": _format_sweep_id(case.id, combo),
                    "input": new_input,
                    "metadata": merged_metadata,
                    "sweep": None,  # Prevent re-expansion
                }
            )
            expanded.append(new_case)

    return expanded


@dataclass
class SweepSummary:
    """Summary of sweep results for a single base case.

    Groups all parameter combination results for analysis.
    """

    base_case_id: str
    total_combinations: int
    passed_combinations: int
    results: list[CaseResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        """Fraction of combinations that passed."""
        if self.total_combinations == 0:
            return 0.0
        return self.passed_combinations / self.total_combinations

    @property
    def best_combination(self) -> dict[str, Any] | None:
        """Parameter combination with the highest score."""
        if not self.results:
            return None
        best = max(self.results, key=lambda r: r.overall_score)
        sweep_meta = best.input_data.get("sweep") or (
            best.output_data.get("sweep") if best.output_data else None
        )
        # Try extracting from case metadata stored in input_data
        if sweep_meta is None:
            # Parse from case_id as fallback
            return self._parse_sweep_params(best.case_id)
        return sweep_meta.get("sweep_params")

    @property
    def worst_combination(self) -> dict[str, Any] | None:
        """Parameter combination with the lowest score."""
        if not self.results:
            return None
        worst = min(self.results, key=lambda r: r.overall_score)
        sweep_meta = worst.input_data.get("sweep") or (
            worst.output_data.get("sweep") if worst.output_data else None
        )
        if sweep_meta is None:
            return self._parse_sweep_params(worst.case_id)
        return sweep_meta.get("sweep_params")

    def score_by_param(self, param_name: str) -> dict[Any, float]:
        """Average score grouped by a specific parameter's values.

        Args:
            param_name: The sweep parameter to group by.

        Returns:
            Mapping of parameter value to average score.
        """
        from collections import defaultdict

        scores: dict[Any, list[float]] = defaultdict(list)
        for result in self.results:
            params = self._get_sweep_params(result)
            if params and param_name in params:
                val = params[param_name]
                scores[val].append(result.overall_score)

        return {val: sum(s) / len(s) for val, s in scores.items()}

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "base_case_id": self.base_case_id,
            "total_combinations": self.total_combinations,
            "passed_combinations": self.passed_combinations,
            "pass_rate": self.pass_rate,
            "best_combination": self.best_combination,
            "worst_combination": self.worst_combination,
        }

    @staticmethod
    def _get_sweep_params(result: CaseResult) -> dict[str, Any] | None:
        """Extract sweep params from a CaseResult."""
        sweep_meta = result.input_data.get("sweep")
        if sweep_meta:
            return sweep_meta.get("sweep_params")
        return SweepSummary._parse_sweep_params(result.case_id)

    @staticmethod
    def _parse_sweep_params(case_id: str) -> dict[str, Any] | None:
        """Parse sweep params from a case ID string."""
        if SWEEP_SEPARATOR not in case_id:
            return None
        _, params_str = case_id.split(SWEEP_SEPARATOR, 1)
        params: dict[str, Any] = {}
        for part in params_str.split("__"):
            if "=" in part:
                key, val = part.split("=", 1)
                # Try to convert to number
                try:
                    params[key] = int(val)
                except ValueError:
                    try:
                        params[key] = float(val)
                    except ValueError:
                        params[key] = val
        return params


def group_sweep_results(
    case_results: list[CaseResult],
) -> tuple[list[SweepSummary], list[CaseResult]]:
    """Group case results by sweep base case ID.

    Separates sweep results (identified by __sweep__ in case_id) from
    non-sweep results, and groups sweep results by their base case ID.

    Args:
        case_results: All case results from a run.

    Returns:
        Tuple of (sweep_summaries, non_sweep_results).
    """
    sweep_groups: dict[str, list[CaseResult]] = {}
    non_sweep: list[CaseResult] = []

    for result in case_results:
        if SWEEP_SEPARATOR in result.case_id:
            base_id = result.case_id.split(SWEEP_SEPARATOR, 1)[0]
            sweep_groups.setdefault(base_id, []).append(result)
        else:
            non_sweep.append(result)

    summaries: list[SweepSummary] = []
    for base_id, results in sweep_groups.items():
        passed = sum(1 for r in results if r.passed)
        summaries.append(
            SweepSummary(
                base_case_id=base_id,
                total_combinations=len(results),
                passed_combinations=passed,
                results=results,
            )
        )

    return summaries, non_sweep
