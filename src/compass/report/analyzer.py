"""Evaluation result analyzer with Transcript/Outcome separation.

Analyzes grading results across multiple tasks, providing:
- Summary statistics split by grader scope (Transcript vs Outcome)
- Per-grader performance breakdown
- Failure pattern identification
- Actionable recommendations based on Transcript/Outcome score gaps
"""

import statistics
from dataclasses import dataclass, field
from typing import Any

from compass.graders.base import GradeResult, GraderScope


def iter_case_dicts(data: Any) -> list[dict[str, Any]]:
    """Flatten a results payload into per-case dicts.

    One reader for every shape Compass writes, so a results file produced by
    any command stays consumable by every analysis command:

    - ``{"results": [EvalResult, ...], "summary": ...}`` — ``compass test --report json``
    - ``{"case_results": [...]}`` — ``EvalResult.to_dict()`` (``compass grade -o``)
    - ``[case, ...]`` / ``{case}`` — bare case records

    Unrecognized payloads come back as a single item so the caller can decide.
    """
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        if isinstance(data.get("results"), list):
            cases: list[dict[str, Any]] = []
            for entry in data["results"]:
                cases.extend(iter_case_dicts(entry))
            return cases
        if isinstance(data.get("case_results"), list):
            return [c for c in data["case_results"] if isinstance(c, dict)]
        return [data]
    return []


@dataclass
class TaskEvalResult:
    """Complete evaluation result for a single task.

    This is the input unit for the analyzer. Each task's GradeResults
    are automatically classified by their grader_scope.
    """

    task_id: str
    passed: bool
    grade_results: list[GradeResult] = field(default_factory=list)
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    category: str = ""
    best_score: float | None = None  # Best-of-k score (max across trials)

    @property
    def effective_best_score(self) -> float:
        """Best score if available, otherwise outcome_score."""
        if self.best_score is not None:
            return self.best_score
        return self.outcome_score

    @property
    def transcript_results(self) -> list[GradeResult]:
        """Grade results from Transcript-scope graders."""
        return [
            r for r in self.grade_results
            if r.grader_scope in (GraderScope.TRANSCRIPT, GraderScope.BOTH)
        ]

    @property
    def outcome_results(self) -> list[GradeResult]:
        """Grade results from Outcome-scope graders."""
        return [
            r for r in self.grade_results
            if r.grader_scope in (GraderScope.OUTCOME, GraderScope.BOTH)
        ]

    @property
    def transcript_score(self) -> float:
        """Average score across Transcript-scope graders."""
        results = self.transcript_results
        if not results:
            return 0.0
        return sum(r.score for r in results) / len(results)

    @property
    def outcome_score(self) -> float:
        """Average score across Outcome-scope graders."""
        results = self.outcome_results
        if not results:
            return 0.0
        return sum(r.score for r in results) / len(results)


@dataclass
class GraderStats:
    """Aggregated statistics for a single grader across all tasks."""

    name: str
    grader_type: str
    grader_scope: str
    passed: int = 0
    failed: int = 0
    scores: list[float] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.passed + self.failed

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    @property
    def avg_score(self) -> float:
        return statistics.mean(self.scores) if self.scores else 0.0

    @property
    def score_std(self) -> float:
        return statistics.stdev(self.scores) if len(self.scores) >= 2 else 0.0

    @property
    def is_bottleneck(self) -> bool:
        """A grader is a bottleneck if its pass rate is below 70%."""
        return self.pass_rate < 0.7 if self.total > 0 else False


@dataclass
class AnalysisReport:
    """Complete analysis report."""

    summary: dict[str, Any]
    transcript_analysis: dict[str, dict[str, Any]]
    outcome_analysis: dict[str, dict[str, Any]]
    scope_comparison: dict[str, Any]
    failure_patterns: list[dict[str, Any]]
    failure_tag_analysis: dict[str, dict[str, Any]]  # Per-tag cluster stats
    recommendations: list[str]
    agreement: dict[str, Any] = field(default_factory=dict)
    dual_axis_data: list[dict[str, Any]] = field(default_factory=list)
    category_analysis: dict[str, dict[str, Any]] = field(default_factory=dict)
    tag_analysis: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "summary": self.summary,
            "transcript_analysis": self.transcript_analysis,
            "outcome_analysis": self.outcome_analysis,
            "scope_comparison": self.scope_comparison,
            "failure_patterns": self.failure_patterns,
            "failure_tag_analysis": self.failure_tag_analysis,
            "recommendations": self.recommendations,
            "agreement": self.agreement,
            "dual_axis_data": self.dual_axis_data,
            "category_analysis": self.category_analysis,
            "tag_analysis": self.tag_analysis,
        }


class EvalResultAnalyzer:
    """Evaluation result analyzer.

    Analyzes grading results with Transcript/Outcome separation,
    identifies failure patterns, and generates actionable recommendations.

    Usage:
        results = [TaskEvalResult(...), ...]
        analyzer = EvalResultAnalyzer(results)
        report = analyzer.analyze()
    """

    def __init__(self, results: list[TaskEvalResult]):
        self.results = results

    def analyze(self) -> AnalysisReport:
        """Generate a complete analysis report."""
        summary = self._compute_summary()
        transcript_analysis = self._analyze_by_scope(GraderScope.TRANSCRIPT)
        outcome_analysis = self._analyze_by_scope(GraderScope.OUTCOME)
        dual_axis_data = self._compute_dual_axis_data()
        scope_comparison = self._compare_scopes(summary, dual_axis_data)
        failure_patterns = self._identify_failure_patterns()
        failure_tag_analysis = self._analyze_failure_tags()
        category_analysis = self._analyze_by_category()
        tag_analysis = self._analyze_by_tag()
        recommendations = self._generate_recommendations(
            summary, transcript_analysis, outcome_analysis,
            scope_comparison, failure_tag_analysis,
        )

        return AnalysisReport(
            summary=summary,
            transcript_analysis=transcript_analysis,
            outcome_analysis=outcome_analysis,
            scope_comparison=scope_comparison,
            failure_patterns=failure_patterns,
            failure_tag_analysis=failure_tag_analysis,
            recommendations=recommendations,
            dual_axis_data=dual_axis_data,
            category_analysis=category_analysis,
            tag_analysis=tag_analysis,
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _compute_summary(self) -> dict[str, Any]:
        """Compute overall summary statistics."""
        total = len(self.results)
        if total == 0:
            return {
                "total_tasks": 0,
                "passed_tasks": 0,
                "pass_rate": 0.0,
                "avg_transcript_score": 0.0,
                "avg_outcome_score": 0.0,
                "transcript_score_std": 0.0,
                "outcome_score_std": 0.0,
                "avg_duration_ms": 0.0,
            }

        passed = sum(1 for r in self.results if r.passed)

        transcript_scores = [r.transcript_score for r in self.results]
        outcome_scores = [r.outcome_score for r in self.results]
        best_scores = [r.effective_best_score for r in self.results]
        has_best = any(r.best_score is not None for r in self.results)

        summary: dict[str, Any] = {
            "total_tasks": total,
            "passed_tasks": passed,
            "pass_rate": passed / total,
            "avg_transcript_score": statistics.mean(transcript_scores),
            "avg_outcome_score": statistics.mean(outcome_scores),
            "transcript_score_std": (
                statistics.stdev(transcript_scores)
                if len(transcript_scores) >= 2 else 0.0
            ),
            "outcome_score_std": (
                statistics.stdev(outcome_scores)
                if len(outcome_scores) >= 2 else 0.0
            ),
            "avg_duration_ms": statistics.mean(
                [r.duration_ms for r in self.results]
            ),
        }

        # Best-of-k metrics (only when multi-trial data is present)
        if has_best:
            summary["avg_best_score"] = statistics.mean(best_scores)
            summary["avg_score"] = statistics.mean(outcome_scores)
            summary["best_vs_avg_gap"] = (
                summary["avg_best_score"] - summary["avg_score"]
            )

        return summary

    # ------------------------------------------------------------------
    # Per-scope analysis
    # ------------------------------------------------------------------

    def _analyze_by_scope(
        self, scope: GraderScope,
    ) -> dict[str, dict[str, Any]]:
        """Analyze grader results filtered by scope."""
        grader_stats: dict[str, GraderStats] = {}

        for result in self.results:
            if scope == GraderScope.TRANSCRIPT:
                grade_results = result.transcript_results
            else:
                grade_results = result.outcome_results

            for gr in grade_results:
                if gr.name not in grader_stats:
                    grader_stats[gr.name] = GraderStats(
                        name=gr.name,
                        grader_type=gr.grader_type.value,
                        grader_scope=gr.grader_scope.value,
                    )
                stats = grader_stats[gr.name]
                stats.scores.append(gr.score)
                if gr.passed:
                    stats.passed += 1
                else:
                    stats.failed += 1

        return {
            name: {
                "pass_rate": stats.pass_rate,
                "avg_score": stats.avg_score,
                "score_std": stats.score_std,
                "total": stats.total,
                "passed": stats.passed,
                "failed": stats.failed,
                "grader_type": stats.grader_type,
                "grader_scope": stats.grader_scope,
                "is_bottleneck": stats.is_bottleneck,
            }
            for name, stats in grader_stats.items()
        }

    # ------------------------------------------------------------------
    # Dual axis data (per-case Outcome vs Transcript scores)
    # ------------------------------------------------------------------

    def _compute_dual_axis_data(self) -> list[dict[str, Any]]:
        """Compute per-case outcome_score and transcript_score for scatter plot."""
        data: list[dict[str, Any]] = []
        for r in self.results:
            data.append({
                "task_id": r.task_id,
                "outcome_score": r.outcome_score,
                "transcript_score": r.transcript_score,
                "passed": r.passed,
            })
        return data

    # ------------------------------------------------------------------
    # Scope comparison
    # ------------------------------------------------------------------

    def _compare_scopes(
        self,
        summary: dict[str, Any],
        dual_axis_data: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Compare Transcript vs Outcome scores to identify gaps."""
        avg_t = summary.get("avg_transcript_score", 0.0)
        avg_o = summary.get("avg_outcome_score", 0.0)
        gap = avg_t - avg_o

        if abs(gap) < 0.1:
            diagnosis = "balanced"
            description = "Transcript 和 Outcome 分数基本一致，Agent 的执行过程和产出质量均衡"
        elif gap > 0.2:
            diagnosis = "process_good_result_bad"
            description = (
                "Transcript 分数明显高于 Outcome 分数，"
                "Agent 过程看起来正确但最终结果不理想，可能存在最后一步执行问题"
            )
        elif gap > 0:
            diagnosis = "process_slightly_better"
            description = "Transcript 分数略高于 Outcome，产出质量有少量提升空间"
        elif gap < -0.2:
            diagnosis = "result_good_process_bad"
            description = (
                "Outcome 分数明显高于 Transcript 分数，"
                "Agent 最终完成了任务但过程不够规范，可以优化效率或合规性"
            )
        else:
            diagnosis = "result_slightly_better"
            description = "Outcome 分数略高于 Transcript，执行过程有少量优化空间"

        # Quadrant distribution (threshold = 0.7)
        threshold = 0.7
        both_good = 0
        result_good_process_bad = 0
        process_good_result_bad = 0
        both_bad = 0
        if dual_axis_data:
            for d in dual_axis_data:
                o = d["outcome_score"]
                t = d["transcript_score"]
                if o >= threshold and t >= threshold:
                    both_good += 1
                elif o >= threshold and t < threshold:
                    result_good_process_bad += 1
                elif o < threshold and t >= threshold:
                    process_good_result_bad += 1
                else:
                    both_bad += 1

        return {
            "avg_transcript_score": avg_t,
            "avg_outcome_score": avg_o,
            "gap": gap,
            "abs_gap": abs(gap),
            "diagnosis": diagnosis,
            "description": description,
            "quadrant_stats": {
                "both_good": both_good,
                "result_good_process_bad": result_good_process_bad,
                "process_good_result_bad": process_good_result_bad,
                "both_bad": both_bad,
            },
        }

    # ------------------------------------------------------------------
    # Failure pattern identification
    # ------------------------------------------------------------------

    def _identify_failure_patterns(self) -> list[dict[str, Any]]:
        """Identify common failure combinations across tasks."""
        failure_combos: dict[tuple[str, ...], list[str]] = {}

        for result in self.results:
            if result.passed:
                continue

            failed_graders: list[str] = []
            for gr in result.grade_results:
                if not gr.passed:
                    scope_prefix = {
                        GraderScope.TRANSCRIPT: "T",
                        GraderScope.OUTCOME: "O",
                        GraderScope.BOTH: "B",
                    }.get(gr.grader_scope, "?")
                    failed_graders.append(f"{scope_prefix}:{gr.name}")

            if failed_graders:
                combo = tuple(sorted(failed_graders))
                if combo not in failure_combos:
                    failure_combos[combo] = []
                failure_combos[combo].append(result.task_id)

        total = len(self.results)
        patterns = []
        for combo, task_ids in sorted(
            failure_combos.items(), key=lambda x: -len(x[1]),
        ):
            count = len(task_ids)
            patterns.append({
                "failed_graders": list(combo),
                "count": count,
                "percentage": count / total * 100 if total > 0 else 0.0,
                "task_ids": task_ids[:5],  # Show up to 5 example task IDs
            })

        return patterns[:10]  # Top 10 patterns

    # ------------------------------------------------------------------
    # Failure tag analysis
    # ------------------------------------------------------------------

    def _analyze_failure_tags(self) -> dict[str, dict[str, Any]]:
        """Cluster failures by their failure_tags across all tasks.

        Returns a dict keyed by tag with count, percentage, source graders,
        and example task IDs — sorted by count descending.
        """
        tag_data: dict[str, dict[str, Any]] = {}
        failed_count = sum(1 for r in self.results if not r.passed)

        for result in self.results:
            if result.passed:
                continue

            for gr in result.grade_results:
                if gr.passed or not gr.failure_tags:
                    continue

                for tag in gr.failure_tags:
                    if tag not in tag_data:
                        tag_data[tag] = {
                            "count": 0,
                            "graders": set(),
                            "task_ids": [],
                        }
                    entry = tag_data[tag]
                    entry["count"] += 1
                    entry["graders"].add(gr.name)
                    if result.task_id not in entry["task_ids"]:
                        entry["task_ids"].append(result.task_id)

        # Convert sets to sorted lists and add percentage
        analysis: dict[str, dict[str, Any]] = {}
        for tag, data in sorted(tag_data.items(), key=lambda x: -x[1]["count"]):
            analysis[tag] = {
                "count": data["count"],
                "percentage": data["count"] / failed_count * 100 if failed_count > 0 else 0.0,
                "graders": sorted(data["graders"]),
                "task_ids": data["task_ids"][:10],  # Limit to 10
            }

        return analysis

    # ------------------------------------------------------------------
    # Category analysis
    # ------------------------------------------------------------------

    def _analyze_by_category(self) -> dict[str, dict[str, Any]]:
        """Aggregate results by category (e.g., backend, fullstack, gym).

        Returns a dict keyed by category with summary stats for each.
        Only includes results that have a non-empty category.
        """
        buckets: dict[str, list[TaskEvalResult]] = {}
        for r in self.results:
            cat = r.category
            if not cat:
                continue
            buckets.setdefault(cat, []).append(r)

        if not buckets:
            return {}

        analysis: dict[str, dict[str, Any]] = {}
        for cat, results in sorted(buckets.items()):
            total = len(results)
            passed = sum(1 for r in results if r.passed)
            scores = [r.outcome_score for r in results]
            t_scores = [r.transcript_score for r in results]
            analysis[cat] = {
                "total": total,
                "passed": passed,
                "failed": total - passed,
                "pass_rate": passed / total if total > 0 else 0.0,
                "avg_score": statistics.mean(scores) if scores else 0.0,
                "avg_outcome_score": statistics.mean(scores) if scores else 0.0,
                "avg_transcript_score": (
                    statistics.mean(t_scores) if t_scores else 0.0
                ),
                "score_std": (
                    statistics.stdev(scores) if len(scores) >= 2 else 0.0
                ),
            }
        return analysis

    # ------------------------------------------------------------------
    # Tag analysis (aggregation by tags)
    # ------------------------------------------------------------------

    def _analyze_by_tag(self) -> dict[str, dict[str, Any]]:
        """Aggregate results by tags.

        A single result may appear in multiple tag buckets.
        Returns a dict keyed by tag with summary stats for each.
        """
        buckets: dict[str, list[TaskEvalResult]] = {}
        for r in self.results:
            for tag in r.tags:
                buckets.setdefault(tag, []).append(r)

        if not buckets:
            return {}

        analysis: dict[str, dict[str, Any]] = {}
        for tag, results in sorted(buckets.items(), key=lambda x: -len(x[1])):
            total = len(results)
            passed = sum(1 for r in results if r.passed)
            scores = [r.outcome_score for r in results]
            analysis[tag] = {
                "total": total,
                "passed": passed,
                "failed": total - passed,
                "pass_rate": passed / total if total > 0 else 0.0,
                "avg_score": statistics.mean(scores) if scores else 0.0,
                "score_std": (
                    statistics.stdev(scores) if len(scores) >= 2 else 0.0
                ),
            }
        return analysis

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    def _generate_recommendations(
        self,
        summary: dict[str, Any],
        transcript_analysis: dict[str, dict[str, Any]],
        outcome_analysis: dict[str, dict[str, Any]],
        scope_comparison: dict[str, Any],
        failure_tag_analysis: dict[str, dict[str, Any]] | None = None,
    ) -> list[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations: list[str] = []

        # 1. Overall pass rate
        pass_rate = summary.get("pass_rate", 0.0)
        if pass_rate < 0.5:
            recommendations.append(
                f"整体通过率仅 {pass_rate:.1%}，建议优先排查最常见的失败模式"
            )

        # 2. Transcript bottlenecks
        for name, stats in transcript_analysis.items():
            if stats.get("is_bottleneck"):
                recommendations.append(
                    f"Transcript 评分器 '{name}' 通过率低 ({stats['pass_rate']:.1%})，"
                    f"建议检查 Agent 的相关执行行为或调整评估标准"
                )

        # 3. Outcome bottlenecks (higher priority)
        for name, stats in outcome_analysis.items():
            if stats.get("pass_rate", 1.0) < 0.7:
                recommendations.append(
                    f"Outcome 评分器 '{name}' 通过率低 ({stats['pass_rate']:.1%})，"
                    f"这是核心功能问题，需要优先解决"
                )

        # 4. Scope gap diagnosis
        diagnosis = scope_comparison.get("diagnosis", "balanced")
        if diagnosis == "process_good_result_bad":
            recommendations.append(
                "Transcript 分数明显高于 Outcome 分数 "
                f"({scope_comparison['avg_transcript_score']:.2f} vs "
                f"{scope_comparison['avg_outcome_score']:.2f})，"
                "说明 Agent 过程看起来正确但最终结果不对，可能存在最后一步执行问题"
            )
        elif diagnosis == "result_good_process_bad":
            recommendations.append(
                "Outcome 分数明显高于 Transcript 分数 "
                f"({scope_comparison['avg_outcome_score']:.2f} vs "
                f"{scope_comparison['avg_transcript_score']:.2f})，"
                "说明 Agent 最终完成了任务但过程不够规范，可以优化执行效率或合规性"
            )

        # 5. High variance warning
        t_std = summary.get("transcript_score_std", 0.0)
        o_std = summary.get("outcome_score_std", 0.0)
        if t_std > 0.3:
            recommendations.append(
                f"Transcript 分数波动较大 (std={t_std:.2f})，"
                "Agent 的执行行为不够稳定，建议检查是否存在随机性过高的决策路径"
            )
        if o_std > 0.3:
            recommendations.append(
                f"Outcome 分数波动较大 (std={o_std:.2f})，"
                "最终产出质量不稳定，建议增加 Trial 次数并分析低分用例的共同特征"
            )

        # 6. Top failure tags
        if failure_tag_analysis:
            top_tags = list(failure_tag_analysis.items())[:3]
            for tag, data in top_tags:
                if data["count"] >= 2:
                    graders_str = ", ".join(data["graders"])
                    recommendations.append(
                        f"失败标签 '{tag}' 出现 {data['count']} 次 "
                        f"({data['percentage']:.0f}% 的失败用例)，"
                        f"来源: {graders_str}，建议针对此类问题优先修复"
                    )

        if not recommendations:
            recommendations.append("所有指标表现良好，暂无改进建议")

        return recommendations
