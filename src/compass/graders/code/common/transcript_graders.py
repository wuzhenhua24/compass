"""Transcript-focused graders for evaluating execution traces.

These graders evaluate HOW the agent worked, not WHAT it produced.
They only access context.transcript and never look at context.outcome.

This is the key capability enabled by the Transcript/Outcome separation.
"""

import fnmatch
import math
from typing import Any

from compass.graders.base import CodeGrader, GradeContext, GradeResult, GraderScope, GraderType
from compass.graders.registry import register_grader


@register_grader("tool_usage")
class ToolUsageGrader(CodeGrader):
    """Evaluate the agent's tool usage patterns from the transcript.

    Scope: TRANSCRIPT - only reads execution trace, never the outcome.

    Checks:
    - Required tools were called
    - Forbidden tools were not called
    - Tool call count within limits
    - No repeated failed tool calls
    """

    name = "tool_usage"
    grader_type = GraderType.CODE
    grader_scope = GraderScope.TRANSCRIPT

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.required_tools = self.config.get("required_tools", [])
        self.forbidden_tools = self.config.get("forbidden_tools", [])
        self.max_tool_calls = self.config.get("max_tool_calls", 50)
        self.max_retries = self.config.get("max_retries", 3)

    async def grade(self, context: GradeContext) -> GradeResult:
        """Grade tool usage from transcript."""
        error = self.validate_context(context)
        if error:
            return GradeResult(
                name=self.name, grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                passed=False, score=0.0, error=error,
            )

        tool_calls = context.tool_calls
        checks = []
        failures = []

        # Check required tools
        called_tools = {tc.tool for tc in tool_calls}
        for required in self.required_tools:
            present = required in called_tools
            checks.append(("required_" + required, present))
            if not present:
                failures.append(f"Required tool '{required}' was not called")

        # Check forbidden tools
        for forbidden in self.forbidden_tools:
            not_called = forbidden not in called_tools
            checks.append(("forbidden_" + forbidden, not_called))
            if not not_called:
                failures.append(f"Forbidden tool '{forbidden}' was called")

        # Check total call count
        within_limit = len(tool_calls) <= self.max_tool_calls
        checks.append(("call_count", within_limit))
        if not within_limit:
            failures.append(
                f"Too many tool calls: {len(tool_calls)} > {self.max_tool_calls}"
            )

        # Check for excessive retries (same tool + same args)
        retry_counts: dict[str, int] = {}
        for tc in tool_calls:
            if tc.error:
                key = f"{tc.tool}:{str(tc.args)}"
                retry_counts[key] = retry_counts.get(key, 0) + 1

        excessive_retries = {k: v for k, v in retry_counts.items() if v > self.max_retries}
        no_excessive = len(excessive_retries) == 0
        checks.append(("retries", no_excessive))
        if not no_excessive:
            failures.append(f"Excessive retries on: {list(excessive_retries.keys())}")

        if not checks:
            return GradeResult(
                name=self.name, grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                passed=True, score=1.0,
            )

        passed_count = sum(1 for _, p in checks if p)
        total_count = len(checks)
        score = passed_count / total_count

        # Extract failure tags from failed checks
        failure_tags = [name for name, passed in checks if not passed]

        return GradeResult(
            name=self.name,
            grader_type=self.grader_type,
            grader_scope=self.grader_scope,
            passed=passed_count == total_count,
            score=score,
            details={
                "checks": {name: passed for name, passed in checks},
                "total_tool_calls": len(tool_calls),
                "unique_tools": list(called_tools),
                "failures": failures,
            },
            failure_tags=failure_tags,
        )


@register_grader("cost_budget")
class CostBudgetGrader(CodeGrader):
    """Evaluate whether the agent stayed within cost budget.

    Scope: TRANSCRIPT - reads tool calls and their durations/costs.

    Checks:
    - Total estimated cost within budget
    - Total token usage within limits
    """

    name = "cost_budget"
    grader_type = GraderType.CODE
    grader_scope = GraderScope.TRANSCRIPT

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.max_cost_usd = self.config.get("max_cost_usd", 1.0)
        self.max_tokens = self.config.get("max_tokens", 100_000)
        self.cost_per_tool: dict[str, float] = self.config.get("cost_per_tool", {})

    async def grade(self, context: GradeContext) -> GradeResult:
        """Grade cost budget from transcript."""
        error = self.validate_context(context)
        if error:
            return GradeResult(
                name=self.name, grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                passed=False, score=0.0, error=error,
            )

        tool_calls = context.tool_calls

        # Estimate cost
        total_cost = 0.0
        cost_breakdown = {}
        for tc in tool_calls:
            # Prefer tc.cost (protocol field)
            if tc.cost is not None:
                tool_cost = tc.cost.total_usd
            else:
                # Fallback to config lookup
                tool_cost = self.cost_per_tool.get(tc.tool, 0.0)
            total_cost += tool_cost
            cost_breakdown[tc.tool] = cost_breakdown.get(tc.tool, 0.0) + tool_cost

        # Estimate tokens from transcript metadata
        total_tokens = 0
        for tc in tool_calls:
            # Prefer tc.tokens (protocol field)
            if tc.tokens is not None:
                total_tokens += tc.tokens.total_tokens
            else:
                # Fallback to legacy result field
                result = tc.result
                if isinstance(result, dict):
                    total_tokens += result.get("tokens_used", 0)

        within_cost = total_cost <= self.max_cost_usd
        within_tokens = total_tokens <= self.max_tokens if self.max_tokens else True

        passed = within_cost and within_tokens

        # Score: how much budget was used (lower is better)
        if self.max_cost_usd > 0:
            cost_ratio = min(1.0, total_cost / self.max_cost_usd)
            score = max(0.0, 1.0 - cost_ratio * 0.5)  # Up to 50% penalty
        else:
            score = 1.0

        # Build failure tags
        failure_tags: list[str] = []
        if not within_cost:
            failure_tags.append("cost_exceeded")
        if not within_tokens:
            failure_tags.append("token_exceeded")

        return GradeResult(
            name=self.name,
            grader_type=self.grader_type,
            grader_scope=self.grader_scope,
            passed=passed,
            score=score if passed else 0.0,
            details={
                "total_cost_usd": total_cost,
                "max_cost_usd": self.max_cost_usd,
                "cost_breakdown": cost_breakdown,
                "total_tokens": total_tokens,
                "max_tokens": self.max_tokens,
            },
            failure_tags=failure_tags,
            reasoning=(
                f"Cost: ${total_cost:.4f} / ${self.max_cost_usd:.2f} budget"
            ),
        )


@register_grader("latency_budget")
class LatencyBudgetGrader(CodeGrader):
    """Evaluate whether the agent met latency requirements.

    Scope: TRANSCRIPT - reads timing data from execution trace.
    """

    name = "latency_budget"
    grader_type = GraderType.CODE
    grader_scope = GraderScope.TRANSCRIPT

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.max_total_ms = self.config.get("max_total_ms", 30_000)
        self.max_per_tool_ms = self.config.get("max_per_tool_ms", 10_000)

    async def grade(self, context: GradeContext) -> GradeResult:
        """Grade latency from transcript timing."""
        error = self.validate_context(context)
        if error:
            return GradeResult(
                name=self.name, grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                passed=False, score=0.0, error=error,
            )

        total_ms = context.total_duration_ms
        tool_calls = context.tool_calls

        # Check total latency
        total_ok = total_ms <= self.max_total_ms

        # Check per-tool latency
        slow_tools = []
        for tc in tool_calls:
            if tc.duration_ms > self.max_per_tool_ms:
                slow_tools.append({
                    "tool": tc.tool,
                    "duration_ms": tc.duration_ms,
                    "limit_ms": self.max_per_tool_ms,
                })

        per_tool_ok = len(slow_tools) == 0
        passed = total_ok and per_tool_ok

        # Score: ratio of budget used
        if self.max_total_ms > 0:
            ratio = min(1.0, total_ms / self.max_total_ms)
            score = max(0.0, 1.0 - ratio * 0.5)
        else:
            score = 1.0

        return GradeResult(
            name=self.name,
            grader_type=self.grader_type,
            grader_scope=self.grader_scope,
            passed=passed,
            score=score if passed else 0.0,
            details={
                "total_duration_ms": total_ms,
                "max_total_ms": self.max_total_ms,
                "slow_tools": slow_tools,
            },
            reasoning=(
                f"Latency: {total_ms:.0f}ms / {self.max_total_ms}ms budget"
            ),
        )


@register_grader("efficiency")
class EfficiencyGrader(CodeGrader):
    """Evaluate execution efficiency from transcript.

    Scope: BOTH - compares transcript effort vs outcome quality.

    This is a combined grader that checks whether the amount of work
    done (transcript) was proportional to the quality achieved (outcome).
    """

    name = "efficiency"
    grader_type = GraderType.CODE
    grader_scope = GraderScope.BOTH

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.max_steps_for_simple = self.config.get("max_steps_for_simple", 5)
        self.expected_tool_count = self.config.get("expected_tool_count")

    async def grade(self, context: GradeContext) -> GradeResult:
        """Grade efficiency by combining transcript and outcome data."""
        error = self.validate_context(context)
        if error:
            return GradeResult(
                name=self.name, grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                passed=False, score=0.0, error=error,
            )

        tool_count = len(context.tool_calls)
        has_output = context.has_output
        error_count = sum(1 for tc in context.tool_calls if tc.error)

        # Score components
        scores = []

        # 1. Did it produce output?
        if has_output:
            scores.append(1.0)
        else:
            scores.append(0.0)

        # 2. Error ratio (fewer errors = better)
        if tool_count > 0:
            error_ratio = error_count / tool_count
            scores.append(max(0.0, 1.0 - error_ratio))
        else:
            scores.append(0.5)

        # 3. Tool count efficiency
        if self.expected_tool_count and self.expected_tool_count > 0:
            ratio = tool_count / self.expected_tool_count
            if ratio <= 1.5:
                scores.append(1.0)
            elif ratio <= 3.0:
                scores.append(0.5)
            else:
                scores.append(0.0)

        overall = sum(scores) / len(scores) if scores else 0.5

        return GradeResult(
            name=self.name,
            grader_type=self.grader_type,
            grader_scope=self.grader_scope,
            passed=overall >= 0.5,
            score=overall,
            details={
                "tool_count": tool_count,
                "error_count": error_count,
                "has_output": has_output,
                "expected_tool_count": self.expected_tool_count,
            },
            reasoning=(
                f"Efficiency: {tool_count} tools, {error_count} errors, "
                f"output={'yes' if has_output else 'no'}"
            ),
        )


@register_grader("loop_detection")
class LoopDetectionGrader(CodeGrader):
    """Detect loops, cycles, and wasteful repetitive patterns in agent execution.

    Scope: TRANSCRIPT - analyzes tool call sequences for inefficient patterns.

    Detection capabilities:
    1. Single tool repetition - same tool called too many times
    2. Exact call repetition - same tool + same input repeated (no progress)
    3. Sequence pattern loops - repeating patterns like A→B→C→A→B→C
    4. Output repetition - same output produced multiple times (wasted work)

    Config:
        max_single_tool_calls: int — Max times a single tool can be called (default 10)
        max_exact_repetitions: int — Max identical calls (tool+input) allowed (default 3)
        max_sequence_repeats: int — Max times a sequence pattern can repeat (default 3)
        min_pattern_length: int — Minimum pattern length to detect (default 2)
        max_pattern_length: int — Maximum pattern length to detect (default 5)
        check_output_repetition: bool — Check for repeated outputs (default True)
        ignore_tools: list[str] — Tools to exclude from loop detection (default [])

    Example YAML:
        graders:
          - name: loop_detection
            config:
              max_single_tool_calls: 10
              max_exact_repetitions: 3
              max_sequence_repeats: 3
              ignore_tools: ["log", "print"]
    """

    name = "loop_detection"
    grader_type = GraderType.CODE
    grader_scope = GraderScope.TRANSCRIPT

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.max_single_tool_calls = self.config.get("max_single_tool_calls", 10)
        self.max_exact_repetitions = self.config.get("max_exact_repetitions", 3)
        self.max_sequence_repeats = self.config.get("max_sequence_repeats", 3)
        self.min_pattern_length = self.config.get("min_pattern_length", 2)
        self.max_pattern_length = self.config.get("max_pattern_length", 5)
        self.check_output_repetition = self.config.get("check_output_repetition", True)
        self.ignore_tools = set(self.config.get("ignore_tools", []))

    async def grade(self, context: GradeContext) -> GradeResult:
        """Detect loops and wasteful patterns in tool call sequence."""
        error = self.validate_context(context)
        if error:
            return GradeResult(
                name=self.name, grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                passed=False, score=0.0, error=error,
            )

        tool_calls = [tc for tc in context.tool_calls if tc.tool not in self.ignore_tools]

        if not tool_calls:
            return GradeResult(
                name=self.name, grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                passed=True, score=1.0,
                details={
                    "message": "No tool calls to analyze",
                    "total_tool_calls": 0,
                    "issues_found": 0,
                    "issues": [],
                },
            )

        issues = []
        severity_scores = []  # 0.0 = severe, 1.0 = no issue

        # 1. Single tool repetition check
        tool_counts = self._count_tools(tool_calls)
        excessive_tools = {
            tool: count for tool, count in tool_counts.items()
            if count > self.max_single_tool_calls
        }
        if excessive_tools:
            issues.append({
                "type": "single_tool_repetition",
                "description": "Tool called too many times",
                "details": excessive_tools,
                "severity": "high",
            })
            # Score: how much over the limit
            max_excess = max(excessive_tools.values()) / self.max_single_tool_calls
            severity_scores.append(max(0.0, 1.0 - (max_excess - 1.0) * 0.5))
        else:
            severity_scores.append(1.0)

        # 2. Exact call repetition check (same tool + same input)
        exact_repetitions = self._find_exact_repetitions(tool_calls)
        excessive_exact = {
            sig: count for sig, count in exact_repetitions.items()
            if count > self.max_exact_repetitions
        }
        if excessive_exact:
            issues.append({
                "type": "exact_call_repetition",
                "description": "Identical call (tool+input) repeated without progress",
                "details": {k: v for k, v in list(excessive_exact.items())[:5]},  # Limit to 5
                "severity": "high",
            })
            max_excess = max(excessive_exact.values()) / self.max_exact_repetitions
            severity_scores.append(max(0.0, 1.0 - (max_excess - 1.0) * 0.5))
        else:
            severity_scores.append(1.0)

        # 3. Sequence pattern loop detection
        patterns = self._detect_sequence_patterns(tool_calls)
        excessive_patterns = [
            p for p in patterns if p["repeats"] > self.max_sequence_repeats
        ]
        if excessive_patterns:
            issues.append({
                "type": "sequence_pattern_loop",
                "description": "Repeating sequence pattern detected",
                "details": excessive_patterns[:3],  # Limit to top 3
                "severity": "critical",
            })
            max_repeats = max(p["repeats"] for p in excessive_patterns)
            severity_scores.append(
                max(0.0, 1.0 - (max_repeats / self.max_sequence_repeats - 1.0) * 0.3)
            )
        else:
            severity_scores.append(1.0)

        # 4. Output repetition check (wasted work)
        if self.check_output_repetition:
            output_repetitions = self._find_output_repetitions(tool_calls)
            if output_repetitions:
                issues.append({
                    "type": "output_repetition",
                    "description": "Same output produced multiple times (wasted work)",
                    "details": output_repetitions[:5],  # Limit to 5
                    "severity": "medium",
                })
                severity_scores.append(0.7)  # Medium penalty
            else:
                severity_scores.append(1.0)

        # Calculate overall score
        overall_score = sum(severity_scores) / len(severity_scores) if severity_scores else 1.0

        # Determine pass/fail
        # Critical or high severity issues cause failure
        has_critical = any(i["severity"] == "critical" for i in issues)
        has_high = any(i["severity"] == "high" for i in issues)
        passed = not has_critical and not has_high and overall_score >= 0.5

        return GradeResult(
            name=self.name,
            grader_type=self.grader_type,
            grader_scope=self.grader_scope,
            passed=passed,
            score=overall_score,
            details={
                "total_tool_calls": len(tool_calls),
                "tool_distribution": dict(tool_counts),
                "issues_found": len(issues),
                "issues": issues,
                "has_critical_issue": has_critical,
                "has_high_severity_issue": has_high,
            },
            reasoning=self._build_reasoning(issues, len(tool_calls)),
        )

    def _count_tools(self, tool_calls: list[Any]) -> dict[str, int]:
        """Count occurrences of each tool."""
        counts: dict[str, int] = {}
        for tc in tool_calls:
            tool = tc.tool
            counts[tool] = counts.get(tool, 0) + 1
        return counts

    def _find_exact_repetitions(self, tool_calls: list[Any]) -> dict[str, int]:
        """Find exact call repetitions (same tool + same input)."""
        import json
        signatures: dict[str, int] = {}
        for tc in tool_calls:
            try:
                # Create a signature from tool name + serialized input
                input_str = json.dumps(tc.input, sort_keys=True, default=str)
                sig = f"{tc.tool}:{input_str[:100]}"  # Truncate for memory
                signatures[sig] = signatures.get(sig, 0) + 1
            except (TypeError, ValueError):
                # If input is not serializable, use tool name only
                signatures[tc.tool] = signatures.get(tc.tool, 0) + 1
        return {k: v for k, v in signatures.items() if v > 1}

    def _detect_sequence_patterns(self, tool_calls: list[Any]) -> list[dict[str, Any]]:
        """Detect repeating sequence patterns like A→B→C→A→B→C."""
        if len(tool_calls) < self.min_pattern_length * 2:
            return []

        tool_sequence = [tc.tool for tc in tool_calls]
        detected_patterns = []

        # Try different pattern lengths
        for pattern_len in range(self.min_pattern_length, self.max_pattern_length + 1):
            if len(tool_sequence) < pattern_len * 2:
                continue

            # Slide through sequence looking for repeating patterns
            for start in range(len(tool_sequence) - pattern_len * 2 + 1):
                pattern = tuple(tool_sequence[start:start + pattern_len])
                repeats = 1

                # Count consecutive repeats
                pos = start + pattern_len
                while pos + pattern_len <= len(tool_sequence):
                    next_segment = tuple(tool_sequence[pos:pos + pattern_len])
                    if next_segment == pattern:
                        repeats += 1
                        pos += pattern_len
                    else:
                        break

                if repeats >= 2:  # At least 2 repeats to be a pattern
                    # Check if this pattern is already detected (subset)
                    pattern_str = "→".join(pattern)
                    if not any(p["pattern"] == pattern_str for p in detected_patterns):
                        detected_patterns.append({
                            "pattern": pattern_str,
                            "length": pattern_len,
                            "repeats": repeats,
                            "start_index": start,
                        })

        # Sort by severity (repeats * length)
        detected_patterns.sort(key=lambda p: p["repeats"] * p["length"], reverse=True)
        return detected_patterns

    def _find_output_repetitions(self, tool_calls: list[Any]) -> list[dict[str, Any]]:
        """Find cases where same output is produced multiple times."""
        import json
        output_map: dict[str, list[int]] = {}

        for i, tc in enumerate(tool_calls):
            if tc.output is None:
                continue
            try:
                output_str = json.dumps(tc.output, sort_keys=True, default=str)[:200]
                output_key = f"{tc.tool}:{output_str}"
                if output_key not in output_map:
                    output_map[output_key] = []
                output_map[output_key].append(i)
            except (TypeError, ValueError):
                continue

        repetitions = []
        for key, indices in output_map.items():
            if len(indices) > 1:
                tool_name = key.split(":")[0]
                repetitions.append({
                    "tool": tool_name,
                    "occurrences": len(indices),
                    "indices": indices[:5],  # Limit stored indices
                })

        return sorted(repetitions, key=lambda x: x["occurrences"], reverse=True)

    def _build_reasoning(self, issues: list[dict[str, Any]], total_calls: int) -> str:
        """Build human-readable reasoning string."""
        if not issues:
            return f"No loop or wasteful patterns detected in {total_calls} tool calls"

        parts = [f"Detected {len(issues)} issue(s) in {total_calls} tool calls:"]
        for issue in issues[:3]:  # Limit to 3 in reasoning
            if issue["type"] == "single_tool_repetition":
                tools = list(issue["details"].keys())[:2]
                parts.append(f"- {issue['type']}: {', '.join(tools)}")
            elif issue["type"] == "sequence_pattern_loop":
                pattern = issue["details"][0]["pattern"] if issue["details"] else "?"
                parts.append(f"- {issue['type']}: {pattern}")
            else:
                parts.append(f"- {issue['type']}")

        return " ".join(parts)


# =====================================================================
# Turn count / interaction efficiency grader
# =====================================================================


@register_grader("turn_count")
class TurnCountGrader(CodeGrader):
    """Evaluate agent interaction efficiency by counting turns.

    Scope: TRANSCRIPT — only evaluates execution trace.

    Inspired by Stripe Agent Benchmark where turn count (17–216) is a key
    efficiency metric.  Completing the same task in fewer turns signals a
    more capable model.

    Scoring modes:

    * **budget** (default) — binary pass/fail against ``max_turns``.
      Score is 1.0 if within budget, else ``max_turns / actual`` (graceful
      degradation, never 0 unless 0 tool calls).
    * **linear** — linearly interpolate between ``min_turns`` (score 1.0)
      and ``max_turns`` (score 0.0).  Scores below ``min_turns`` are
      clamped to 1.0; scores above ``max_turns`` are clamped to 0.0.
    * **logarithmic** — penalise excess turns on a log-scale.  Tolerant
      of moderate overruns, harsh on extreme ones.  Uses
      ``1 - log(actual/ideal) / log(max/ideal)`` where *ideal* is
      ``min_turns`` (or ``expected_turns``).

    Config:
        max_turns:       int   — hard ceiling (default 50).
        min_turns:       int   — ideal/minimum turns (default 1).
        expected_turns:  int   — optional expected turns for efficiency
                                 ratio reporting.
        scoring:         str   — "budget" | "linear" | "logarithmic"
                                 (default "budget").
        count_filter:    str   — which tool calls to count:
                                 "all" (default) | "llm" | "non-error".
        pass_threshold:  float — minimum score to pass (default 0.7).
    """

    name = "turn_count"
    grader_type = GraderType.CODE
    grader_scope = GraderScope.TRANSCRIPT

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.max_turns: int = self.config.get("max_turns", 50)
        self.min_turns: int = self.config.get("min_turns", 1)
        self.expected_turns: int | None = self.config.get("expected_turns")
        self.scoring: str = self.config.get("scoring", "budget")
        self.count_filter: str = self.config.get("count_filter", "all")
        self.pass_threshold: float = self.config.get("pass_threshold", 0.7)

    # -----------------------------------------------------------------

    async def grade(self, context: GradeContext) -> GradeResult:
        error = self.validate_context(context)
        if error:
            return self._error_result(error)

        turns = self._count_turns(context)
        score = self._compute_score(turns)
        passed = score >= self.pass_threshold

        details: dict[str, Any] = {
            "turn_count": turns,
            "max_turns": self.max_turns,
            "min_turns": self.min_turns,
            "scoring": self.scoring,
            "count_filter": self.count_filter,
        }
        if self.expected_turns is not None:
            details["expected_turns"] = self.expected_turns
            details["efficiency_ratio"] = (
                round(turns / self.expected_turns, 2)
                if self.expected_turns > 0
                else None
            )

        reasoning = self._build_reasoning(turns, score)

        return GradeResult(
            name=self.name,
            grader_type=self.grader_type,
            grader_scope=self.grader_scope,
            passed=passed,
            score=round(score, 4),
            details=details,
            reasoning=reasoning,
        )

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _count_turns(self, context: GradeContext) -> int:
        """Count tool calls according to the configured filter."""
        tool_calls = context.tool_calls
        if self.count_filter == "llm":
            return sum(
                1 for tc in tool_calls
                if getattr(tc, "tool_type", None) == "llm"
            )
        if self.count_filter == "non-error":
            return sum(
                1 for tc in tool_calls if not tc.error
            )
        return len(tool_calls)

    def _compute_score(self, turns: int) -> float:
        """Compute a 0–1 score from the turn count."""
        if turns == 0:
            return 0.0

        if self.scoring == "linear":
            return self._score_linear(turns)
        if self.scoring == "logarithmic":
            return self._score_logarithmic(turns)
        # default: budget
        return self._score_budget(turns)

    def _score_budget(self, turns: int) -> float:
        if turns <= self.max_turns:
            return 1.0
        # Graceful degradation: the more you exceed, the lower the score
        return min(1.0, self.max_turns / turns)

    def _score_linear(self, turns: int) -> float:
        if turns <= self.min_turns:
            return 1.0
        if turns >= self.max_turns:
            return 0.0
        span = self.max_turns - self.min_turns
        if span <= 0:
            return 1.0 if turns <= self.min_turns else 0.0
        return 1.0 - (turns - self.min_turns) / span

    def _score_logarithmic(self, turns: int) -> float:
        ideal = self.min_turns or 1
        if turns <= ideal:
            return 1.0
        if turns >= self.max_turns:
            return 0.0
        log_range = math.log(self.max_turns / ideal)
        if log_range <= 0:
            return 1.0
        return max(0.0, 1.0 - math.log(turns / ideal) / log_range)

    def _build_reasoning(self, turns: int, score: float) -> str:
        parts = [f"Turn count: {turns}"]
        if self.expected_turns:
            ratio = turns / self.expected_turns if self.expected_turns else 0
            parts.append(f"(expected {self.expected_turns}, ratio {ratio:.1f}x)")
        parts.append(f"limit: {self.max_turns}")
        parts.append(f"scoring: {self.scoring}")
        parts.append(f"score: {score:.2f}")
        return " | ".join(parts)

    def _error_result(self, error: str) -> GradeResult:
        return GradeResult(
            name=self.name,
            grader_type=self.grader_type,
            grader_scope=self.grader_scope,
            passed=False,
            score=0.0,
            error=error,
        )


@register_grader("leak_detection")
class LeakDetectionGrader(CodeGrader):
    """Detect answer leakage by scanning transcripts for sentinel markers.

    Scope: TRANSCRIPT — only reads execution trace.

    Inspired by Stripe's agent benchmark: each eval embeds unique UUID
    markers in grader/solution files.  If the agent reads those files
    during execution, the markers appear in its transcript (tool-call
    outputs, inputs, or reasoning steps), proving it "peeked" at the
    answer.  The trial is then invalidated with score 0.

    Markers can come from two sources (merged at check time):
    1. ``GradeContext.leak_markers`` — set by the runner from Scenario /
       TestCase YAML (``leak_markers`` field).
    2. ``config.leak_patterns`` — set on this grader's own config.

    Config:
        leak_patterns:  list[str] — extra patterns beyond context markers.
        invalidate:     bool      — if True (default), leaked trial gets
                                    score 0; if False, score is reduced
                                    proportionally by number of leaks.

    YAML example::

        # Scenario-level markers (auto-injected into GradeContext)
        leak_markers:
          - "COMPASS_LEAK_a1b2c3d4e5f6"

        cases:
          - id: task_1
            leak_markers:
              - "TASK1_SECRET_UUID_789"
            graders:
              - name: leak_detection
    """

    name = "leak_detection"
    grader_type = GraderType.CODE
    grader_scope = GraderScope.TRANSCRIPT

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.leak_patterns: list[str] = self.config.get("leak_patterns", [])
        self.invalidate: bool = self.config.get("invalidate", True)

    async def grade(self, context: GradeContext) -> GradeResult:
        if not context.has_transcript:
            return GradeResult(
                name=self.name,
                grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                passed=True,
                score=1.0,
                reasoning="No transcript available — skipping leak check.",
            )

        # Merge context-level markers with grader-config patterns
        result = context.check_leaks(extra_patterns=self.leak_patterns)

        if result.has_leaks:
            score = 0.0 if self.invalidate else max(
                0.0,
                1.0 - len(result.leaked) / max(len(result.searched_patterns), 1),
            )
            return GradeResult(
                name=self.name,
                grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                passed=False,
                score=score,
                details=result.to_dict(),
                failure_tags=["answer_leak"],
                reasoning=(
                    f"Trial invalidated: agent accessed grader/solution "
                    f"data (leaked {len(result.leaked)} of "
                    f"{len(result.searched_patterns)} markers: "
                    f"{result.leaked})"
                ),
            )

        return GradeResult(
            name=self.name,
            grader_type=self.grader_type,
            grader_scope=self.grader_scope,
            passed=True,
            score=1.0,
            details=result.to_dict(),
            reasoning=(
                f"No leaks detected ({len(result.searched_patterns)} "
                f"markers checked)."
            ),
        )


@register_grader("state_delta")
class StateDeltaGrader(CodeGrader):
    """Guard the environment changes (state delta) recorded in the transcript.

    Scope: TRANSCRIPT - reads ``ToolCall.state_delta``, never the outcome.

    The final answer and the state delta can disagree ("scheduled the meeting"
    vs. a duplicate invite in the calendar). This grader checks the delta side:
    what the agent actually changed in the world.

    Checks (all optional, driven by config):
    - ``readonly``: no state changes at all (strong guard for read-only agents)
    - ``forbid``: no recorded change may match any of these matchers
    - ``require``: each matcher must match at least one recorded change
    - ``max_changes``: total number of recorded changes within limit

    A matcher is a dict with optional keys ``kind`` / ``op`` / ``target``:
    ``kind`` and ``op`` match exactly, ``target`` is an fnmatch glob
    (e.g. ``{"kind": "file", "op": "delete", "target": "/etc/*"}``).
    A missing key matches anything.

    Caveat: this grader sees only what was *recorded*. An empty state delta
    means "nothing captured", not "nothing changed" - capturing deltas is the
    adapter/harness/importer's job. ``require`` matchers therefore double as a
    capture check: they fail when the expected change was not recorded.
    """

    name = "state_delta"
    grader_type = GraderType.CODE
    grader_scope = GraderScope.TRANSCRIPT

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.readonly = self.config.get("readonly", False)
        self.forbid = self.config.get("forbid", [])
        self.require = self.config.get("require", [])
        self.max_changes = self.config.get("max_changes")

    @staticmethod
    def _matches(matcher: dict[str, Any], change: Any) -> bool:
        """Whether a state change matches a {kind, op, target} matcher."""
        if "kind" in matcher and change.kind != matcher["kind"]:
            return False
        if "op" in matcher and change.op != matcher["op"]:
            return False
        if "target" in matcher and not fnmatch.fnmatch(
            change.target, matcher["target"]
        ):
            return False
        return True

    @staticmethod
    def _describe(matcher: dict[str, Any]) -> str:
        """Human-readable form of a matcher for failure messages."""
        return (
            f"kind={matcher.get('kind', '*')} "
            f"op={matcher.get('op', '*')} "
            f"target={matcher.get('target', '*')}"
        )

    async def grade(self, context: GradeContext) -> GradeResult:
        """Grade the recorded state delta against configured constraints."""
        error = self.validate_context(context)
        if error:
            return GradeResult(
                name=self.name, grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                passed=False, score=0.0, error=error,
            )

        # Keep (call_id, change) pairs so violations attribute to the step
        # that caused them - outcome-level checks can't do that.
        changes = [
            (tc.call_id, sc)
            for tc in context.tool_calls
            for sc in tc.state_delta
        ]

        checks: list[tuple[str, bool]] = []
        failures: list[str] = []
        violations: list[dict[str, Any]] = []

        if self.readonly:
            is_clean = len(changes) == 0
            checks.append(("readonly", is_clean))
            if not is_clean:
                failures.append(
                    f"Expected no state changes, but {len(changes)} were recorded"
                )
                violations.extend(
                    {"rule": "readonly", "call_id": cid, **sc.to_dict()}
                    for cid, sc in changes
                )

        for matcher in self.forbid:
            hits = [(cid, sc) for cid, sc in changes if self._matches(matcher, sc)]
            ok = len(hits) == 0
            checks.append((f"forbid[{self._describe(matcher)}]", ok))
            if not ok:
                failures.append(
                    f"Forbidden change matched ({self._describe(matcher)}): "
                    f"{[sc.target for _, sc in hits]}"
                )
                violations.extend(
                    {"rule": f"forbid[{self._describe(matcher)}]",
                     "call_id": cid, **sc.to_dict()}
                    for cid, sc in hits
                )

        for matcher in self.require:
            found = any(self._matches(matcher, sc) for _, sc in changes)
            checks.append((f"require[{self._describe(matcher)}]", found))
            if not found:
                failures.append(
                    f"Required change not recorded ({self._describe(matcher)})"
                )

        if self.max_changes is not None:
            within = len(changes) <= self.max_changes
            checks.append(("max_changes", within))
            if not within:
                failures.append(
                    f"Too many state changes: {len(changes)} > {self.max_changes}"
                )

        if not checks:
            return GradeResult(
                name=self.name, grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                passed=True, score=1.0,
                reasoning="No state delta constraints configured.",
                details={"total_changes": len(changes)},
            )

        passed_count = sum(1 for _, p in checks if p)
        score = passed_count / len(checks)
        failure_tags = [name for name, passed in checks if not passed]

        return GradeResult(
            name=self.name,
            grader_type=self.grader_type,
            grader_scope=self.grader_scope,
            passed=passed_count == len(checks),
            score=score,
            details={
                "checks": {name: passed for name, passed in checks},
                "total_changes": len(changes),
                "violations": violations,
                "failures": failures,
            },
            failure_tags=failure_tags,
        )
