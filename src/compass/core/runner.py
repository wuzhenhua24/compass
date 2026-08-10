"""Test runner for executing scenarios."""

import asyncio
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from compass.adapters import get_adapter, AgentInput, AgentOutput
from compass.core.checkpoint import CheckpointStore, scenario_fingerprint
from compass.core.regrade import grader_fingerprint
from compass.core.result import CaseResult, EvalResult, EvaluatorResult, TestStatus
from compass.core.sweep import expand_sweeps
from compass.core.scenario import (
    AggregationConfig,
    GraderConfig,
    GraderType,
    MetricsConfig,
    Scenario,
    ShortCircuitMode,
    TestCase,
)
from compass.core.transcript import Outcome, Transcript, TranscriptRecorder
from compass.core.trial import TrialManager, TaskResult, TrialResult
from compass.graders import get_grader, GradeContext, GradeResult

logger = logging.getLogger(__name__)


class Compass:
    """Main test runner class."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the test runner.

        Args:
            config: Optional configuration dictionary.
        """
        self.config = config or {}
        self._results: list[EvalResult] = []

    async def run(
        self,
        scenario: Scenario | str | Path,
        case_ids: list[str] | None = None,
        stages: list[str] | None = None,
        categories: list[str] | None = None,
        parallel: bool = False,
        max_workers: int = 4,
        trace_dir: str | Path | None = None,
        trace_format: str = "json",
        resume: bool = False,
    ) -> EvalResult:
        """Run a test scenario.

        Args:
            scenario: Scenario object or path to YAML file.
            case_ids: Optional list of specific case IDs to run.
            stages: Optional list of stages to filter by (OR logic between stages,
                AND logic with case_ids).
            categories: Optional list of categories to filter by.
            parallel: Whether to run cases in parallel.
            max_workers: Maximum number of parallel workers.
            trace_dir: Optional directory to save execution traces.
            trace_format: Trace file format ("json" or "jsonl").
            resume: If True, resume from a previous checkpoint in trace_dir.

        Returns:
            EvalResult containing all case results.
        """
        # Prepare trace directory
        trace_path: Path | None = None
        if trace_dir:
            trace_path = Path(trace_dir)
            trace_path.mkdir(parents=True, exist_ok=True)
        if isinstance(scenario, (str, Path)):
            scenario = Scenario.from_yaml(scenario)

        # Filter cases if specific IDs provided
        cases = scenario.cases
        if case_ids:
            cases = [c for c in cases if c.id in case_ids]

        # Filter cases by stage
        if stages:
            cases = [c for c in cases if c.stage in stages]

        # Filter cases by category
        if categories:
            cases = [
                c for c in cases
                if scenario.get_category_for_case(c) in categories
            ]

        # Expand sweep configurations into concrete cases
        cases = expand_sweeps(cases, scenario.sweep)

        # ── Checkpoint handling ──
        all_case_ids = [c.id for c in cases]
        fingerprint = scenario_fingerprint(scenario)
        run_id = uuid.uuid4().hex[:12]
        completed_results: dict[str, CaseResult] = {}
        existing_trials: dict[str, list[TrialResult]] = {}
        store: CheckpointStore | None = None

        if trace_path is not None:
            if resume:
                store = CheckpointStore.load(trace_path)
                if store is not None and store.validate_scenario(fingerprint):
                    cp = store.get_checkpoint()
                    completed_results = store.load_completed_results()
                    # Partially-sampled multi-trial cases resume from the
                    # attempts already paid for, rather than starting over.
                    existing_trials = store.load_trials()
                    run_id = cp.run_id
                    logger.info(
                        "Resuming run %s — %s completed, %d remaining",
                        run_id,
                        cp.progress,
                        len(cp.remaining_case_ids),
                    )
                else:
                    if store is not None:
                        logger.warning(
                            "Scenario changed since last checkpoint, starting fresh"
                        )
                    store = CheckpointStore.create(
                        trace_path, run_id, fingerprint, scenario.name, all_case_ids,
                    )
            else:
                store = CheckpointStore.create(
                    trace_path, run_id, fingerprint, scenario.name, all_case_ids,
                )

        # Filter out already-completed cases
        remaining_cases = [
            c for c in cases if c.id not in completed_results
        ]

        start_time = time.time()

        # Multi-trial cases go through the round-major executor so samples stay
        # balanced under interruption; single-trial cases have nothing to
        # balance and keep the simpler per-case path.
        multi_trial_cases = [
            c for c in remaining_cases if scenario.get_trials_for_case(c) > 1
        ]
        single_trial_cases = [
            c for c in remaining_cases if scenario.get_trials_for_case(c) <= 1
        ]

        new_results: list[CaseResult] = []
        if single_trial_cases:
            if parallel:
                new_results += await self._run_parallel(
                    scenario, single_trial_cases, max_workers, trace_path, trace_format,
                    checkpoint_store=store,
                    run_id=run_id, config_hash=fingerprint,
                )
            else:
                new_results += await self._run_sequential(
                    scenario, single_trial_cases, trace_path, trace_format,
                    checkpoint_store=store,
                    run_id=run_id, config_hash=fingerprint,
                )
        if multi_trial_cases:
            new_results += await self._run_trial_rounds(
                scenario, multi_trial_cases,
                trace_dir=trace_path, trace_format=trace_format,
                checkpoint_store=store,
                run_id=run_id, config_hash=fingerprint,
                parallel=parallel, max_workers=max_workers,
                existing_trials=existing_trials,
            )

        duration_ms = (time.time() - start_time) * 1000

        # Build a lookup from new results
        new_results_map = {r.case_id: r for r in new_results}

        # Merge completed + new in original case order
        case_results: list[CaseResult] = []
        for case in cases:
            if case.id in completed_results:
                case_results.append(completed_results[case.id])
            elif case.id in new_results_map:
                case_results.append(new_results_map[case.id])

        if store is not None:
            store.mark_completed()

        # Aggregate results
        passed = sum(1 for r in case_results if r.passed)
        failed = sum(1 for r in case_results if r.status == TestStatus.FAILED)
        errors = sum(1 for r in case_results if r.status == TestStatus.ERROR)

        result = EvalResult(
            scenario_name=scenario.name,
            run_id=run_id,
            config_hash=fingerprint,
            total_cases=len(case_results),
            passed_cases=passed,
            failed_cases=failed,
            error_cases=errors,
            case_results=case_results,
            duration_ms=duration_ms,
            timestamp=datetime.now(),
        )

        self._results.append(result)
        return result

    async def _run_sequential(
        self,
        scenario: Scenario,
        cases: list[TestCase],
        trace_dir: Path | None = None,
        trace_format: str = "json",
        checkpoint_store: CheckpointStore | None = None,
        run_id: str = "",
        config_hash: str = "",
    ) -> list[CaseResult]:
        """Run test cases sequentially."""
        results = []
        for case in cases:
            result = await self._run_case(
                scenario, case, trace_dir, trace_format,
                run_id=run_id, config_hash=config_hash,
            )
            if checkpoint_store is not None:
                checkpoint_store.save_case_result(case.id, result)
            results.append(result)
        return results

    async def _run_parallel(
        self,
        scenario: Scenario,
        cases: list[TestCase],
        max_workers: int,
        trace_dir: Path | None = None,
        trace_format: str = "json",
        checkpoint_store: CheckpointStore | None = None,
        run_id: str = "",
        config_hash: str = "",
    ) -> list[CaseResult]:
        """Run test cases in parallel."""
        semaphore = asyncio.Semaphore(max_workers)

        async def run_with_semaphore(case: TestCase) -> CaseResult:
            async with semaphore:
                result = await self._run_case(
                    scenario, case, trace_dir, trace_format,
                    run_id=run_id, config_hash=config_hash,
                )
                if checkpoint_store is not None:
                    checkpoint_store.save_case_result(case.id, result)
                return result

        tasks = [run_with_semaphore(case) for case in cases]
        return await asyncio.gather(*tasks)

    async def _run_case(
        self,
        scenario: Scenario,
        case: TestCase,
        trace_dir: Path | None = None,
        trace_format: str = "json",
        run_id: str = "",
        config_hash: str = "",
    ) -> CaseResult:
        """Run a single-trial test case.

        Cases configured for multiple trials go through
        :meth:`_run_trial_rounds` instead, which samples round-major so an
        interrupted run leaves balanced samples.
        """
        start_time = time.time()
        # Content hash of the grading contract applied here — the same
        # fingerprint `compass grade` writes, so a live result and a re-graded
        # one are directly comparable (and `compass compare` can tell when a
        # score delta is the grader moving rather than the agent).
        fingerprint = grader_fingerprint(scenario, case)

        try:
            passed, score, evaluator_results, output_data, _, transcript = await self._run_single_trial(
                scenario, case, run_id=run_id, config_hash=config_hash
            )

            # Save trace if trace_dir is specified
            if trace_dir and transcript:
                self._save_trace(transcript, trace_dir, case.id, trace_format)
            duration_ms = (time.time() - start_time) * 1000

            return CaseResult(
                case_id=case.id,
                status=TestStatus.PASSED if passed else TestStatus.FAILED,
                passed=passed,
                overall_score=score,
                evaluator_results=evaluator_results,
                input_data=case.input.model_dump(),
                output_data=output_data,
                duration_ms=duration_ms,
                tags=case.tags,
                category=scenario.get_category_for_case(case),
                total_trials=1,
                passed_trials=1 if passed else 0,
                grader_fingerprint=fingerprint,
            )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return CaseResult(
                case_id=case.id,
                status=TestStatus.ERROR,
                passed=False,
                overall_score=0.0,
                input_data=case.input.model_dump(),
                duration_ms=duration_ms,
                tags=case.tags,
                category=scenario.get_category_for_case(case),
                error=str(e),
                total_trials=1,
                passed_trials=0,
                error_trials=1,
                grader_fingerprint=fingerprint,
            )

    async def _run_trial_rounds(
        self,
        scenario: Scenario,
        cases: list[TestCase],
        *,
        trace_dir: Path | None = None,
        trace_format: str = "json",
        checkpoint_store: CheckpointStore | None = None,
        run_id: str = "",
        config_hash: str = "",
        parallel: bool = False,
        max_workers: int = 4,
        existing_trials: dict[str, list[TrialResult]] | None = None,
    ) -> list[CaseResult]:
        """Execute multi-trial cases round-major, topping up to the target.

        Two properties the naive "finish case A's k trials, then case B's" loop
        cannot give you:

        **Balanced samples.** Trials run in full passes over every case, so an
        interrupted run leaves each case with a comparable number of attempts
        instead of the first cases fully sampled and the last ones untouched.
        pass^k computed over a lopsided sample is not the metric it claims to be.

        **Top-up, not re-run.** The target is "have N evaluated trials", not
        "execute N trials", so re-running the same command after an interruption
        executes only the shortfall and is a no-op once every case is complete.
        Trials lost to harness errors do not count toward the target — but the
        shortfall is attempted only once per invocation, so a persistently
        broken adapter fails the run instead of spinning in a retry loop.
        """
        existing_trials = existing_trials or {}
        trials: dict[str, list[TrialResult]] = {
            case.id: list(existing_trials.get(case.id, [])) for case in cases
        }
        shortfall = {
            case.id: max(
                0,
                scenario.get_trials_for_case(case)
                - sum(1 for t in trials[case.id] if t.error is None),
            )
            for case in cases
        }
        # Trial numbers continue past whatever a previous invocation recorded
        next_number = {
            case.id: max((t.trial_number for t in trials[case.id]), default=0) + 1
            for case in cases
        }

        max_rounds = max(shortfall.values(), default=0)
        semaphore = asyncio.Semaphore(max_workers) if parallel else None

        for round_index in range(1, max_rounds + 1):
            round_cases = [c for c in cases if shortfall[c.id] >= round_index]
            if not round_cases:
                continue

            async def run_one(case: TestCase) -> tuple[str, TrialResult]:
                number = next_number[case.id]
                next_number[case.id] = number + 1
                if semaphore is not None:
                    async with semaphore:
                        trial = await self._execute_one_trial(
                            scenario, case, number,
                            trace_dir, trace_format, run_id, config_hash,
                        )
                else:
                    trial = await self._execute_one_trial(
                        scenario, case, number,
                        trace_dir, trace_format, run_id, config_hash,
                    )
                return case.id, trial

            if parallel:
                completed = await asyncio.gather(*(run_one(c) for c in round_cases))
            else:
                completed = [await run_one(c) for c in round_cases]

            for case_id, trial in completed:
                trials[case_id].append(trial)
                if checkpoint_store is not None:
                    checkpoint_store.save_trial_result(case_id, trial)

        # Fold each case's trials into a CaseResult
        results: list[CaseResult] = []
        for case in cases:
            task_result = TaskResult(
                task_id=case.id,
                expect=case.expect,
                expect_reason=case.expect_reason,
                trials=trials[case.id],
                input_data=case.input.model_dump(),
            )
            case_result = self._task_result_to_case_result(task_result, case.metrics)
            case_result.tags = case.tags
            case_result.category = scenario.get_category_for_case(case)
            case_result.grader_fingerprint = grader_fingerprint(scenario, case)
            results.append(case_result)
            if checkpoint_store is not None:
                checkpoint_store.save_case_result(case.id, case_result)

        return results

    async def _execute_one_trial(
        self,
        scenario: Scenario,
        case: TestCase,
        trial_number: int,
        trace_dir: Path | None,
        trace_format: str,
        run_id: str,
        config_hash: str,
    ) -> TrialResult:
        """Run one attempt, capturing harness errors as an unscored trial."""
        captured: list[Transcript] = []

        async def run_fn():
            result = await self._run_single_trial(
                scenario, case, run_id=run_id, config_hash=config_hash
            )
            if len(result) >= 6 and result[5] is not None:
                captured.append(result[5])
            return result

        trial = await TrialManager(num_trials=1).execute_trial(
            case.id, trial_number, run_fn
        )
        if trace_dir and captured:
            self._save_trace(
                captured[0], trace_dir, f"{case.id}_trial{trial_number}", trace_format
            )
        return trial

    async def _run_single_trial(
        self,
        scenario: Scenario,
        case: TestCase,
        run_id: str = "",
        config_hash: str = "",
    ) -> tuple[bool, float, list[EvaluatorResult], dict[str, Any], dict[str, Any], Transcript]:
        """Execute a single trial.

        Returns:
            Tuple of (passed, score, evaluator_results, output_data, environment, transcript).
        """
        trial_id = f"{case.id}_trial_{uuid.uuid4().hex[:8]}"

        with TranscriptRecorder(task_id=case.id, trial_id=trial_id) as transcript:
            transcript.input_prompt = case.input.prompt
            transcript.input_params = case.input.params
            transcript.run_id = run_id
            transcript.config_hash = config_hash

            try:
                # Get adapter and run agent
                adapter = get_adapter(scenario.agent.adapter)(scenario.agent.config)
                agent_input = AgentInput(
                    prompt=case.input.prompt,
                    negative_prompt=case.input.negative_prompt,
                    params=case.input.params,
                    context={"transcript": transcript},
                )
                agent_output = await adapter.run(agent_input)

                # Check if adapter returned an error (not raised as exception)
                if agent_output.error:
                    transcript.set_outcome(
                        output_data=agent_output.to_dict(),
                        metadata={"agent_error": agent_output.error},
                    )
                    transcript.finalize(
                        grader_results=[],
                        final_score=0.0,
                        final_passed=False,
                    )
                    # Raise to trigger ERROR status in caller
                    raise RuntimeError(f"Agent error: {agent_output.error}")

                # Build outcome (also stored on transcript)
                transcript.set_outcome(
                    image=agent_output.image,
                    output_data=agent_output.to_dict(),
                    artifacts=list(agent_output.artifacts),
                )
                outcome = transcript.outcome

                passed, overall_score, evaluator_results = await self.grade_transcript(
                    scenario, case, transcript, outcome=outcome
                )

                transcript.finalize(
                    grader_results=[
                        {
                            "name": r.name,
                            "score": r.score,
                            "passed": r.passed,
                            "weight": r.weight,
                            "required": r.required,
                            "gate": r.gate,
                            "grader_type": r.grader_type,
                            "grader_scope": r.grader_scope,
                            "grader_version": r.grader_version,
                            "metadata": r.metadata,
                            "failure_tags": r.failure_tags,
                            "error": r.error,
                        }
                        for r in evaluator_results
                    ],
                    final_score=overall_score,
                    final_passed=passed,
                )

                return (
                    passed,
                    overall_score,
                    evaluator_results,
                    agent_output.to_dict(),
                    {},  # environment placeholder
                    transcript,
                )

            except Exception as e:
                transcript.set_outcome(
                    output_data={"error": str(e)},
                    metadata={"error": str(e)},
                )
                transcript.finalize(
                    grader_results=[],
                    final_score=0.0,
                    final_passed=False,
                )
                # Re-raise to let TrialManager handle error recording
                raise

    async def grade_transcript(
        self,
        scenario: Scenario,
        case: TestCase,
        transcript: Transcript,
        outcome: Outcome | None = None,
    ) -> tuple[bool, float, list[EvaluatorResult]]:
        """Grade an already-executed transcript against a case's graders.

        This is the single grading path, shared by live runs (``compass test``)
        and offline re-grading (``compass grade``) so the two can never drift:
        a score produced by re-grading a recorded trace is the score the live
        run would have produced from the same evidence.

        Args:
            scenario: The scenario supplying default graders, aggregation,
                leak markers and category.
            case: The test case whose graders/aggregation apply.
            transcript: The execution trace (live or loaded from disk).
            outcome: The outcome to grade; defaults to ``transcript.outcome``.

        Returns:
            Tuple of (passed, overall_score, evaluator_results). ``passed``
            already accounts for ``expect=fail`` negative tests.
        """
        if outcome is None:
            outcome = transcript.outcome

        # Load named reference images
        reference_images = self._load_reference_images(case.input.reference_images)

        # Back-compat: if exactly one reference image, also set
        # the legacy single reference_image field
        ref_image = None
        if len(reference_images) == 1:
            ref_image = next(iter(reference_images.values()))

        # Build grade context
        grade_context = GradeContext(
            prompt=case.input.prompt,
            negative_prompt=case.input.negative_prompt,
            params=case.input.params,
            transcript=transcript,
            outcome=outcome,
            reference_image=ref_image,
            reference_images=reference_images,
            leak_markers=scenario.get_leak_markers_for_case(case),
            metadata=case.metadata,
        )

        # Get graders for this case (default_graders + case graders)
        grader_configs = scenario.get_graders_for_case(case)

        # Get aggregation config (needed for short-circuit mode)
        aggregation = scenario.get_aggregation_for_case(case)

        # Run graders (with Code-First short-circuit support)
        evaluator_results = await self._run_graders(
            grader_configs, grade_context, aggregation
        )
        overall_score, graders_passed = self._aggregate_results(
            evaluator_results,
            pass_threshold=aggregation.pass_threshold,
            required_graders=aggregation.required_graders,
        )

        # Handle expect=fail (negative tests)
        # For negative tests, we expect the agent to be blocked or fail
        if case.is_negative_test:
            # Negative test passes if: agent was blocked OR graders failed
            agent_blocked = outcome.blocked if outcome else False
            passed = agent_blocked or not graders_passed
        else:
            # Positive test: normal pass logic
            passed = graders_passed

        return passed, overall_score, evaluator_results

    def _save_trace(
        self,
        transcript,
        trace_dir: Path,
        case_id: str,
        trace_format: str,
    ) -> None:
        """Save transcript trace to file.

        Args:
            transcript: The Transcript object to save.
            trace_dir: Directory to save the trace file.
            case_id: Case identifier for filename.
            trace_format: Format to save ("json" or "jsonl").
        """
        import json

        # Sanitize case_id for filename
        safe_case_id = case_id.replace("/", "_").replace("\\", "_")

        if trace_format == "jsonl":
            trace_file = trace_dir / f"{safe_case_id}.jsonl"
            transcript.save_jsonl(trace_file)
        else:
            # Default to JSON format
            trace_file = trace_dir / f"{safe_case_id}.json"
            with open(trace_file, "w", encoding="utf-8") as f:
                json.dump(transcript.to_dict(), f, ensure_ascii=False, indent=2, default=str)

        # Save artifact binaries
        if transcript.outcome and (
            transcript.outcome.image is not None
            or transcript.outcome.artifacts
        ):
            from compass.core.artifact_store import ArtifactStore

            store = ArtifactStore(trace_dir)
            saved = store.save_trial_artifacts(transcript, case_id)
            if saved:
                logger.info("Saved %d artifact(s) for case %s", len(saved), case_id)

    @staticmethod
    def _load_reference_images(paths: dict[str, str]) -> dict:
        """Load PIL Images from a name-to-path mapping.

        Skips entries that fail to load and logs a warning.
        """
        from PIL import Image

        images: dict = {}
        for name, path in paths.items():
            try:
                images[name] = Image.open(path).copy()
            except Exception:
                logger.warning(
                    "Failed to load reference image '%s' from %s", name, path
                )
        return images

    def _task_result_to_case_result(
        self,
        task_result: TaskResult,
        metrics_config: MetricsConfig | None = None,
    ) -> CaseResult:
        """Convert TaskResult to CaseResult.

        ``metrics_config`` (the case's ``metrics:`` block) drives which
        ``pass@k`` / ``pass^k`` values are reported and whether the consistency
        score is included.
        """
        metrics = task_result.metrics
        last_trial = task_result.trials[-1]
        # Grader detail comes from the last trial that actually graded something:
        # if the final attempt died in the harness it carries no grader results,
        # and reporting an empty breakdown would hide the earlier evidence.
        evaluated = task_result.evaluated_trials
        detail_trial = evaluated[-1] if evaluated else last_trial

        # Extract evaluator_results from grader_results
        evaluator_results: list[EvaluatorResult] = []
        if detail_trial.grader_results:
            for r in detail_trial.grader_results:
                if isinstance(r, EvaluatorResult):
                    evaluator_results.append(r)
                elif isinstance(r, dict):
                    evaluator_results.append(EvaluatorResult.from_dict(r))

        # Determine status. A case is ERROR only when *no* attempt produced
        # evidence — if some trials survived, the harness noise is reported via
        # error_trials and the surviving samples still decide pass/fail.
        if not task_result.evaluated_trials:
            status = TestStatus.ERROR
        elif task_result.overall_passed:
            status = TestStatus.PASSED
        else:
            status = TestStatus.FAILED

        return CaseResult(
            case_id=task_result.task_id,
            status=status,
            passed=task_result.overall_passed,
            overall_score=metrics.score_mean,
            evaluator_results=evaluator_results,
            input_data=task_result.input_data,
            output_data=detail_trial.outcome,
            duration_ms=sum(t.duration_ms for t in task_result.trials),
            error=last_trial.error,
            total_trials=task_result.total_trials,
            passed_trials=task_result.passed_trials,
            error_trials=task_result.error_trials,
            trial_metrics=metrics.to_dict(
                pass_at_k=metrics_config.pass_at_k if metrics_config else None,
                include_consistency=(
                    metrics_config.consistency if metrics_config else True
                ),
            ),
        )

    async def _run_graders(
        self,
        grader_configs: list[GraderConfig],
        context: GradeContext,
        aggregation: AggregationConfig | None = None,
    ) -> list[EvaluatorResult]:
        """Run graders with Code-First short-circuit support.

        When short_circuit is enabled:
        1. Phase 1: Run all Code Graders first (fast, cheap, deterministic)
        2. Check short-circuit condition
        3. Phase 2: Run Model Graders only if short-circuit not triggered

        Args:
            grader_configs: Grader configurations to run.
            context: Grade context with outcome and transcript.
            aggregation: Aggregation config with short-circuit settings.

        Returns:
            List of EvaluatorResult (for backward compatibility with CaseResult).
        """
        # Determine short-circuit mode
        short_circuit_mode = ShortCircuitMode.DISABLED
        if aggregation is not None:
            short_circuit_mode = aggregation.short_circuit

        # If short-circuit disabled, run all graders sequentially
        if short_circuit_mode == ShortCircuitMode.DISABLED:
            return await self._run_graders_sequential(grader_configs, context)

        # Split graders into Code and Model phases by their *actual* registered
        # type — YAML `type:` defaults to model and is easy to get wrong, and a
        # miscategorized grader silently lands in the wrong short-circuit phase.
        code_graders = []
        model_graders = []
        other_graders = []
        for c in grader_configs:
            actual = self._resolve_grader_type(c)
            if actual == GraderType.CODE.value:
                code_graders.append(c)
            elif actual == GraderType.MODEL.value:
                model_graders.append(c)
            else:
                other_graders.append(c)

        # Phase 1: Run Code Graders
        results = await self._run_graders_sequential(code_graders, context)

        # Check short-circuit condition
        should_short_circuit = self._check_short_circuit(
            results, short_circuit_mode, aggregation
        )

        if should_short_circuit:
            # Mark the Model Graders — and any others (e.g. Human) — as skipped
            for config in model_graders + other_graders:
                results.append(
                    EvaluatorResult(
                        name=config.name,
                        # Never ran, so never measured: unscored, not zero.
                        score=None,
                        passed=False,
                        weight=config.weight,
                        required=config.required,
                        gate=config.gate,
                        grader_type=config.type.value,
                        grader_scope="outcome",
                        skipped=True,
                        skip_reason="short_circuit",
                        # Informational copy; the aggregate reads the field above,
                        # never this dict (which a user grader can populate).
                        metadata={"skipped": True, "reason": "short_circuit"},
                        error="Skipped due to Code-First short-circuit",
                    )
                )
        else:
            # Phase 2: Run Model Graders
            model_results = await self._run_graders_sequential(model_graders, context)
            results.extend(model_results)

            # Run other graders (e.g., Human)
            other_results = await self._run_graders_sequential(other_graders, context)
            results.extend(other_results)

        return results

    @staticmethod
    def _resolve_grader_type(config: GraderConfig) -> str:
        """Resolve a grader config's effective type ("code"/"model"/"human").

        Prefers the registered grader class's ``grader_type`` over the YAML
        ``type:`` declaration; warns when the two disagree. Falls back to the
        declared type for graders not present in the registry.
        """
        try:
            grader_cls = get_grader(config.name)
        except Exception:
            return config.type.value

        actual = getattr(grader_cls, "grader_type", None)
        actual_value = getattr(actual, "value", None) or (
            actual if isinstance(actual, str) else None
        )
        if actual_value is None:
            return config.type.value

        # Only warn on an *explicit* wrong declaration — an omitted `type:`
        # falls back to the model default and is not a user mistake.
        explicitly_set = "type" in getattr(config, "model_fields_set", set())
        if explicitly_set and actual_value != config.type.value:
            logger.warning(
                "Grader '%s' is declared as type '%s' in the scenario but is "
                "registered as '%s' — using '%s' for phase classification.",
                config.name, config.type.value, actual_value, actual_value,
            )
        return actual_value

    def _check_short_circuit(
        self,
        code_results: list[EvaluatorResult],
        mode: ShortCircuitMode,
        aggregation: AggregationConfig | None,
    ) -> bool:
        """Check if short-circuit condition is met.

        Args:
            code_results: Results from Code Graders phase.
            mode: Short-circuit mode.
            aggregation: Aggregation config with required_graders list.

        Returns:
            True if Model Graders should be skipped.
        """
        if mode == ShortCircuitMode.DISABLED:
            return False

        if mode == ShortCircuitMode.CODE_FAIL:
            # Short-circuit if ANY Code Grader fails
            return any(not r.passed for r in code_results)

        if mode == ShortCircuitMode.CODE_PASS:
            # Short-circuit if ALL Code Graders pass (run Model only when Code fails)
            # Use case: Code rules confirm quality, skip expensive Model evaluation
            # If Code fails, it might be edge case needing Model's intelligent judgment
            if not code_results:
                return False  # No Code graders, run Model graders
            return all(r.passed for r in code_results)

        if mode == ShortCircuitMode.REQUIRED_FAIL:
            # Short-circuit if any REQUIRED Code Grader fails
            required_names = set(aggregation.required_graders) if aggregation else set()

            for result in code_results:
                # Check required by name (from aggregation config)
                if result.name in required_names and not result.passed:
                    return True
                # Check required flag on individual grader
                if result.required and not result.passed:
                    return True

            return False

        return False

    async def _run_graders_sequential(
        self,
        grader_configs: list[GraderConfig],
        context: GradeContext,
    ) -> list[EvaluatorResult]:
        """Run graders sequentially and return results.

        Args:
            grader_configs: Grader configurations to run.
            context: Grade context with outcome and transcript.

        Returns:
            List of EvaluatorResult.
        """
        results: list[EvaluatorResult] = []

        for config in grader_configs:
            try:
                grader_cls = get_grader(config.name)
                grader = grader_cls(config.config)
                grade_result = await grader.grade(context)

                # Get grader type and scope from the grader class
                grader_type = getattr(grader, "grader_type", None)
                grader_scope = getattr(grader, "grader_scope", None)

                # Handle both enum and string types for grader_type/grader_scope
                if grader_type is not None:
                    grader_type_value = grader_type.value if hasattr(grader_type, "value") else str(grader_type)
                else:
                    grader_type_value = config.type.value

                if grader_scope is not None:
                    grader_scope_value = grader_scope.value if hasattr(grader_scope, "value") else str(grader_scope)
                else:
                    grader_scope_value = "outcome"

                results.append(
                    EvaluatorResult(
                        name=config.name,
                        score=grade_result.score,
                        passed=grade_result.passed,
                        weight=config.weight,
                        required=config.required,
                        gate=config.gate,
                        grader_type=grader_type_value,
                        grader_scope=grader_scope_value,
                        grader_version=(
                            grade_result.grader_version
                            or getattr(grader, "version", "")
                        ),
                        metadata=grade_result.details,
                        failure_tags=grade_result.failure_tags,
                        error=grade_result.error,
                    )
                )
            except Exception as e:
                results.append(
                    EvaluatorResult(
                        name=config.name,
                        # The grader blew up, so it measured nothing. Recording
                        # 0.0 here would make an LLM-judge timeout look exactly
                        # like an agent that genuinely scored zero.
                        score=None,
                        passed=False,
                        weight=config.weight,
                        required=config.required,
                        gate=config.gate,
                        grader_type=config.type.value,
                        grader_scope="outcome",
                        failure_tags=["grader_error"],
                        error=str(e),
                    )
                )

        return results

    @staticmethod
    def _aggregate_results(
        results: list[EvaluatorResult],
        pass_threshold: float,
        required_graders: list[str] | None = None,
    ) -> tuple[float, bool]:
        """Aggregate evaluator results into an overall score.

        Three populations, deliberately kept apart:

        - **Skipped** (``skipped=True``, short-circuited): never ran. Excluded
          from everything, and does not hold the case back — skipping is a
          decision the aggregate already made.
        - **Unscored** (``score is None``): ran but measured nothing (crash,
          timeout). Excluded from the score denominator, because averaging in a
          zero would let a judge outage read as a bad answer. But it **does**
          fail the case: a pass cannot be certified from an evaluation that did
          not finish.
        - **Scored**: contributes to the weighted mean as usual.

        Gate graders (``gate=True``) are hard pass/fail checks: all must pass,
        and they are excluded from the weighted score so they cannot drag the
        aggregate number down.

        Args:
            results: Evaluator results.
            pass_threshold: Minimum score to pass.
            required_graders: List of grader names that must pass (from AggregationConfig).

        Returns:
            Tuple of (overall_score, passed).
        """
        if not results:
            return 0.0, False

        required_graders = required_graders or []

        # Skipped graders drop out of every check. Read the core-owned field,
        # never `metadata` — that dict is the grader's own `details` payload and
        # a user grader could otherwise silently exclude itself from scoring.
        active_results = [r for r in results if not r.skipped]

        # Separate gate graders from graded (scored) graders
        gate_results = [r for r in active_results if r.gate]
        graded_results = [r for r in active_results if not r.gate]

        # No active signal at all (e.g. every grader was short-circuited): fail
        # closed rather than vacuously pass.
        if not graded_results and not gate_results:
            return 0.0, False

        # Gate check: every gate grader must pass
        gates_passed = all(r.passed for r in gate_results)

        # An evaluation that did not finish cannot certify a pass.
        all_scored = all(r.scored for r in active_results)

        # Only actual measurements enter the mean
        scored_results = [r for r in graded_results if r.scored]

        # Calculate weighted score from graded (non-gate), scored graders only
        total_weight = sum(r.weight for r in scored_results if r.weight > 0)
        if total_weight > 0:
            score = sum(r.weighted_score for r in scored_results) / total_weight
        elif scored_results:
            # Every scored grader has zero weight: fall back to an unweighted
            # mean of their scores so they still yield a pass/fail signal, rather
            # than forcing the case to score 0 and fail (and still applying the
            # gate/required checks below).
            score = sum(r.score or 0.0 for r in scored_results) / len(scored_results)
        elif gate_results:
            # Only gate graders produced anything; their pass/fail decides.
            score = 1.0 if gates_passed else 0.0
        else:
            # Graded graders exist but none of them measured anything.
            score = 0.0

        # Threshold check (only based on graded score)
        # When only gate graders exist, skip threshold check
        threshold_passed = score >= pass_threshold if scored_results else True

        # Check required graders by name (from AggregationConfig.required_graders)
        # Only check non-gate, non-skipped graders
        required_by_name_passed = True
        for grader_name in required_graders:
            matching = [r for r in graded_results if r.name == grader_name]
            if matching and not matching[0].passed:
                required_by_name_passed = False
                break

        # Check required flag on individual graders (from GraderConfig.required)
        # Only check non-gate, non-skipped graders
        required_by_flag_passed = True
        for r in graded_results:
            if r.required and not r.passed:
                required_by_flag_passed = False
                break

        # Overall pass: gates AND threshold AND required graders AND a
        # complete evaluation (an unscored grader means we don't actually know)
        passed = (
            gates_passed
            and threshold_passed
            and required_by_name_passed
            and required_by_flag_passed
            and all_scored
        )
        return score, passed

    def get_results(self) -> list[EvalResult]:
        """Get all accumulated results."""
        return self._results

    def clear_results(self) -> None:
        """Clear accumulated results."""
        self._results.clear()
