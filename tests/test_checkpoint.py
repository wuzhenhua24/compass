"""Tests for checkpoint.py - Resumable evaluation support."""

from __future__ import annotations

import pytest

from compass.core.checkpoint import (
    CheckpointStore,
    RunCheckpoint,
    _case_result_to_dict,
    _dict_to_case_result,
    find_checkpoints,
    scenario_fingerprint,
)
from compass.core.result import CaseResult, EvaluatorResult, TestStatus

# ===================================================================
# Helper factories
# ===================================================================


def _make_case_result(
    case_id: str = "case_1",
    passed: bool = True,
    score: float = 0.9,
    status: TestStatus = TestStatus.PASSED,
) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        status=status,
        passed=passed,
        overall_score=score,
        evaluator_results=[
            EvaluatorResult(
                name="test_grader",
                score=score,
                passed=passed,
                weight=1.0,
                grader_type="code",
                grader_scope="outcome",
                metadata={"key": "value"},
                failure_tags=["tag1"] if not passed else [],
            ),
        ],
        input_data={"prompt": "test prompt"},
        output_data={"result": "ok"},
        duration_ms=150.0,
        total_trials=1,
        passed_trials=1 if passed else 0,
    )


def _make_scenario():
    """Create a minimal Scenario for fingerprinting tests."""
    from compass.core.scenario import (
        AgentConfig,
        GraderConfig,
        InputConfig,
        Scenario,
        TestCase,
    )

    return Scenario(
        name="test_scenario",
        agent=AgentConfig(adapter="image", endpoint="http://localhost:8000"),
        default_graders=[
            GraderConfig(name="semantic_match", config={"threshold": 0.25}),
        ],
        cases=[
            TestCase(
                id="case_1",
                input=InputConfig(prompt="Generate a cat"),
            ),
            TestCase(
                id="case_2",
                input=InputConfig(prompt="Generate a dog"),
            ),
        ],
    )


# ===================================================================
# Serialization round-trip
# ===================================================================


class TestSerialization:
    def test_case_result_round_trip(self):
        """CaseResult survives serialize → deserialize."""
        original = _make_case_result(case_id="rt_1", passed=True, score=0.85)
        data = _case_result_to_dict(original)
        restored = _dict_to_case_result(data)

        assert restored.case_id == original.case_id
        assert restored.passed == original.passed
        assert restored.overall_score == original.overall_score
        assert restored.status == original.status
        assert restored.duration_ms == original.duration_ms
        assert restored.total_trials == original.total_trials
        assert restored.passed_trials == original.passed_trials
        assert len(restored.evaluator_results) == 1
        er = restored.evaluator_results[0]
        assert er.name == "test_grader"
        assert er.score == 0.85
        assert er.metadata == {"key": "value"}

    def test_case_result_round_trip_preserves_tags_and_category(self):
        """Regression: tags/category must survive the checkpoint round-trip so a
        resumed run keeps its per-category / per-tag analysis intact."""
        original = _make_case_result(case_id="rt_tags", passed=True, score=0.9)
        original.tags = ["safety", "smoke"]
        original.category = "backend"

        restored = _dict_to_case_result(_case_result_to_dict(original))

        assert restored.tags == ["safety", "smoke"]
        assert restored.category == "backend"
        # Timestamp is preserved too (not reset to "now").
        assert restored.timestamp == original.timestamp

    def test_case_result_round_trip_preserves_grader_version(self):
        """Regression: grader_version must survive the checkpoint round-trip so
        resumed runs keep the verifier-version audit trail intact."""
        original = _make_case_result(case_id="rt_ver", passed=True, score=0.9)
        original.evaluator_results[0].grader_version = "2.1"

        restored = _dict_to_case_result(_case_result_to_dict(original))

        assert restored.evaluator_results[0].grader_version == "2.1"

    def test_case_result_round_trip_with_error(self):
        """CaseResult with error survives round-trip."""
        original = CaseResult(
            case_id="err_1",
            status=TestStatus.ERROR,
            passed=False,
            overall_score=0.0,
            error="Something broke",
            duration_ms=50.0,
        )
        data = _case_result_to_dict(original)
        restored = _dict_to_case_result(data)

        assert restored.status == TestStatus.ERROR
        assert restored.error == "Something broke"
        assert restored.evaluator_results == []

    def test_case_result_round_trip_with_failure_tags(self):
        """EvaluatorResult failure_tags survive round-trip."""
        original = _make_case_result(passed=False, score=0.3, status=TestStatus.FAILED)
        data = _case_result_to_dict(original)
        restored = _dict_to_case_result(data)

        er = restored.evaluator_results[0]
        assert er.failure_tags == ["tag1"]
        assert er.passed is False


# ===================================================================
# scenario_fingerprint
# ===================================================================


class TestScenarioFingerprint:
    def test_same_scenario_same_fingerprint(self):
        """Identical scenarios produce the same fingerprint."""
        s1 = _make_scenario()
        s2 = _make_scenario()
        assert scenario_fingerprint(s1) == scenario_fingerprint(s2)

    def test_different_prompt_different_fingerprint(self):
        """Changing a case prompt changes the fingerprint."""
        from compass.core.scenario import (
            AgentConfig,
            GraderConfig,
            InputConfig,
            Scenario,
            TestCase,
        )

        s1 = _make_scenario()
        s2 = Scenario(
            name="test_scenario",
            agent=AgentConfig(adapter="image", endpoint="http://localhost:8000"),
            default_graders=[
                GraderConfig(name="semantic_match", config={"threshold": 0.25}),
            ],
            cases=[
                TestCase(id="case_1", input=InputConfig(prompt="Generate a cat")),
                TestCase(id="case_2", input=InputConfig(prompt="Generate a BIRD")),
            ],
        )
        assert scenario_fingerprint(s1) != scenario_fingerprint(s2)

    def test_cosmetic_changes_same_fingerprint(self):
        """Changing description/tags does NOT change the fingerprint."""
        from compass.core.scenario import (
            AgentConfig,
            GraderConfig,
            InputConfig,
            Scenario,
            TestCase,
        )

        s1 = _make_scenario()
        s2 = Scenario(
            name="RENAMED scenario",
            description="Added description",
            agent=AgentConfig(adapter="image", endpoint="http://localhost:8000"),
            default_graders=[
                GraderConfig(name="semantic_match", config={"threshold": 0.25}),
            ],
            cases=[
                TestCase(
                    id="case_1",
                    description="Added case description",
                    input=InputConfig(prompt="Generate a cat"),
                    tags=["new_tag"],
                ),
                TestCase(
                    id="case_2",
                    input=InputConfig(prompt="Generate a dog"),
                ),
            ],
            tags=["scenario_tag"],
        )
        assert scenario_fingerprint(s1) == scenario_fingerprint(s2)

    def test_changing_case_aggregation_changes_fingerprint(self):
        """Regression: editing a case's aggregation (e.g. pass_threshold) must
        invalidate the checkpoint so a resume does not reuse stale verdicts."""
        s1 = _make_scenario()
        s2 = _make_scenario()
        s2.cases[0].aggregation.pass_threshold = 0.95
        assert scenario_fingerprint(s1) != scenario_fingerprint(s2)

    def test_changing_short_circuit_changes_fingerprint(self):
        from compass.core.scenario import ShortCircuitMode

        s1 = _make_scenario()
        s2 = _make_scenario()
        s2.cases[0].aggregation.short_circuit = ShortCircuitMode.CODE_FAIL
        assert scenario_fingerprint(s1) != scenario_fingerprint(s2)

    def test_changing_trials_changes_fingerprint(self):
        """Regression: changing trial count changes pass@k/pass^k, so it must
        invalidate the checkpoint."""
        s1 = _make_scenario()
        s2 = _make_scenario()
        s2.cases[0].trials = 5
        assert scenario_fingerprint(s1) != scenario_fingerprint(s2)

    def test_changing_defaults_trials_changes_fingerprint(self):
        s1 = _make_scenario()
        s2 = _make_scenario()
        s2.defaults.trials = 3
        assert scenario_fingerprint(s1) != scenario_fingerprint(s2)


# ===================================================================
# RunCheckpoint
# ===================================================================


class TestRunCheckpoint:
    def test_remaining_case_ids(self):
        cp = RunCheckpoint(
            run_id="r1",
            scenario_fingerprint="fp",
            scenario_name="test",
            total_cases=["a", "b", "c", "d"],
            completed_cases={"a": {"file": "a.json"}, "c": {"file": "c.json"}},
        )
        assert cp.remaining_case_ids == ["b", "d"]

    def test_progress(self):
        cp = RunCheckpoint(
            run_id="r1",
            scenario_fingerprint="fp",
            scenario_name="test",
            total_cases=["a", "b", "c"],
            completed_cases={"a": {"file": "a.json"}},
        )
        assert cp.progress == "1/3"


# ===================================================================
# CheckpointStore
# ===================================================================


class TestCheckpointStore:
    def test_create_and_load(self, tmp_path):
        """Create a checkpoint, then load it back."""
        store = CheckpointStore.create(
            tmp_path, "run_1", "fp_abc", "My Scenario", ["c1", "c2", "c3"],
        )
        cp = store.get_checkpoint()
        assert cp.run_id == "run_1"
        assert cp.scenario_fingerprint == "fp_abc"
        assert cp.scenario_name == "My Scenario"
        assert cp.total_cases == ["c1", "c2", "c3"]
        assert cp.completed_cases == {}
        assert cp.status == "in_progress"

        # Load from same directory
        loaded = CheckpointStore.load(tmp_path)
        assert loaded is not None
        assert loaded.get_checkpoint().run_id == "run_1"

    def test_load_returns_none_when_missing(self, tmp_path):
        """Load returns None when no checkpoint exists."""
        assert CheckpointStore.load(tmp_path) is None

    def test_validate_scenario(self, tmp_path):
        """validate_scenario checks fingerprint match."""
        store = CheckpointStore.create(
            tmp_path, "r1", "fp_123", "test", ["c1"],
        )
        assert store.validate_scenario("fp_123") is True
        assert store.validate_scenario("fp_999") is False

    def test_save_and_load_case_result(self, tmp_path):
        """Save a case result and load it back."""
        store = CheckpointStore.create(
            tmp_path, "r1", "fp", "test", ["c1", "c2"],
        )

        result = _make_case_result(case_id="c1", passed=True, score=0.9)
        store.save_case_result("c1", result)

        # Check index updated
        cp = store.get_checkpoint()
        assert "c1" in cp.completed_cases
        assert cp.remaining_case_ids == ["c2"]

        # Load back
        loaded = store.load_completed_results()
        assert "c1" in loaded
        assert loaded["c1"].case_id == "c1"
        assert loaded["c1"].passed is True
        assert loaded["c1"].overall_score == 0.9

    def test_save_multiple_case_results(self, tmp_path):
        """Save multiple case results incrementally."""
        store = CheckpointStore.create(
            tmp_path, "r1", "fp", "test", ["c1", "c2", "c3"],
        )

        store.save_case_result("c1", _make_case_result("c1", True, 1.0))
        store.save_case_result("c3", _make_case_result("c3", False, 0.3, TestStatus.FAILED))

        cp = store.get_checkpoint()
        assert cp.remaining_case_ids == ["c2"]

        loaded = store.load_completed_results()
        assert len(loaded) == 2
        assert loaded["c1"].passed is True
        assert loaded["c3"].passed is False

    def test_mark_completed(self, tmp_path):
        """mark_completed updates status."""
        store = CheckpointStore.create(
            tmp_path, "r1", "fp", "test", ["c1"],
        )
        store.save_case_result("c1", _make_case_result("c1"))
        store.mark_completed()

        cp = store.get_checkpoint()
        assert cp.status == "completed"

    def test_missing_case_file_skipped(self, tmp_path):
        """load_completed_results skips cases with missing files."""
        store = CheckpointStore.create(
            tmp_path, "r1", "fp", "test", ["c1"],
        )
        store.save_case_result("c1", _make_case_result("c1"))

        # Delete the case file to simulate corruption
        cases_dir = tmp_path / ".checkpoint_cases"
        for f in cases_dir.iterdir():
            f.unlink()

        loaded = store.load_completed_results()
        assert len(loaded) == 0


# ===================================================================
# Integration: CheckpointStore with runner resume flow
# ===================================================================


class TestCheckpointResumeFlow:
    """Tests simulating the resume flow without running actual agents."""

    def test_full_resume_flow(self, tmp_path):
        """Simulate: run 2/3 cases → interrupt → resume → run remaining 1."""
        # --- First run: complete 2 cases ---
        store = CheckpointStore.create(
            tmp_path, "run_abc", "fp_xyz", "scenario", ["c1", "c2", "c3"],
        )
        store.save_case_result("c1", _make_case_result("c1", True, 1.0))
        store.save_case_result("c2", _make_case_result("c2", False, 0.4, TestStatus.FAILED))
        # c3 not saved (simulates interrupt)

        # --- Resume ---
        store2 = CheckpointStore.load(tmp_path)
        assert store2 is not None
        assert store2.validate_scenario("fp_xyz")

        cp = store2.get_checkpoint()
        assert cp.run_id == "run_abc"
        assert cp.remaining_case_ids == ["c3"]

        completed = store2.load_completed_results()
        assert len(completed) == 2
        assert completed["c1"].passed is True
        assert completed["c2"].passed is False

        # Run remaining case
        store2.save_case_result("c3", _make_case_result("c3", True, 0.8))
        store2.mark_completed()

        cp = store2.get_checkpoint()
        assert cp.remaining_case_ids == []
        assert cp.status == "completed"

    def test_resume_with_changed_scenario_starts_fresh(self, tmp_path):
        """If scenario fingerprint changed, old checkpoint is invalid."""
        CheckpointStore.create(
            tmp_path, "run_old", "fp_OLD", "scenario", ["c1"],
        )

        store = CheckpointStore.load(tmp_path)
        assert store is not None
        assert store.validate_scenario("fp_OLD") is True
        assert store.validate_scenario("fp_NEW") is False

    def test_resume_no_checkpoint_creates_new(self, tmp_path):
        """Resume on empty dir creates a fresh checkpoint."""
        store = CheckpointStore.load(tmp_path)
        assert store is None

        # Caller would create fresh
        store = CheckpointStore.create(
            tmp_path, "run_new", "fp", "test", ["c1"],
        )
        assert store.get_checkpoint().status == "in_progress"


# ===================================================================
# Integration: Runner with resume
# ===================================================================


class TestRunnerResume:
    """Integration tests for Compass.run() with resume=True."""

    @pytest.fixture(autouse=True)
    def setup_adapter(self):
        """Register test adapter."""
        from compass.adapters.coding import CodingAdapter
        from compass.adapters.registry import register_adapter, unregister_adapter

        @register_adapter("_test_checkpoint_adapter")
        class _TestCPAdapter(CodingAdapter):
            name = "_test_checkpoint_adapter"

        yield
        try:
            unregister_adapter("_test_checkpoint_adapter")
        except Exception:
            pass

    def _make_scenario(self, num_cases: int = 3):
        from compass.core.scenario import (
            AgentConfig,
            AggregationConfig,
            DefaultsConfig,
            GraderConfig,
            InputConfig,
            Scenario,
            TestCase,
        )

        cases = [
            TestCase(
                id=f"case_{i}",
                input=InputConfig(
                    prompt=f"test prompt {i}",
                    params={
                        "files": [{"path": "main.py", "content": "print('hello')"}],
                    },
                ),
                graders=[GraderConfig(name="exit_code_check", config={})],
                aggregation=AggregationConfig(pass_threshold=0.5),
            )
            for i in range(1, num_cases + 1)
        ]
        return Scenario(
            name="checkpoint_test",
            agent=AgentConfig(
                adapter="_test_checkpoint_adapter",
                config={"run_command": "python main.py"},
            ),
            cases=cases,
            defaults=DefaultsConfig(trials=1),
        )

    @pytest.mark.asyncio
    async def test_run_creates_checkpoint(self, tmp_path):
        """Normal run with trace_dir creates a checkpoint."""
        from compass.core.runner import Compass

        scenario = self._make_scenario(num_cases=2)
        runner = Compass()
        result = await runner.run(scenario, trace_dir=str(tmp_path))

        assert result.total_cases == 2
        # Checkpoint should exist
        store = CheckpointStore.load(tmp_path)
        assert store is not None
        cp = store.get_checkpoint()
        assert cp.status == "completed"
        assert len(cp.completed_cases) == 2

    @pytest.mark.asyncio
    async def test_resume_skips_completed_cases(self, tmp_path):
        """Resume skips already-completed cases from checkpoint."""
        from compass.core.runner import Compass

        scenario = self._make_scenario(num_cases=3)
        fp = scenario_fingerprint(scenario)

        # Pre-populate checkpoint with 2 completed cases
        store = CheckpointStore.create(
            tmp_path,
            "pre_run",
            fp,
            scenario.name,
            ["case_1", "case_2", "case_3"],
        )
        store.save_case_result(
            "case_1", _make_case_result("case_1", True, 1.0)
        )
        store.save_case_result(
            "case_2", _make_case_result("case_2", True, 0.8)
        )

        # Resume — should only run case_3
        runner = Compass()
        result = await runner.run(
            scenario, trace_dir=str(tmp_path), resume=True
        )

        assert result.total_cases == 3
        # First two should have scores from checkpoint
        assert result.case_results[0].case_id == "case_1"
        assert result.case_results[0].overall_score == 1.0
        assert result.case_results[1].case_id == "case_2"
        assert result.case_results[1].overall_score == 0.8
        # Third was actually run
        assert result.case_results[2].case_id == "case_3"

    @pytest.mark.asyncio
    async def test_resume_all_completed(self, tmp_path):
        """Resume when all cases already completed returns cached results."""
        from compass.core.runner import Compass

        scenario = self._make_scenario(num_cases=2)
        fp = scenario_fingerprint(scenario)

        store = CheckpointStore.create(
            tmp_path, "done_run", fp, scenario.name, ["case_1", "case_2"],
        )
        store.save_case_result(
            "case_1", _make_case_result("case_1", True, 1.0)
        )
        store.save_case_result(
            "case_2", _make_case_result("case_2", True, 0.9)
        )

        runner = Compass()
        result = await runner.run(
            scenario, trace_dir=str(tmp_path), resume=True
        )

        assert result.total_cases == 2
        assert result.passed_cases == 2
        # Both from checkpoint
        assert result.case_results[0].overall_score == 1.0
        assert result.case_results[1].overall_score == 0.9

    @pytest.mark.asyncio
    async def test_resume_ignores_stale_checkpoint(self, tmp_path):
        """Resume with changed scenario starts fresh."""
        from compass.core.runner import Compass

        # Create checkpoint with old fingerprint
        CheckpointStore.create(
            tmp_path, "old_run", "stale_fingerprint", "old", ["c1"],
        )

        scenario = self._make_scenario(num_cases=2)
        runner = Compass()
        result = await runner.run(
            scenario, trace_dir=str(tmp_path), resume=True
        )

        # Should have run all cases fresh
        assert result.total_cases == 2

    @pytest.mark.asyncio
    async def test_no_checkpoint_without_trace_dir(self):
        """Run without trace_dir doesn't create checkpoints."""
        from compass.core.runner import Compass

        scenario = self._make_scenario(num_cases=1)
        runner = Compass()
        result = await runner.run(scenario)

        assert result.total_cases == 1
        assert result.passed_cases == 1


# ===================================================================
# CheckpointStore.clean()
# ===================================================================


class TestCheckpointClean:
    def test_clean_removes_all_files(self, tmp_path):
        """clean() removes checkpoint index and case files."""
        store = CheckpointStore.create(
            tmp_path, "r1", "fp", "test", ["c1", "c2"],
        )
        store.save_case_result("c1", _make_case_result("c1"))
        store.save_case_result("c2", _make_case_result("c2"))

        deleted = store.clean()
        # 2 case files + 1 checkpoint.json = 3
        assert deleted == 3
        assert not (tmp_path / "checkpoint.json").exists()
        assert not (tmp_path / ".checkpoint_cases").exists()

    def test_clean_empty_checkpoint(self, tmp_path):
        """clean() works on checkpoint with no completed cases."""
        store = CheckpointStore.create(
            tmp_path, "r1", "fp", "test", ["c1"],
        )
        deleted = store.clean()
        # 0 case files + 1 checkpoint.json = 1
        assert deleted == 1
        assert not (tmp_path / "checkpoint.json").exists()

    def test_clean_idempotent(self, tmp_path):
        """clean() on already-cleaned dir doesn't error."""
        store = CheckpointStore.create(
            tmp_path, "r1", "fp", "test", ["c1"],
        )
        store.clean()
        # Second clean — files already gone, but should not raise
        # (load returns None because checkpoint.json is gone)
        assert CheckpointStore.load(tmp_path) is None

    def test_clean_preserves_other_files(self, tmp_path):
        """clean() only removes checkpoint files, not trace files."""
        store = CheckpointStore.create(
            tmp_path, "r1", "fp", "test", ["c1"],
        )
        store.save_case_result("c1", _make_case_result("c1"))

        # Create a non-checkpoint file in the same directory
        other_file = tmp_path / "c1.json"
        other_file.write_text('{"trace": true}')

        store.clean()
        # The trace file should still exist
        assert other_file.exists()


# ===================================================================
# find_checkpoints
# ===================================================================


class TestFindCheckpoints:
    def test_find_in_flat_dir(self, tmp_path):
        """find_checkpoints discovers checkpoint in the given directory."""
        CheckpointStore.create(tmp_path, "r1", "fp", "test", ["c1"])

        stores = find_checkpoints(tmp_path)
        assert len(stores) == 1
        assert stores[0].get_checkpoint().run_id == "r1"

    def test_find_recursive(self, tmp_path):
        """find_checkpoints discovers checkpoints in subdirectories."""
        dir_a = tmp_path / "run_a"
        dir_b = tmp_path / "run_b"
        CheckpointStore.create(dir_a, "r_a", "fp_a", "scenario_a", ["c1"])
        CheckpointStore.create(dir_b, "r_b", "fp_b", "scenario_b", ["c1", "c2"])

        stores = find_checkpoints(tmp_path)
        assert len(stores) == 2
        run_ids = {s.get_checkpoint().run_id for s in stores}
        assert run_ids == {"r_a", "r_b"}

    def test_find_no_recursive(self, tmp_path):
        """find_checkpoints with recursive=False only looks in the given dir."""
        sub = tmp_path / "sub"
        CheckpointStore.create(sub, "r_sub", "fp", "test", ["c1"])

        # Non-recursive from parent: should not find it
        stores = find_checkpoints(tmp_path, recursive=False)
        assert len(stores) == 0

        # Non-recursive from the actual dir: should find it
        stores = find_checkpoints(sub, recursive=False)
        assert len(stores) == 1

    def test_find_empty_dir(self, tmp_path):
        """find_checkpoints returns empty list on dir without checkpoints."""
        stores = find_checkpoints(tmp_path)
        assert stores == []

    def test_find_nested_deep(self, tmp_path):
        """find_checkpoints discovers deeply nested checkpoints."""
        deep = tmp_path / "a" / "b" / "c"
        CheckpointStore.create(deep, "r_deep", "fp", "deep_scenario", ["c1"])

        stores = find_checkpoints(tmp_path)
        assert len(stores) == 1
        assert stores[0].get_checkpoint().scenario_name == "deep_scenario"


# ===================================================================
# CLI: compass checkpoint list / clean
# ===================================================================


class TestCheckpointCLI:
    """Test the CLI subcommands using click's CliRunner."""

    @pytest.fixture
    def runner(self):
        from click.testing import CliRunner
        # Use a wide terminal so Rich tables don't truncate cell contents
        return CliRunner(env={"COLUMNS": "200"})

    @pytest.fixture
    def cli(self):
        from compass.cli.main import cli
        return cli

    def test_checkpoint_list_empty(self, runner, cli, tmp_path):
        """checkpoint list on empty dir shows 'no checkpoints'."""
        result = runner.invoke(cli, ["checkpoint", "list", str(tmp_path)])
        assert result.exit_code == 0
        assert "No checkpoints found" in result.output

    def test_checkpoint_list_shows_checkpoints(self, runner, cli, tmp_path):
        """checkpoint list shows discovered checkpoints."""
        dir_a = tmp_path / "run_a"
        CheckpointStore.create(dir_a, "r_aaa", "fp_a", "ScenarioAlpha", ["c1", "c2"])
        store_a = CheckpointStore.load(dir_a)
        store_a.save_case_result("c1", _make_case_result("c1"))
        store_a.mark_completed()

        dir_b = tmp_path / "run_b"
        CheckpointStore.create(dir_b, "r_bbb", "fp_b", "ScenarioBeta", ["c1", "c2", "c3"])

        result = runner.invoke(cli, ["checkpoint", "list", str(tmp_path)])
        assert result.exit_code == 0
        assert "r_aaa" in result.output
        assert "ScenarioAlpha" in result.output
        assert "completed" in result.output
        assert "r_bbb" in result.output
        assert "ScenarioBeta" in result.output
        assert "in_progr" in result.output  # Rich may truncate "in_progress"

    def test_checkpoint_clean_completed_only(self, runner, cli, tmp_path):
        """checkpoint clean without --all only removes completed checkpoints."""
        dir_done = tmp_path / "done"
        store_done = CheckpointStore.create(dir_done, "r_done", "fp", "done", ["c1"])
        store_done.save_case_result("c1", _make_case_result("c1"))
        store_done.mark_completed()

        dir_wip = tmp_path / "wip"
        CheckpointStore.create(dir_wip, "r_wip", "fp", "wip", ["c1", "c2"])

        result = runner.invoke(
            cli, ["checkpoint", "clean", str(tmp_path), "--force"]
        )
        assert result.exit_code == 0
        assert "1 checkpoint(s)" in result.output

        # Completed checkpoint should be gone
        assert CheckpointStore.load(dir_done) is None
        # In-progress checkpoint should still be there
        assert CheckpointStore.load(dir_wip) is not None

    def test_checkpoint_clean_all(self, runner, cli, tmp_path):
        """checkpoint clean --all removes both completed and in-progress."""
        dir_done = tmp_path / "done"
        store_done = CheckpointStore.create(dir_done, "r1", "fp", "done", ["c1"])
        store_done.mark_completed()

        dir_wip = tmp_path / "wip"
        CheckpointStore.create(dir_wip, "r2", "fp", "wip", ["c1"])

        result = runner.invoke(
            cli, ["checkpoint", "clean", str(tmp_path), "--all", "--force"]
        )
        assert result.exit_code == 0
        assert "2 checkpoint(s)" in result.output

        assert CheckpointStore.load(dir_done) is None
        assert CheckpointStore.load(dir_wip) is None

    def test_checkpoint_clean_nothing_to_clean(self, runner, cli, tmp_path):
        """checkpoint clean when no checkpoints match shows message."""
        # Only in-progress, no --all
        dir_wip = tmp_path / "wip"
        CheckpointStore.create(dir_wip, "r1", "fp", "wip", ["c1"])

        result = runner.invoke(
            cli, ["checkpoint", "clean", str(tmp_path), "--force"]
        )
        assert result.exit_code == 0
        assert "No checkpoints to clean" in result.output

    def test_checkpoint_clean_empty_dir(self, runner, cli, tmp_path):
        """checkpoint clean on empty dir shows 'no checkpoints'."""
        result = runner.invoke(
            cli, ["checkpoint", "clean", str(tmp_path), "--force"]
        )
        assert result.exit_code == 0
        assert "No checkpoints found" in result.output
