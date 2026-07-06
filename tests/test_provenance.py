"""Tests for audit provenance (protocol 1.4): run_id / config_hash / grader_version.

A minimal auditable trace record ties every transcript back to the run that
produced it (run_id), the scenario configuration it executed under
(config_hash), and the verifier version that scored it (grader_version).
"""

from __future__ import annotations

import json

import pytest

from compass.adapters.registry import register_adapter
from compass.core.checkpoint import scenario_fingerprint
from compass.core.result import EvalResult, EvaluatorResult
from compass.core.runner import Compass
from compass.core.scenario import (
    AgentConfig,
    AggregationConfig,
    GraderConfig,
    InputConfig,
    Scenario,
    TestCase,
)
from compass.core.transcript import Transcript
from compass.graders.base import Grader, GradeResult

# ===================================================================
# Transcript-level provenance
# ===================================================================


class TestTranscriptProvenance:
    def test_defaults_empty(self):
        t = Transcript(task_id="t1", trial_id="tr1")
        assert t.run_id == ""
        assert t.config_hash == ""

    def test_to_dict_includes_provenance(self):
        t = Transcript(task_id="t1", trial_id="tr1", run_id="run123", config_hash="abc")
        d = t.to_dict()
        assert d["run_id"] == "run123"
        assert d["config_hash"] == "abc"

    def test_json_save_load_round_trip(self, tmp_path):
        t = Transcript(task_id="t1", trial_id="tr1", run_id="run123", config_hash="abc")
        path = t.save(tmp_path / "trace.json")
        loaded = Transcript.load(path)
        assert loaded.run_id == "run123"
        assert loaded.config_hash == "abc"

    def test_legacy_json_without_provenance_loads(self, tmp_path):
        """Pre-1.4 transcripts (no run_id/config_hash) load with empty strings."""
        t = Transcript(task_id="t1", trial_id="tr1")
        path = t.save(tmp_path / "trace.json")
        data = json.loads(path.read_text())
        del data["run_id"]
        del data["config_hash"]
        path.write_text(json.dumps(data))
        loaded = Transcript.load(path)
        assert loaded.run_id == ""
        assert loaded.config_hash == ""

    def test_jsonl_round_trip(self):
        t = Transcript(task_id="t1", trial_id="tr1", run_id="run123", config_hash="abc")
        restored = Transcript.from_jsonl(t.to_jsonl())
        assert restored.run_id == "run123"
        assert restored.config_hash == "abc"

    def test_jsonl_omits_empty_provenance(self):
        t = Transcript(task_id="t1", trial_id="tr1")
        started = json.loads(t.to_jsonl().split("\n")[0])
        assert started["type"] == "transcript.started"
        assert "run_id" not in started
        assert "config_hash" not in started


# ===================================================================
# Verifier version
# ===================================================================


class TestGraderVersion:
    def test_grader_has_default_version(self):
        assert Grader.version == "1.0"

    def test_grade_result_carries_version(self):
        r = GradeResult(name="g", grader_version="2.1")
        assert r.to_dict()["grader_version"] == "2.1"

    def test_evaluator_result_carries_version(self):
        r = EvaluatorResult(name="g", score=1.0, passed=True, grader_version="2.1")
        assert r.to_dict()["grader_version"] == "2.1"


# ===================================================================
# End-to-end: runner stamps provenance
# ===================================================================


_registered = False


def _ensure_test_adapter():
    global _registered
    if _registered:
        return

    from compass.adapters.coding import CodingAdapter

    @register_adapter("_test_provenance_runner")
    class _TestAdapter(CodingAdapter):
        name = "_test_provenance_runner"

    _registered = True


def _make_scenario() -> Scenario:
    case = TestCase(
        id="case_1",
        input=InputConfig(
            prompt="test",
            params={"files": [{"path": "main.py", "content": "print('hi')"}]},
        ),
        graders=[GraderConfig(name="exit_code_check", config={})],
        aggregation=AggregationConfig(pass_threshold=0.5),
    )
    return Scenario(
        name="provenance_scenario",
        agent=AgentConfig(
            adapter="_test_provenance_runner",
            config={"run_command": "python main.py"},
        ),
        cases=[case],
    )


class TestRunnerProvenance:
    @pytest.mark.asyncio
    async def test_run_stamps_provenance_everywhere(self, tmp_path):
        _ensure_test_adapter()
        scenario = _make_scenario()
        runner = Compass()
        result = await runner.run(scenario, trace_dir=tmp_path)

        assert isinstance(result, EvalResult)

        # EvalResult carries run identity + config hash
        assert result.run_id != ""
        assert result.config_hash == scenario_fingerprint(scenario)
        assert result.to_dict()["run_id"] == result.run_id

        # Saved trace ties back to the same run and config
        trace = json.loads((tmp_path / "case_1.json").read_text())
        assert trace["run_id"] == result.run_id
        assert trace["config_hash"] == result.config_hash

        # Every grade carries the verifier version that produced it
        case = result.case_results[0]
        assert case.evaluator_results, "expected grader results"
        for er in case.evaluator_results:
            assert er.grader_version != ""

        # ... and the version is persisted in the trace's grading record
        for gr in trace["grading"]["results"]:
            assert gr["grader_version"] != ""
