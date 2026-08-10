"""Tests for crash-safe writes and the completion-marker ordering.

A truncated JSON file is worse than a missing one: a missing checkpoint means
"start fresh", a corrupt one means the resume path reads garbage. Two defences:

1. every record is written all-at-once (temp file + atomic rename)
2. reads tolerate damage — one bad file costs that record, not the whole resume

Plus the ordering rule: a trace file is written *after* the artifacts it
references, so its presence means the evidence it points at is really there.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from compass.adapters.base import Adapter, AgentInput, AgentOutput
from compass.adapters.registry import register_adapter, unregister_adapter
from compass.core.artifact_store import ArtifactStore
from compass.core.artifacts import ImageArtifact
from compass.core.checkpoint import (
    _CHECKPOINT_FILE,
    CheckpointStore,
    scenario_fingerprint,
)
from compass.core.fileio import (
    atomic_path,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
)
from compass.core.regrade import load_trace
from compass.core.result import CaseResult, TestStatus
from compass.core.runner import Compass
from compass.core.scenario import (
    AgentConfig,
    AggregationConfig,
    DefaultsConfig,
    GraderConfig,
    InputConfig,
    Scenario,
    TestCase,
)
from compass.core.trial import TrialResult

# ===================================================================
# Atomic write primitives
# ===================================================================


class TestAtomicWrites:
    def test_write_text_creates_the_file(self, tmp_path):
        path = atomic_write_text(tmp_path / "a.txt", "hello")
        assert path.read_text() == "hello"

    def test_parent_directories_are_created(self, tmp_path):
        atomic_write_text(tmp_path / "deep" / "nested" / "a.txt", "hi")
        assert (tmp_path / "deep" / "nested" / "a.txt").read_text() == "hi"

    def test_no_temp_files_survive_a_successful_write(self, tmp_path):
        atomic_write_json(tmp_path / "a.json", {"k": 1})
        assert [p.name for p in tmp_path.iterdir()] == ["a.json"]

    def test_a_failed_rename_leaves_the_original_intact(self, tmp_path, monkeypatch):
        """The whole point: readers see the old complete file, never a stump."""
        path = tmp_path / "a.json"
        atomic_write_json(path, {"version": 1})

        def boom(src, dst):
            raise OSError("disk full")

        monkeypatch.setattr(os, "replace", boom)
        with pytest.raises(OSError):
            atomic_write_json(path, {"version": 2})

        assert json.loads(path.read_text()) == {"version": 1}
        # ...and no debris left behind
        assert [p.name for p in tmp_path.iterdir()] == ["a.json"]

    def test_an_unserializable_payload_does_not_touch_the_target(self, tmp_path):
        """Serialize before opening anything: a bad payload is a no-op."""
        path = tmp_path / "a.json"
        atomic_write_json(path, {"version": 1})

        class Hostile:
            def __str__(self):
                raise RuntimeError("nope")

        with pytest.raises(RuntimeError):
            atomic_write_json(path, {"bad": Hostile()})

        assert json.loads(path.read_text()) == {"version": 1}
        assert [p.name for p in tmp_path.iterdir()] == ["a.json"]

    def test_atomic_path_renames_on_success(self, tmp_path):
        target = tmp_path / "img.png"
        with atomic_path(target) as tmp:
            tmp.write_bytes(b"\x89PNG")
            assert not target.exists()  # not visible until the block ends
        assert target.read_bytes() == b"\x89PNG"

    def test_atomic_path_cleans_up_on_failure(self, tmp_path):
        target = tmp_path / "img.png"
        with pytest.raises(ValueError):
            with atomic_path(target) as tmp:
                tmp.write_bytes(b"partial")
                raise ValueError("render failed")

        assert not target.exists()
        assert list(tmp_path.iterdir()) == []

    def test_write_bytes_round_trips(self, tmp_path):
        atomic_write_bytes(tmp_path / "b.bin", b"\x00\xff")
        assert (tmp_path / "b.bin").read_bytes() == b"\x00\xff"


# ===================================================================
# Checkpoint durability
# ===================================================================


def _store(tmp_path: Path) -> CheckpointStore:
    return CheckpointStore.create(tmp_path, "run1", "fp1", "scn", ["c1", "c2"])


def _case(case_id: str) -> CaseResult:
    return CaseResult(
        case_id=case_id, status=TestStatus.PASSED, passed=True, overall_score=1.0
    )


class TestCheckpointDurability:
    def test_index_and_records_are_written_atomically(self, tmp_path):
        store = _store(tmp_path)
        store.save_case_result("c1", _case("c1"))
        store.save_trial_result(
            "c2", TrialResult(trial_id="t", trial_number=1, passed=True, score=1.0)
        )

        # No stray temp files anywhere under the checkpoint dir
        strays = [p.name for p in tmp_path.rglob("*") if p.name.startswith(".")
                  and p.is_file() and p.suffix not in (".json",)]
        assert strays == []
        assert store.load_completed_results()["c1"].passed is True

    def test_a_corrupt_index_starts_fresh_instead_of_crashing(self, tmp_path):
        """A missing checkpoint means "start over"; a corrupt one must too."""
        _store(tmp_path)
        (tmp_path / _CHECKPOINT_FILE).write_text("{not json")

        assert CheckpointStore.load(tmp_path) is None

    def test_an_index_with_the_wrong_shape_also_starts_fresh(self, tmp_path):
        _store(tmp_path)
        (tmp_path / _CHECKPOINT_FILE).write_text(json.dumps({"unexpected": True}))

        assert CheckpointStore.load(tmp_path) is None

    def test_one_corrupt_case_record_does_not_sink_the_rest(self, tmp_path):
        store = _store(tmp_path)
        store.save_case_result("c1", _case("c1"))
        store.save_case_result("c2", _case("c2"))

        (tmp_path / ".checkpoint_cases" / "c1.json").write_text("{truncated")

        results = store.load_completed_results()
        assert "c1" not in results  # discarded, will be re-run
        assert results["c2"].passed is True  # the rest of the progress survives

    def test_a_missing_case_record_is_skipped(self, tmp_path):
        store = _store(tmp_path)
        store.save_case_result("c1", _case("c1"))
        (tmp_path / ".checkpoint_cases" / "c1.json").unlink()

        assert store.load_completed_results() == {}

    def test_one_corrupt_trial_record_does_not_sink_the_rest(self, tmp_path):
        store = _store(tmp_path)
        for n in (1, 2):
            store.save_trial_result(
                "c1",
                TrialResult(trial_id=f"t{n}", trial_number=n, passed=True, score=1.0),
            )

        (tmp_path / ".checkpoint_trials" / "c1__t1.json").write_text("nonsense")

        trials = store.load_trials()
        assert [t.trial_number for t in trials["c1"]] == [2]

    def test_a_malformed_case_payload_is_discarded_not_raised(self, tmp_path):
        store = _store(tmp_path)
        store.save_case_result("c1", _case("c1"))
        # Valid JSON, wrong shape — missing the required keys
        (tmp_path / ".checkpoint_cases" / "c1.json").write_text(json.dumps({"a": 1}))

        assert store.load_completed_results() == {}


# ===================================================================
# Resume survives damage end to end
# ===================================================================


class _CountingAdapter(Adapter):
    name = "_dur_counting"
    calls: list[str] = []

    async def run(self, input: AgentInput) -> AgentOutput:
        type(self).calls.append(input.params.get("case_id", "?"))
        return AgentOutput()


@pytest.fixture
def counting_adapter():
    _CountingAdapter.calls = []
    register_adapter("_dur_counting")(_CountingAdapter)
    yield _CountingAdapter
    unregister_adapter("_dur_counting")


def _scenario(adapter: str = "_dur_counting", trials: int = 1) -> Scenario:
    return Scenario(
        name="durability",
        agent=AgentConfig(adapter=adapter),
        defaults=DefaultsConfig(trials=trials),
        cases=[
            TestCase(
                id=cid,
                input=InputConfig(prompt="go", params={"case_id": cid}),
                graders=[
                    GraderConfig(name="tool_usage", type="code", config={})
                ],
                aggregation=AggregationConfig(pass_threshold=0.0),
            )
            for cid in ("c1", "c2")
        ],
    )


class TestResumeSurvivesDamage:
    async def test_a_corrupt_index_reruns_instead_of_exploding(
        self, tmp_path, counting_adapter
    ):
        scenario = _scenario()
        await Compass().run(scenario, trace_dir=tmp_path)
        assert len(counting_adapter.calls) == 2

        (tmp_path / _CHECKPOINT_FILE).write_text("}{ corrupt")
        counting_adapter.calls = []

        result = await Compass().run(scenario, trace_dir=tmp_path, resume=True)

        # Degraded to a fresh run rather than raising
        assert len(counting_adapter.calls) == 2
        assert result.total_cases == 2

    async def test_a_corrupt_case_record_only_costs_that_case(
        self, tmp_path, counting_adapter
    ):
        scenario = _scenario()
        await Compass().run(scenario, trace_dir=tmp_path)

        (tmp_path / ".checkpoint_cases" / "c1.json").write_text("{truncated")
        counting_adapter.calls = []

        result = await Compass().run(scenario, trace_dir=tmp_path, resume=True)

        assert counting_adapter.calls == ["c1"]  # only the damaged one re-ran
        assert result.total_cases == 2

    def test_the_checkpoint_fingerprint_still_gates_reuse(self, tmp_path):
        """Durability must not weaken the staleness check."""
        scenario = _scenario()
        store = CheckpointStore.create(
            tmp_path, "run1", scenario_fingerprint(scenario), "s", ["c1"]
        )
        assert store.validate_scenario(scenario_fingerprint(scenario)) is True
        assert store.validate_scenario("something-else") is False


# ===================================================================
# Completion-marker ordering
# ===================================================================


class _ImageAdapter(Adapter):
    name = "_dur_image"

    async def run(self, input: AgentInput) -> AgentOutput:
        from PIL import Image

        img = Image.new("RGB", (4, 4), "red")
        return AgentOutput(image=img, artifacts=[ImageArtifact(image=img)])


@pytest.fixture
def image_adapter():
    register_adapter("_dur_image")(_ImageAdapter)
    yield
    unregister_adapter("_dur_image")


class TestTraceIsWrittenLast:
    async def test_the_trace_records_artifact_paths(self, tmp_path, image_adapter):
        """Artifacts are saved first because saving back-fills image_path.

        Written the other way round, the trace stored image_path: null and no
        re-grade could ever find the image again.
        """
        await Compass().run(_scenario(adapter="_dur_image"), trace_dir=tmp_path)

        data = json.loads((tmp_path / "c1.json").read_text())
        artifact = data["outcome"]["artifacts"][0]
        assert artifact["image_path"] is not None
        assert Path(artifact["image_path"]).is_file()

    async def test_a_regraded_trace_can_rehydrate_the_image(
        self, tmp_path, image_adapter
    ):
        await Compass().run(_scenario(adapter="_dur_image"), trace_dir=tmp_path)

        loaded = load_trace(tmp_path / "c1.json")
        artifact = loaded.transcript.outcome.artifacts[0]
        assert artifact.image is not None
        assert artifact.image.size == (4, 4)

    async def test_the_artifacts_exist_by_the_time_the_trace_does(
        self, tmp_path, image_adapter
    ):
        """Trace presence is the completion marker for the whole record."""
        await Compass().run(_scenario(adapter="_dur_image"), trace_dir=tmp_path)

        assert (tmp_path / "c1.json").is_file()
        saved = sorted(p.name for p in (tmp_path / "c1").iterdir() if p.is_file())
        assert saved == ["artifact_0_image.png", "manifest.json", "output.png"]

    def test_artifact_images_are_written_atomically(self, tmp_path):
        from PIL import Image

        ArtifactStore.save_image(Image.new("RGB", (2, 2)), tmp_path / "x.png")
        assert [p.name for p in tmp_path.iterdir()] == ["x.png"]
