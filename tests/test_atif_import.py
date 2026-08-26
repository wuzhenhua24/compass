"""Tests for the ATIF (Harbor Agent Trajectory Interchange Format) importer.

No Harbor package is required: trajectories are hand-built to the ATIF-v1.7
schema. The shapes here mirror what Harbor's own agents write to
``<trial_dir>/agent/trajectory.json``.
"""

from __future__ import annotations

import json

import pytest

from compass.integrations import (
    ATIFImportError,
    atif_to_transcript,
    import_atif_dir,
    import_atif_file,
    load_atif,
    looks_like_atif,
)
from compass.llm import register_pricing, reset_pricing


@pytest.fixture(autouse=True)
def _pricing_defaults():
    reset_pricing()
    register_pricing("gpt-4o", (2.50, 10.00))
    yield
    reset_pricing()


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _traj(steps, **kw):
    traj = {
        "schema_version": "ATIF-v1.7",
        "session_id": "sess-1",
        "agent": {"name": "terminus-2", "version": "2.0.0", "model_name": "gpt-4o"},
        "steps": steps,
    }
    traj.update(kw)
    return traj


def _step(step_id, source, message, **kw):
    step = {"step_id": step_id, "source": source, "message": message}
    step.update(kw)
    return step


def _agent(step_id, message, *, tools=None, results=None, metrics=None, **kw):
    step = _step(step_id, "agent", message, **kw)
    if tools is not None:
        step["tool_calls"] = tools
    if results is not None:
        step["observation"] = {"results": results}
    if metrics is not None:
        step["metrics"] = metrics
    return step


def _tool(call_id, name, args=None):
    return {"tool_call_id": call_id, "function_name": name, "arguments": args or {}}


def _metrics(prompt=100, completion=20, cached=0, **kw):
    m = {"prompt_tokens": prompt, "completion_tokens": completion, "cached_tokens": cached}
    m.update(kw)
    return m


def _write(tmp_path, name, payload):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Core mapping
# ---------------------------------------------------------------------------


class TestBasicMapping:
    def test_identity_prompt_and_final_output(self):
        t = atif_to_transcript(
            _traj([
                _step(1, "user", "Fix the failing test."),
                _agent(2, "Done — the test passes now.", metrics=_metrics()),
            ])
        )
        assert t.task_id == "sess-1"
        assert t.trial_id == "sess-1"
        assert t.input_prompt == "Fix the failing test."
        assert t.outcome.output_data["final_output"] == "Done — the test passes now."
        assert t.environment.model_version == "gpt-4o"
        assert t.environment.adapter_version == "terminus-2@2.0.0"

    def test_agent_step_becomes_one_llm_generation(self):
        t = atif_to_transcript(
            _traj([
                _agent(1, "first", metrics=_metrics()),
                _agent(2, "second", metrics=_metrics()),
            ])
        )
        llm = [c for c in t.tool_calls if c.tool_name == "llm.generation"]
        assert len(llm) == 2
        assert [c.turn_index for c in llm] == [1, 2]
        assert llm[0].output == "first"
        assert t.metadata["atif"]["llm_steps"] == 2

    def test_tool_call_and_its_observation(self):
        t = atif_to_transcript(
            _traj([
                _agent(
                    1,
                    "running the tests",
                    tools=[_tool("c1", "bash", {"cmd": "pytest -q"})],
                    results=[{"source_call_id": "c1", "content": "3 passed"}],
                    metrics=_metrics(),
                ),
            ])
        )
        call = next(c for c in t.tool_calls if c.tool_name == "bash")
        assert call.call_id == "c1"
        assert call.input == {"cmd": "pytest -q"}
        assert call.output == "3 passed"
        assert call.tool_type == "code"
        assert call.turn_index == 1

    def test_several_results_for_one_call_are_all_kept(self):
        t = atif_to_transcript(
            _traj([
                _agent(
                    1,
                    "go",
                    tools=[_tool("c1", "bash")],
                    results=[
                        {"source_call_id": "c1", "content": "line one"},
                        {"source_call_id": "c1", "content": "line two"},
                    ],
                ),
            ])
        )
        call = next(c for c in t.tool_calls if c.tool_name == "bash")
        assert call.output == "line one\nline two"

    def test_result_without_source_call_id_is_a_reasoning_step(self):
        t = atif_to_transcript(
            _traj([_agent(1, "go", results=[{"content": "terminal was resized"}])])
        )
        assert any("[observation] terminal was resized" in s for s in t.reasoning_steps)

    def test_reasoning_content_becomes_a_reasoning_step(self):
        t = atif_to_transcript(
            _traj([_agent(1, "answer", reasoning_content="let me think")])
        )
        assert any(s.startswith("[reasoning] let me think") for s in t.reasoning_steps)

    def test_unknown_tool_names_are_plain_functions(self):
        t = atif_to_transcript(
            _traj([_agent(1, "x", tools=[_tool("c1", "some_domain_tool")])])
        )
        call = next(c for c in t.tool_calls if c.tool_name == "some_domain_tool")
        assert call.tool_type == "function"

    def test_atif_records_no_failure_so_every_call_reads_ok(self):
        """A documented gap: an all-ok transcript means "not recorded"."""
        t = atif_to_transcript(
            _traj([
                _agent(
                    1,
                    "x",
                    tools=[_tool("c1", "bash")],
                    results=[{"source_call_id": "c1", "content": "error: exit 1"}],
                ),
            ])
        )
        assert all(c.status == "ok" for c in t.tool_calls)
        assert all(not c.state_delta for c in t.tool_calls)


class TestUserAndSystemSteps:
    def test_later_user_turns_are_trace_steps_not_the_prompt(self):
        t = atif_to_transcript(
            _traj([
                _step(1, "user", "the task"),
                _agent(2, "working"),
                _step(3, "user", "actually, also handle empty input"),
            ])
        )
        assert t.input_prompt == "the task"
        assert any("[user] actually, also handle empty input" in s for s in t.reasoning_steps)

    def test_opening_system_step_is_the_system_prompt(self):
        t = atif_to_transcript(
            _traj([_step(1, "system", "You are a helpful agent."), _step(2, "user", "go")])
        )
        assert t.input_params["system_prompt"] == "You are a helpful agent."
        assert not any(s.startswith("[system]") for s in t.reasoning_steps)

    def test_mid_run_system_step_is_an_event_not_configuration(self):
        t = atif_to_transcript(
            _traj([
                _step(1, "user", "go"),
                _agent(2, "working"),
                _step(3, "system", "Performed context summarization."),
            ])
        )
        assert "system_prompt" not in t.input_params
        assert any("[system] Performed context summarization." in s for s in t.reasoning_steps)

    def test_a_long_system_prompt_is_truncated_and_says_so(self):
        t = atif_to_transcript(_traj([_step(1, "system", "x" * 9000)]))
        assert t.input_params["system_prompt_truncated"] is True
        assert len(t.input_params["system_prompt"]) < 9000


# ---------------------------------------------------------------------------
# Tokens and cost
# ---------------------------------------------------------------------------


class TestTokensAndCost:
    def test_cached_tokens_come_out_of_input_tokens(self):
        t = atif_to_transcript(
            _traj([_agent(1, "x", metrics=_metrics(prompt=1000, completion=200, cached=800))])
        )
        tokens = t.sum_tokens()
        assert tokens.input_tokens == 200      # fresh input only
        assert tokens.cache_read_tokens == 800
        assert tokens.total_tokens == 1200
        call = t.tool_calls[0]
        assert call.tokens.metadata["prompt_tokens_including_cache"] == 1000

    def test_cost_computed_from_the_pricing_table(self):
        t = atif_to_transcript(
            _traj([_agent(1, "x", metrics=_metrics(prompt=1_000_000, completion=1_000_000))])
        )
        assert t.sum_cost().total_usd == pytest.approx(12.50)

    def test_reported_cost_wins_over_the_pricing_table(self):
        t = atif_to_transcript(
            _traj([_agent(1, "x", metrics=_metrics(prompt=1_000_000, cost_usd=0.42))])
        )
        assert t.sum_cost().total_usd == pytest.approx(0.42)
        assert t.tool_calls[0].cost.metadata["source"] == "atif"

    def test_an_unpriced_model_records_no_cost_rather_than_zero(self):
        traj = _traj([_agent(1, "x", metrics=_metrics())])
        traj["agent"]["model_name"] = "some-unreleased-model"
        t = atif_to_transcript(traj)
        assert t.tool_calls[0].cost is None

    def test_model_override_beats_the_trajectory(self):
        traj = _traj([_agent(1, "x", metrics=_metrics(prompt=1_000_000, completion=0))])
        traj["agent"]["model_name"] = "some-unreleased-model"
        t = atif_to_transcript(traj, model="gpt-4o")
        assert t.sum_cost().total_usd == pytest.approx(2.50)
        assert t.tool_calls[0].metadata["model"] == "gpt-4o"

    def test_a_step_without_metrics_records_no_tokens(self):
        t = atif_to_transcript(_traj([_agent(1, "x")]))
        assert t.tool_calls[0].tokens is None
        assert t.sum_tokens().total_tokens == 0


# ---------------------------------------------------------------------------
# ATIF-v1.5 / v1.7 semantics
# ---------------------------------------------------------------------------


class TestCopiedContext:
    def test_copied_steps_are_dropped_and_counted(self):
        t = atif_to_transcript(
            _traj([
                _agent(1, "carried over", metrics=_metrics(prompt=5000), is_copied_context=True),
                _agent(2, "carried over too", metrics=_metrics(prompt=5000),
                       is_copied_context=True),
                _agent(3, "actual work", metrics=_metrics(prompt=100, completion=20)),
            ])
        )
        assert t.metadata["atif"]["copied_context_steps"] == 2
        assert len([c for c in t.tool_calls if c.tool_name == "llm.generation"]) == 1
        assert t.sum_tokens().total_tokens == 120  # not billed three times

    def test_copied_steps_do_not_leave_phantom_tool_calls(self):
        t = atif_to_transcript(
            _traj([
                _agent(1, "x", tools=[_tool("c1", "bash")], is_copied_context=True),
                _agent(2, "x", tools=[_tool("c2", "bash")]),
            ])
        )
        assert [c.call_id for c in t.tool_calls if c.tool_name == "bash"] == ["c2"]


class TestLLMCallCount:
    def test_deterministic_dispatch_emits_no_llm_call(self):
        t = atif_to_transcript(
            _traj([_agent(1, "routing", llm_call_count=0, tools=[_tool("c1", "bash")])])
        )
        assert not [c for c in t.tool_calls if c.tool_name == "llm.generation"]
        assert [c.tool_name for c in t.tool_calls] == ["bash"]

    def test_aggregated_inferences_are_reported_not_split(self):
        t = atif_to_transcript(
            _traj([_agent(1, "x", llm_call_count=3, metrics=_metrics())])
        )
        llm = [c for c in t.tool_calls if c.tool_name == "llm.generation"]
        assert len(llm) == 1
        assert llm[0].metadata["llm_call_count"] == 3


class TestMultimodalContent:
    def test_content_parts_flatten_to_text_and_count_images(self):
        t = atif_to_transcript(
            _traj([
                _step(1, "user", [
                    {"type": "text", "text": "What is in this screenshot?"},
                    {"type": "image",
                     "source": {"media_type": "image/png", "path": "shot.png"}},
                ]),
                _agent(2, "A terminal."),
            ])
        )
        assert "What is in this screenshot?" in t.input_prompt
        assert "[image: shot.png]" in t.input_prompt
        assert t.metadata["atif"]["image_content_parts"] == 1


# ---------------------------------------------------------------------------
# Subagents
# ---------------------------------------------------------------------------


def _delegating(ref):
    """A trajectory whose step 2 delegates to *ref*."""
    return _traj(
        [
            _step(1, "user", "do it"),
            _agent(
                2,
                "delegating",
                tools=[_tool("c1", "Task")],
                results=[{"source_call_id": "c1", "content": "done",
                          "subagent_trajectory_ref": [ref]}],
                metrics=_metrics(prompt=100, completion=20),
            ),
        ]
    )


def _sub(trajectory_id="sub-1", name="researcher"):
    return {
        "schema_version": "ATIF-v1.7",
        "trajectory_id": trajectory_id,
        "agent": {"name": name, "version": "1.0", "model_name": "gpt-4o"},
        "steps": [_agent(1, "found it", tools=[_tool("s1", "grep")],
                         metrics=_metrics(prompt=500, completion=50))],
    }


class TestSubagents:
    def test_embedded_subagent_is_flattened_with_its_agent_name(self):
        traj = _delegating({"trajectory_id": "sub-1"})
        traj["subagent_trajectories"] = [_sub()]
        t = atif_to_transcript(traj)

        assert t.metadata["atif"]["subagents"] == ["researcher"]
        sub_calls = [c for c in t.tool_calls if c.agent_name == "researcher"]
        assert {c.tool_name for c in sub_calls} == {"llm.generation", "grep"}
        # A delegation is one turn of the parent, whatever happened inside it.
        assert all(c.turn_index == 2 for c in sub_calls)
        # The subagent's bill lands in the parent's total, as ATIF defines it.
        assert t.sum_tokens().total_tokens == 670

    def test_the_subagent_does_not_hijack_the_final_output(self):
        traj = _delegating({"trajectory_id": "sub-1"})
        traj["subagent_trajectories"] = [_sub()]
        traj["steps"].append(_agent(3, "here is the answer"))
        t = atif_to_transcript(traj)
        assert t.outcome.output_data["final_output"] == "here is the answer"

    def test_subagents_can_be_excluded(self):
        traj = _delegating({"trajectory_id": "sub-1"})
        traj["subagent_trajectories"] = [_sub()]
        t = atif_to_transcript(traj, include_subagents=False)
        assert not [c for c in t.tool_calls if c.agent_name]
        assert t.metadata["atif"]["subagents_excluded"] is True
        assert t.sum_tokens().total_tokens == 120

    def test_a_dangling_embedded_ref_is_reported_not_silently_dropped(self):
        traj = _delegating({"trajectory_id": "nope"})
        traj["subagent_trajectories"] = [_sub()]
        t = atif_to_transcript(traj)
        unresolved = t.metadata["atif"]["unresolved_subagent_refs"]
        assert unresolved[0]["trajectory_id"] == "nope"
        assert "not found" in unresolved[0]["reason"]

    def test_external_trajectory_path_resolves_beside_the_source(self, tmp_path):
        _write(tmp_path, "sub.json", _sub())
        path = _write(tmp_path, "trajectory.json", _delegating({"trajectory_path": "sub.json"}))
        t = import_atif_file(path)
        assert t.metadata["atif"]["subagents"] == ["researcher"]
        assert t.sum_tokens().total_tokens == 670

    def test_a_path_escaping_the_source_directory_is_not_read(self, tmp_path):
        _write(tmp_path, "outside.json", _sub())
        run = tmp_path / "run"
        path = _write(run, "trajectory.json",
                      _delegating({"trajectory_path": "../outside.json"}))
        t = import_atif_file(path)
        assert "subagents" not in t.metadata["atif"]
        reason = t.metadata["atif"]["unresolved_subagent_refs"][0]["reason"]
        assert "outside the source directory" in reason

    def test_a_missing_external_file_is_reported(self, tmp_path):
        path = _write(tmp_path, "trajectory.json",
                      _delegating({"trajectory_path": "gone.json"}))
        t = import_atif_file(path)
        assert t.metadata["atif"]["unresolved_subagent_refs"][0]["reason"] == (
            "trajectory_path does not exist"
        )

    def test_a_cycle_terminates(self, tmp_path):
        """A trajectory that delegates to itself must not recurse forever."""
        path = _write(tmp_path, "trajectory.json",
                      _delegating({"trajectory_path": "trajectory.json"}))
        t = import_atif_file(path)
        # Followed once, refused the second time round.
        assert any("cycle" in r["reason"]
                   for r in t.metadata["atif"]["unresolved_subagent_refs"])

    def test_in_memory_refs_need_a_base_dir(self):
        t = atif_to_transcript(_delegating({"trajectory_path": "sub.json"}))
        reason = t.metadata["atif"]["unresolved_subagent_refs"][0]["reason"]
        assert "no directory" in reason


# ---------------------------------------------------------------------------
# Provenance, timing, detection
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_reported_final_metrics_are_kept_beside_the_computed_sums(self):
        traj = _traj(
            [_agent(1, "x", metrics=_metrics(prompt=100, completion=20))],
            final_metrics={"total_prompt_tokens": 100, "total_completion_tokens": 20,
                           "total_cost_usd": 0.001},
            notes="produced by a bridge",
        )
        atif = atif_to_transcript(traj).metadata["atif"]
        assert atif["final_metrics_reported"]["total_cost_usd"] == 0.001
        assert atif["notes"] == "produced by a bridge"
        assert atif["schema_version"] == "ATIF-v1.7"

    def test_tool_definitions_become_the_available_tool_list(self):
        traj = _traj([_agent(1, "x")])
        traj["agent"]["tool_definitions"] = [
            {"type": "function", "function": {"name": "bash"}},
            {"name": "read_file"},
        ]
        assert atif_to_transcript(traj).metadata["atif"]["available_tools"] == [
            "bash", "read_file"
        ]

    def test_timestamps_set_the_wall_clock(self):
        t = atif_to_transcript(
            _traj([
                _agent(1, "a", timestamp="2026-01-01T00:00:00Z"),
                _agent(2, "b", timestamp="2026-01-01T00:00:30Z"),
            ])
        )
        assert t.total_duration_ms == pytest.approx(30_000)
        assert t.end_time > t.start_time

    def test_mixed_offset_and_naive_timestamps_do_not_explode(self):
        t = atif_to_transcript(
            _traj([
                _agent(1, "a", timestamp="2026-01-01T00:00:00+00:00"),
                _agent(2, "b", timestamp="2026-01-01T00:00:10"),
            ])
        )
        assert t.end_time is not None

    def test_no_timestamps_means_no_invented_durations(self):
        t = atif_to_transcript(_traj([_agent(1, "a", tools=[_tool("c1", "bash")])]))
        assert t.total_duration_ms == 0.0
        assert all(c.duration_ms == 0.0 for c in t.tool_calls)
        assert "per-call durations not recorded" in t.metadata["atif"]["timings"]


class TestSniffing:
    def test_recognizes_a_declared_trajectory(self):
        assert looks_like_atif(_traj([_agent(1, "x")]))

    def test_recognizes_one_without_a_schema_version(self):
        traj = _traj([_agent(1, "x")])
        del traj["schema_version"]
        assert looks_like_atif(traj)

    @pytest.mark.parametrize(
        "payload",
        [
            {"type": "session", "id": "s"},                       # pi
            {"type": "assistant", "message": {}},                 # claude
            {"type": "thread.started", "thread_id": "t"},         # codex
            {"resourceSpans": []},                                # otlp
            {"agent": {"name": "a"}, "steps": []},                # no steps
            [],
            "nope",
        ],
    )
    def test_rejects_other_shapes(self, payload):
        assert not looks_like_atif(payload)


class TestErrors:
    def test_a_non_atif_payload_raises(self):
        with pytest.raises(ATIFImportError, match="not an ATIF trajectory"):
            atif_to_transcript({"resourceSpans": []})

    def test_invalid_json_raises(self):
        with pytest.raises(ATIFImportError, match="not valid JSON"):
            load_atif("{not json")

    def test_an_empty_payload_raises(self):
        with pytest.raises(ATIFImportError, match="empty payload"):
            load_atif("   ")

    def test_a_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            import_atif_file(tmp_path / "nothing.json")


# ---------------------------------------------------------------------------
# Harbor job directories
# ---------------------------------------------------------------------------


def _harbor_trial(root, task, trial, *, reward=1.0):
    trial_dir = root / task / trial
    _write(trial_dir, "agent/trajectory.json",
           _traj([_step(1, "user", f"task {task}"), _agent(2, "done", metrics=_metrics())]))
    _write(trial_dir, "results.json", {
        "task_name": task,
        "trial_name": trial,
        "trial_uri": f"local://{task}/{trial}",
        "verifier_result": {"rewards": {"reward": reward}},
    })
    return trial_dir


class TestHarborLayout:
    def test_the_sidecar_names_the_task_and_carries_the_reward(self, tmp_path):
        trial = _harbor_trial(tmp_path, "fix-bug", "trial-1", reward=1.0)
        t = import_atif_file(trial / "agent/trajectory.json")
        assert t.task_id == "fix-bug"
        assert t.trial_id == "trial-1"
        assert t.metadata["harbor"]["rewards"] == {"reward": 1.0}

    def test_the_reward_is_recorded_not_turned_into_a_compass_score(self, tmp_path):
        trial = _harbor_trial(tmp_path, "fix-bug", "trial-1", reward=1.0)
        t = import_atif_file(trial / "agent/trajectory.json")
        assert t.final_score == 0.0
        assert t.final_passed is False
        assert not t.grader_results

    def test_a_multi_step_trial_still_finds_its_sidecar(self, tmp_path):
        trial_dir = tmp_path / "task" / "trial-1"
        _write(trial_dir, "steps/build/agent/trajectory.json", _traj([_agent(1, "x")]))
        _write(trial_dir, "results.json",
               {"task_name": "task", "trial_name": "trial-1",
                "verifier_result": {"rewards": {"reward": 0.0}}})
        t = import_atif_file(trial_dir / "steps/build/agent/trajectory.json")
        assert t.metadata["harbor"]["rewards"] == {"reward": 0.0}

    def test_a_whole_job_directory_imports_every_trial(self, tmp_path):
        _harbor_trial(tmp_path, "task-a", "trial-1")
        _harbor_trial(tmp_path, "task-a", "trial-2", reward=0.0)
        _harbor_trial(tmp_path, "task-b", "trial-1")
        transcripts = import_atif_dir(tmp_path)
        assert len(transcripts) == 3
        assert {t.task_id for t in transcripts} == {"task-a", "task-b"}

    def test_a_stray_file_does_not_kill_the_sweep(self, tmp_path):
        _harbor_trial(tmp_path, "task-a", "trial-1")
        _write(tmp_path, "other/agent/trajectory.json", {"not": "atif"})
        assert len(import_atif_dir(tmp_path)) == 1

    def test_sweeping_a_non_directory_raises(self, tmp_path):
        path = _write(tmp_path, "trajectory.json", _traj([_agent(1, "x")]))
        with pytest.raises(ATIFImportError, match="not a directory"):
            import_atif_dir(path)


# ---------------------------------------------------------------------------
# The point of the exercise: grading a Harbor run's process
# ---------------------------------------------------------------------------


class TestGradesLikeAnyTranscript:
    async def test_a_transcript_scope_grader_reads_an_imported_run(self):
        from compass.graders import GradeContext, get_grader

        t = atif_to_transcript(
            _traj([
                _step(1, "user", "run the suite"),
                _agent(2, "x", tools=[_tool("c1", "bash")],
                       metrics=_metrics(prompt=1_000_000, completion=1_000_000)),
            ])
        )
        grader = get_grader("cost_budget")({"max_cost_usd": 1.0})
        result = await grader.grade(GradeContext(transcript=t))
        assert result.passed is False  # $12.50 against a $1 budget
