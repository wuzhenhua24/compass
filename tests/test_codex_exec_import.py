"""Tests for the Codex CLI (``codex exec --json``) -> Compass Transcript mapping.

No codex install is required: the event stream is hand-built from the real
format — a ``thread.started`` header, ``item.started``/``item.completed`` pairs
per step, and one ``turn.completed`` carrying the whole turn's usage.
"""

from __future__ import annotations

import json

import pytest

from compass.graders import GradeContext, get_grader
from compass.integrations import (
    CodexStreamError,
    CodexStreamReconstructor,
    import_codex_stream_json,
    load_codex_stream,
    looks_like_codex_stream,
)
from compass.llm import register_pricing, reset_pricing


@pytest.fixture(autouse=True)
def _pricing_defaults():
    """A known rate for the test model, so cost asserts are deterministic."""
    reset_pricing()
    register_pricing("codex-test-model", {"input": 1.0, "output": 10.0, "cached": 0.1})
    yield
    reset_pricing()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _stream(*events: dict) -> str:
    return "\n".join(json.dumps(e) for e in events) + "\n"


def _usage(prompt=10_000, cached=8_000, write=0, output=500, reasoning=120) -> dict:
    return {
        "input_tokens": prompt,
        "cached_input_tokens": cached,
        "cache_write_input_tokens": write,
        "output_tokens": output,
        "reasoning_output_tokens": reasoning,
    }


def _run(*, cwd="/repo", edited="pricing.py") -> str:
    """A realistic one-turn run: narrate, probe, patch, verify, answer."""
    return _stream(
        {"type": "thread.started", "thread_id": "019ff681-8132"},
        {"type": "turn.started"},
        {"type": "item.completed",
         "item": {"id": "item_0", "type": "reasoning", "text": "**Planning edits**"}},
        {"type": "item.completed",
         "item": {"id": "item_1", "type": "agent_message",
                  "text": "I'm inspecting the module first."}},
        {"type": "item.started",
         "item": {"id": "item_2", "type": "command_execution",
                  "command": "/bin/zsh -lc 'rg -n def'", "aggregated_output": "",
                  "exit_code": None, "status": "in_progress"}},
        {"type": "item.completed",
         "item": {"id": "item_2", "type": "command_execution",
                  "command": "/bin/zsh -lc 'rg -n def'",
                  "aggregated_output": "pricing.py:1:def line_total\n",
                  "exit_code": 0, "status": "completed"}},
        {"type": "item.started",
         "item": {"id": "item_3", "type": "file_change",
                  "changes": [{"path": f"{cwd}/{edited}", "kind": "update"},
                              {"path": f"{cwd}/money.py", "kind": "add"}],
                  "status": "in_progress"}},
        {"type": "item.completed",
         "item": {"id": "item_3", "type": "file_change",
                  "changes": [{"path": f"{cwd}/{edited}", "kind": "update"},
                              {"path": f"{cwd}/money.py", "kind": "add"}],
                  "status": "completed"}},
        {"type": "item.started",
         "item": {"id": "item_4", "type": "command_execution",
                  "command": "/bin/zsh -lc 'pytest -q'", "aggregated_output": "",
                  "exit_code": None, "status": "in_progress"}},
        {"type": "item.completed",
         "item": {"id": "item_4", "type": "command_execution",
                  "command": "/bin/zsh -lc 'pytest -q'",
                  "aggregated_output": "1 failed, 3 passed", "exit_code": 1,
                  "status": "completed"}},
        {"type": "item.completed",
         "item": {"id": "item_5", "type": "agent_message",
                  "text": "Rounded to cents in line_total."}},
        {"type": "turn.completed", "usage": _usage()},
    )


# ---------------------------------------------------------------------------
# The mapping
# ---------------------------------------------------------------------------


def test_the_run_becomes_tool_calls_in_the_order_they_started():
    t = load_codex_stream(_run(), model="codex-test-model", cwd="/repo")

    assert [c.tool_name for c in t.tool_calls] == [
        "shell", "apply_patch", "shell", "llm.generation"
    ]
    assert t.trial_id == "019ff681-8132"
    assert t.task_id == "019ff681-8132"
    assert t.metadata["thread_id"] == "019ff681-8132"
    assert t.metadata["turns"] == 1


def test_a_shell_call_carries_its_command_output_and_exit_code():
    t = load_codex_stream(_run(), model="codex-test-model", cwd="/repo")
    probe, verify = (c for c in t.tool_calls if c.tool_name == "shell")

    assert probe.input == {"command": "/bin/zsh -lc 'rg -n def'"}
    assert "pricing.py:1:def line_total" in probe.output
    assert probe.status == "ok" and probe.metadata["exit_code"] == 0
    assert probe.call_id == "item_2"      # the item id pairs start with completion

    # A command that exited non-zero reads as an error step, the way a failed
    # Bash does on a Claude trace.
    assert verify.status == "error"
    assert verify.error["exit_code"] == 1


def test_narration_and_thinking_land_in_the_trajectory_in_order():
    """codex talks while it works: the messages are trajectory, and only the last
    one is the answer."""
    t = load_codex_stream(_run(), model="codex-test-model", cwd="/repo")

    assert t.reasoning_steps == [
        "[reasoning] **Planning edits**",
        "[message] I'm inspecting the module first.",
        "[message] Rounded to cents in line_total.",
    ]
    assert t.outcome.output_data["final_output"] == "Rounded to cents in line_total."


def test_the_turn_becomes_one_llm_call_holding_the_whole_token_bill():
    """codex aggregates usage per turn and never reports the requests inside it,
    so a `codex exec` transcript has exactly one llm.generation call."""
    t = load_codex_stream(_run(), model="codex-test-model", cwd="/repo")
    llm = t.tool_calls[-1]

    assert llm.tool_name == "llm.generation" and llm.tool_type == "llm"
    assert llm.output == "Rounded to cents in line_total."
    assert llm.turn_index == 1
    assert t.environment.model_version == "codex-test-model"


def test_cached_prompt_traffic_is_kept_out_of_input_tokens():
    """codex's `input_tokens` is the whole prompt bill, cache included; Compass
    reports the fresh part separately because it is priced differently."""
    t = load_codex_stream(
        _stream(
            {"type": "thread.started", "thread_id": "th"},
            {"type": "turn.started"},
            {"type": "turn.completed",
             "usage": _usage(prompt=10_000, cached=8_000, write=500, output=500)},
        ),
        model="codex-test-model",
    )
    tokens = t.tool_calls[0].tokens

    assert tokens.input_tokens == 1_500          # 10k - 8k cached - 500 written
    assert tokens.cache_read_tokens == 8_000
    assert tokens.cache_creation_tokens == 500
    assert tokens.output_tokens == 500
    assert tokens.total_tokens == 10_500         # everything the model processed
    assert tokens.metadata["reasoning_output_tokens"] == 120


def test_cost_comes_from_the_pricing_table_because_codex_reports_none():
    t = load_codex_stream(_run(), model="codex-test-model", cwd="/repo")

    assert t.sum_cost().total_usd == pytest.approx(
        2_000 / 1e6 * 1.0 + 500 / 1e6 * 10.0 + 8_000 / 1e6 * 0.1
    )
    assert "cost_unpriced_model" not in t.metadata


def test_an_unpriced_model_says_so_instead_of_reporting_a_small_number():
    """$0.00 would pass a cost_budget grader having measured nothing."""
    t = load_codex_stream(_run(), model="gpt-nobody-priced-this", cwd="/repo")

    assert t.sum_cost().total_usd == 0.0
    assert t.metadata["cost_unpriced_model"] == "gpt-nobody-priced-this"


def test_no_model_means_no_cost_at_all():
    """An imported stream does not know what ran — the format never records it."""
    t = load_codex_stream(_run(), cwd="/repo")

    assert t.tool_calls[-1].cost is None
    assert t.environment.model_version in ("", None)


# ---------------------------------------------------------------------------
# State delta
# ---------------------------------------------------------------------------


def test_file_changes_become_state_deltas_relative_to_the_cwd():
    t = load_codex_stream(_run(), model="codex-test-model", cwd="/repo")
    changes = t.all_state_changes()

    assert [(c.kind, c.op, c.target) for c in changes] == [
        ("file", "update", "pricing.py"),
        ("file", "create", "money.py"),
    ]
    assert changes[0].metadata["absolute_path"] == "/repo/pricing.py"
    assert changes[0].metadata["tool"] == "apply_patch"


def test_a_path_outside_the_cwd_stays_absolute():
    """A path is normalized before it is relativized, so it cannot come out as
    ``../elsewhere/x.py`` — a target pointing outside a workspace the change
    never left, which no honest glob could describe."""
    t = load_codex_stream(
        _run(cwd="/repo", edited="../elsewhere/x.py"), model="codex-test-model",
        cwd="/repo",
    )
    targets = [c.target for c in t.all_state_changes()]

    assert targets == ["/elsewhere/x.py", "money.py"]


def test_an_unfinished_change_records_nothing():
    """Until the item completes the patch has not been applied — and it may never
    be (a rejected patch, a killed run)."""
    t = load_codex_stream(
        _stream(
            {"type": "thread.started", "thread_id": "th"},
            {"type": "turn.started"},
            {"type": "item.started",
             "item": {"id": "i1", "type": "file_change",
                      "changes": [{"path": "/repo/app.py", "kind": "update"}],
                      "status": "in_progress"}},
        ),
        model="codex-test-model",
        cwd="/repo",
    )

    assert t.all_state_changes() == []
    assert t.metadata["incomplete_items"] == ["i1"]
    assert t.tool_calls[0].metadata["incomplete"] is True


async def test_state_delta_reaches_the_grader():
    t = load_codex_stream(
        _run(edited="tests/test_pricing.py"), model="codex-test-model", cwd="/repo"
    )
    grader = get_grader("state_delta")(
        {"forbid": [{"kind": "file", "target": "tests/*"}]}
    )

    result = await grader.grade(GradeContext(transcript=t, outcome=t.outcome))

    assert result.passed is False


# ---------------------------------------------------------------------------
# The rest of the vocabulary
# ---------------------------------------------------------------------------


def test_plans_searches_and_mcp_calls_use_codexs_own_tool_names():
    t = load_codex_stream(
        _stream(
            {"type": "thread.started", "thread_id": "th"},
            {"type": "turn.started"},
            {"type": "item.started",
             "item": {"id": "p", "type": "todo_list",
                      "items": [{"text": "Look", "completed": False}]}},
            {"type": "item.updated",
             "item": {"id": "p", "type": "todo_list",
                      "items": [{"text": "Look", "completed": True}]}},
            {"type": "item.completed",
             "item": {"id": "p", "type": "todo_list",
                      "items": [{"text": "Look", "completed": True}]}},
            {"type": "item.completed",
             "item": {"id": "w", "type": "web_search", "query": "decimal rounding"}},
            {"type": "item.completed",
             "item": {"id": "m", "type": "mcp_tool_call", "server": "docs",
                      "tool": "lookup", "arguments": {"q": "ROUND_HALF_UP"},
                      "result": "…", "status": "completed"}},
        ),
        model="codex-test-model",
    )

    assert [c.tool_name for c in t.tool_calls] == [
        "update_plan", "web_search", "docs.lookup"
    ]
    plan = t.tool_calls[0]
    assert plan.input["items"] == [{"text": "Look", "completed": True}]
    assert plan.metadata["updates"] == 1
    assert t.tool_calls[1].input == {"query": "decimal rounding"}
    assert t.tool_calls[2].input == {"q": "ROUND_HALF_UP"}


def test_a_failed_turn_is_recorded_as_a_failure_not_a_quiet_stop():
    t = load_codex_stream(
        _stream(
            {"type": "thread.started", "thread_id": "th"},
            {"type": "item.completed",
             "item": {"id": "e", "type": "error", "message": "Model metadata missing"}},
            {"type": "turn.started"},
            {"type": "error", "message": "{\"status\": 400}"},
            {"type": "turn.failed", "error": {"message": "model not supported"}},
        ),
        model="codex-test-model",
    )

    assert t.metadata["turn_failures"] == ["model not supported"]
    assert [e["source"] for e in t.metadata["errors"]] == ["item", "stream"]
    llm = t.tool_calls[0]
    assert llm.status == "error" and llm.error["message"] == "model not supported"


def test_an_unknown_event_is_counted_not_dropped():
    """A count answers the question that matters: did something happen during
    this run that the transcript does not explain?"""
    t = load_codex_stream(
        _stream(
            {"type": "thread.started", "thread_id": "th"},
            {"type": "turn.started"},
            {"type": "item.completed", "item": {"id": "x", "type": "future_item"}},
            {"type": "thread.compacted"},
            {"type": "turn.completed", "usage": _usage()},
        ),
        model="codex-test-model",
    )

    assert t.metadata["unhandled_events"] == {"item:future_item": 1, "thread.compacted": 1}


def test_a_stream_killed_mid_turn_still_yields_what_happened():
    """The reconstructor is fed line by line precisely so a timeout keeps the
    steps that completed."""
    r = CodexStreamReconstructor(task_id="fix_rounding", model="codex-test-model")
    for line in _run().splitlines()[:-1]:      # everything but turn.completed
        r.feed_line(line)
    t = r.finish()

    assert [c.tool_name for c in t.tool_calls] == ["shell", "apply_patch", "shell"]
    assert t.task_id == "fix_rounding"         # an explicit task id wins
    assert t.outcome.output_data["final_output"] == "Rounded to cents in line_total."


def test_a_non_json_line_is_reported_not_raised():
    r = CodexStreamReconstructor(model="codex-test-model")

    assert r.feed_line("Reading additional input from stdin...") is False
    assert r.feed_line("") is False
    assert r.feed_line('{"type": "thread.started", "thread_id": "th"}') is True
    assert r.feed_line('{"type": "assistant", "message": {}}') is False  # not codex


# ---------------------------------------------------------------------------
# File / format handling
# ---------------------------------------------------------------------------


def test_importing_a_saved_stream_from_disk(tmp_path):
    path = tmp_path / "case.stream.jsonl"
    path.write_text(_run(), encoding="utf-8")

    t = import_codex_stream_json(path, task_id="fix_rounding", model="codex-test-model",
                                 cwd="/repo")

    assert t.task_id == "fix_rounding"
    assert len(t.tool_calls) == 4


def test_a_replayed_stream_records_no_durations():
    """The events carry no timestamps, so the only thing an offline import could
    measure is the replay itself."""
    t = load_codex_stream(_run(), model="codex-test-model", cwd="/repo")

    assert all(c.duration_ms == 0.0 for c in t.tool_calls)
    assert t.total_duration_ms == 0.0
    assert "no timestamps" in t.metadata["timings"]


def test_a_file_with_no_codex_events_is_an_error():
    with pytest.raises(CodexStreamError, match="no `codex exec --json` events"):
        load_codex_stream('{"type": "session", "id": "pi-1"}\n')


def test_the_format_sniffer_recognizes_codex_events_only():
    assert looks_like_codex_stream({"type": "thread.started"}) is True
    assert looks_like_codex_stream({"type": "item.completed", "item": {}}) is True
    assert looks_like_codex_stream({"type": "turn.failed"}) is True
    assert looks_like_codex_stream({"type": "session"}) is False        # pi
    assert looks_like_codex_stream({"type": "assistant"}) is False      # claude
    assert looks_like_codex_stream(["not", "a", "dict"]) is False
