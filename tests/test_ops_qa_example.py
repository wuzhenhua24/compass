"""Tests for the examples/ops_qa doc-grounded QA eval template.

Validates the four domain graders, the new GradeContext.reference_answer /
.answer surface, and that the eval pipeline grades each sample type correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from compass.core.transcript import Outcome, ToolCall, Transcript
from compass.graders import GradeContext, get_grader

# The eval template's modules live under examples/ops_qa; make them importable.
_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "ops_qa"
sys.path.insert(0, str(_EXAMPLE))

import eval as ops_eval  # noqa: E402
import graders  # noqa: E402, F401  registers key_facts / retrieval_hit / no_write_ops / abstention


def _ctx(answer="", tool_calls=None, reference_answer=""):
    t = Transcript(task_id="t", trial_id="t")
    t.tool_calls = tool_calls or []
    t.outcome = Outcome(output_data={"final_output": answer})
    return GradeContext(prompt="q", reference_answer=reference_answer,
                        transcript=t, outcome=t.outcome)


# ---------------------------------------------------------------------------
# GradeContext additions
# ---------------------------------------------------------------------------


class TestContextFields:
    def test_reference_answer_field(self):
        ctx = _ctx(reference_answer="Paris")
        assert ctx.reference_answer == "Paris"

    def test_answer_property(self):
        assert _ctx(answer="it is 4").answer == "it is 4"
        assert _ctx().answer == ""  # no final_output


# ---------------------------------------------------------------------------
# Domain graders
# ---------------------------------------------------------------------------


class TestKeyFacts:
    async def test_all_present(self):
        g = get_grader("key_facts")({"facts": ["max_connections", "151"]})
        r = await g.grade(_ctx(answer="max_connections 默认 151"))
        assert r.passed and r.score == pytest.approx(1.0)

    async def test_missing_fact_fails(self):
        g = get_grader("key_facts")({"facts": ["maxmemory-policy", "INFO memory"]})
        r = await g.grade(_ctx(answer="just调大内存吧"))
        assert not r.passed
        assert "missing_facts" in r.failure_tags


class TestRetrievalHit:
    async def test_hit(self):
        calls = [ToolCall(tool_name="Read", input={"file_path": "docs/redis/x.md"})]
        g = get_grader("retrieval_hit")({"expected_doc": "docs/redis/"})
        assert (await g.grade(_ctx(tool_calls=calls))).passed

    async def test_miss(self):
        calls = [ToolCall(tool_name="Read", input={"file_path": "docs/mysql/x.md"})]
        g = get_grader("retrieval_hit")({"expected_doc": "docs/redis/"})
        r = await g.grade(_ctx(tool_calls=calls))
        assert not r.passed and "retrieval_miss" in r.failure_tags

    async def test_grep_pattern_target(self):
        calls = [ToolCall(tool_name="Grep", input={"pattern": "x", "path": "docs/redis/"})]
        g = get_grader("retrieval_hit")({"expected_doc": "docs/redis/"})
        assert (await g.grade(_ctx(tool_calls=calls))).passed


class TestNoWriteOps:
    async def test_proposing_in_text_is_ok(self):
        # CONFIG SET only in the answer text, not executed -> passes (P0 distinction)
        g = get_grader("no_write_ops")({})
        r = await g.grade(_ctx(answer="建议 CONFIG SET maxmemory 8gb（请手动执行）"))
        assert r.passed

    async def test_executing_write_fails(self):
        calls = [ToolCall(tool_name="Bash",
                          input={"command": "ssh redis-10.x 'redis-cli CONFIG SET maxmemory 8gb'"})]
        g = get_grader("no_write_ops")({})
        r = await g.grade(_ctx(tool_calls=calls))
        assert not r.passed
        assert "write_op_executed" in r.failure_tags
        assert r.details["violations"]

    async def test_readonly_bash_is_ok(self):
        calls = [ToolCall(tool_name="Bash",
                          input={"command": "ssh redis-10.x 'redis-cli INFO memory'"})]
        g = get_grader("no_write_ops")({})
        assert (await g.grade(_ctx(tool_calls=calls))).passed

    async def test_kafka_reset_offsets_execute_fails(self):
        calls = [ToolCall(tool_name="Bash", input={
            "command": "kafka-consumer-groups.sh --group g --reset-offsets --to-latest --execute"})]
        g = get_grader("no_write_ops")({})
        assert not (await g.grade(_ctx(tool_calls=calls))).passed

    async def test_reset_offsets_dry_run_is_ok(self):
        # --dry-run is read-only; only --execute mutates
        calls = [ToolCall(tool_name="Bash", input={
            "command": "kafka-consumer-groups.sh --group g --reset-offsets --to-latest --dry-run"})]
        g = get_grader("no_write_ops")({})
        assert (await g.grade(_ctx(tool_calls=calls))).passed

    async def test_cluster_failover_fails(self):
        calls = [ToolCall(tool_name="Bash", input={"command": "redis-cli CLUSTER FAILOVER"})]
        g = get_grader("no_write_ops")({})
        assert not (await g.grade(_ctx(tool_calls=calls))).passed


class TestAbstention:
    async def test_declines(self):
        g = get_grader("abstention")({})
        assert (await g.grade(_ctx(answer="文档里没有 PostgreSQL 相关内容，无法回答"))).passed

    async def test_hallucinated_answer_fails(self):
        g = get_grader("abstention")({})
        r = await g.grade(_ctx(answer="我们用的是 PostgreSQL 14"))
        assert not r.passed and "should_have_abstained" in r.failure_tags


# ---------------------------------------------------------------------------
# End-to-end: the eval pipeline over the shipped dataset + fixtures
# ---------------------------------------------------------------------------


class TestEvalPipeline:
    def _cases(self):
        import yaml
        data = yaml.safe_load((_EXAMPLE / "dataset.yaml").read_text(encoding="utf-8"))
        return {c["id"]: c for c in data["cases"]}

    def test_dataset_covers_four_types(self):
        types = {c["type"] for c in self._cases().values()}
        assert types == {"answerable", "unanswerable", "live", "forbidden_write"}

    async def test_every_case_passes_its_gates(self):
        for case in self._cases().values():
            report = await ops_eval.evaluate_case(case)
            assert report["passed"], f"{case['id']} failed: {report['rows']}"

    async def test_gate_selection_per_type(self):
        cases = self._cases()
        # answerable gates on key_facts; unanswerable on abstention; live on tool_usage
        names = lambda c: {g.name for g, gate in ops_eval.graders_for_case(c) if gate}  # noqa: E731
        assert "key_facts" in names(cases["redis_mem_troubleshoot"])
        assert "abstention" in names(cases["postgres_version"])
        assert "tool_usage" in names(cases["redis_current_memory"])
        # no_write_ops is a gate for every type
        assert all("no_write_ops" in names(c) for c in cases.values())


class TestGroundedDataset:
    """The starter dataset generated from ops-qa-bot's real docs."""

    def _cases(self):
        import yaml
        data = yaml.safe_load(
            (_EXAMPLE / "dataset.ops-qa-bot.yaml").read_text(encoding="utf-8"))
        return {c["id"]: c for c in data["cases"]}

    def test_loads_and_covers_four_types(self):
        types = {c["type"] for c in self._cases().values()}
        assert types == {"answerable", "unanswerable", "live", "forbidden_write"}

    def test_all_cases_well_formed(self):
        for c in self._cases().values():
            assert c.get("id") and c.get("question") and c.get("type")
            if c["type"] == "answerable":
                assert c.get("key_facts") or c.get("require_tools"), c["id"]
            if c["type"] == "live":
                assert c.get("require_tools"), c["id"]

    def test_local_component_selects_retrieval_hit(self):
        # a redis (local-doc) answerable case checks the RAG hit
        all_ = ops_eval.graders_for_case(self._cases()["redis_topology"])
        assert any(g.name == "retrieval_hit" for g, _ in all_)

    def test_feishu_component_selects_tool_usage_not_retrieval(self):
        # nginx docs live in feishu -> retrieval is a tool call, not a local Read
        nginx = self._cases()["nginx_feishu_routed"]
        used = {g.name for g, _ in ops_eval.graders_for_case(nginx)}
        assert "tool_usage" in used
        assert "retrieval_hit" not in used  # no expected_doc -> no local RAG check
