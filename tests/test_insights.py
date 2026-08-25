"""Tests for the grounded insights judge.

No API key and no LLM: the judge's reply is injected, because the thing under
test is not the model's prose — it is the gate the prose has to pass. Every
test here is a claim the model could plausibly write and a statement about
whether this run's data lets it stand.
"""

from __future__ import annotations

import json

import pytest

from compass.report.compare import compare_results, load_case_records
from compass.report.insights import (
    CLAIM_TYPES,
    CaseFacts,
    InsightsJudge,
    build_prompt,
    collect_facts,
    ground,
)
from compass.report.site import collect_run_payload

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _case(case_id, *, passed=True, status=None, scope="outcome", failure_tags=None):
    score = 1.0 if passed else 0.0
    return {
        "case_id": case_id,
        "task_id": case_id,
        "status": status or ("passed" if passed else "failed"),
        "passed": passed,
        "overall_score": score,
        "best_score": score,
        "duration_ms": 1000.0,
        "evaluator_results": [
            {
                "name": "checker",
                "score": score,
                "passed": passed,
                "grader_scope": scope,
                "grader_type": "code",
                "failure_tags": failure_tags or [],
            }
        ],
    }


def _doc(cases, name="qa"):
    payload = {
        "results": [
            {
                "scenario_name": name,
                "run_id": "run-1",
                "config_hash": "cfg-1",
                "timestamp": "2026-08-25T00:00:00",
                "duration_ms": 2000.0,
                "case_results": cases,
            }
        ]
    }
    return collect_run_payload(payload, name=name, generated="2026-08-25T00:00:00+00:00")


#: One passing case, one plain failure, one process failure, one harness error.
STANDARD = [
    _case("pos_explicit"),
    _case("neg_chart", passed=False),
    _case("pos_implicit", passed=False, scope="transcript", failure_tags=["not_triggered"]),
    _case("boom", passed=False, status="error"),
]


def _claim(claim_type, severity, title, message, case_ids=()):
    return {
        "claim_type": claim_type,
        "severity": severity,
        "title": title,
        "message": message,
        "case_ids": list(case_ids),
    }


def _judge(claims, **kwargs):
    async def call(prompt, schema):
        return {"claims": claims}

    return InsightsJudge(call=call, max_claims=kwargs.pop("max_claims", 10), **kwargs)


async def _run(claims, cases=None, comparison=None, **kwargs):
    subject = STANDARD if cases is None else cases
    return await _judge(claims, **kwargs).run(_doc(subject), comparison)


# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------


class TestFacts:
    def test_a_case_is_read_into_what_a_claim_can_lean_on(self):
        facts = {f.case_id: f for f in collect_facts(_doc(STANDARD))}

        assert facts["pos_explicit"].passed is True
        assert facts["neg_chart"].passed is False
        assert facts["neg_chart"].errored is False
        assert facts["boom"].errored is True
        assert facts["pos_implicit"].process_failed is True
        assert facts["pos_implicit"].failure_tags == ("not_triggered",)

    def test_a_failing_outcome_grader_is_not_a_process_defect(self):
        facts = {f.case_id: f for f in collect_facts(_doc(STANDARD))}

        assert facts["neg_chart"].process_failed is False
        assert facts["neg_chart"].outcome_failed is True

    def test_flips_come_from_the_comparison_not_the_document(self):
        before = [_case("c1", passed=True), _case("c2", passed=False)]
        after = [_case("c1", passed=False), _case("c2", passed=True)]
        comparison = compare_results(
            {c["case_id"]: r for c, r in zip(before, _records(before), strict=True)},
            {c["case_id"]: r for c, r in zip(after, _records(after), strict=True)},
            label_a="v1",
            label_b="v2",
        )

        facts = {f.case_id: f for f in collect_facts(_doc(after), comparison)}

        assert facts["c1"].flip == "regressed"
        assert facts["c2"].flip == "improved"


def _records(cases):
    """CaseRecords for a case list, via the same loader `compare` uses."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "r.json"
        path.write_text(json.dumps({"case_results": cases}), encoding="utf-8")
        records = load_case_records(path)
    return [records[c["case_id"]] for c in cases]


# ---------------------------------------------------------------------------
# Grounding: what the data refuses
# ---------------------------------------------------------------------------


class TestGrounding:
    async def test_a_supported_claim_survives(self):
        report = await _run(
            [_claim("failure_pattern", "fail", "neg_chart fails",
                    "The case neg_chart did not pass.", ["neg_chart"])]
        )

        assert [c.title for c in report.claims] == ["neg_chart fails"]
        assert report.dropped == []

    async def test_a_claim_citing_a_case_that_passed_is_dropped(self):
        report = await _run(
            [_claim("failure_pattern", "fail", "pos_explicit fails",
                    "pos_explicit also failed.", ["pos_explicit"])]
        )

        assert report.claims == []
        assert "it passed" in report.dropped[0].reason

    async def test_a_harness_error_is_not_evidence_of_an_agent_failure(self):
        """Compass keeps crashed cases out of every denominator. "3 of 15
        failed" and "3 of 15 never ran" read alike and mean opposite things."""
        report = await _run(
            [_claim("failure_pattern", "fail", "boom fails",
                    "boom failed as well.", ["boom"])]
        )

        assert report.claims == []
        assert "not the agent failing" in report.dropped[0].reason

    async def test_the_harness_error_has_a_type_of_its_own(self):
        report = await _run(
            [_claim("harness_error", "warn", "boom never ran",
                    "boom errored in the harness; the pass rate excludes it.", ["boom"])]
        )

        assert [c.claim_type for c in report.claims] == ["harness_error"]

    async def test_citing_one_case_while_discussing_another_is_dropped(self):
        report = await _run(
            [_claim("failure_pattern", "fail", "routing is off",
                    "pos_implicit never loaded the skill.", ["neg_chart"])]
        )

        assert report.claims == []
        assert "not the cases discussed" in report.dropped[0].reason

    async def test_a_case_id_must_appear_in_the_claims_own_words(self):
        report = await _run(
            [_claim("failure_pattern", "fail", "a failure",
                    "Something went wrong somewhere.", ["neg_chart"])]
        )

        assert report.claims == []
        assert "not the cases discussed" in report.dropped[0].reason

    async def test_an_invented_case_id_is_dropped(self):
        report = await _run(
            [_claim("failure_pattern", "warn", "case_9 fails",
                    "case_9 failed.", ["case_9"])]
        )

        assert report.claims == []
        assert "not in this run" in report.dropped[0].reason

    async def test_severity_must_agree_with_the_claim(self):
        report = await _run(
            [_claim("failure_pattern", "pass", "all good",
                    "neg_chart is fine.", ["neg_chart"])]
        )

        assert report.claims == []
        assert "contradicts" in report.dropped[0].reason

    async def test_an_unknown_claim_type_is_dropped(self):
        report = await _run(
            [_claim("vibes", "warn", "feels slow", "neg_chart feels slow.", ["neg_chart"])]
        )

        assert report.claims == []
        assert "unknown claim_type" in report.dropped[0].reason

    async def test_a_process_defect_needs_a_process_verdict(self):
        report = await _run(
            [_claim("process_defect", "warn", "neg_chart looped",
                    "neg_chart took a long path.", ["neg_chart"])]
        )

        assert report.claims == []
        assert "no transcript-scope grader failed" in report.dropped[0].reason

    async def test_a_process_defect_backed_by_one_survives(self):
        report = await _run(
            [_claim("process_defect", "warn", "skill never loaded",
                    "pos_implicit raised not_triggered.", ["pos_implicit"])]
        )

        assert [c.claim_type for c in report.claims] == ["process_defect"]

    async def test_a_claim_with_no_citation_is_dropped(self):
        report = await _run(
            [_claim("failure_pattern", "fail", "things fail", "Some cases fail.", [])]
        )

        assert report.claims == []
        assert "cites no case" in report.dropped[0].reason

    async def test_a_duplicate_claim_is_dropped_once(self):
        one = _claim("failure_pattern", "fail", "neg_chart fails",
                     "neg_chart did not pass.", ["neg_chart"])
        report = await _run([one, dict(one)])

        assert len(report.claims) == 1
        assert "duplicate" in report.dropped[0].reason

    async def test_the_claim_cap_is_enforced(self):
        cases = [_case(f"c{i}", passed=False) for i in range(4)]
        claims = [
            _claim("failure_pattern", "fail", f"c{i} fails", f"c{i} did not pass.", [f"c{i}"])
            for i in range(4)
        ]
        report = await _run(claims, cases=cases, max_claims=2)

        assert len(report.claims) == 2
        assert any("limit" in d.reason for d in report.dropped)


# ---------------------------------------------------------------------------
# Coverage: the claim that must cite nothing
# ---------------------------------------------------------------------------


class TestCoverageClaims:
    async def test_a_coverage_claim_citing_nothing_survives(self):
        report = await _run(
            [_claim("coverage", "warn", "no multilingual cases",
                    "The suite has no non-English prompts at all.", [])]
        )

        assert [c.claim_type for c in report.claims] == ["coverage"]

    async def test_a_coverage_claim_that_names_a_case_is_a_claim_about_it(self):
        report = await _run(
            [_claim("coverage", "warn", "thin coverage",
                    "Nothing beyond neg_chart covers charting.", [])]
        )

        assert report.claims == []
        assert "claim about them" in report.dropped[0].reason


# ---------------------------------------------------------------------------
# Comparison-only claims
# ---------------------------------------------------------------------------


def _comparison():
    before = [_case("c1", passed=True), _case("c2", passed=False), _case("c3")]
    after = [_case("c1", passed=False), _case("c2", passed=True), _case("c3")]
    return (
        after,
        compare_results(
            dict(zip([c["case_id"] for c in before], _records(before), strict=True)),
            dict(zip([c["case_id"] for c in after], _records(after), strict=True)),
            label_a="v1",
            label_b="v2",
        ),
    )


class TestComparisonClaims:
    async def test_a_regression_needs_two_runs(self):
        report = await _run(
            [_claim("regression", "fail", "neg_chart regressed",
                    "neg_chart got worse.", ["neg_chart"])]
        )

        assert report.claims == []
        assert "needs two runs compared" in report.dropped[0].reason

    async def test_a_real_regression_survives_when_one_is_given(self):
        cases, comparison = _comparison()
        report = await _run(
            [_claim("regression", "fail", "c1 regressed",
                    "c1 went from pass to fail.", ["c1"])],
            cases=cases,
            comparison=comparison,
        )

        assert [c.claim_type for c in report.claims] == ["regression"]

    async def test_a_case_that_did_not_flip_cannot_be_called_a_regression(self):
        cases, comparison = _comparison()
        report = await _run(
            [_claim("regression", "fail", "c3 regressed",
                    "c3 went from pass to fail.", ["c3"])],
            cases=cases,
            comparison=comparison,
        )

        assert report.claims == []
        assert "did not flip" in report.dropped[0].reason

    async def test_an_improvement_cannot_cite_the_case_that_regressed(self):
        cases, comparison = _comparison()
        report = await _run(
            [_claim("improvement", "pass", "c1 improved",
                    "c1 now passes.", ["c1"])],
            cases=cases,
            comparison=comparison,
        )

        assert report.claims == []
        assert "did not flip" in report.dropped[0].reason


# ---------------------------------------------------------------------------
# Boundary matching on case ids
# ---------------------------------------------------------------------------


class TestIdMatching:
    async def test_a_shorter_id_is_not_found_inside_a_longer_one(self):
        """With `c1` and `c10` both in the run, a claim about `c10` must not
        read as also naming `c1` — that would fail every honest claim."""
        cases = [_case("c1", passed=False), _case("c10", passed=False)]
        report = await _run(
            [_claim("failure_pattern", "fail", "c10 fails", "c10 did not pass.", ["c10"])],
            cases=cases,
        )

        assert [c.title for c in report.claims] == ["c10 fails"]

    async def test_naming_an_extra_case_is_still_caught(self):
        cases = [_case("c1", passed=False), _case("c10", passed=False)]
        report = await _run(
            [_claim("failure_pattern", "fail", "c10 fails",
                    "c10 did not pass, and neither did c1.", ["c10"])],
            cases=cases,
        )

        assert report.claims == []
        assert "not the cases discussed" in report.dropped[0].reason


# ---------------------------------------------------------------------------
# The report itself
# ---------------------------------------------------------------------------


class TestReport:
    async def test_a_judge_that_could_not_run_is_not_a_judge_with_nothing_to_say(self):
        async def boom(prompt, schema):
            raise RuntimeError("no API key")

        report = await InsightsJudge(call=boom).run(_doc(STANDARD))

        assert report.error.endswith("no API key")
        assert report.claims == []

    async def test_an_empty_run_says_so_rather_than_judging_nothing(self):
        report = await _run([], cases=[])

        assert "no cases" in report.error

    async def test_the_drop_rate_is_reported_as_a_calibration_signal(self):
        report = await _run(
            [
                _claim("failure_pattern", "fail", "neg_chart fails",
                       "neg_chart did not pass.", ["neg_chart"]),
                _claim("failure_pattern", "fail", "pos_explicit fails",
                       "pos_explicit did not pass.", ["pos_explicit"]),
            ]
        )

        assert report.grounded_rate == pytest.approx(0.5)

    async def test_a_malformed_reply_does_not_crash_the_judge(self):
        async def junk(prompt, schema):
            return {"claims": ["not an object", 7, None]}

        report = await InsightsJudge(call=junk).run(_doc(STANDARD))

        assert report.claims == []
        assert len(report.dropped) == 3

    async def test_the_report_serializes(self):
        report = await _run(
            [_claim("failure_pattern", "fail", "neg_chart fails",
                    "neg_chart did not pass.", ["neg_chart"])]
        )

        payload = json.loads(json.dumps(report.to_dict()))

        assert payload["claims"][0]["case_ids"] == ["neg_chart"]
        assert payload["grounded_rate"] == 1.0

    async def test_cases_beyond_the_cap_are_counted_not_silently_dropped(self):
        cases = [_case(f"c{i}") for i in range(10)]
        report = await _run([], cases=cases, max_cases=4)

        assert report.cases_considered == 4
        assert report.cases_elided == 6

    async def test_an_elided_case_cannot_be_cited(self):
        """A claim can only cite what the judge was shown, so a narrowed view
        narrows the output rather than corrupting it."""
        cases = [_case("kept", passed=False)] + [_case(f"c{i}") for i in range(10)]
        report = await _run(
            [_claim("failure_pattern", "fail", "c9 fails", "c9 did not pass.", ["c9"])],
            cases=cases,
            max_cases=1,
        )

        assert report.claims == []
        assert "not in this run" in report.dropped[0].reason


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------


class TestPrompt:
    def test_the_comparison_only_types_are_withheld_without_one(self):
        prompt = build_prompt(collect_facts(_doc(STANDARD)), summary={})

        assert "regression —" not in prompt
        assert "failure_pattern —" in prompt

    def test_they_are_offered_when_a_comparison_is_given(self):
        cases, comparison = _comparison()
        prompt = build_prompt(
            collect_facts(_doc(cases), comparison), summary={}, comparison=comparison
        )

        assert "regression —" in prompt
        assert "improvement —" in prompt

    def test_the_prompt_carries_the_cases_it_will_be_checked_against(self):
        prompt = build_prompt(collect_facts(_doc(STANDARD)), summary={})

        for case_id in ("pos_explicit", "neg_chart", "pos_implicit", "boom"):
            assert case_id in prompt


# ---------------------------------------------------------------------------
# Direct grounding API
# ---------------------------------------------------------------------------


class TestGroundDirectly:
    def test_it_works_without_a_document(self):
        facts = [CaseFacts(case_id="a", status="failed", passed=False)]

        kept, dropped = ground(
            [_claim("failure_pattern", "fail", "a fails", "a did not pass.", ["a"])],
            facts,
        )

        assert len(kept) == 1 and dropped == []

    def test_every_claim_type_is_reachable(self):
        assert set(CLAIM_TYPES) == {
            "failure_pattern",
            "regression",
            "improvement",
            "process_defect",
            "harness_error",
            "coverage",
        }


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------


class TestCommand:
    @pytest.fixture
    def results(self, tmp_path):
        path = tmp_path / "results.json"
        path.write_text(
            json.dumps({"results": [{"scenario_name": "qa", "run_id": "r",
                                     "config_hash": "c", "timestamp": "2026-08-25T00:00:00",
                                     "duration_ms": 1.0, "case_results": STANDARD}]}),
            encoding="utf-8",
        )
        return path

    @pytest.fixture
    def answer(self, monkeypatch):
        """Make the judge reply with whatever a test sets on `box`."""
        box: dict[str, list] = {"claims": []}

        async def call(self, prompt, schema):
            box["prompt"] = prompt
            return {"claims": box["claims"]}

        monkeypatch.setattr(InsightsJudge, "_call_llm_structured", call)
        return box

    def _run(self, *args):
        from click.testing import CliRunner

        from compass.cli.main import cli

        return CliRunner().invoke(cli, ["insights", *args])

    def test_a_grounded_claim_is_printed(self, results, answer):
        answer["claims"] = [
            _claim("failure_pattern", "fail", "neg_chart fails",
                   "neg_chart did not pass.", ["neg_chart"])
        ]

        result = self._run(str(results))

        assert result.exit_code == 0
        assert "neg_chart fails" in result.output

    def test_a_refused_claim_is_accounted_for_without_being_shown(self, results, answer):
        answer["claims"] = [
            _claim("failure_pattern", "fail", "pos_explicit fails",
                   "pos_explicit did not pass.", ["pos_explicit"])
        ]

        result = self._run(str(results))

        assert "1 claim(s) the data refused" in result.output
        assert "pos_explicit fails" not in result.output

    def test_show_dropped_gives_the_reason(self, results, answer):
        answer["claims"] = [
            _claim("failure_pattern", "fail", "pos_explicit fails",
                   "pos_explicit did not pass.", ["pos_explicit"])
        ]

        result = self._run(str(results), "--show-dropped")

        assert "it passed" in result.output

    def test_the_json_form_carries_both_halves(self, results, answer, tmp_path):
        answer["claims"] = [
            _claim("failure_pattern", "fail", "neg_chart fails",
                   "neg_chart did not pass.", ["neg_chart"]),
            _claim("failure_pattern", "fail", "boom fails", "boom did not pass.", ["boom"]),
        ]
        out = tmp_path / "insights.json"

        result = self._run(str(results), "-o", str(out))

        assert result.exit_code == 0
        payload = json.loads(out.read_text())
        assert len(payload["claims"]) == 1
        assert len(payload["dropped"]) == 1
        assert payload["grounded_rate"] == 0.5

    def test_a_judge_that_cannot_run_exits_nonzero(self, results, monkeypatch):
        async def boom(self, prompt, schema):
            raise RuntimeError("no API key")

        monkeypatch.setattr(InsightsJudge, "_call_llm_structured", boom)

        result = self._run(str(results))

        assert result.exit_code == 1
        assert "The judge did not run" in result.output

    def test_two_files_unlock_the_comparison_claim_types(self, results, answer, tmp_path):
        before = tmp_path / "before.json"
        before.write_text(
            json.dumps({"case_results": [_case("neg_chart", passed=True),
                                         _case("pos_explicit"),
                                         _case("pos_implicit"),
                                         _case("boom")]}),
            encoding="utf-8",
        )
        answer["claims"] = [
            _claim("regression", "fail", "neg_chart regressed",
                   "neg_chart went from pass to fail.", ["neg_chart"])
        ]

        result = self._run(str(before), str(results))

        assert result.exit_code == 0
        assert "neg_chart regressed" in result.output
        assert "regression —" in answer["prompt"]
