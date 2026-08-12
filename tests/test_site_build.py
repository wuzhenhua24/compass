"""Tests for the static site builder (``compass site build``).

The property that matters most here is the merge: a build writes one slug and
leaves every other entry alone. That is the whole basis for aggregating runs
from separate repositories into one directory without a server.
"""

import json

import pytest
from click.testing import CliRunner

from compass.cli.main import cli
from compass.report.site import (
    SCHEMA,
    app_html,
    build_site,
    collect_run_payload,
    load_index,
    publish_doc,
    slugify,
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _case(case_id, passed=True, status=None, score=1.0, **extra):
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
                "name": "semantic_match",
                "score": score,
                "passed": passed,
                "grader_scope": "outcome",
                "grader_type": "model",
                "metadata": {"reasoning": "the model said something quotable"},
            }
        ],
        **extra,
    }


def _payload(name="qa", cases=None):
    return {
        "results": [
            {
                "scenario_name": name,
                "run_id": "run-1",
                "config_hash": "cfg-1",
                "timestamp": "2026-08-10T00:00:00",
                "duration_ms": 2000.0,
                "case_results": cases if cases is not None else [_case("c1")],
            }
        ]
    }


def _doc(name="qa", cases=None, generated="2026-08-10T00:00:00+00:00"):
    return collect_run_payload(_payload(name, cases), name=name, generated=generated)


def _index(site_dir):
    return json.loads((site_dir / "index.json").read_text())


# ------------------------------------------------------------------
# Reading serialized results
# ------------------------------------------------------------------

class TestCollectRunPayload:
    def test_reads_a_test_report_bundle(self):
        doc = collect_run_payload(_payload())

        assert doc["schema"] == SCHEMA
        assert doc["run"]["total_cases"] == 1
        assert doc["scenarios"][0]["name"] == "qa"
        assert doc["scenarios"][0]["run_id"] == "run-1"

    def test_reads_a_single_eval_result(self):
        doc = collect_run_payload(
            {"scenario_name": "solo", "case_results": [_case("c1"), _case("c2", passed=False)]}
        )

        assert [s["name"] for s in doc["scenarios"]] == ["solo"]
        assert doc["run"]["passed_cases"] == 1
        assert doc["run"]["failed_cases"] == 1

    def test_reads_a_bare_case_list(self):
        """No scenario grouping to keep — it becomes one unnamed scenario."""
        doc = collect_run_payload([_case("c1"), _case("c2")])

        assert len(doc["scenarios"]) == 1
        assert doc["scenarios"][0]["name"] == ""
        assert doc["run"]["total_cases"] == 2

    def test_aggregates_are_derived_not_trusted(self):
        """A stored counter that disagrees with the cases does not win."""
        doc = collect_run_payload(
            {
                "scenario_name": "lying",
                "passed_cases": 99,
                "total_cases": 99,
                "case_results": [_case("c1"), _case("c2", passed=False)],
            }
        )

        assert doc["run"]["total_cases"] == 2
        assert doc["run"]["passed_cases"] == 1
        assert doc["run"]["pass_rate"] == 0.5


# ------------------------------------------------------------------
# Redaction
# ------------------------------------------------------------------

class TestMultiTrialCasesPublishTheMean:
    """A published grader number must be the mean over trials, not one sample.

    ``evaluator_results`` holds one trial's detail by design. Publishing a score
    straight from it put a *sampled* number next to ``overall_score``, which is
    the mean — so the two disagreed. Seen on a real 3-trial run whose agent
    edited ``tests/`` on one attempt: the site published ``state_delta 1.00``,
    a clean integrity gate, while ``grader_summary`` in the same document said
    0.667.
    """

    @staticmethod
    def _multi_trial_case():
        return _case(
            "c1",
            score=0.9,
            total_trials=3,
            passed_trials=2,
            evaluator_results=[
                {
                    "name": "state_delta",
                    # The sampled trial happened to be a clean one.
                    "score": 1.0,
                    "weight": 1.0,
                    "weighted_score": 1.0,
                    "passed": True,
                    "gate": True,
                    "grader_scope": "transcript",
                    "grader_type": "code",
                },
                {
                    "name": "integration_test",
                    "score": 1.0,
                    "weight": 1.0,
                    "weighted_score": 1.0,
                    "passed": True,
                    "grader_scope": "outcome",
                    "grader_type": "code",
                },
                {
                    "name": "integration_test",
                    "score": 1.0,
                    "weight": 1.0,
                    "weighted_score": 1.0,
                    "passed": True,
                    "grader_scope": "outcome",
                    "grader_type": "code",
                },
            ],
            grader_summary={
                "state_delta": {
                    "score_mean": 2 / 3,
                    "pass_fraction": 2 / 3,
                    "trials": 3,
                    "metrics": {},
                },
                "integration_test": {
                    "score_mean": 1.0,
                    "pass_fraction": 1.0,
                    "trials": 3,
                    "metrics": {},
                },
                "integration_test#2": {
                    "score_mean": 0.5,
                    "pass_fraction": 0.0,
                    "trials": 3,
                    "metrics": {},
                },
            },
        )

    def _case_row(self):
        doc = _doc(cases=[self._multi_trial_case()])
        return doc["scenarios"][0]["cases"][0]

    def test_breakdown_reports_the_trial_mean(self):
        row = self._case_row()
        assert row["breakdown"]["state_delta"] == pytest.approx(2 / 3)

    def test_the_grader_row_carries_the_mean_and_its_denominator(self):
        row = self._case_row()
        delta = next(
            g for g in row["evaluator_results"] if g["name"] == "state_delta"
        )
        assert delta["score"] == pytest.approx(2 / 3)
        assert delta["pass_fraction"] == pytest.approx(2 / 3)
        assert delta["trials"] == 3

    def test_unlabelled_repeats_keep_their_positional_key(self):
        """Two `integration_test` graders are two signals, not one."""
        row = self._case_row()
        assert row["breakdown"]["integration_test"] == pytest.approx(1.0)
        assert row["breakdown"]["integration_test#2"] == pytest.approx(0.5)

    def test_scope_scores_follow_the_averaged_numbers(self):
        row = self._case_row()
        assert row["outcome_score"] == pytest.approx(0.75)  # (1.0 + 0.5) / 2
        assert row["transcript_score"] == pytest.approx(2 / 3)

    def test_a_single_trial_case_is_passed_through_untouched(self):
        """No grader_summary and no trials to average — nothing to correct, and
        nothing invented either: the row keeps exactly the fields it arrived
        with."""
        case = _case("c1", score=0.4, breakdown={"semantic_match": 0.4})
        row = _doc(cases=[case])["scenarios"][0]["cases"][0]

        assert row["breakdown"] == {"semantic_match": 0.4}
        assert row["evaluator_results"] == case["evaluator_results"]


class TestPublishDoc:
    def test_grader_metadata_is_dropped_by_default(self):
        published = publish_doc(_doc())
        grader = published["scenarios"][0]["cases"][0]["evaluator_results"][0]

        assert "metadata" not in grader
        assert published["details_redacted"] is True

    def test_include_details_keeps_it(self):
        published = publish_doc(_doc(), include_details=True)
        grader = published["scenarios"][0]["cases"][0]["evaluator_results"][0]

        assert grader["metadata"]["reasoning"]
        assert "details_redacted" not in published

    def test_source_document_is_not_mutated(self):
        doc = _doc()
        publish_doc(doc)

        assert doc["scenarios"][0]["cases"][0]["evaluator_results"][0]["metadata"]

    def test_scores_survive_redaction(self):
        """Redaction removes free-form payloads, not the measurements."""
        published = publish_doc(_doc())
        grader = published["scenarios"][0]["cases"][0]["evaluator_results"][0]

        assert grader["score"] == 1.0
        assert published["run"]["pass_rate"] == 1.0


# ------------------------------------------------------------------
# Slugs
# ------------------------------------------------------------------

class TestSlugify:
    def test_normalizes_names(self):
        assert slugify("Image Evals / nightly") == "image-evals-nightly"
        assert slugify("QA_2026") == "qa_2026"

    def test_folds_path_separators_and_traversal(self):
        """A slug becomes a directory name — it may not escape the site."""
        assert "/" not in slugify("../../etc/passwd")
        assert ".." not in slugify("..")
        assert slugify("../..") == "run"

    def test_never_empty(self):
        assert slugify("") == "run"
        assert slugify("!!!") == "run"


# ------------------------------------------------------------------
# Building and merging
# ------------------------------------------------------------------

class TestBuildSite:
    def test_writes_the_site_layout(self, tmp_path):
        result = build_site(_doc(), tmp_path / "site", slug="qa")

        site = tmp_path / "site"
        assert (site / "index.html").exists()
        assert (site / "index.json").exists()
        assert (site / "runs" / "qa" / "run.json").exists()
        assert result.slug == "qa"
        assert result.runs == 1

    def test_run_json_is_the_published_document(self, tmp_path):
        build_site(_doc(), tmp_path, slug="qa")
        doc = json.loads((tmp_path / "runs" / "qa" / "run.json").read_text())

        assert doc["schema"] == SCHEMA
        assert doc["run"]["pass_rate"] == 1.0
        assert doc["details_redacted"] is True

    def test_index_entry_matches_the_document(self, tmp_path):
        cases = [_case("c1"), _case("c2", passed=False, score=0.0)]
        build_site(_doc(cases=cases), tmp_path, slug="qa")
        entry = _index(tmp_path)["runs"][0]

        assert entry["slug"] == "qa"
        assert entry["total_cases"] == 2
        assert entry["passed_cases"] == 1
        assert entry["pass_rate"] == 0.5
        assert entry["scopes"] == {"outcome": True, "transcript": False}

    def test_other_runs_are_left_untouched(self, tmp_path):
        """The aggregation property: two repositories, one directory."""
        build_site(_doc(name="repo-a"), tmp_path, slug="repo-a")
        build_site(_doc(name="repo-b"), tmp_path, slug="repo-b")

        slugs = [e["slug"] for e in _index(tmp_path)["runs"]]
        assert slugs == ["repo-a", "repo-b"]
        assert (tmp_path / "runs" / "repo-a" / "run.json").exists()
        assert (tmp_path / "runs" / "repo-b" / "run.json").exists()

    def test_rebuilding_a_slug_replaces_only_that_entry(self, tmp_path):
        build_site(_doc(name="repo-a"), tmp_path, slug="repo-a")
        build_site(_doc(name="repo-b"), tmp_path, slug="repo-b")
        build_site(
            _doc(name="repo-a", cases=[_case("c1", passed=False, score=0.0)]),
            tmp_path,
            slug="repo-a",
        )

        entries = {e["slug"]: e for e in _index(tmp_path)["runs"]}
        assert len(entries) == 2
        assert entries["repo-a"]["pass_rate"] == 0.0
        assert entries["repo-b"]["pass_rate"] == 1.0

    def test_stale_files_do_not_survive_a_rebuild(self, tmp_path):
        build_site(_doc(), tmp_path, slug="qa")
        stale = tmp_path / "runs" / "qa" / "stale.json"
        stale.write_text("{}")

        build_site(_doc(), tmp_path, slug="qa")
        assert not stale.exists()

    def test_a_corrupt_index_is_treated_as_absent(self, tmp_path):
        """Refusing to build because another writer left half a file behind
        would break aggregation in exactly the case it exists for."""
        tmp_path.mkdir(exist_ok=True)
        (tmp_path / "index.json").write_text("{not json")

        result = build_site(_doc(), tmp_path, slug="qa")
        assert result.runs == 1
        assert load_index(tmp_path)[0]["slug"] == "qa"

    def test_load_index_on_a_fresh_directory(self, tmp_path):
        assert load_index(tmp_path / "nothing-here") == []


class TestHistory:
    def test_previous_builds_become_a_trail(self, tmp_path):
        build_site(_doc(generated="2026-08-01T00:00:00+00:00"), tmp_path, slug="qa")
        build_site(
            _doc(
                cases=[_case("c1", passed=False, score=0.0)],
                generated="2026-08-02T00:00:00+00:00",
            ),
            tmp_path,
            slug="qa",
        )
        entry = _index(tmp_path)["runs"][0]

        assert entry["pass_rate"] == 0.0
        assert len(entry["history"]) == 1
        assert entry["history"][0]["pass_rate"] == 1.0
        assert entry["history"][0]["generated"] == "2026-08-01T00:00:00+00:00"

    def test_trail_is_newest_first_and_bounded(self, tmp_path):
        for day in range(1, 6):
            build_site(
                _doc(generated=f"2026-08-0{day}T00:00:00+00:00"),
                tmp_path,
                slug="qa",
                history=2,
            )
        entry = _index(tmp_path)["runs"][0]

        assert [h["generated"] for h in entry["history"]] == [
            "2026-08-04T00:00:00+00:00",
            "2026-08-03T00:00:00+00:00",
        ]

    def test_history_zero_keeps_none(self, tmp_path):
        build_site(_doc(), tmp_path, slug="qa", history=0)
        build_site(_doc(), tmp_path, slug="qa", history=0)

        assert _index(tmp_path)["runs"][0]["history"] == []

    def test_history_only_carries_summary_numbers(self, tmp_path):
        """A trend line, not an archive — no case rows in the manifest."""
        build_site(_doc(), tmp_path, slug="qa")
        build_site(_doc(), tmp_path, slug="qa")
        snapshot = _index(tmp_path)["runs"][0]["history"][0]

        assert "cases" not in snapshot
        assert "scenarios" not in snapshot
        assert set(snapshot) == {
            "generated", "pass_rate", "average_score", "best_of_k_score",
            "total_cases", "evaluated_cases", "passed_cases", "contract",
        }


class TestContract:
    """A trend line is a comparison stretched over time, so it inherits
    Compass's rule that scores are only comparable under one grading
    contract."""

    def test_same_grading_gives_the_same_id(self):
        cases = [_case("c1", grader_fingerprint="abc"), _case("c2", grader_fingerprint="def")]

        assert _doc(cases=cases)["run"]["contract"] == _doc(cases=cases)["run"]["contract"]

    def test_a_changed_fingerprint_changes_the_id(self):
        before = _doc(cases=[_case("c1", grader_fingerprint="abc")])
        after = _doc(cases=[_case("c1", grader_fingerprint="xyz")])

        assert before["run"]["contract"] != after["run"]["contract"]

    def test_order_does_not_matter(self):
        a = _doc(cases=[_case("c1", grader_fingerprint="abc"),
                        _case("c2", grader_fingerprint="def")])
        b = _doc(cases=[_case("c2", grader_fingerprint="def"),
                        _case("c1", grader_fingerprint="abc")])

        assert a["run"]["contract"] == b["run"]["contract"]

    def test_falls_back_to_the_scenario_config_hash(self):
        """Older results predate per-case fingerprints but still identify the
        configuration they ran under."""
        doc = collect_run_payload({
            "results": [{"scenario_name": "qa", "config_hash": "cfg-1",
                         "case_results": [_case("c1")]}]
        })

        assert doc["run"]["contract"]

    def test_unknown_is_reported_as_unknown(self):
        """Never as "unchanged" — that would draw a straight line through a
        break nobody can see."""
        doc = collect_run_payload({"results": [{"scenario_name": "qa",
                                               "case_results": [_case("c1")]}]})

        assert doc["run"]["contract"] == ""

    def test_history_records_the_contract_of_each_build(self, tmp_path):
        build_site(_doc(cases=[_case("c1", grader_fingerprint="abc")]), tmp_path, slug="qa")
        build_site(_doc(cases=[_case("c1", grader_fingerprint="xyz")]), tmp_path, slug="qa")
        entry = _index(tmp_path)["runs"][0]

        assert entry["contract"] != entry["history"][0]["contract"]
        assert entry["history"][0]["contract"]


class TestTraces:
    def test_traces_are_opt_in(self, tmp_path):
        """A transcript carries the full prompts and outputs of a run, so it
        is never published unless named."""
        result = build_site(_doc(), tmp_path / "site", slug="qa")

        assert result.trace_files == 0
        assert not (tmp_path / "site" / "runs" / "qa" / "traces").exists()
        doc = json.loads((tmp_path / "site" / "runs" / "qa" / "run.json").read_text())
        assert doc["traces"] == []

    def test_named_traces_are_copied_and_listed(self, tmp_path):
        traces = tmp_path / "traces"
        (traces / "nested").mkdir(parents=True)
        (traces / "c1.json").write_text('{"case_id": "c1"}')
        (traces / "nested" / "c2.json").write_text('{"case_id": "c2"}')

        result = build_site(_doc(), tmp_path / "site", slug="qa", trace_dir=traces)

        assert result.trace_files == 2
        assert result.trace_bytes > 0
        assert (tmp_path / "site" / "runs" / "qa" / "traces" / "c1.json").exists()
        doc = json.loads((tmp_path / "site" / "runs" / "qa" / "run.json").read_text())
        assert doc["traces"] == ["c1.json", "nested/c2.json"]
        assert _index(tmp_path / "site")["runs"][0]["traces"] == 2


class TestViewer:
    def test_app_html_is_packaged(self):
        html = app_html()

        assert html.startswith("<!DOCTYPE html>")
        assert "index.json" in html

    def test_viewer_and_builder_agree_on_the_schema(self):
        """The page refuses data it cannot read — that check is worthless if
        the constant drifts."""
        assert f'SCHEMA = "{SCHEMA}"' in app_html()


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

class TestSiteBuildCommand:
    def _write_results(self, tmp_path, name="results", cases=None):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(_payload(cases=cases)))
        return path

    def test_builds_from_a_results_file(self, tmp_path):
        results = self._write_results(tmp_path)
        site = tmp_path / "site"

        out = CliRunner().invoke(cli, ["site", "build", str(results), "-o", str(site)])

        assert out.exit_code == 0, out.output
        assert (site / "index.json").exists()
        assert _index(site)["runs"][0]["slug"] == "results"

    def test_slug_defaults_to_the_file_stem(self, tmp_path):
        results = self._write_results(tmp_path, name="Nightly Image Evals")
        site = tmp_path / "site"

        CliRunner().invoke(cli, ["site", "build", str(results), "-o", str(site)])

        assert _index(site)["runs"][0]["slug"] == "nightly-image-evals"

    def test_slug_option_overrides(self, tmp_path):
        results = self._write_results(tmp_path)
        site = tmp_path / "site"

        CliRunner().invoke(
            cli, ["site", "build", str(results), "-o", str(site), "--slug", "custom"]
        )

        assert _index(site)["runs"][0]["slug"] == "custom"

    def test_several_files_become_several_runs(self, tmp_path):
        a = self._write_results(tmp_path, name="alpha")
        b = self._write_results(tmp_path, name="beta")
        site = tmp_path / "site"

        out = CliRunner().invoke(cli, ["site", "build", str(a), str(b), "-o", str(site)])

        assert out.exit_code == 0, out.output
        assert [e["slug"] for e in _index(site)["runs"]] == ["alpha", "beta"]

    def test_slug_with_several_files_is_rejected(self, tmp_path):
        a = self._write_results(tmp_path, name="alpha")
        b = self._write_results(tmp_path, name="beta")

        out = CliRunner().invoke(
            cli, ["site", "build", str(a), str(b), "-o", str(tmp_path / "s"), "--slug", "x"]
        )

        assert out.exit_code == 1
        assert "--slug takes a single results file" in out.output

    def test_redaction_is_reported(self, tmp_path):
        results = self._write_results(tmp_path)

        out = CliRunner().invoke(
            cli, ["site", "build", str(results), "-o", str(tmp_path / "site")]
        )

        assert "redacted" in out.output

    def test_publishing_traces_says_so(self, tmp_path):
        results = self._write_results(tmp_path)
        traces = tmp_path / "traces"
        traces.mkdir()
        (traces / "c1.json").write_text("{}")

        out = CliRunner().invoke(cli, [
            "site", "build", str(results), "-o", str(tmp_path / "site"),
            "--trace-dir", str(traces),
        ])

        assert out.exit_code == 0, out.output
        assert "prompts and model output" in out.output

    def test_empty_results_are_refused(self, tmp_path):
        empty = tmp_path / "empty.json"
        empty.write_text(json.dumps({"results": []}))

        out = CliRunner().invoke(
            cli, ["site", "build", str(empty), "-o", str(tmp_path / "site")]
        )

        assert out.exit_code == 1
        assert "Nothing to publish" in out.output

    def test_unreadable_results_fail_loudly(self, tmp_path):
        broken = tmp_path / "broken.json"
        broken.write_text("{not json")

        out = CliRunner().invoke(
            cli, ["site", "build", str(broken), "-o", str(tmp_path / "site")]
        )

        assert out.exit_code == 1
        assert "Could not read" in out.output
