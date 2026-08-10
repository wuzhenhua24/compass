"""Tests for the static site builder (``compass site build``).

The property that matters most here is the merge: a build writes one slug and
leaves every other entry alone. That is the whole basis for aggregating runs
from separate repositories into one directory without a server.
"""

import json

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
            "total_cases", "evaluated_cases", "passed_cases",
        }


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
