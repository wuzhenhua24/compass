"""Tests for the live site server (``compass site serve``).

What separates serving from building is that nothing is frozen: every response
is recomputed from the files on disk, so a run still being written shows its
progress. These tests rewrite the source between requests and check that the
next response has moved.
"""

import json
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager

import pytest
from click.testing import CliRunner

from compass.cli.main import cli
from compass.report.site import (
    SCHEMA,
    LiveSource,
    is_built_site,
    is_loopback,
    make_server,
    make_static_server,
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _case(case_id, passed=True, score=1.0):
    return {
        "case_id": case_id,
        "task_id": case_id,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "overall_score": score,
        "best_score": score,
        "duration_ms": 100.0,
        "evaluator_results": [
            {
                "name": "semantic_match",
                "score": score,
                "passed": passed,
                "grader_scope": "outcome",
                "grader_type": "model",
                "metadata": {"reasoning": "quotable model output"},
            }
        ],
    }


def _write_results(path, cases):
    path.write_text(json.dumps({
        "results": [{"scenario_name": "qa", "case_results": cases}]
    }))
    return path


@contextmanager
def _running(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _get(base, path):
    with urllib.request.urlopen(base + path, timeout=5) as res:
        return res.status, res.read(), res.headers.get("Content-Type")


def _get_json(base, path):
    return json.loads(_get(base, path)[1])


@pytest.fixture
def results(tmp_path):
    return _write_results(tmp_path / "qa.json", [_case("c1")])


# ------------------------------------------------------------------
# Serving live results
# ------------------------------------------------------------------

class TestLiveServer:
    def test_serves_the_viewer_at_the_root(self, results):
        with _running(make_server([LiveSource("qa", results)])) as base:
            status, body, ctype = _get(base, "/")

            assert status == 200
            assert b"<!DOCTYPE html>" in body
            assert "text/html" in ctype

    def test_manifest_says_live(self, results):
        """The flag is the page's cue to keep polling — a built site is a
        snapshot and says the opposite."""
        with _running(make_server([LiveSource("qa", results)])) as base:
            index = _get_json(base, "/index.json")

            assert index["live"] is True
            assert index["schema"] == SCHEMA
            assert [e["slug"] for e in index["runs"]] == ["qa"]

    def test_serves_the_same_shape_build_writes(self, results):
        with _running(make_server([LiveSource("qa", results)])) as base:
            doc = _get_json(base, "/runs/qa/run.json")

            assert doc["schema"] == SCHEMA
            assert doc["run"]["pass_rate"] == 1.0
            assert doc["scenarios"][0]["cases"][0]["case_id"] == "c1"

    def test_rewriting_the_source_changes_the_next_response(self, results):
        """The whole point of serving instead of building."""
        with _running(make_server([LiveSource("qa", results)])) as base:
            assert _get_json(base, "/index.json")["runs"][0]["total_cases"] == 1

            _write_results(results, [_case("c1"), _case("c2", passed=False, score=0.0)])

            index = _get_json(base, "/index.json")
            assert index["runs"][0]["total_cases"] == 2
            assert index["runs"][0]["pass_rate"] == 0.5
            doc = _get_json(base, "/runs/qa/run.json")
            assert len(doc["scenarios"][0]["cases"]) == 2

    def test_several_runs_are_served_together(self, tmp_path):
        a = _write_results(tmp_path / "a.json", [_case("c1")])
        b = _write_results(tmp_path / "b.json", [_case("c1", passed=False, score=0.0)])

        with _running(make_server([LiveSource("a", a), LiveSource("b", b)])) as base:
            index = _get_json(base, "/index.json")

            assert [e["slug"] for e in index["runs"]] == ["a", "b"]
            assert _get_json(base, "/runs/b/run.json")["run"]["pass_rate"] == 0.0

    def test_unknown_run_is_a_404(self, results):
        with _running(make_server([LiveSource("qa", results)])) as base:
            with pytest.raises(urllib.error.HTTPError) as exc:
                _get(base, "/runs/nope/run.json")

            assert exc.value.code == 404

    def test_a_half_written_source_reports_rather_than_crashing(self, tmp_path):
        """A results file caught mid-rewrite is normal when polling."""
        broken = tmp_path / "qa.json"
        broken.write_text('{"results": [')

        with _running(make_server([LiveSource("qa", broken)])) as base:
            with pytest.raises(urllib.error.HTTPError) as exc:
                _get(base, "/index.json")

            assert exc.value.code == 503

    def test_responses_are_never_cached(self, results):
        """A cached poll would defeat the point."""
        with _running(make_server([LiveSource("qa", results)])) as base:
            with urllib.request.urlopen(base + "/index.json", timeout=5) as res:
                assert res.headers.get("Cache-Control") == "no-store"


class TestServedRedaction:
    def test_details_are_redacted_by_default(self, results):
        with _running(make_server([LiveSource("qa", results)])) as base:
            doc = _get_json(base, "/runs/qa/run.json")

            grader = doc["scenarios"][0]["cases"][0]["evaluator_results"][0]
            assert "metadata" not in grader
            assert doc["details_redacted"] is True

    def test_include_details_serves_them(self, results):
        server = make_server([LiveSource("qa", results)], include_details=True)
        with _running(server) as base:
            doc = _get_json(base, "/runs/qa/run.json")

            grader = doc["scenarios"][0]["cases"][0]["evaluator_results"][0]
            assert grader["metadata"]["reasoning"]


class TestServedTraces:
    def test_traces_are_listed_and_readable(self, tmp_path, results):
        traces = tmp_path / "traces"
        traces.mkdir()
        (traces / "c1.json").write_text('{"case_id": "c1"}')

        source = LiveSource("qa", results, trace_dir=traces)
        with _running(make_server([source])) as base:
            assert _get_json(base, "/runs/qa/run.json")["traces"] == ["c1.json"]
            status, body, ctype = _get(base, "/runs/qa/traces/c1.json")

            assert status == 200
            assert b"c1" in body
            assert "text/plain" in ctype

    def test_a_new_trace_appears_without_a_restart(self, tmp_path, results):
        """A trace lands without the results file necessarily being rewritten,
        so the listing must not ride on that file's mtime."""
        traces = tmp_path / "traces"
        traces.mkdir()

        source = LiveSource("qa", results, trace_dir=traces)
        with _running(make_server([source])) as base:
            assert _get_json(base, "/runs/qa/run.json")["traces"] == []

            (traces / "c1.json").write_text("{}")

            assert _get_json(base, "/runs/qa/run.json")["traces"] == ["c1.json"]
            assert _get_json(base, "/index.json")["runs"][0]["traces"] == 1

    def test_traces_cannot_be_escaped(self, tmp_path, results):
        """The trace path comes off the wire; it may not reach outside."""
        traces = tmp_path / "traces"
        traces.mkdir()
        (tmp_path / "secret.txt").write_text("not yours")

        source = LiveSource("qa", results, trace_dir=traces)
        with _running(make_server([source])) as base:
            with pytest.raises(urllib.error.HTTPError) as exc:
                _get(base, "/runs/qa/traces/..%2Fsecret.txt")

            assert exc.value.code == 404


# ------------------------------------------------------------------
# Serving a built site
# ------------------------------------------------------------------

class TestStaticServer:
    @pytest.fixture
    def built(self, tmp_path):
        results = _write_results(tmp_path / "qa.json", [_case("c1")])
        site = tmp_path / "site"
        CliRunner().invoke(cli, ["site", "build", str(results), "-o", str(site)])
        return site

    def test_is_built_site_detects_one(self, built, tmp_path):
        assert is_built_site(built)
        assert not is_built_site(tmp_path)
        assert not is_built_site(tmp_path / "qa.json")

    def test_serves_the_built_files(self, built):
        with _running(make_static_server(built)) as base:
            assert b"<!DOCTYPE html>" in _get(base, "/index.html")[1]
            index = _get_json(base, "/index.json")

            assert index["live"] is False
            assert index["runs"][0]["slug"] == "qa"
            assert _get_json(base, "/runs/qa/run.json")["run"]["pass_rate"] == 1.0

    def test_a_rebuild_shows_up_without_a_restart(self, built, tmp_path):
        with _running(make_static_server(built)) as base:
            assert _get_json(base, "/index.json")["runs"][0]["total_cases"] == 1

            results = _write_results(tmp_path / "qa.json", [_case("c1"), _case("c2")])
            CliRunner().invoke(cli, ["site", "build", str(results), "-o", str(built)])

            assert _get_json(base, "/index.json")["runs"][0]["total_cases"] == 2


# ------------------------------------------------------------------
# Binding
# ------------------------------------------------------------------

class TestIsLoopback:
    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "127.0.0.5", ""])
    def test_local_addresses(self, host):
        assert is_loopback(host)

    @pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "::", "example.com"])
    def test_everything_else_is_publishing(self, host):
        assert not is_loopback(host)


# ------------------------------------------------------------------
# CLI wiring
# ------------------------------------------------------------------

class TestSiteServeCommand:
    """The command blocks in serve_forever, so these check what it decides
    before it gets there."""

    def _invoke(self, monkeypatch, args):
        captured = {}

        def fake_make_server(sources, *, host, port, include_details=False):
            captured.update(
                sources=list(sources), host=host, port=port,
                include_details=include_details, static=False,
            )
            return _StubServer(port)

        def fake_static(root, *, host, port):
            captured.update(root=root, host=host, port=port, static=True)
            return _StubServer(port)

        monkeypatch.setattr("compass.report.site.make_server", fake_make_server)
        monkeypatch.setattr("compass.report.site.make_static_server", fake_static)
        result = CliRunner().invoke(cli, ["site", "serve", *args])
        return result, captured

    def test_serves_a_results_file(self, tmp_path, monkeypatch):
        results = _write_results(tmp_path / "qa.json", [_case("c1")])

        out, captured = self._invoke(monkeypatch, [str(results)])

        assert out.exit_code == 0, out.output
        assert captured["static"] is False
        assert [s.slug for s in captured["sources"]] == ["qa"]
        assert captured["host"] == "127.0.0.1"
        assert captured["port"] == 7001

    def test_a_directory_becomes_one_run_per_file(self, tmp_path, monkeypatch):
        _write_results(tmp_path / "alpha.json", [_case("c1")])
        _write_results(tmp_path / "beta.json", [_case("c1")])

        out, captured = self._invoke(monkeypatch, [str(tmp_path)])

        assert out.exit_code == 0, out.output
        assert sorted(s.slug for s in captured["sources"]) == ["alpha", "beta"]

    def test_a_built_site_is_served_statically(self, tmp_path, monkeypatch):
        results = _write_results(tmp_path / "qa.json", [_case("c1")])
        site = tmp_path / "site"
        CliRunner().invoke(cli, ["site", "build", str(results), "-o", str(site)])

        out, captured = self._invoke(monkeypatch, [str(site)])

        assert out.exit_code == 0, out.output
        assert captured["static"] is True

    def test_localhost_serves_details_but_says_nothing_alarming(self, tmp_path, monkeypatch):
        results = _write_results(tmp_path / "qa.json", [_case("c1")])

        out, captured = self._invoke(monkeypatch, [str(results)])

        assert captured["include_details"] is True
        assert "redacted" not in out.output

    def test_binding_elsewhere_redacts_and_warns(self, tmp_path, monkeypatch):
        """Serving off-loopback is publishing, and gets publishing's default."""
        results = _write_results(tmp_path / "qa.json", [_case("c1")])

        out, captured = self._invoke(monkeypatch, [str(results), "--host", "0.0.0.0"])

        assert captured["include_details"] is False
        assert "redacted" in out.output
        assert "anyone who can reach this machine" in out.output

    def test_include_details_overrides_the_bind_default(self, tmp_path, monkeypatch):
        results = _write_results(tmp_path / "qa.json", [_case("c1")])

        out, captured = self._invoke(
            monkeypatch, [str(results), "--host", "0.0.0.0", "--include-details"]
        )

        assert captured["include_details"] is True
        assert "anyone who can reach this machine" in out.output

    def test_redact_overrides_on_localhost(self, tmp_path, monkeypatch):
        results = _write_results(tmp_path / "qa.json", [_case("c1")])

        _, captured = self._invoke(monkeypatch, [str(results), "--redact"])

        assert captured["include_details"] is False

    def test_slug_with_several_sources_is_rejected(self, tmp_path, monkeypatch):
        _write_results(tmp_path / "alpha.json", [_case("c1")])
        _write_results(tmp_path / "beta.json", [_case("c1")])

        out, _ = self._invoke(monkeypatch, [str(tmp_path), "--slug", "x"])

        assert out.exit_code == 1
        assert "take a single results file" in out.output

    def test_an_empty_directory_is_refused(self, tmp_path, monkeypatch):
        empty = tmp_path / "empty"
        empty.mkdir()

        out, _ = self._invoke(monkeypatch, [str(empty)])

        assert out.exit_code == 1
        assert "No results files found" in out.output


class _StubServer:
    """Stands in for a real server: reports its port, then stops."""

    def __init__(self, port):
        self.server_port = port

    def serve_forever(self):
        raise KeyboardInterrupt

    def server_close(self):
        pass
