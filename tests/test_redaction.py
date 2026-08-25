"""Tests for secret scrubbing on the way into a published site.

Two halves, and both matter. The scrub has to *catch* a credential in the one
place a published site really carries them — a grader's evidence and a trace's
tool-call arguments — and it has to *leave alone* the ordinary text that lives
beside them, because a page whose numbers have been eaten by an over-eager
regex is a different kind of broken.

Every key in here is a made-up shape, not a real credential.
"""

from __future__ import annotations

import json

import pytest

from compass.report.redaction import (
    PLACEHOLDER,
    env_secret_values,
    redact_json,
    redact_text,
    summarize,
)
from compass.report.site import build_site, collect_run_payload, publish_doc

# Invented, but shaped like the real thing — which is the whole point.
OPENAI = "sk-proj-AbC123dEf456GhI789jKl012MnO345"
ANTHROPIC = "sk-ant-api03-AbC123dEf456GhI789jKl012MnO345pQr"
NVIDIA = "nvapi-AbC123dEf456GhI789jKl012MnO"
GITHUB = "ghp_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789"
AWS = "AKIAIOSFODNN7EXAMPLE"
GOOGLE = "AIzaSyA1bC2dE3fG4hI5jK6lM7nO8pQ9rS0tU1vW"
SLACK = "xoxb-1234567890-AbCdEfGhIjKl"
JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftJeZ4CVPmB92K27uhbUJU1p1r_wW1gFWFO"
OPENSHIFT = "sha256~AbC123dEf456GhI789jKl012"


# ---------------------------------------------------------------------------
# What comes out
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "secret",
    [OPENAI, ANTHROPIC, NVIDIA, GITHUB, AWS, GOOGLE, SLACK, JWT, OPENSHIFT],
)
def test_a_credential_does_not_survive(secret):
    assert secret not in redact_text(f"the call was made with {secret} attached")


def test_a_bearer_header_keeps_the_scheme_and_drops_the_token():
    out = redact_text("curl -H 'Authorization: Bearer aB3dEfGhIjKlMnOpQrStUvWxYz012345'")

    assert "Bearer" in out
    assert "aB3dEfGhIjKlMnOpQrStUvWxYz012345" not in out


def test_an_in_house_token_is_caught_by_its_name():
    """No recognizable shape — only the credential-shaped variable name."""
    out = redact_text("export ACME_SERVICE_TOKEN=hunter2hunter2hunter")

    assert "hunter2hunter2hunter" not in out
    assert "ACME_SERVICE_TOKEN=" in out


def test_a_credential_shaped_field_is_caught_in_json_and_yaml():
    assert "realvaluehere123" not in redact_text('"api_key": "realvaluehere123"')
    assert "realvaluehere123" not in redact_text("  api_key: realvaluehere123")


def test_the_prefix_survives_so_the_owner_knows_what_to_rotate():
    out = redact_text(f"export ANTHROPIC_API_KEY={ANTHROPIC}")

    assert out == f"export ANTHROPIC_API_KEY=sk-{PLACEHOLDER}"


def test_one_secret_is_counted_once():
    """Overlapping patterns must not double-count — the number goes in a report."""
    _, counts = redact_json({"e": f"export ANTHROPIC_API_KEY={ANTHROPIC}"}, extra_values=())

    assert sum(counts.values()) == 1


# ---------------------------------------------------------------------------
# What stays
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "the task-granularity of this skill is fine",          # `sk-` inside a word
        "digest 9f2a1c0b7e5d4a63 for report-writer",           # lowercase hex
        '{"total_tokens": 1234567890123}',                     # a long number
        "check the token: authentication is handled upstream",  # prose, not a field
        "Authorization: Bearer $GITHUB_TOKEN",                 # a reference
        "api_key: <your-key-here>",                            # a placeholder
        "export OPENAI_API_KEY=$OPENAI_API_KEY",               # a reference again
        "cat .claude/skills/report-writer/SKILL.md",           # an ordinary command
        "https://github.com/harbor-framework/harbor",          # an ordinary URL
    ],
)
def test_ordinary_text_is_left_alone(text):
    assert redact_text(text) == text


def test_a_second_pass_changes_nothing():
    once = redact_text(f"KEY={OPENAI} and {JWT}")

    assert redact_text(once) == once


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_only_values_are_scrubbed_never_keys():
    doc = {f"api_key_{OPENAI}": "fine"}

    scrubbed, _ = redact_json(doc, extra_values=())

    assert list(scrubbed) == list(doc)


def test_a_pattern_cannot_match_across_two_json_fields():
    """Parsed structure is why `total_tokens` next to a long number is safe."""
    doc = {"token": 12345678901234567890, "note": "unrelated"}

    scrubbed, counts = redact_json(doc, extra_values=())

    assert scrubbed == doc
    assert not counts


def test_nested_lists_and_dicts_are_walked():
    doc = {"cases": [{"evidence": [{"match": f"Bearer {ANTHROPIC}"}]}]}

    scrubbed, counts = redact_json(doc, extra_values=())

    assert ANTHROPIC not in json.dumps(scrubbed)
    assert sum(counts.values()) == 1


# ---------------------------------------------------------------------------
# Known values from the environment
# ---------------------------------------------------------------------------


def test_a_known_value_is_redacted_whatever_its_shape():
    doc = {"cmd": "curl -H 'X-Acme-Auth: zzz9-inhouse-value-xyz'"}

    scrubbed, counts = redact_json(doc, extra_values=("zzz9-inhouse-value-xyz",))

    assert "zzz9-inhouse-value-xyz" not in json.dumps(scrubbed)
    assert counts["known_value"] == 1


def test_the_environment_supplies_credential_shaped_names():
    env = {
        "ACME_API_KEY": "value-from-the-environment",
        "MY_PASSWORD": "another-one-entirely",
        "PATH": "/usr/bin",
        "SHORT_TOKEN": "abc",  # below the length floor
    }

    values = env_secret_values(env)

    assert set(values) == {"value-from-the-environment", "another-one-entirely"}


def test_publishing_uses_the_environment(monkeypatch):
    monkeypatch.setenv("ACME_API_KEY", "value-from-the-environment")
    doc = {"scenarios": [], "note": "ran with value-from-the-environment"}

    published = publish_doc(doc, include_details=True)

    assert "value-from-the-environment" not in published["note"]


# ---------------------------------------------------------------------------
# The publish path
# ---------------------------------------------------------------------------


def _case_with(secret: str) -> dict:
    return {
        "case_id": "c1",
        "task_id": "c1",
        "status": "passed",
        "passed": True,
        "overall_score": 1.0,
        "best_score": 1.0,
        "duration_ms": 1000.0,
        "evaluator_results": [
            {
                "name": "skill_trigger",
                "score": 1.0,
                "passed": True,
                "grader_scope": "transcript",
                "grader_type": "code",
                "metadata": {"evidence": [{"match": f"curl -H 'Authorization: Bearer {secret}'"}]},
            }
        ],
    }


def _doc(secret: str) -> dict:
    payload = {
        "results": [
            {
                "scenario_name": "qa",
                "run_id": "run-1",
                "config_hash": "cfg-1",
                "timestamp": "2026-08-10T00:00:00",
                "duration_ms": 2000.0,
                "case_results": [_case_with(secret)],
            }
        ]
    }
    return collect_run_payload(payload, name="qa", generated="2026-08-10T00:00:00+00:00")


def test_include_details_publishes_the_evidence_without_the_credential():
    published = publish_doc(_doc(ANTHROPIC), include_details=True)
    text = json.dumps(published)

    assert ANTHROPIC not in text
    assert "curl -H" in text  # the evidence itself survives
    assert published["secrets_redacted"] == {"openai_key": 1}


def test_a_clean_document_is_not_labelled():
    published = publish_doc(_doc("nothing-secret-here"), include_details=True)

    assert "secrets_redacted" not in published


def test_a_published_trace_is_scrubbed(tmp_path):
    traces = tmp_path / "traces"
    traces.mkdir()
    (traces / "c1.json").write_text(
        json.dumps(
            {
                "task_id": "c1",
                "tool_calls": [{"input": {"command": f"export OPENAI_API_KEY={OPENAI}"}}],
                "tokens": {"total_tokens": 1234567890123},
            }
        ),
        encoding="utf-8",
    )
    (traces / "run.log").write_text(f"launching with {OPENAI}\n", encoding="utf-8")

    result = build_site(
        _doc("clean"), tmp_path / "site", slug="qa", trace_dir=traces, include_details=True
    )

    published = (tmp_path / "site/runs/qa/traces/c1.json").read_text()
    assert OPENAI not in published
    assert "1234567890123" in published  # the token count is not a credential
    assert OPENAI not in (tmp_path / "site/runs/qa/traces/run.log").read_text()
    assert result.redactions


def test_a_jsonl_trace_is_scrubbed_line_by_line(tmp_path):
    traces = tmp_path / "traces"
    traces.mkdir()
    (traces / "c1.jsonl").write_text(
        "\n".join(json.dumps({"cmd": f"KEY={s}"}) for s in (OPENAI, "harmless-value")) + "\n",
        encoding="utf-8",
    )

    build_site(_doc("clean"), tmp_path / "site", slug="qa", trace_dir=traces)

    lines = (tmp_path / "site/runs/qa/traces/c1.jsonl").read_text().splitlines()
    assert OPENAI not in lines[0]
    assert json.loads(lines[1])["cmd"] == "KEY=harmless-value"


def test_a_binary_trace_file_is_copied_untouched(tmp_path):
    traces = tmp_path / "traces"
    traces.mkdir()
    blob = b"\x89PNG\r\n\x1a\n" + OPENAI.encode()
    (traces / "shot.png").write_bytes(blob)

    build_site(_doc("clean"), tmp_path / "site", slug="qa", trace_dir=traces)

    assert (tmp_path / "site/runs/qa/traces/shot.png").read_bytes() == blob


def test_a_truncated_json_trace_falls_back_to_text_scrubbing(tmp_path):
    """A run killed mid-write still holds the command line that leaked."""
    traces = tmp_path / "traces"
    traces.mkdir()
    (traces / "c1.json").write_text(
        f'{{"tool_calls": [{{"input": {{"command": "export OPENAI_API_KEY={OPENAI}"',
        encoding="utf-8",
    )

    build_site(_doc("clean"), tmp_path / "site", slug="qa", trace_dir=traces)

    assert OPENAI not in (tmp_path / "site/runs/qa/traces/c1.json").read_text()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_a_scrub_reports_what_it_hit():
    _, counts = redact_json({"a": f"KEY={OPENAI}", "b": JWT}, extra_values=())

    line = summarize(counts)

    assert line.startswith("redacted 2 secrets")
    assert "openai_key" in line and "jwt" in line


def test_finding_nothing_says_nothing():
    _, counts = redact_json({"a": "ordinary text"}, extra_values=())

    assert summarize(counts) == ""
# ---------------------------------------------------------------------------
# The live server
# ---------------------------------------------------------------------------


def _fetch_trace(tmp_path, host: str) -> str:
    """The body `compass site serve` returns for a trace file, bound to *host*."""
    import threading
    from urllib.request import urlopen

    from compass.report.site import LiveSource, make_server

    results = tmp_path / "results.json"
    results.write_text(json.dumps(_doc("clean")), encoding="utf-8")
    traces = tmp_path / "traces"
    traces.mkdir()
    (traces / "c1.json").write_text(
        json.dumps({"cmd": f"export OPENAI_API_KEY={OPENAI}"}), encoding="utf-8"
    )

    server = make_server(
        [LiveSource(slug="qa", path=results, trace_dir=traces)], host=host, port=0
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        with urlopen(f"http://127.0.0.1:{port}/runs/qa/traces/c1.json") as reply:
            return reply.read().decode()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_serving_on_loopback_keeps_the_trace_as_it_is(tmp_path):
    """A private view of files already on this disk. Scrubbing them would only
    make `serve` disagree with `build` about what the trace says."""
    assert OPENAI in _fetch_trace(tmp_path, "127.0.0.1")


def test_serving_off_loopback_scrubs_the_trace(tmp_path):
    """Bound to an address someone else can reach, serving a raw transcript is
    publishing it — the rule `include_details` already follows."""
    assert OPENAI not in _fetch_trace(tmp_path, "0.0.0.0")
