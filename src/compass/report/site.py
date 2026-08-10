"""The report data layer, and the static site built on top of it.

``collect_run`` turns a run into a plain, versioned JSON document. Everything
that shows a run to a person renders that document and nothing else — the HTML
report, and the static site this module builds.

The site is deliberately dumb::

    site/
      index.html              the viewer (one self-contained file)
      index.json              the manifest: one entry per run, merged in place
      runs/<slug>/run.json    the document for that run
      runs/<slug>/traces/…    its transcripts and artifacts, if published

``build_site`` writes exactly one slug and leaves every other entry in
``index.json`` untouched. That single property is what makes cross-repository
aggregation work without a server: several projects can build into the same
output directory — a shared ``gh-pages`` branch, one bucket prefix — and the
manifest accumulates. There is no central node to run, because the merge point
is a file that can be rewritten idempotently.

``make_server`` serves that same shape without building it, recomputing each
response from the results files on disk. The viewer cannot tell the two apart
except by the manifest's ``live`` flag, which is its cue to keep polling — so
one page shows both a finished run and a run still being written.

Conventions the document commits to:

- Scores and rates are fractions in ``0..1``, never percentages. Formatting is
  a viewer's job.
- Harness errors are excluded from every denominator, matching
  ``EvalResult.pass_rate``: a crashed adapter is not the agent getting it wrong,
  and counting it as one lets infrastructure flakiness masquerade as a
  regression.
- A grader that was skipped, or that ran without producing a number, is left
  out of averages rather than counted as zero. "Not measured" is not
  "measured as zero".
- Every aggregate is derived from the case rows, at every level. Stored
  counters are never trusted, so a hand-written results file and a
  Compass-written one summarize the same way.

Case rows are a near-copy of ``EvalResult.to_dict()``'s case records, so
anything that reads a Compass results file (``analyzer.iter_case_dicts`` and
everything downstream of it) reads these too. The differences are deliberate:
the ``grade_results`` alias is dropped (it duplicates ``evaluator_results``
verbatim, and a published document pays for every byte twice), and each row
gains its scenario name plus the two scope axes.
"""

import json
import re
import shutil
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, SimpleHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from ipaddress import ip_address
from mimetypes import guess_type
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from compass.core.result import EvalResult, TestStatus
from compass.report.analyzer import iter_case_dicts

#: Bumped when the document shape changes incompatibly. A viewer reads this
#: before anything else, so an old page can refuse a new document instead of
#: rendering it wrong.
SCHEMA = "compass.run/1"

#: How many previous builds of a slug the manifest remembers. Only the summary
#: numbers are kept — enough for a trend line, and small enough that a site
#: aggregating many runs stays a manifest rather than an archive.
DEFAULT_HISTORY = 20

_OUTCOME_SCOPES = ("outcome", "both")
_TRANSCRIPT_SCOPES = ("transcript", "both")
_ERROR = TestStatus.ERROR.value

#: Summary fields carried from an entry into the history trail. ``contract``
#: rides along on purpose: a trend line whose points were graded under
#: different rules is not a trend, and the viewer has to be able to say where
#: the break is rather than drawing straight through it.
_SNAPSHOT_FIELDS = (
    "generated",
    "pass_rate",
    "average_score",
    "best_of_k_score",
    "total_cases",
    "evaluated_cases",
    "passed_cases",
    "contract",
)


def now_iso() -> str:
    """UTC timestamp, second precision — unambiguous once a page is shared."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _measured(grader: Mapping[str, Any]) -> bool:
    """Whether a grader record carries a usable measurement.

    Mirrors ``EvaluatorResult.scored`` but reads the serialized record, so the
    same rule applies to results loaded back from disk.
    """
    return not grader.get("skipped", False) and grader.get("score") is not None


def scope_scores(graders: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Project one case's grader scores onto the Outcome and Transcript axes.

    A ``both``-scope grader contributes to both axes. Axis presence is reported
    separately from the average: a run where every outcome grader legitimately
    scored 0.0 still *has* an outcome axis, and a viewer must be able to tell
    that apart from a run with no outcome graders at all.
    """
    outcome: list[float] = []
    transcript: list[float] = []

    for grader in graders:
        if not _measured(grader):
            continue
        scope = grader.get("grader_scope", "outcome")
        score = float(grader["score"])
        if scope in _OUTCOME_SCOPES:
            outcome.append(score)
        if scope in _TRANSCRIPT_SCOPES:
            transcript.append(score)

    return {
        "outcome_score": _mean(outcome),
        "transcript_score": _mean(transcript),
        "has_outcome": bool(outcome),
        "has_transcript": bool(transcript),
    }


def iter_cases(doc: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    """Every case row in the document, in scenario order.

    Rows live under their scenario so the document has exactly one home for
    each; this is the flat view a chart or a filter wants.
    """
    for scenario in doc.get("scenarios", []):
        yield from scenario.get("cases", [])


def category_rows(cases: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Per-category breakdown, sorted by category name.

    Uncategorized cases are omitted rather than bucketed under a placeholder —
    an empty result means "nobody labelled anything", which is a viewer's cue
    to hide the table entirely.
    """
    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for case in cases:
        category = case.get("category")
        if category:
            buckets[category].append(case)

    rows = []
    for category in sorted(buckets):
        group = buckets[category]
        stats = _stats(group)
        rows.append(
            {
                "category": category,
                "total": stats["total_cases"],
                "errors": stats["error_cases"],
                "evaluated": stats["evaluated_cases"],
                "passed": stats["passed_cases"],
                "failed": stats["failed_cases"],
                "pass_rate": stats["pass_rate"],
                "average_score": stats["average_score"],
            }
        )
    return rows


def _stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Counts and means over case rows.

    Every level of the document aggregates here — run, scenario, category — so
    the three can never tell different stories about the same cases.
    """
    evaluated = [r for r in rows if r.get("status") != _ERROR]
    passed = sum(1 for r in evaluated if r.get("passed"))
    return {
        "total_cases": len(rows),
        "passed_cases": passed,
        "failed_cases": len(evaluated) - passed,
        "error_cases": len(rows) - len(evaluated),
        "evaluated_cases": len(evaluated),
        "pass_rate": _rate(passed, len(evaluated)),
        "average_score": _mean([float(r.get("overall_score") or 0.0) for r in evaluated]),
        "best_of_k_score": _mean([float(r.get("best_score") or 0.0) for r in evaluated]),
    }


def contract_id(
    cases: Sequence[Mapping[str, Any]], scenarios: Sequence[Mapping[str, Any]]
) -> str:
    """A short id for the grading rules this run was scored under.

    Compass already refuses to call scores comparable across different grading
    contracts (see ``CaseResult.grader_fingerprint``). A trend line is a
    comparison stretched over time, so it inherits that rule: when this id
    changes between builds, the points on either side were not measured the
    same way and a viewer must say so rather than drawing through it.

    Derived from the per-case fingerprints, falling back to the scenario config
    hashes. Empty when a results file carries neither — an unknown contract is
    reported as unknown, never as "unchanged".
    """
    parts = sorted({str(c.get("grader_fingerprint") or "") for c in cases} - {""})
    if not parts:
        parts = sorted({str(s.get("config_hash") or "") for s in scenarios} - {""})
    if not parts:
        return ""
    return sha256("\n".join(parts).encode()).hexdigest()[:12]


def _case_row(record: Mapping[str, Any], scenario: str) -> dict[str, Any]:
    row = dict(record)
    row.pop("grade_results", None)  # verbatim alias of evaluator_results
    row["scenario"] = scenario
    row.setdefault("case_id", row.get("task_id", ""))
    row.setdefault("task_id", row["case_id"])
    row.update(scope_scores(row.get("evaluator_results") or []))
    return row


def _scenario_block(record: Mapping[str, Any]) -> dict[str, Any]:
    """One scenario's summary and its case rows."""
    name = str(record.get("scenario_name") or "")
    rows = [_case_row(case, name) for case in record.get("case_results") or []]
    return {
        "name": name,
        "run_id": record.get("run_id", ""),
        "config_hash": record.get("config_hash", ""),
        "timestamp": record.get("timestamp", ""),
        "duration_ms": float(record.get("duration_ms") or 0.0),
        **_stats(rows),
        "cases": rows,
    }


def _scenario_records(payload: Any) -> list[Mapping[str, Any]]:
    """Find the scenario records in any results shape Compass writes.

    Mirrors ``analyzer.iter_case_dicts``, but keeps the scenario grouping that
    a report needs; only a bare case list has no grouping to keep, and becomes
    a single unnamed scenario.
    """
    if isinstance(payload, Mapping):
        if isinstance(payload.get("results"), list):
            records = []
            for entry in payload["results"]:
                records.extend(_scenario_records(entry))
            return records
        if isinstance(payload.get("case_results"), list):
            return [payload]
    return [{"scenario_name": "", "case_results": iter_case_dicts(payload)}]


def collect_run_payload(
    payload: Any,
    *,
    name: str = "",
    generated: str | None = None,
) -> dict[str, Any]:
    """Build the document from a serialized results payload.

    Accepts every shape ``analyze`` and ``compare`` accept, so a results file
    written by any Compass command can be published without being replayed.
    """
    scenarios = [_scenario_block(record) for record in _scenario_records(payload)]
    cases = [case for scenario in scenarios for case in scenario["cases"]]

    return {
        "schema": SCHEMA,
        "generated": generated or now_iso(),
        "run": {
            "name": name,
            "scenarios": len(scenarios),
            # Means over cases, not over scenario means: scenarios differ in
            # size, so averaging their averages would over-weight small ones.
            **_stats(cases),
            "duration_ms": sum(s["duration_ms"] for s in scenarios),
            "contract": contract_id(cases, scenarios),
        },
        "scenarios": scenarios,
        "categories": category_rows(cases),
        "scopes": {
            "outcome": any(c["has_outcome"] for c in cases),
            "transcript": any(c["has_transcript"] for c in cases),
        },
    }


def collect_run(
    results: Sequence[EvalResult],
    *,
    name: str = "",
    generated: str | None = None,
) -> dict[str, Any]:
    """Everything a report needs about one run, as one plain document.

    Args:
        results: The run's scenario results.
        name: Optional label for the run as a whole (a viewer's heading, and
            the name a static site indexes it under).
        generated: Override the timestamp — for reproducible output.

    Returns:
        A JSON-serializable dict; see the module docstring for the conventions
        its numbers follow.
    """
    return collect_run_payload(
        {"results": [result.to_dict() for result in results]},
        name=name,
        generated=generated,
    )


# --- static site ---------------------------------------------------------


_SLUG_STRIP = re.compile(r"[^a-z0-9._-]+")


def slugify(text: str) -> str:
    """Turn a name into a safe single path segment.

    A slug becomes a directory under the site, so it may not carry separators
    or resolve upwards; anything that would is folded to ``-``.
    """
    slug = _SLUG_STRIP.sub("-", text.strip().lower()).strip("-.")
    return slug or "run"


def app_html() -> str:
    """The viewer, as shipped with the package."""
    return (files("compass.report") / "app.html").read_text(encoding="utf-8")


def publish_doc(doc: Mapping[str, Any], *, include_details: bool = False) -> dict[str, Any]:
    """The copy of a document that is safe to publish.

    Grader ``metadata`` is the one free-form field in a case row: it carries
    whatever a grader chose to put in ``details``, which routinely means model
    output, prompts, or rendered reasoning. Publishing is not local debugging,
    so it comes out by default and goes back in only when asked for.
    """
    published: dict[str, Any] = json.loads(json.dumps(doc))
    if include_details:
        return published

    for case in iter_cases(published):
        for grader in case.get("evaluator_results") or []:
            grader.pop("metadata", None)
    published["details_redacted"] = True
    return published


def index_entry(slug: str, doc: Mapping[str, Any]) -> dict[str, Any]:
    """One run's row in the manifest — the index page renders only these."""
    run = dict(doc["run"])
    run.pop("name", None)
    return {
        "slug": slug,
        "name": doc["run"].get("name") or slug,
        "generated": doc["generated"],
        **run,
        "scopes": doc.get("scopes", {}),
        "categories": [row["category"] for row in doc.get("categories", [])],
        "traces": 0,
        "history": [],
    }


def _snapshot(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {key: entry.get(key) for key in _SNAPSHOT_FIELDS}


def load_index(site_dir: Path) -> list[dict[str, Any]]:
    """Existing manifest entries, or none.

    A corrupt manifest is treated as absent rather than fatal: refusing to
    build because some other writer left half a file behind would make the
    aggregation property useless in exactly the situation it exists for.
    """
    index_file = Path(site_dir) / "index.json"
    if not index_file.exists():
        return []
    try:
        data = json.loads(index_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    entries = data.get("runs") if isinstance(data, Mapping) else None
    return [e for e in entries if isinstance(e, dict)] if isinstance(entries, list) else []


@dataclass
class BuildResult:
    """What one ``build_site`` call did."""

    slug: str
    site_dir: Path
    run_dir: Path
    entry: dict[str, Any]
    #: Every run now in the manifest, this one included.
    runs: int
    trace_files: int = 0
    trace_bytes: int = 0
    #: Slug's previous builds still remembered, for the trend line.
    history: int = 0
    published_paths: list[str] = field(default_factory=list)


def _copy_traces(source: Path, target: Path) -> tuple[list[str], int]:
    """Copy a trace directory into the site, reporting what was published."""
    shutil.copytree(source, target)
    paths, total = [], 0
    for path in sorted(target.rglob("*")):
        if path.is_file():
            paths.append(path.relative_to(target).as_posix())
            total += path.stat().st_size
    return paths, total


def build_site(
    doc: Mapping[str, Any],
    site_dir: str | Path,
    *,
    slug: str,
    trace_dir: str | Path | None = None,
    include_details: bool = False,
    history: int = DEFAULT_HISTORY,
) -> BuildResult:
    """Add or refresh one run in a static site directory.

    Only this slug is written. Entries for other runs are carried through
    untouched, which is what lets separate repositories build into one shared
    output directory and have the manifest accumulate.

    Args:
        doc: Document from ``collect_run`` / ``collect_run_payload``.
        site_dir: Output directory; created if absent, merged into if not.
        slug: Name to index this run under. Slugified before use.
        trace_dir: Transcripts and artifacts to publish alongside the run.
            Opt-in: a trace carries the full prompts and outputs of a run, so
            it is never published unless named.
        include_details: Keep grader ``metadata`` in the published document.
        history: How many previous builds of this slug to remember.

    Returns:
        A ``BuildResult`` describing what was written.
    """
    site_dir = Path(site_dir)
    slug = slugify(slug)

    run_dir = (site_dir / "runs" / slug).resolve()
    runs_root = (site_dir / "runs").resolve()
    if not run_dir.is_relative_to(runs_root):
        raise ValueError(f"unsafe slug: {slug!r}")

    published = publish_doc(doc, include_details=include_details)

    site_dir.mkdir(parents=True, exist_ok=True)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    trace_paths: list[str] = []
    trace_bytes = 0
    if trace_dir is not None:
        trace_paths, trace_bytes = _copy_traces(Path(trace_dir), run_dir / "traces")
    published["traces"] = trace_paths

    (run_dir / "run.json").write_text(
        json.dumps(published, ensure_ascii=False), encoding="utf-8"
    )

    entry = index_entry(slug, published)
    entry["traces"] = len(trace_paths)

    entries = load_index(site_dir)
    previous = next((e for e in entries if e.get("slug") == slug), None)
    if previous and history > 0:
        trail = [_snapshot(previous), *(previous.get("history") or [])]
        entry["history"] = trail[:history]

    entries = [e for e in entries if e.get("slug") != slug]
    entries.append(entry)
    entries.sort(key=lambda e: str(e.get("slug", "")))

    (site_dir / "index.json").write_text(
        json.dumps(
            {"schema": SCHEMA, "generated": now_iso(), "live": False, "runs": entries},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (site_dir / "index.html").write_text(app_html(), encoding="utf-8")

    return BuildResult(
        slug=slug,
        site_dir=site_dir,
        run_dir=run_dir,
        entry=entry,
        runs=len(entries),
        trace_files=len(trace_paths),
        trace_bytes=trace_bytes,
        history=len(entry["history"]),
        published_paths=trace_paths,
    )


# --- live server ---------------------------------------------------------


@dataclass
class LiveSource:
    """One results file the server watches.

    ``build_site`` freezes a run; this points at the file it was frozen from,
    so a page can show a run that is still being written.
    """

    slug: str
    path: Path
    name: str = ""
    trace_dir: Path | None = None


def _list_traces(trace_dir: Path | None) -> list[str]:
    if trace_dir is None or not trace_dir.is_dir():
        return []
    return sorted(
        p.relative_to(trace_dir).as_posix() for p in trace_dir.rglob("*") if p.is_file()
    )


def _read_source(
    source: LiveSource, cache: dict[str, tuple[int, dict[str, Any]]]
) -> dict[str, Any]:
    """The document for one source, recomputed whenever its file changes.

    Parsing is keyed on mtime rather than a timer: polling has to be cheap
    enough for the page to do it every few seconds, and a run that has not
    advanced should cost nothing to re-serve. The trace listing is *not*
    cached with it — a trace lands without the results file necessarily being
    rewritten, and a transcript you cannot open until something else changes
    is worse than a directory scan.
    """
    key = str(source.path)
    mtime = source.path.stat().st_mtime_ns
    hit = cache.get(key)
    if hit and hit[0] == mtime:
        doc = hit[1]
    else:
        payload = json.loads(source.path.read_text(encoding="utf-8"))
        doc = collect_run_payload(payload, name=source.name or source.slug)
        cache[key] = (mtime, doc)

    doc["traces"] = _list_traces(source.trace_dir)
    return doc


def make_server(
    sources: Sequence[LiveSource],
    *,
    host: str = "127.0.0.1",
    port: int = 7001,
    include_details: bool = False,
) -> ThreadingHTTPServer:
    """A server that renders results straight from disk.

    Serves the same shape ``build_site`` writes, so the viewer cannot tell the
    difference — except that the manifest says ``live``, which is the page's
    cue to keep polling. Every response is recomputed from the files, so a run
    that is still being written shows its progress.

    Returns the server without starting it; call ``serve_forever()``.
    """
    by_slug = {source.slug: source for source in sources}
    cache: dict[str, tuple[int, dict[str, Any]]] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802  (BaseHTTPRequestHandler's API)
            path = unquote(self.path.split("?")[0])
            try:
                if path in ("/", "/index.html"):
                    return self._reply(200, app_html().encode(), "text/html; charset=utf-8")
                if path == "/index.json":
                    return self._reply_json(self._manifest())
                if path.startswith("/runs/"):
                    return self._serve_run(path.removeprefix("/runs/"))
            except FileNotFoundError:
                return self._reply(404, b"gone from disk", "text/plain; charset=utf-8")
            except (json.JSONDecodeError, OSError) as e:
                # A results file being rewritten is a normal thing to catch
                # mid-poll; say so rather than dropping the connection.
                return self._reply(503, str(e).encode(), "text/plain; charset=utf-8")
            self._reply(404, b"not found", "text/plain; charset=utf-8")

        def _manifest(self) -> dict[str, Any]:
            entries = []
            for slug, source in sorted(by_slug.items()):
                doc = _read_source(source, cache)
                entry = index_entry(slug, doc)
                entry["traces"] = len(doc.get("traces") or [])
                entries.append(entry)
            return {"schema": SCHEMA, "generated": now_iso(), "live": True, "runs": entries}

        def _serve_run(self, rest: str) -> None:
            slug, _, tail = rest.partition("/")
            source = by_slug.get(slug)
            if source is None:
                return self._reply(404, b"no such run", "text/plain; charset=utf-8")
            if tail == "run.json":
                doc = _read_source(source, cache)
                return self._reply_json(publish_doc(doc, include_details=include_details))
            if tail.startswith("traces/") and source.trace_dir:
                return self._serve_trace(source.trace_dir, tail.removeprefix("traces/"))
            self._reply(404, b"not found", "text/plain; charset=utf-8")

        def _serve_trace(self, trace_dir: Path, rest: str) -> None:
            root = trace_dir.resolve()
            target = (root / rest).resolve()
            if not target.is_relative_to(root) or not target.is_file():
                return self._reply(404, b"no such trace", "text/plain; charset=utf-8")
            # Transcripts are JSON/JSONL; render them inline rather than
            # prompting a download the reader did not ask for.
            ctype = (
                "text/plain; charset=utf-8"
                if target.suffix in (".json", ".jsonl", ".yaml", ".yml", ".txt", ".log")
                else guess_type(target.name)[0] or "application/octet-stream"
            )
            self._reply(200, target.read_bytes(), ctype)

        def _reply_json(self, data: Mapping[str, Any]) -> None:
            self._reply(
                200, json.dumps(data, ensure_ascii=False).encode(), "application/json"
            )

        def _reply(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: Any) -> None:
            """Silence per-request logging — the page polls every few seconds."""

    return ThreadingHTTPServer((host, port), Handler)


def make_static_server(
    root: str | Path, *, host: str = "127.0.0.1", port: int = 7001
) -> ThreadingHTTPServer:
    """Serve an already-built site directory.

    Browsers will not fetch JSON from ``file://``, so a built site needs a
    server even to be looked at locally. Files are read per request, so a
    concurrent ``site build`` shows up on the next reload.
    """
    directory = str(Path(root).resolve())

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=directory, **kwargs)

        def end_headers(self) -> None:
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def log_message(self, *args: Any) -> None:
            """Silence per-request logging."""

    return ThreadingHTTPServer((host, port), Handler)


def is_built_site(path: str | Path) -> bool:
    """Whether a directory is a site rather than a pile of results files."""
    path = Path(path)
    return path.is_dir() and (path / "index.json").is_file() and (path / "runs").is_dir()


def is_loopback(host: str) -> bool:
    """Whether binding this host keeps the server on this machine.

    Serving to the loopback interface is local debugging; serving to anything
    else is publishing, and the two deserve different defaults.
    """
    if host in ("", "localhost"):
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False
