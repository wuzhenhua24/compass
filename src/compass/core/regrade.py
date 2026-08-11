"""Offline grading — apply a scenario's graders to already-recorded transcripts.

Running and grading are separate verbs in Compass::

    compass test  scenario.yaml --trace-dir ./traces   # data plane: execute + record
    compass grade ./traces --scenario scenario.yaml    # control plane: score + re-score

A recorded trace is immutable evidence; grading only ever *adds* files under
``<trace_dir>/grades/<name>/``. The same trace can therefore be scored by
several grader sets side by side (a cheap deterministic one and an expensive
LLM judge, say), and re-scored after a rubric edit — without paying to re-run
the agent, and without the agent's non-determinism making the before/after
scores incomparable.

On-disk layout::

    traces/
      cat_on_sofa.json              # the trace, never modified
      cat_on_sofa/output.png        # artifact binaries (written by ArtifactStore)
      grades/
        <grade-set>/
          _spec.json                # snapshot of the grader spec, per case
          cat_on_sofa.json          # the grade record

Every grade record carries the fingerprint of the grader spec that produced it,
so ``compass grade`` can distinguish an up-to-date grade from one produced by an
older version of the spec and prompt for ``--regrade`` — rather than trusting a
hand-maintained version string.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from compass.core.artifact_store import _safe_filename
from compass.core.artifacts import ImageArtifact
from compass.core.fileio import atomic_write_json
from compass.core.result import CaseResult, EvalResult, EvaluatorResult, TestStatus
from compass.core.scenario import Scenario, TestCase
from compass.core.transcript import Transcript
from compass.core.trial import TaskResult, TrialResult

logger = logging.getLogger(__name__)

GRADES_DIRNAME = "grades"
SPEC_FILENAME = "_spec.json"

# Top-level files in a trace dir that are known not to be transcripts.
_NON_TRACE_FILES = {"checkpoint.json", "manifest.json"}


# ----------------------------------------------------------------------
# Grader spec + fingerprint
# ----------------------------------------------------------------------


def case_grader_spec(scenario: Scenario, case: TestCase) -> dict[str, Any]:
    """The grading contract that applies to one case.

    Everything that can move a score: the resolved grader list (scenario
    defaults + ``expected`` + explicit graders), the aggregation rules, the
    leak markers, and ``expect`` (which inverts pass/fail for negative tests).
    Deliberately excludes the agent config and the prompt — those describe how
    the evidence was produced, not how it is judged, and a trace is graded as
    the evidence it already is.
    """
    return {
        "case_id": case.id,
        "expect": case.expect,
        # ``label`` is excluded: it names a grader instance so results can tell
        # two runs of the same grader apart, and cannot move a score. Hashing it
        # would report "regraded" — the signal that a diff is not attributable
        # to the agent — for a purely cosmetic edit.
        "graders": [
            g.model_dump(mode="json", exclude={"label"})
            for g in scenario.get_graders_for_case(case)
        ],
        "aggregation": scenario.get_aggregation_for_case(case).model_dump(mode="json"),
        "leak_markers": scenario.get_leak_markers_for_case(case),
    }


def spec_fingerprint(spec: dict[str, Any]) -> str:
    """Content hash of a grader spec.

    Automatic, unlike the hand-maintained ``Grader.version``: editing a
    threshold or a rubric changes the fingerprint whether or not anyone
    remembered to bump a version string.
    """
    payload = json.dumps(spec, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def grader_fingerprint(scenario: Scenario, case: TestCase) -> str:
    """Fingerprint of the grading contract for one case.

    Stamped onto every CaseResult by both the live runner and ``compass
    grade``, so two results are known to be comparable — or known not to be —
    without trusting a hand-maintained version string.
    """
    return spec_fingerprint(case_grader_spec(scenario, case))


# ----------------------------------------------------------------------
# Trace discovery + loading
# ----------------------------------------------------------------------


@dataclass
class LoadedTrace:
    """A transcript read back from disk, plus what can be graded from it."""

    path: Path
    transcript: Transcript
    #: JSONL is an event stream, not a full-fidelity record: the ``outcome.set``
    #: event carries no ``output_data`` and no artifact payloads, so OUTCOME and
    #: BOTH scoped graders cannot be honestly re-run against it.
    supports_outcome: bool = True

    @property
    def stem(self) -> str:
        return self.path.stem

    @property
    def task_id(self) -> str:
        return self.transcript.task_id


def discover_traces(path: Path) -> list[Path]:
    """Trace files at *path*: the file itself, or the top level of a directory.

    Scanning only the top level is deliberate — it skips ``grades/``, the
    per-case artifact directories and ``.checkpoint_cases/`` without needing to
    special-case them.
    """
    path = Path(path)
    if path.is_file():
        return [path]

    return sorted(
        p
        for p in path.iterdir()
        if p.is_file()
        and p.suffix in (".json", ".jsonl")
        and p.name not in _NON_TRACE_FILES
        and not p.name.startswith(".")
    )


def load_trace(path: Path, trace_dir: Path | None = None) -> LoadedTrace | None:
    """Load one trace file, or return None if it is not a transcript.

    Raises ValueError if the file *looks* like a transcript but cannot be
    parsed — a corrupt trace is worth reporting, an unrelated JSON file is not.
    """
    path = Path(path)
    trace_dir = trace_dir or path.parent

    if path.suffix == ".jsonl":
        try:
            first = next(
                (ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()),
                "",
            )
            head = json.loads(first) if first else {}
        except (OSError, json.JSONDecodeError):
            return None
        if head.get("type") != "transcript.started":
            return None
        try:
            transcript = Transcript.load_jsonl(path)
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller
            raise ValueError(f"unreadable JSONL trace: {exc}") from exc
        supports_outcome = False
    else:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict) or "task_id" not in data or "trial_id" not in data:
            return None
        try:
            transcript = Transcript.load(path)
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller
            raise ValueError(f"unreadable trace: {exc}") from exc
        supports_outcome = True

    loaded = LoadedTrace(
        path=path, transcript=transcript, supports_outcome=supports_outcome
    )
    if supports_outcome:
        _rehydrate_artifacts(transcript, trace_dir)
    return loaded


def _rehydrate_artifacts(transcript: Transcript, trace_dir: Path) -> None:
    """Reload artifact binaries that JSON serialization could not carry.

    ``Outcome.to_dict()`` stores paths, not pixels. Without this, every image
    grader would see ``context.image is None`` on a re-graded trace and score it
    as a missing output.
    """
    outcome = transcript.outcome
    if outcome is None:
        return

    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - PIL is a hard dependency
        return

    def _open(p: Path) -> Any:
        try:
            return Image.open(p).copy()
        except Exception:  # noqa: BLE001
            logger.warning("Could not reload artifact image from %s", p)
            return None

    for art in outcome.artifacts:
        if isinstance(art, ImageArtifact) and art.image is None and art.image_path:
            art.image = _open(Path(art.image_path))

    # Legacy Outcome.image: ArtifactStore writes it to <trace_dir>/<case>/output.png
    if outcome.image is None:
        legacy = trace_dir / _safe_filename(transcript.task_id) / "output.png"
        if legacy.exists():
            outcome.image = _open(legacy)


# ----------------------------------------------------------------------
# Grade store
# ----------------------------------------------------------------------


class GradeStore:
    """Reads and writes ``<trace_dir>/grades/<name>/``.

    Grades are additive: nothing under the store ever modifies a trace, and
    multiple grade sets coexist under their own names.
    """

    def __init__(self, trace_dir: Path | str, grade_set: str):
        self.trace_dir = Path(trace_dir)
        self.grade_set = grade_set
        self.dir = self.trace_dir / GRADES_DIRNAME / grade_set

    def record_path(self, trace_stem: str) -> Path:
        return self.dir / f"{trace_stem}.json"

    def workspace_path(self, trace_stem: str) -> Path:
        """Shared grade workspace for one trace, beside its grade record."""
        return self.dir / trace_stem

    def list_evidence(self, trace_stem: str) -> list[str]:
        """Files the graders left in that trace's workspace, sorted."""
        path = self.workspace_path(trace_stem)
        if not path.is_dir():
            return []
        return sorted(p.name for p in path.iterdir() if p.is_file())

    def load_record(self, trace_stem: str) -> dict[str, Any] | None:
        path = self.record_path(trace_stem)
        if not path.exists():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Discarding unreadable grade record: %s", path)
            return None
        return record if isinstance(record, dict) else None

    def write_record(self, trace_stem: str, record: dict[str, Any]) -> Path:
        return atomic_write_json(self.record_path(trace_stem), record)

    def write_spec(self, scenario: Scenario, specs: dict[str, dict[str, Any]]) -> Path:
        """Snapshot the grader spec that produced this round of grades."""
        return atomic_write_json(
            self.dir / SPEC_FILENAME,
            {
                "grade_set": self.grade_set,
                "scenario_name": scenario.name,
                "written_at": datetime.now(timezone.utc).isoformat(),
                "cases": {
                    case_id: {
                        "fingerprint": spec_fingerprint(spec),
                        "spec": spec,
                    }
                    for case_id, spec in sorted(specs.items())
                },
            },
        )

    def clear(self) -> None:
        """Discard every grade in this set, so nothing stale survives a --regrade."""
        if self.dir.exists():
            shutil.rmtree(self.dir)


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------


@dataclass
class GradeReport:
    """Outcome of a ``compass grade`` invocation."""

    result: EvalResult
    grade_set: str
    grade_dir: Path
    #: Traces scored during this invocation.
    graded: int = 0
    #: Traces left alone because an up-to-date grade already existed.
    skipped: int = 0
    #: Reused grades whose spec fingerprint no longer matches the scenario.
    stale: int = 0
    #: Trace task_ids with no matching case in the scenario.
    unmatched: list[str] = field(default_factory=list)
    #: (path, reason) for traces that could not be graded.
    unreadable: list[tuple[str, str]] = field(default_factory=list)
    #: Cases in the scenario with no trace at all.
    missing_traces: list[str] = field(default_factory=list)

    @property
    def has_stale(self) -> bool:
        return self.stale > 0


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------


async def grade_traces(
    scenario: Scenario,
    trace_dir: Path | str,
    *,
    grade_set: str,
    case_ids: list[str] | None = None,
    regrade: bool = False,
    runner: Any | None = None,
) -> GradeReport:
    """Apply *scenario*'s graders to the transcripts recorded in *trace_dir*.

    Args:
        scenario: Supplies the graders, aggregation and expectations.
        trace_dir: Directory of trace files (or a single trace file).
        grade_set: Name of the grade set — grades land in ``grades/<name>/``.
        case_ids: Only grade these cases; None means every case.
        regrade: Discard existing grades in this set and redo them.
        runner: Optional ``Compass`` instance; one is created if omitted.

    Returns:
        A GradeReport whose ``result`` is an ordinary EvalResult — so
        ``compass analyze`` and ``compass compare`` consume re-graded output
        exactly like freshly-run output.
    """
    from compass.core.runner import Compass

    runner = runner or Compass()
    trace_path = Path(trace_dir)
    root = trace_path if trace_path.is_dir() else trace_path.parent
    store = GradeStore(root, grade_set)

    if regrade:
        store.clear()

    started = datetime.now()

    # --- load traces -------------------------------------------------
    loaded: list[LoadedTrace] = []
    unreadable: list[tuple[str, str]] = []
    for path in discover_traces(trace_path):
        try:
            trace = load_trace(path, trace_dir=root)
        except ValueError as exc:
            unreadable.append((str(path), str(exc)))
            continue
        if trace is not None:
            loaded.append(trace)

    # --- match traces to cases ---------------------------------------
    wanted = set(case_ids) if case_ids else None
    by_case: dict[str, list[LoadedTrace]] = {}
    unmatched: list[str] = []
    for trace in loaded:
        case = scenario.get_case(trace.task_id)
        if case is None:
            if trace.task_id not in unmatched:
                unmatched.append(trace.task_id)
            continue
        if wanted is not None and case.id not in wanted:
            continue
        by_case.setdefault(case.id, []).append(trace)

    for traces in by_case.values():
        traces.sort(key=lambda t: t.path.name)

    # --- grade --------------------------------------------------------
    specs: dict[str, dict[str, Any]] = {}
    case_results: list[CaseResult] = []
    graded = skipped = stale = 0

    for case in scenario.cases:
        if wanted is not None and case.id not in wanted:
            continue
        traces = by_case.get(case.id) or []
        if not traces:
            continue

        spec = case_grader_spec(scenario, case)
        fingerprint = spec_fingerprint(spec)
        specs[case.id] = spec

        needs_outcome = _spec_needs_outcome(scenario, case)
        trials: list[TrialResult] = []

        for index, trace in enumerate(traces, start=1):
            if needs_outcome and not trace.supports_outcome:
                unreadable.append(
                    (
                        str(trace.path),
                        "JSONL traces carry no outcome payload; case has "
                        "OUTCOME/BOTH scoped graders — re-record with "
                        "--trace-format json",
                    )
                )
                continue

            record = store.load_record(trace.stem)
            if record is not None:
                if record.get("spec_fingerprint") == fingerprint:
                    skipped += 1
                else:
                    stale += 1
                trials.append(_trial_from_record(record, trace, index))
                continue

            # Graders chain through this directory and whatever they leave
            # behind is kept beside the grade record as scoring evidence.
            workspace = store.workspace_path(trace.stem)
            passed, score, evaluator_results = await runner.grade_transcript(
                scenario, case, trace.transcript, workspace=workspace
            )
            record = _build_record(
                scenario=scenario,
                grade_set=grade_set,
                trace=trace,
                fingerprint=fingerprint,
                passed=passed,
                score=score,
                evaluator_results=evaluator_results,
                evidence=store.list_evidence(trace.stem),
            )
            store.write_record(trace.stem, record)
            graded += 1
            trials.append(
                TrialResult(
                    trial_id=trace.transcript.trial_id or trace.stem,
                    trial_number=index,
                    passed=passed,
                    score=score,
                    grader_results=list(evaluator_results),
                    duration_ms=trace.transcript.total_duration_ms,
                    outcome=_outcome_payload(trace.transcript),
                    transcript_id=trace.transcript.trial_id,
                )
            )

        if not trials:
            case_results.append(
                CaseResult(
                    case_id=case.id,
                    status=TestStatus.ERROR,
                    passed=False,
                    overall_score=0.0,
                    input_data=case.input.model_dump(),
                    tags=case.tags,
                    category=scenario.get_category_for_case(case),
                    error="No gradable trace for this case",
                    total_trials=0,
                    passed_trials=0,
                )
            )
            continue

        task_result = TaskResult(
            task_id=case.id,
            expect=case.expect,
            expect_reason=case.expect_reason,
            trials=trials,
            input_data=case.input.model_dump(),
        )
        case_result = runner._task_result_to_case_result(task_result, case.metrics)
        case_result.tags = case.tags
        case_result.category = scenario.get_category_for_case(case)
        # Same fingerprint the live runner stamps, so `compass compare` can tell
        # a re-grade under a changed contract from an agent-side change.
        case_result.grader_fingerprint = fingerprint
        case_results.append(case_result)

    if specs:
        store.write_spec(scenario, specs)

    missing = [
        case.id
        for case in scenario.cases
        if (wanted is None or case.id in wanted) and case.id not in by_case
    ]

    result = EvalResult(
        scenario_name=scenario.name,
        run_id=_common_field(loaded, "run_id"),
        config_hash=_common_field(loaded, "config_hash"),
        total_cases=len(case_results),
        passed_cases=sum(1 for r in case_results if r.passed),
        failed_cases=sum(1 for r in case_results if r.status == TestStatus.FAILED),
        error_cases=sum(1 for r in case_results if r.status == TestStatus.ERROR),
        case_results=case_results,
        duration_ms=(datetime.now() - started).total_seconds() * 1000,
    )

    return GradeReport(
        result=result,
        grade_set=grade_set,
        grade_dir=store.dir,
        graded=graded,
        skipped=skipped,
        stale=stale,
        unmatched=unmatched,
        unreadable=unreadable,
        missing_traces=missing,
    )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _spec_needs_outcome(scenario: Scenario, case: TestCase) -> bool:
    """Does any grader for this case need the outcome payload?"""
    from compass.graders import get_grader

    for config in scenario.get_graders_for_case(case):
        try:
            grader_cls = get_grader(config.name)
        except Exception:  # noqa: BLE001 - unknown grader: assume it needs outcome
            return True
        scope = getattr(grader_cls, "grader_scope", None)
        value = getattr(scope, "value", scope)
        if value in ("outcome", "both"):
            return True
    return False


def _outcome_payload(transcript: Transcript) -> dict[str, Any]:
    outcome = transcript.outcome
    return dict(outcome.output_data) if outcome else {}


def _build_record(
    *,
    scenario: Scenario,
    grade_set: str,
    trace: LoadedTrace,
    fingerprint: str,
    passed: bool,
    score: float,
    evaluator_results: list[EvaluatorResult],
    evidence: list[str] | None = None,
) -> dict[str, Any]:
    """The on-disk grade record: the verdict plus everything needed to audit it."""
    return {
        "grade_set": grade_set,
        "scenario_name": scenario.name,
        "graded_at": datetime.now(timezone.utc).isoformat(),
        "trace_file": trace.path.name,
        "task_id": trace.transcript.task_id,
        "trial_id": trace.transcript.trial_id,
        # Provenance of the evidence, carried over from the trace so a grade
        # record alone answers "which run produced what I scored?"
        "run_id": trace.transcript.run_id,
        "config_hash": trace.transcript.config_hash,
        # Provenance of the judgement.
        "spec_fingerprint": fingerprint,
        "passed": passed,
        "score": score,
        # Union of what the graders observed — cheap faceting without having
        # to walk into every grade_result.
        "tags": sorted({t for r in evaluator_results for t in r.tags}),
        "grade_results": [r.to_dict() for r in evaluator_results],
        # Files the graders produced while scoring — the visual/derived
        # evidence behind the verdict, which `details` cannot hold.
        "evidence": evidence or [],
    }


def _trial_from_record(
    record: dict[str, Any], trace: LoadedTrace, index: int
) -> TrialResult:
    """Rebuild a TrialResult from a stored grade, so reused grades still report."""
    return TrialResult(
        trial_id=record.get("trial_id") or trace.stem,
        trial_number=index,
        passed=bool(record.get("passed")),
        score=float(record.get("score") or 0.0),
        grader_results=[
            EvaluatorResult.from_dict(r) for r in record.get("grade_results", [])
        ],
        duration_ms=trace.transcript.total_duration_ms,
        outcome=_outcome_payload(trace.transcript),
        transcript_id=record.get("trial_id"),
    )


def _common_field(traces: list[LoadedTrace], attr: str) -> str:
    """The shared value of *attr* across traces, or "" when they disagree."""
    values = {getattr(t.transcript, attr, "") for t in traces}
    values.discard("")
    return values.pop() if len(values) == 1 else ""
