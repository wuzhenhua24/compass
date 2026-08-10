"""Checkpoint management for resumable evaluations.

Enables long-running evaluation runs to be interrupted and resumed
without re-executing already-completed cases.

Storage layout::

    {checkpoint_dir}/
        checkpoint.json          # RunCheckpoint state index
        .checkpoint_cases/       # Individual CaseResult files
            {case_id}.json
        .checkpoint_trials/      # Individual TrialResult files
            {case_id}__t{n}.json

Two granularities on purpose: a finished case resumes from its CaseResult,
while a multi-trial case interrupted midway resumes from the attempts it
already made — the alternative is paying for those samples twice.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from compass.core.result import CaseResult, EvaluatorResult, TestStatus
from compass.core.trial import TrialResult

logger = logging.getLogger(__name__)

# Internal directory name for checkpoint case results
_CASES_DIR = ".checkpoint_cases"
# Individual trials, so an interrupted multi-trial case resumes from the
# attempts it already made instead of discarding them
_TRIALS_DIR = ".checkpoint_trials"
_CHECKPOINT_FILE = "checkpoint.json"


def scenario_fingerprint(scenario: Any) -> str:
    """Generate a deterministic hash of scenario configuration.

    Covers everything that affects *how a case is executed and scored*: agent
    config, grader configs, defaults (trials/timeout), default aggregation, and
    per-case input, graders, aggregation, trial count and expect.
    Does NOT cover metadata, tags, or descriptions so cosmetic edits
    don't invalidate checkpoints.
    """
    hasher = hashlib.sha256()
    # Agent config
    hasher.update(
        json.dumps(scenario.agent.model_dump(), sort_keys=True).encode()
    )
    # Default graders
    hasher.update(
        json.dumps(
            [g.model_dump() for g in scenario.default_graders], sort_keys=True
        ).encode()
    )
    # Defaults (trials, timeout, environment) — changing trial count changes
    # pass@k / pass^k, so it must invalidate the checkpoint.
    hasher.update(
        json.dumps(scenario.defaults.model_dump(), sort_keys=True).encode()
    )
    # Default aggregation
    hasher.update(
        json.dumps(
            scenario.default_aggregation.model_dump(), sort_keys=True
        ).encode()
    )
    # Each case: id + input + graders + aggregation + trials + expect
    for case in scenario.cases:
        hasher.update(case.id.encode())
        hasher.update(
            json.dumps(case.input.model_dump(), sort_keys=True).encode()
        )
        hasher.update(
            json.dumps(
                [g.model_dump() for g in case.graders], sort_keys=True
            ).encode()
        )
        # Aggregation and trial count change the scoring verdict, so a resume
        # after editing them must not silently reuse stale case results.
        hasher.update(
            json.dumps(case.aggregation.model_dump(), sort_keys=True).encode()
        )
        hasher.update(str(case.trials).encode())
        hasher.update(case.expect.encode())
    return hasher.hexdigest()[:16]


def _safe_filename(case_id: str) -> str:
    """Sanitize case_id for use as filename."""
    return case_id.replace("/", "_").replace("\\", "_").replace(" ", "_")


# ── Data structures ──


@dataclass
class RunCheckpoint:
    """Persistent state of an evaluation run."""

    run_id: str
    scenario_fingerprint: str
    scenario_name: str
    total_cases: list[str]  # Ordered list of all case IDs
    completed_cases: dict[str, dict[str, Any]]  # case_id → {"file": ...}
    # case_id → {trial_number (as str): filename} for cases still in progress
    completed_trials: dict[str, dict[str, str]] = field(default_factory=dict)
    status: str = "in_progress"  # "in_progress" | "completed"
    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    @property
    def remaining_case_ids(self) -> list[str]:
        return [
            cid for cid in self.total_cases if cid not in self.completed_cases
        ]

    @property
    def progress(self) -> str:
        return f"{len(self.completed_cases)}/{len(self.total_cases)}"


# ── Serialization helpers ──


def _case_result_to_dict(result: CaseResult) -> dict[str, Any]:
    """Serialize a CaseResult to a JSON-safe dict."""
    return {
        "case_id": result.case_id,
        "status": result.status.value,
        "passed": result.passed,
        "overall_score": result.overall_score,
        "evaluator_results": [er.to_dict() for er in result.evaluator_results],
        "input_data": result.input_data,
        "output_data": result.output_data,
        "duration_ms": result.duration_ms,
        "error": result.error,
        # Classification — must be persisted or resumed runs lose their
        # tags/category and break per-category / per-tag analysis.
        "tags": result.tags,
        "category": result.category,
        "timestamp": result.timestamp.isoformat(),
        "total_trials": result.total_trials,
        "passed_trials": result.passed_trials,
        "error_trials": result.error_trials,
        "trial_metrics": result.trial_metrics,
        "grader_fingerprint": result.grader_fingerprint,
    }


def _trial_result_to_dict(trial: TrialResult) -> dict[str, Any]:
    """Serialize a TrialResult to a JSON-safe dict."""
    return {
        "trial_id": trial.trial_id,
        "trial_number": trial.trial_number,
        "passed": trial.passed,
        "score": trial.score,
        "grader_results": [
            er.to_dict() if isinstance(er, EvaluatorResult) else er
            for er in trial.grader_results
        ],
        "duration_ms": trial.duration_ms,
        "error": trial.error,
        "timestamp": trial.timestamp.isoformat(),
        "outcome": trial.outcome,
        "environment": trial.environment,
        "transcript_id": trial.transcript_id,
    }


def _dict_to_trial_result(data: dict[str, Any]) -> TrialResult:
    """Deserialize a TrialResult from a dict."""
    kwargs: dict[str, Any] = dict(
        trial_id=data["trial_id"],
        trial_number=data["trial_number"],
        passed=data["passed"],
        # An absent score means "not measured" — preserve the null rather than
        # resurrecting it as a zero on resume.
        score=data.get("score"),
        grader_results=[
            EvaluatorResult.from_dict(er) for er in data.get("grader_results", [])
        ],
        duration_ms=data.get("duration_ms", 0.0),
        error=data.get("error"),
        outcome=data.get("outcome", {}),
        environment=data.get("environment", {}),
        transcript_id=data.get("transcript_id"),
    )
    ts = data.get("timestamp")
    if ts:
        try:
            kwargs["timestamp"] = datetime.fromisoformat(ts)
        except (ValueError, TypeError):
            pass
    return TrialResult(**kwargs)


def _dict_to_case_result(data: dict[str, Any]) -> CaseResult:
    """Deserialize a CaseResult from a dict."""
    # from_dict keeps new EvaluatorResult fields (skipped, skip_reason, a null
    # score) alive across a resume instead of quietly defaulting them.
    evaluator_results = [
        EvaluatorResult.from_dict(er) for er in data.get("evaluator_results", [])
    ]

    kwargs: dict[str, Any] = dict(
        case_id=data["case_id"],
        status=TestStatus(data["status"]),
        passed=data["passed"],
        overall_score=data["overall_score"],
        evaluator_results=evaluator_results,
        input_data=data.get("input_data", {}),
        output_data=data.get("output_data", {}),
        duration_ms=data.get("duration_ms", 0.0),
        error=data.get("error"),
        tags=data.get("tags", []),
        category=data.get("category", ""),
        total_trials=data.get("total_trials", 1),
        passed_trials=data.get("passed_trials", 0),
        error_trials=data.get("error_trials", 0),
        trial_metrics=data.get("trial_metrics"),
        grader_fingerprint=data.get("grader_fingerprint", ""),
    )
    # Restore the original timestamp when present (older checkpoints omit it).
    ts = data.get("timestamp")
    if ts:
        try:
            kwargs["timestamp"] = datetime.fromisoformat(ts)
        except (ValueError, TypeError):
            pass
    return CaseResult(**kwargs)


# ── CheckpointStore ──


class CheckpointStore:
    """Manages checkpoint persistence for a single evaluation run.

    Each completed case is written to its own JSON file immediately,
    providing crash-safe incremental progress.  A top-level index
    (``checkpoint.json``) tracks overall run state.
    """

    def __init__(self, checkpoint_dir: Path):
        self._dir = checkpoint_dir
        self._cases_dir = checkpoint_dir / _CASES_DIR
        self._trials_dir = checkpoint_dir / _TRIALS_DIR

    # ── Creation ──

    @classmethod
    def create(
        cls,
        checkpoint_dir: Path,
        run_id: str,
        fingerprint: str,
        scenario_name: str,
        case_ids: list[str],
    ) -> "CheckpointStore":
        """Initialise a fresh checkpoint for a new run."""
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        store = cls(checkpoint_dir)
        store._cases_dir.mkdir(exist_ok=True)

        cp = RunCheckpoint(
            run_id=run_id,
            scenario_fingerprint=fingerprint,
            scenario_name=scenario_name,
            total_cases=case_ids,
            completed_cases={},
            completed_trials={},
        )
        store._write_index(cp)
        return store

    # ── Loading / querying ──

    @classmethod
    def load(cls, checkpoint_dir: Path) -> "CheckpointStore | None":
        """Load an existing checkpoint.  Returns *None* if none exists."""
        if not (checkpoint_dir / _CHECKPOINT_FILE).exists():
            return None
        return cls(checkpoint_dir)

    def get_checkpoint(self) -> RunCheckpoint:
        """Read the current checkpoint index."""
        data = json.loads(
            (self._dir / _CHECKPOINT_FILE).read_text(encoding="utf-8")
        )
        return RunCheckpoint(**data)

    def validate_scenario(self, fingerprint: str) -> bool:
        """Return True if the stored fingerprint matches *fingerprint*."""
        return self.get_checkpoint().scenario_fingerprint == fingerprint

    # ── Recording results ──

    def save_case_result(self, case_id: str, result: CaseResult) -> None:
        """Persist a single completed case (called after each case finishes)."""
        filename = f"{_safe_filename(case_id)}.json"
        case_file = self._cases_dir / filename
        case_file.write_text(
            json.dumps(
                _case_result_to_dict(result),
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        # Update the index atomically
        cp = self.get_checkpoint()
        cp.completed_cases[case_id] = {"file": filename}
        cp.updated_at = datetime.now().isoformat()
        self._write_index(cp)

    def save_trial_result(self, case_id: str, trial: TrialResult) -> None:
        """Persist one finished trial (called as each attempt completes).

        Trial-level granularity is what makes an interrupted multi-trial run
        resumable *without* throwing away the attempts it already paid for.
        """
        self._trials_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{_safe_filename(case_id)}__t{trial.trial_number}.json"
        (self._trials_dir / filename).write_text(
            json.dumps(
                _trial_result_to_dict(trial),
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        cp = self.get_checkpoint()
        cp.completed_trials.setdefault(case_id, {})[str(trial.trial_number)] = filename
        cp.updated_at = datetime.now().isoformat()
        self._write_index(cp)

    def load_trials(self) -> dict[str, list[TrialResult]]:
        """Load previously recorded trials, keyed by case id and ordered."""
        cp = self.get_checkpoint()
        trials: dict[str, list[TrialResult]] = {}

        for case_id, entries in cp.completed_trials.items():
            loaded: list[TrialResult] = []
            for filename in entries.values():
                trial_file = self._trials_dir / filename
                if not trial_file.exists():
                    logger.warning(
                        "Checkpoint references missing trial file %s for case %s",
                        trial_file,
                        case_id,
                    )
                    continue
                data = json.loads(trial_file.read_text(encoding="utf-8"))
                loaded.append(_dict_to_trial_result(data))
            if loaded:
                trials[case_id] = sorted(loaded, key=lambda t: t.trial_number)

        return trials

    def load_completed_results(self) -> dict[str, CaseResult]:
        """Load all previously completed CaseResults from disk."""
        cp = self.get_checkpoint()
        results: dict[str, CaseResult] = {}

        for case_id, meta in cp.completed_cases.items():
            case_file = self._cases_dir / meta["file"]
            if not case_file.exists():
                logger.warning(
                    "Checkpoint references missing file %s for case %s",
                    case_file,
                    case_id,
                )
                continue
            data = json.loads(case_file.read_text(encoding="utf-8"))
            results[case_id] = _dict_to_case_result(data)

        return results

    def mark_completed(self) -> None:
        """Mark the run as fully completed."""
        cp = self.get_checkpoint()
        cp.status = "completed"
        cp.updated_at = datetime.now().isoformat()
        self._write_index(cp)

    # ── Cleanup ──

    def clean(self) -> int:
        """Remove checkpoint index and all cached case files.

        Returns the number of files deleted.
        """
        deleted = 0

        # Remove individual case and trial files
        for cache_dir in (self._cases_dir, self._trials_dir):
            if cache_dir.is_dir():
                for f in cache_dir.iterdir():
                    f.unlink()
                    deleted += 1
                cache_dir.rmdir()

        # Remove checkpoint index
        index_file = self._dir / _CHECKPOINT_FILE
        if index_file.exists():
            index_file.unlink()
            deleted += 1

        return deleted

    @property
    def dir(self) -> Path:
        """The checkpoint directory path."""
        return self._dir

    # ── Internal ──

    def _write_index(self, cp: RunCheckpoint) -> None:
        index_file = self._dir / _CHECKPOINT_FILE
        index_file.write_text(
            json.dumps(cp.__dict__, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


# ── Discovery ──


def find_checkpoints(search_dir: Path, recursive: bool = True) -> list[CheckpointStore]:
    """Find all checkpoint.json files under *search_dir*.

    Args:
        search_dir: Root directory to search.
        recursive: If True, search subdirectories recursively.

    Returns:
        List of CheckpointStore instances, one per discovered checkpoint.
    """
    pattern = f"**/{_CHECKPOINT_FILE}" if recursive else _CHECKPOINT_FILE
    stores: list[CheckpointStore] = []
    for cp_file in sorted(search_dir.glob(pattern)):
        stores.append(CheckpointStore(cp_file.parent))
    return stores
