"""Artifact binary storage and baseline management.

Provides persistent storage of artifact binaries (images, code files, text)
alongside transcript JSON files, and supports baseline management for
regression testing.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from compass.core.artifacts import (
    CodeArtifact,
    ImageArtifact,
    TextArtifact,
)
from compass.core.fileio import atomic_path, atomic_write_json, atomic_write_text

if TYPE_CHECKING:
    from PIL import Image

    from compass.core.transcript import Transcript

logger = logging.getLogger(__name__)


@dataclass
class BaselineComparison:
    """Result of comparing current trial artifacts with a baseline."""

    identical: bool
    baseline_path: Path | None
    current_path: Path | None
    hash_match: bool | None = None
    details: dict[str, Any] = field(default_factory=dict)


class ArtifactStore:
    """Manages persistent storage of artifact binaries."""

    def __init__(self, base_dir: Path | str):
        self.base_dir = Path(base_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_trial_artifacts(
        self,
        transcript: Transcript,
        case_id: str,
    ) -> list[str]:
        """Save all artifact binaries from a trial.

        Creates a directory ``{base_dir}/{safe_case_id}/`` and writes each
        artifact to disk.  Returns a list of saved file paths (relative to
        *base_dir*).

        Args:
            transcript: The Transcript whose outcome artifacts should be saved.
            case_id: Case identifier used to build the directory name.

        Returns:
            List of relative paths to saved files (empty if nothing saved).
        """
        outcome = transcript.outcome
        if outcome is None:
            return []

        has_legacy_image = outcome.image is not None
        has_artifacts = bool(outcome.artifacts)

        if not has_legacy_image and not has_artifacts:
            return []

        safe_id = _safe_filename(case_id)
        artifact_dir = self.base_dir / safe_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        saved: list[str] = []

        # Legacy image (Outcome.image)
        if outcome.image is not None:
            path = artifact_dir / "output.png"
            self.save_image(outcome.image, path)
            saved.append(str(path.relative_to(self.base_dir)))

        # Typed artifacts
        for i, artifact in enumerate(outcome.artifacts):
            if isinstance(artifact, ImageArtifact) and artifact.image is not None:
                fname = f"artifact_{i}_image.png"
                path = artifact_dir / fname
                self.save_image(artifact.image, path)
                artifact.image_path = str(path)
                saved.append(str(path.relative_to(self.base_dir)))

            elif isinstance(artifact, CodeArtifact):
                for j, gf in enumerate(artifact.files):
                    safe_name = _safe_filename(Path(gf.path).name) if gf.path else f"file_{j}"
                    fname = f"artifact_{i}_code_{j}_{safe_name}"
                    path = artifact_dir / fname
                    atomic_write_text(path, gf.content)
                    saved.append(str(path.relative_to(self.base_dir)))

            elif isinstance(artifact, TextArtifact) and artifact.content:
                fname = f"artifact_{i}_text.txt"
                path = artifact_dir / fname
                atomic_write_text(path, artifact.content)
                saved.append(str(path.relative_to(self.base_dir)))

        # Write manifest
        if saved:
            manifest = {
                "case_id": case_id,
                "trial_id": transcript.trial_id,
                "timestamp": _get_timestamp(),
                "artifacts": saved,
                "image_hash": outcome.image_hash,
            }
            atomic_write_json(artifact_dir / "manifest.json", manifest)

        return saved

    @staticmethod
    def save_image(image: Image.Image, path: Path) -> None:
        """Save a PIL Image to *path* as PNG, atomically."""
        with atomic_path(path, suffix=".png") as tmp:
            image.save(str(tmp), format="PNG")

    # ------------------------------------------------------------------
    # Grade workspace
    # ------------------------------------------------------------------

    def grade_workspace(self, case_id: str, create: bool = True) -> Path:
        """The shared workspace directory for one trial's graders.

        Sits beside that trial's output artifacts (``{base_dir}/{case_id}/``)
        so evidence produced *while scoring* — a rendered image, an extracted
        document, a judge's raw response — is archived next to the output it
        was derived from, rather than living in a temp dir that disappears.
        """
        path = self.base_dir / _safe_filename(case_id) / "grade"
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def list_grade_evidence(self, case_id: str) -> list[str]:
        """Filenames left in a trial's grade workspace, sorted."""
        path = self.grade_workspace(case_id, create=False)
        if not path.is_dir():
            return []
        return sorted(p.name for p in path.iterdir() if p.is_file())

    # ------------------------------------------------------------------
    # Baseline management
    # ------------------------------------------------------------------

    def set_baseline(self, case_id: str, trial_dir: Path | str | None = None) -> Path:
        """Copy a trial's artifacts as the baseline for *case_id*.

        If *trial_dir* is ``None``, the current artifact directory for
        *case_id* is used.

        Returns:
            Path to the baseline directory.
        """
        safe_id = _safe_filename(case_id)
        source = Path(trial_dir) if trial_dir else self.base_dir / safe_id
        if not source.is_dir():
            raise FileNotFoundError(f"Trial directory not found: {source}")

        baseline_dir = self.base_dir / "baselines" / safe_id
        if baseline_dir.exists():
            shutil.rmtree(baseline_dir)
        shutil.copytree(source, baseline_dir)
        return baseline_dir

    def load_baseline(self, case_id: str) -> dict[str, Any] | None:
        """Load the baseline manifest for *case_id*.

        Returns:
            Manifest dict, or ``None`` if no baseline exists.
        """
        safe_id = _safe_filename(case_id)
        manifest_path = self.base_dir / "baselines" / safe_id / "manifest.json"
        if not manifest_path.exists():
            return None
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def compare_with_baseline(self, case_id: str) -> BaselineComparison:
        """Compare the latest trial artifacts with the baseline.

        Uses ``image_hash`` from both manifests for a fast comparison.
        """
        safe_id = _safe_filename(case_id)
        current_dir = self.base_dir / safe_id
        baseline_dir = self.base_dir / "baselines" / safe_id

        current_manifest_path = current_dir / "manifest.json"
        baseline_manifest_path = baseline_dir / "manifest.json"

        if not baseline_manifest_path.exists():
            return BaselineComparison(
                identical=False,
                baseline_path=None,
                current_path=current_dir if current_dir.exists() else None,
                details={"error": "no baseline found"},
            )

        if not current_manifest_path.exists():
            return BaselineComparison(
                identical=False,
                baseline_path=baseline_dir,
                current_path=None,
                details={"error": "no current trial artifacts found"},
            )

        baseline_manifest = json.loads(
            baseline_manifest_path.read_text(encoding="utf-8")
        )
        current_manifest = json.loads(
            current_manifest_path.read_text(encoding="utf-8")
        )

        baseline_hash = baseline_manifest.get("image_hash")
        current_hash = current_manifest.get("image_hash")

        hash_match: bool | None = None
        if baseline_hash is not None and current_hash is not None:
            hash_match = baseline_hash == current_hash

        identical = hash_match is True

        return BaselineComparison(
            identical=identical,
            baseline_path=baseline_dir,
            current_path=current_dir,
            hash_match=hash_match,
            details={
                "baseline_hash": baseline_hash,
                "current_hash": current_hash,
                "baseline_trial_id": baseline_manifest.get("trial_id"),
                "current_trial_id": current_manifest.get("trial_id"),
            },
        )

    def list_trials(self, case_id: str) -> list[dict[str, Any]]:
        """List stored trial manifests for *case_id*.

        Returns:
            List of manifest dicts (empty if none found).
        """
        safe_id = _safe_filename(case_id)
        case_dir = self.base_dir / safe_id
        if not case_dir.is_dir():
            return []

        manifest_path = case_dir / "manifest.json"
        if not manifest_path.exists():
            return []

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return [manifest]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _safe_filename(name: str) -> str:
    """Replace characters unsafe for filenames."""
    return re.sub(r'[/\\:*?"<>|\s]', "_", name)


def _get_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()
