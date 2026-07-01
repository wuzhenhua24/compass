"""Tests for ArtifactStore — artifact binary persistence and baseline management."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compass.core.artifact_store import ArtifactStore, _safe_filename
from compass.core.artifacts import CodeArtifact, GeneratedFile, ImageArtifact, TextArtifact
from compass.core.transcript import Outcome, Transcript

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pil_image(width: int = 4, height: int = 4):
    """Create a tiny PIL Image for testing (avoids heavy fixture)."""
    from PIL import Image

    return Image.new("RGB", (width, height), color="red")


def _make_transcript(
    case_id: str = "test_case",
    trial_id: str = "trial_1",
    image=None,
    image_hash: str | None = None,
    artifacts=None,
) -> Transcript:
    t = Transcript(task_id=case_id, trial_id=trial_id)
    t.outcome = Outcome(
        image=image,
        image_hash=image_hash,
        artifacts=artifacts or [],
    )
    return t


# ---------------------------------------------------------------------------
# TestArtifactStore — basics
# ---------------------------------------------------------------------------


class TestArtifactStore:
    def test_init(self, tmp_path: Path):
        store = ArtifactStore(tmp_path / "store")
        assert store.base_dir == tmp_path / "store"

    def test_init_accepts_string(self, tmp_path: Path):
        store = ArtifactStore(str(tmp_path / "store"))
        assert store.base_dir == tmp_path / "store"


# ---------------------------------------------------------------------------
# TestSaveTrialArtifacts
# ---------------------------------------------------------------------------


class TestSaveTrialArtifacts:
    def test_saves_legacy_image(self, tmp_path: Path):
        img = _make_pil_image()
        transcript = _make_transcript(image=img)
        store = ArtifactStore(tmp_path)

        saved = store.save_trial_artifacts(transcript, "case1")

        assert len(saved) >= 1
        assert (tmp_path / "case1" / "output.png").exists()

    def test_saves_image_artifact(self, tmp_path: Path):
        img = _make_pil_image()
        art = ImageArtifact(image=img)
        transcript = _make_transcript(artifacts=[art])
        store = ArtifactStore(tmp_path)

        saved = store.save_trial_artifacts(transcript, "case1")

        assert any("artifact_0_image.png" in s for s in saved)
        assert (tmp_path / "case1" / "artifact_0_image.png").exists()
        # image_path should be updated on the artifact
        assert art.image_path is not None

    def test_saves_code_artifact(self, tmp_path: Path):
        gf = GeneratedFile(path="src/main.py", content="print('hello')", language="python")
        art = CodeArtifact(files=[gf])
        transcript = _make_transcript(artifacts=[art])
        store = ArtifactStore(tmp_path)

        saved = store.save_trial_artifacts(transcript, "case1")

        assert len(saved) == 1
        saved_file = tmp_path / "case1" / "artifact_0_code_0_main.py"
        assert saved_file.exists()
        assert saved_file.read_text() == "print('hello')"

    def test_saves_text_artifact(self, tmp_path: Path):
        art = TextArtifact(content="Hello World")
        transcript = _make_transcript(artifacts=[art])
        store = ArtifactStore(tmp_path)

        saved = store.save_trial_artifacts(transcript, "case1")

        assert len(saved) == 1
        saved_file = tmp_path / "case1" / "artifact_0_text.txt"
        assert saved_file.exists()
        assert saved_file.read_text() == "Hello World"

    def test_generates_manifest(self, tmp_path: Path):
        img = _make_pil_image()
        transcript = _make_transcript(image=img, image_hash="abc123")
        store = ArtifactStore(tmp_path)

        store.save_trial_artifacts(transcript, "case1")

        manifest_path = tmp_path / "case1" / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["case_id"] == "case1"
        assert manifest["image_hash"] == "abc123"
        assert len(manifest["artifacts"]) >= 1

    def test_mixed_artifacts(self, tmp_path: Path):
        img = _make_pil_image()
        artifacts = [
            ImageArtifact(image=img),
            CodeArtifact(files=[GeneratedFile(path="app.js", content="console.log(1)")]),
            TextArtifact(content="report content"),
        ]
        transcript = _make_transcript(artifacts=artifacts)
        store = ArtifactStore(tmp_path)

        saved = store.save_trial_artifacts(transcript, "mixed")

        assert len(saved) == 3
        assert (tmp_path / "mixed" / "artifact_0_image.png").exists()
        assert (tmp_path / "mixed" / "artifact_1_code_0_app.js").exists()
        assert (tmp_path / "mixed" / "artifact_2_text.txt").exists()

    def test_no_artifacts_returns_empty(self, tmp_path: Path):
        transcript = _make_transcript()
        store = ArtifactStore(tmp_path)

        saved = store.save_trial_artifacts(transcript, "empty")

        assert saved == []
        assert not (tmp_path / "empty").exists()

    def test_no_outcome_returns_empty(self, tmp_path: Path):
        t = Transcript(task_id="x", trial_id="y")
        t.outcome = Outcome()  # no image, no artifacts
        store = ArtifactStore(tmp_path)

        saved = store.save_trial_artifacts(t, "none")

        assert saved == []

    def test_empty_text_artifact_skipped(self, tmp_path: Path):
        art = TextArtifact(content="")
        transcript = _make_transcript(artifacts=[art])
        store = ArtifactStore(tmp_path)

        saved = store.save_trial_artifacts(transcript, "case1")

        assert saved == []

    def test_image_artifact_without_image_skipped(self, tmp_path: Path):
        art = ImageArtifact(image=None, image_path="/some/old/path.png")
        transcript = _make_transcript(artifacts=[art])
        store = ArtifactStore(tmp_path)

        saved = store.save_trial_artifacts(transcript, "case1")

        assert saved == []


# ---------------------------------------------------------------------------
# TestBaseline
# ---------------------------------------------------------------------------


class TestBaseline:
    def test_set_baseline_copies_files(self, tmp_path: Path):
        # Create trial artifacts
        img = _make_pil_image()
        transcript = _make_transcript(image=img, image_hash="deadbeef")
        store = ArtifactStore(tmp_path)
        store.save_trial_artifacts(transcript, "case1")

        # Set baseline
        baseline_dir = store.set_baseline("case1")

        assert baseline_dir.exists()
        assert (baseline_dir / "output.png").exists()
        assert (baseline_dir / "manifest.json").exists()

    def test_load_baseline_returns_manifest(self, tmp_path: Path):
        img = _make_pil_image()
        transcript = _make_transcript(image=img, image_hash="deadbeef")
        store = ArtifactStore(tmp_path)
        store.save_trial_artifacts(transcript, "case1")
        store.set_baseline("case1")

        manifest = store.load_baseline("case1")

        assert manifest is not None
        assert manifest["case_id"] == "case1"
        assert manifest["image_hash"] == "deadbeef"

    def test_load_baseline_not_exists_returns_none(self, tmp_path: Path):
        store = ArtifactStore(tmp_path)

        assert store.load_baseline("nonexistent") is None

    def test_compare_hash_match(self, tmp_path: Path):
        img = _make_pil_image()
        transcript = _make_transcript(image=img, image_hash="same_hash")
        store = ArtifactStore(tmp_path)
        store.save_trial_artifacts(transcript, "case1")
        store.set_baseline("case1")

        result = store.compare_with_baseline("case1")

        assert result.identical is True
        assert result.hash_match is True

    def test_compare_hash_mismatch(self, tmp_path: Path):
        img = _make_pil_image()
        # Set baseline with one hash
        transcript = _make_transcript(image=img, image_hash="hash_v1")
        store = ArtifactStore(tmp_path)
        store.save_trial_artifacts(transcript, "case1")
        store.set_baseline("case1")

        # Overwrite current trial with a different hash
        transcript2 = _make_transcript(image=img, image_hash="hash_v2")
        store.save_trial_artifacts(transcript2, "case1")

        result = store.compare_with_baseline("case1")

        assert result.identical is False
        assert result.hash_match is False

    def test_compare_no_baseline(self, tmp_path: Path):
        img = _make_pil_image()
        transcript = _make_transcript(image=img)
        store = ArtifactStore(tmp_path)
        store.save_trial_artifacts(transcript, "case1")

        result = store.compare_with_baseline("case1")

        assert result.identical is False
        assert result.baseline_path is None

    def test_set_baseline_not_found_raises(self, tmp_path: Path):
        store = ArtifactStore(tmp_path)
        with pytest.raises(FileNotFoundError):
            store.set_baseline("missing_case")


# ---------------------------------------------------------------------------
# TestListTrials
# ---------------------------------------------------------------------------


class TestListTrials:
    def test_list_trials_returns_manifest(self, tmp_path: Path):
        img = _make_pil_image()
        transcript = _make_transcript(image=img, trial_id="t1")
        store = ArtifactStore(tmp_path)
        store.save_trial_artifacts(transcript, "case1")

        trials = store.list_trials("case1")

        assert len(trials) == 1
        assert trials[0]["trial_id"] == "t1"

    def test_list_trials_empty(self, tmp_path: Path):
        store = ArtifactStore(tmp_path)

        assert store.list_trials("nope") == []


# ---------------------------------------------------------------------------
# TestSafeFilename
# ---------------------------------------------------------------------------


class TestSafeFilename:
    def test_replaces_special_chars(self):
        assert _safe_filename('a/b\\c:d*e?"f<g>h|i') == "a_b_c_d_e__f_g_h_i"

    def test_preserves_safe_chars(self):
        assert _safe_filename("hello-world_v2.txt") == "hello-world_v2.txt"

    def test_replaces_spaces(self):
        assert _safe_filename("my case id") == "my_case_id"
