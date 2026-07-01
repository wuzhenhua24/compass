"""Tests for the typed artifact system."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from compass.core.artifacts import (
    Artifact,
    CodeArtifact,
    ExecutionResult,
    GeneratedFile,
    ImageArtifact,
    TextArtifact,
    _artifact_registry,
    deserialize_artifact,
    get_artifact_class,
    register_artifact,
)
from compass.core.transcript import Outcome, Transcript
from compass.graders.base import GradeContext


# ===================================================================
# Registry tests
# ===================================================================


class TestArtifactRegistry:
    """Tests for the artifact registry."""

    def test_builtin_types_registered(self):
        """Built-in artifact types are registered at import time."""
        assert "image" in _artifact_registry
        assert "code" in _artifact_registry
        assert "text" in _artifact_registry

    def test_get_artifact_class(self):
        assert get_artifact_class("image") is ImageArtifact
        assert get_artifact_class("code") is CodeArtifact
        assert get_artifact_class("text") is TextArtifact

    def test_get_unknown_type_raises(self):
        with pytest.raises(KeyError, match="not found"):
            get_artifact_class("nonexistent_type")

    def test_register_custom_artifact(self):
        """Registering a custom artifact type works and is retrievable."""

        @register_artifact("custom_test")
        @dataclass
        class CustomArtifact(Artifact):
            artifact_type: str = "custom_test"
            value: int = 0

            @property
            def has_output(self) -> bool:
                return self.value != 0

            def to_dict(self) -> dict[str, Any]:
                return {
                    "artifact_type": self.artifact_type,
                    "value": self.value,
                    "metadata": self.metadata,
                }

            @classmethod
            def from_dict(cls, data: dict[str, Any]) -> "CustomArtifact":
                return cls(
                    value=data.get("value", 0),
                    metadata=data.get("metadata", {}),
                )

        try:
            assert get_artifact_class("custom_test") is CustomArtifact

            # Round-trip
            original = CustomArtifact(value=42)
            restored = deserialize_artifact(original.to_dict())
            assert isinstance(restored, CustomArtifact)
            assert restored.value == 42
        finally:
            _artifact_registry.pop("custom_test", None)


# ===================================================================
# Artifact serialization round-trip tests
# ===================================================================


class TestImageArtifact:
    def test_to_dict_from_dict_roundtrip(self):
        original = ImageArtifact(
            image_path="/tmp/test.png",
            image_hash="abc123",
            metadata={"source": "test"},
        )
        data = original.to_dict()
        restored = ImageArtifact.from_dict(data)

        assert restored.image_path == "/tmp/test.png"
        assert restored.image_hash == "abc123"
        assert restored.metadata == {"source": "test"}
        assert data["artifact_type"] == "image"

    def test_has_output_with_image(self):
        img = Image.new("RGB", (10, 10))
        art = ImageArtifact(image=img)
        assert art.has_output is True

    def test_has_output_with_path(self):
        art = ImageArtifact(image_path="/some/path.png")
        assert art.has_output is True

    def test_has_output_empty(self):
        art = ImageArtifact()
        assert art.has_output is False


class TestCodeArtifact:
    def test_to_dict_from_dict_roundtrip(self):
        original = CodeArtifact(
            files=[
                GeneratedFile(path="main.py", content="print('hi')", language="python"),
                GeneratedFile(path="util.py", content="pass", language="python"),
            ],
            language="python",
            diff="--- a\n+++ b",
            execution=ExecutionResult(
                exit_code=0, stdout="hi\n", stderr="", duration_ms=123.4
            ),
            metadata={"version": "1"},
        )
        data = original.to_dict()
        restored = CodeArtifact.from_dict(data)

        assert len(restored.files) == 2
        assert restored.files[0].path == "main.py"
        assert restored.files[0].content == "print('hi')"
        assert restored.language == "python"
        assert restored.diff == "--- a\n+++ b"
        assert restored.execution is not None
        assert restored.execution.exit_code == 0
        assert restored.execution.stdout == "hi\n"
        assert restored.execution.duration_ms == 123.4
        assert restored.metadata == {"version": "1"}
        assert data["artifact_type"] == "code"

    def test_has_output_with_files(self):
        art = CodeArtifact(files=[GeneratedFile(path="a.py", content="x")])
        assert art.has_output is True

    def test_has_output_empty(self):
        art = CodeArtifact()
        assert art.has_output is False

    def test_no_execution(self):
        original = CodeArtifact(files=[GeneratedFile(path="a.py")])
        data = original.to_dict()
        assert data["execution"] is None
        restored = CodeArtifact.from_dict(data)
        assert restored.execution is None


class TestTextArtifact:
    def test_to_dict_from_dict_roundtrip(self):
        original = TextArtifact(
            content="Hello world",
            format="markdown",
            word_count=2,
            metadata={"lang": "en"},
        )
        data = original.to_dict()
        restored = TextArtifact.from_dict(data)

        assert restored.content == "Hello world"
        assert restored.format == "markdown"
        assert restored.word_count == 2
        assert restored.metadata == {"lang": "en"}
        assert data["artifact_type"] == "text"

    def test_has_output_with_content(self):
        art = TextArtifact(content="some text")
        assert art.has_output is True

    def test_has_output_empty(self):
        art = TextArtifact()
        assert art.has_output is False


class TestDeserializeArtifact:
    def test_deserialize_image(self):
        data = {"artifact_type": "image", "image_path": "/x.png", "image_hash": "h"}
        art = deserialize_artifact(data)
        assert isinstance(art, ImageArtifact)
        assert art.image_path == "/x.png"

    def test_deserialize_code(self):
        data = {
            "artifact_type": "code",
            "files": [{"path": "a.py", "content": "x", "language": "python"}],
            "language": "python",
            "diff": "",
            "execution": None,
        }
        art = deserialize_artifact(data)
        assert isinstance(art, CodeArtifact)
        assert len(art.files) == 1

    def test_deserialize_unknown_raises(self):
        with pytest.raises(KeyError):
            deserialize_artifact({"artifact_type": "unknown_xyz"})


# ===================================================================
# Outcome integration tests
# ===================================================================


class TestOutcomeArtifacts:
    def test_empty_artifacts_by_default(self):
        outcome = Outcome()
        assert outcome.artifacts == []
        assert outcome.has_output is False

    def test_has_output_with_artifact(self):
        outcome = Outcome(
            artifacts=[TextArtifact(content="hello")]
        )
        assert outcome.has_output is True

    def test_has_output_false_for_empty_artifacts(self):
        outcome = Outcome(
            artifacts=[TextArtifact(content="")]
        )
        assert outcome.has_output is False

    def test_has_image_legacy_field(self):
        img = Image.new("RGB", (10, 10))
        outcome = Outcome(image=img)
        assert outcome.has_image is True

    def test_has_image_via_artifact(self):
        outcome = Outcome(
            artifacts=[ImageArtifact(image_path="/test.png")]
        )
        assert outcome.has_image is True

    def test_has_image_false(self):
        outcome = Outcome()
        assert outcome.has_image is False

    def test_get_artifact(self):
        code_art = CodeArtifact(files=[GeneratedFile(path="a.py")])
        text_art = TextArtifact(content="hello")
        outcome = Outcome(artifacts=[code_art, text_art])

        assert outcome.get_artifact(CodeArtifact) is code_art
        assert outcome.get_artifact(TextArtifact) is text_art
        assert outcome.get_artifact(ImageArtifact) is None

    def test_get_artifacts_multiple(self):
        art1 = TextArtifact(content="a")
        art2 = TextArtifact(content="b")
        art3 = CodeArtifact(files=[GeneratedFile(path="x.py")])
        outcome = Outcome(artifacts=[art1, art2, art3])

        texts = outcome.get_artifacts(TextArtifact)
        assert len(texts) == 2
        assert texts[0] is art1
        assert texts[1] is art2

    def test_to_dict_includes_artifacts(self):
        outcome = Outcome(
            artifacts=[TextArtifact(content="hi", word_count=1)]
        )
        d = outcome.to_dict()
        assert "artifacts" in d
        assert len(d["artifacts"]) == 1
        assert d["artifacts"][0]["artifact_type"] == "text"
        assert d["artifacts"][0]["content"] == "hi"


# ===================================================================
# set_outcome backward compatibility tests
# ===================================================================


class TestSetOutcomeBackwardCompat:
    def test_legacy_image_params_create_image_artifact(self):
        transcript = Transcript(task_id="t1", trial_id="r1")
        img = Image.new("RGB", (10, 10))
        transcript.set_outcome(image=img, image_path="/out.png", image_bytes=b"fake")

        assert transcript.outcome.image is img
        assert transcript.outcome.image_path == "/out.png"
        assert transcript.outcome.image_hash is not None

        # ImageArtifact auto-created
        assert len(transcript.outcome.artifacts) == 1
        art = transcript.outcome.artifacts[0]
        assert isinstance(art, ImageArtifact)
        assert art.image is img
        assert art.image_path == "/out.png"

    def test_legacy_image_no_duplicate_if_artifact_given(self):
        transcript = Transcript(task_id="t1", trial_id="r1")
        img = Image.new("RGB", (10, 10))
        existing_art = ImageArtifact(image=img, image_path="/out.png")
        transcript.set_outcome(
            image=img, image_path="/out.png", artifacts=[existing_art]
        )

        # Should NOT create a second ImageArtifact
        assert len(transcript.outcome.artifacts) == 1
        assert transcript.outcome.artifacts[0] is existing_art

    def test_no_image_no_artifact(self):
        transcript = Transcript(task_id="t1", trial_id="r1")
        transcript.set_outcome(blocked=True, blocked_reason="safety")

        assert transcript.outcome.artifacts == []
        assert transcript.outcome.blocked is True

    def test_set_outcome_with_code_artifact(self):
        transcript = Transcript(task_id="t1", trial_id="r1")
        code_art = CodeArtifact(
            files=[GeneratedFile(path="main.py", content="print(1)")],
            language="python",
        )
        transcript.set_outcome(artifacts=[code_art])

        assert len(transcript.outcome.artifacts) == 1
        assert isinstance(transcript.outcome.artifacts[0], CodeArtifact)
        assert transcript.outcome.has_output is True


# ===================================================================
# Transcript save/load round-trip tests
# ===================================================================


class TestTranscriptSaveLoadArtifacts:
    def test_roundtrip_with_artifacts(self, tmp_path):
        transcript = Transcript(task_id="t1", trial_id="r1")
        code_art = CodeArtifact(
            files=[GeneratedFile(path="app.py", content="x=1", language="python")],
            language="python",
            execution=ExecutionResult(exit_code=0, stdout="ok"),
        )
        text_art = TextArtifact(content="Summary", format="markdown", word_count=1)
        transcript.set_outcome(
            output_data={"key": "value"},
            artifacts=[code_art, text_art],
        )

        path = transcript.save(tmp_path / "transcript.json")
        loaded = Transcript.load(path)

        assert len(loaded.outcome.artifacts) == 2
        loaded_code = loaded.outcome.get_artifact(CodeArtifact)
        assert loaded_code is not None
        assert loaded_code.files[0].path == "app.py"
        assert loaded_code.execution.exit_code == 0

        loaded_text = loaded.outcome.get_artifact(TextArtifact)
        assert loaded_text is not None
        assert loaded_text.content == "Summary"
        assert loaded_text.format == "markdown"

    def test_roundtrip_with_image_artifact(self, tmp_path):
        transcript = Transcript(task_id="t1", trial_id="r1")
        transcript.set_outcome(image_path="/out.png", image_bytes=b"data")

        path = transcript.save(tmp_path / "transcript.json")
        loaded = Transcript.load(path)

        assert len(loaded.outcome.artifacts) == 1
        art = loaded.outcome.artifacts[0]
        assert isinstance(art, ImageArtifact)
        assert art.image_path == "/out.png"

    def test_load_old_format_without_artifacts(self, tmp_path):
        """Loading a JSON file that has no 'artifacts' key should not crash."""
        old_data = {
            "task_id": "t1",
            "trial_id": "r1",
            "input": {"prompt": "test", "params": {}},
            "tool_calls": [],
            "reasoning_steps": [],
            "outcome": {
                "image_path": "/legacy.png",
                "image_hash": "legacyhash",
                "blocked": False,
                "blocked_reason": "",
                "output_data": {},
                "metadata": {},
            },
            "environment": {},
            "timing": {
                "start_time": "2024-01-01T00:00:00",
                "end_time": None,
                "total_duration_ms": 0,
            },
            "grading": {"results": [], "final_score": 0.0, "final_passed": False},
        }
        path = tmp_path / "old_transcript.json"
        with open(path, "w") as f:
            json.dump(old_data, f)

        loaded = Transcript.load(path)
        assert loaded.outcome.artifacts == []
        assert loaded.outcome.image_path == "/legacy.png"

    def test_load_with_unknown_artifact_type(self, tmp_path):
        """Unknown artifact types are skipped with a warning, not crash."""
        data = {
            "task_id": "t1",
            "trial_id": "r1",
            "input": {"prompt": "", "params": {}},
            "tool_calls": [],
            "reasoning_steps": [],
            "outcome": {
                "image_path": None,
                "image_hash": None,
                "blocked": False,
                "blocked_reason": "",
                "output_data": {},
                "metadata": {},
                "artifacts": [
                    {"artifact_type": "unknown_future_type", "data": "stuff"},
                    {"artifact_type": "text", "content": "kept", "format": "plain", "word_count": 1},
                ],
            },
            "timing": {
                "start_time": "2024-01-01T00:00:00",
                "end_time": None,
                "total_duration_ms": 0,
            },
            "grading": {"results": [], "final_score": 0.0, "final_passed": False},
        }
        path = tmp_path / "future_transcript.json"
        with open(path, "w") as f:
            json.dump(data, f)

        loaded = Transcript.load(path)
        # Unknown type skipped, known type preserved
        assert len(loaded.outcome.artifacts) == 1
        assert isinstance(loaded.outcome.artifacts[0], TextArtifact)
        assert loaded.outcome.artifacts[0].content == "kept"


# ===================================================================
# GradeContext accessor tests
# ===================================================================


class TestGradeContextAccessors:
    def test_image_legacy(self):
        img = Image.new("RGB", (10, 10))
        ctx = GradeContext(outcome=Outcome(image=img))
        assert ctx.image is img

    def test_image_from_artifact(self):
        img = Image.new("RGB", (10, 10))
        ctx = GradeContext(
            outcome=Outcome(artifacts=[ImageArtifact(image=img)])
        )
        assert ctx.image is img

    def test_image_legacy_takes_priority(self):
        img_legacy = Image.new("RGB", (10, 10), color="red")
        img_artifact = Image.new("RGB", (10, 10), color="blue")
        ctx = GradeContext(
            outcome=Outcome(
                image=img_legacy,
                artifacts=[ImageArtifact(image=img_artifact)],
            )
        )
        assert ctx.image is img_legacy

    def test_image_none(self):
        ctx = GradeContext(outcome=Outcome())
        assert ctx.image is None

    def test_code_artifact(self):
        code = CodeArtifact(files=[GeneratedFile(path="a.py")])
        ctx = GradeContext(outcome=Outcome(artifacts=[code]))
        assert ctx.code_artifact is code

    def test_code_artifact_none(self):
        ctx = GradeContext(outcome=Outcome())
        assert ctx.code_artifact is None

    def test_text_artifact(self):
        text = TextArtifact(content="hello")
        ctx = GradeContext(outcome=Outcome(artifacts=[text]))
        assert ctx.text_artifact is text

    def test_text_artifact_none(self):
        ctx = GradeContext(outcome=Outcome())
        assert ctx.text_artifact is None

    def test_generic_artifact_method(self):
        text = TextArtifact(content="hello")
        ctx = GradeContext(outcome=Outcome(artifacts=[text]))
        assert ctx.artifact(TextArtifact) is text
        assert ctx.artifact(CodeArtifact) is None

    def test_has_output_true(self):
        ctx = GradeContext(
            outcome=Outcome(artifacts=[TextArtifact(content="x")])
        )
        assert ctx.has_output is True

    def test_has_output_false(self):
        ctx = GradeContext(outcome=Outcome())
        assert ctx.has_output is False

    def test_has_output_no_outcome(self):
        ctx = GradeContext()
        assert ctx.has_output is False

    def test_reference_artifact(self):
        ref = TextArtifact(content="reference")
        ctx = GradeContext(reference_artifact=ref)
        assert ctx.reference_artifact is ref
        assert ctx.reference_artifact.content == "reference"

    def test_accessors_with_no_outcome(self):
        ctx = GradeContext()
        assert ctx.code_artifact is None
        assert ctx.text_artifact is None
        assert ctx.artifact(ImageArtifact) is None


# ===================================================================
# Helper dataclass tests
# ===================================================================


class TestGeneratedFile:
    def test_roundtrip(self):
        gf = GeneratedFile(path="src/main.py", content="x=1", language="python")
        data = gf.to_dict()
        restored = GeneratedFile.from_dict(data)
        assert restored.path == "src/main.py"
        assert restored.content == "x=1"
        assert restored.language == "python"


class TestExecutionResult:
    def test_roundtrip(self):
        er = ExecutionResult(exit_code=1, stdout="out", stderr="err", duration_ms=42.5)
        data = er.to_dict()
        restored = ExecutionResult.from_dict(data)
        assert restored.exit_code == 1
        assert restored.stdout == "out"
        assert restored.stderr == "err"
        assert restored.duration_ms == 42.5
