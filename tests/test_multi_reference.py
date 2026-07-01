"""Tests for multi-reference image support in GradeContext, InputConfig, and Runner."""

from pathlib import Path

from PIL import Image

from compass.core.runner import Compass
from compass.core.scenario import InputConfig, Scenario
from compass.graders.base import GradeContext

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_image(width: int = 64, height: int = 64, color: str = "red") -> Image.Image:
    """Create a small solid-color PIL Image for testing."""
    return Image.new("RGB", (width, height), color)


def _save_image(img: Image.Image, path: Path) -> None:
    img.save(str(path))


# ===========================================================================
# GradeContext tests
# ===========================================================================


class TestGradeContextReferenceImages:
    """Tests for GradeContext.reference_images field and convenience methods."""

    def test_reference_images_default_empty(self):
        ctx = GradeContext()
        assert ctx.reference_images == {}

    def test_reference_images_populated(self):
        person = _make_image(color="blue")
        product = _make_image(color="green")
        ctx = GradeContext(reference_images={"person": person, "product": product})

        assert len(ctx.reference_images) == 2
        assert ctx.reference_images["person"] is person
        assert ctx.reference_images["product"] is product

    def test_get_reference_image_exists(self):
        img = _make_image()
        ctx = GradeContext(reference_images={"person": img})
        assert ctx.get_reference_image("person") is img

    def test_get_reference_image_missing(self):
        ctx = GradeContext(reference_images={"person": _make_image()})
        assert ctx.get_reference_image("nonexistent") is None

    def test_has_reference_images_with_dict(self):
        ctx = GradeContext(reference_images={"person": _make_image()})
        assert ctx.has_reference_images is True

    def test_has_reference_images_with_single(self):
        ctx = GradeContext(reference_image=_make_image())
        assert ctx.has_reference_images is True

    def test_has_reference_images_empty(self):
        ctx = GradeContext()
        assert ctx.has_reference_images is False

    def test_backward_compat_reference_image(self):
        """Old reference_image field still works independently."""
        single = _make_image(color="yellow")
        ctx = GradeContext(reference_image=single)

        assert ctx.reference_image is single
        assert ctx.reference_images == {}
        assert ctx.has_reference_images is True


# ===========================================================================
# InputConfig tests
# ===========================================================================


class TestInputConfigReferenceImages:
    """Tests for InputConfig.reference_images field."""

    def test_input_config_default_no_ref_images(self):
        cfg = InputConfig(prompt="test")
        assert cfg.reference_images == {}

    def test_input_config_with_ref_images(self):
        cfg = InputConfig(
            prompt="test",
            reference_images={"person": "person.png", "product": "jacket.png"},
        )
        assert cfg.reference_images == {
            "person": "person.png",
            "product": "jacket.png",
        }

    def test_input_config_yaml_roundtrip(self):
        """Serialize and deserialize preserves reference_images."""
        cfg = InputConfig(
            prompt="hello",
            reference_images={"a": "a.png", "b": "b.png"},
        )
        dumped = cfg.model_dump()
        restored = InputConfig.model_validate(dumped)
        assert restored.reference_images == cfg.reference_images

    def test_input_config_backward_compat(self):
        """Old-style dict without reference_images still parses fine."""
        cfg = InputConfig.model_validate({"prompt": "test"})
        assert cfg.reference_images == {}


# ===========================================================================
# Runner._load_reference_images tests
# ===========================================================================


class TestLoadReferenceImages:
    """Tests for Compass._load_reference_images static method."""

    def test_load_reference_images_success(self, tmp_path: Path):
        img = _make_image(color="red")
        p = tmp_path / "ref.png"
        _save_image(img, p)

        result = Compass._load_reference_images({"person": str(p)})

        assert "person" in result
        assert isinstance(result["person"], Image.Image)
        assert result["person"].size == (64, 64)

    def test_load_reference_images_missing_file(self, tmp_path: Path):
        result = Compass._load_reference_images(
            {"missing": str(tmp_path / "does_not_exist.png")}
        )
        assert result == {}

    def test_load_reference_images_empty(self):
        result = Compass._load_reference_images({})
        assert result == {}

    def test_load_reference_images_multiple(self, tmp_path: Path):
        for name, color in [("person", "blue"), ("product", "green")]:
            _save_image(_make_image(color=color), tmp_path / f"{name}.png")

        result = Compass._load_reference_images(
            {
                "person": str(tmp_path / "person.png"),
                "product": str(tmp_path / "product.png"),
            }
        )
        assert len(result) == 2
        assert "person" in result
        assert "product" in result


# ===========================================================================
# Runner integration: GradeContext built with reference images
# ===========================================================================


class TestGradeContextBuiltWithRefImages:
    """Integration-level: verify GradeContext is correctly built by the runner."""

    def test_grade_context_built_with_ref_images(self, tmp_path: Path):
        """Simulate the runner logic for building GradeContext with ref images."""
        # Prepare reference images on disk
        _save_image(_make_image(color="blue"), tmp_path / "person.png")
        _save_image(_make_image(color="green"), tmp_path / "product.png")

        paths = {
            "person": str(tmp_path / "person.png"),
            "product": str(tmp_path / "product.png"),
        }
        reference_images = Compass._load_reference_images(paths)

        ref_image = None
        if len(reference_images) == 1:
            ref_image = next(iter(reference_images.values()))

        ctx = GradeContext(
            prompt="Put the jacket on",
            reference_image=ref_image,
            reference_images=reference_images,
        )

        assert len(ctx.reference_images) == 2
        assert ctx.reference_image is None  # >1 image, no single fallback
        assert ctx.get_reference_image("person") is not None
        assert ctx.get_reference_image("product") is not None
        assert ctx.has_reference_images is True

    def test_single_ref_image_sets_legacy_field(self, tmp_path: Path):
        """When only one reference image, it should populate reference_image."""
        _save_image(_make_image(color="red"), tmp_path / "only.png")

        paths = {"only": str(tmp_path / "only.png")}
        reference_images = Compass._load_reference_images(paths)

        ref_image = None
        if len(reference_images) == 1:
            ref_image = next(iter(reference_images.values()))

        ctx = GradeContext(
            prompt="test",
            reference_image=ref_image,
            reference_images=reference_images,
        )

        assert ctx.reference_image is not None
        assert ctx.reference_image is ref_image
        assert ctx.has_reference_images is True


# ===========================================================================
# Scenario YAML tests
# ===========================================================================


class TestScenarioReferenceImages:
    """Tests for Scenario YAML parsing with reference_images."""

    def test_scenario_with_reference_images(self, tmp_path: Path):
        yaml_content = """\
name: Virtual Tryon
agent:
  adapter: image
cases:
  - id: tryon_001
    input:
      prompt: "Put this jacket on the person"
      reference_images:
        person: fixtures/person.png
        product: fixtures/jacket.png
    graders:
      - name: semantic_match
"""
        yaml_file = tmp_path / "scenario.yaml"
        yaml_file.write_text(yaml_content)

        scenario = Scenario.from_yaml(yaml_file)
        case = scenario.cases[0]

        assert case.input.reference_images == {
            "person": "fixtures/person.png",
            "product": "fixtures/jacket.png",
        }

    def test_scenario_without_reference_images(self, tmp_path: Path):
        yaml_content = """\
name: Basic
agent:
  adapter: image
cases:
  - id: basic_001
    input:
      prompt: "Generate a cat"
    graders:
      - name: semantic_match
"""
        yaml_file = tmp_path / "scenario.yaml"
        yaml_file.write_text(yaml_content)

        scenario = Scenario.from_yaml(yaml_file)
        case = scenario.cases[0]

        assert case.input.reference_images == {}
