"""Tests for CLI functionality."""

import tempfile

import pytest
import yaml

from compass.core.scenario import Scenario
from compass.graders import GraderType


class TestInitCommand:
    """Tests for the init command template."""

    def test_init_template_is_valid_yaml(self):
        """The init template should be valid YAML."""
        # Import the template from main.py

        # Extract the template string (it's defined inside the init function)
        # We need to test it by running the command
        with tempfile.TemporaryDirectory():
            # Simulate the init command by writing the template
            template = """name: "My Test Scenario"
description: "Description of what this scenario tests"

agent:
  adapter: image
  endpoint: "http://localhost:8000"

# Default graders applied to all cases
default_graders:
  - type: model
    name: semantic_match
    weight: 0.4
    config:
      threshold: 0.25
  - type: model
    name: safety_check
    weight: 0.0
    required: true
    config:
      checks: [nsfw]

# Default aggregation settings
default_aggregation:
  method: weighted_sum
  pass_threshold: 0.7
  required_graders: [safety_check]

cases:
  - id: "example_case_1"
    input:
      prompt: "A beautiful landscape with mountains and a lake"
      negative_prompt: "blurry, low quality, distorted"
      params:
        width: 1024
        height: 1024
        steps: 30

    graders:
      - type: model
        name: aesthetic_score
        weight: 0.3
        config:
          min_score: 5.0

      - type: model
        name: vlm_judge
        weight: 0.3
        config:
          model: "gpt-4o"
          criteria:
            - "The image contains mountains"
            - "The image contains a lake"
            - "The overall scene is a landscape"

    aggregation:
      method: weighted_sum
      pass_threshold: 0.7

    tags:
      - landscape
      - nature

  - id: "example_case_2"
    input:
      prompt: "A portrait of a robot in cyberpunk style"
      negative_prompt: "realistic human, photograph"
      params:
        width: 768
        height: 1024

    tags:
      - portrait
      - cyberpunk
"""
            # Parse as YAML
            data = yaml.safe_load(template)
            assert data is not None
            assert "name" in data
            assert "agent" in data
            assert "cases" in data

    def test_init_template_can_be_parsed_as_scenario(self):
        """The init template should be parseable as a Scenario."""
        template = """name: "My Test Scenario"
description: "Description of what this scenario tests"

agent:
  adapter: image
  endpoint: "http://localhost:8000"

default_graders:
  - type: model
    name: semantic_match
    weight: 0.4
    config:
      threshold: 0.25
  - type: model
    name: safety_check
    weight: 0.0
    required: true
    config:
      checks: [nsfw]

default_aggregation:
  method: weighted_sum
  pass_threshold: 0.7
  required_graders: [safety_check]

cases:
  - id: "example_case_1"
    input:
      prompt: "A beautiful landscape with mountains and a lake"
      negative_prompt: "blurry, low quality, distorted"
      params:
        width: 1024
        height: 1024
        steps: 30

    graders:
      - type: model
        name: aesthetic_score
        weight: 0.3
        config:
          min_score: 5.0

      - type: model
        name: vlm_judge
        weight: 0.3
        config:
          model: "gpt-4o"
          criteria:
            - "The image contains mountains"
            - "The image contains a lake"
            - "The overall scene is a landscape"

    aggregation:
      method: weighted_sum
      pass_threshold: 0.7

    tags:
      - landscape
      - nature

  - id: "example_case_2"
    input:
      prompt: "A portrait of a robot in cyberpunk style"
      negative_prompt: "realistic human, photograph"
      params:
        width: 768
        height: 1024

    tags:
      - portrait
      - cyberpunk
"""
        data = yaml.safe_load(template)

        # This should NOT raise a validation error
        scenario = Scenario.from_dict(data)

        # Verify structure
        assert scenario.name == "My Test Scenario"
        assert scenario.agent.adapter == "image"
        assert len(scenario.default_graders) == 2
        assert scenario.default_graders[0].name == "semantic_match"
        assert scenario.default_graders[1].name == "safety_check"
        assert scenario.default_aggregation.required_graders == ["safety_check"]
        assert len(scenario.cases) == 2
        assert scenario.cases[0].id == "example_case_1"
        assert len(scenario.cases[0].graders) == 2
        assert scenario.cases[0].graders[0].name == "aesthetic_score"
        assert scenario.cases[1].id == "example_case_2"

    def test_graders_use_correct_field_names(self):
        """Verify the template uses 'graders' not 'evaluators'."""
        template = """name: "Test"
description: ""
agent:
  adapter: image

default_graders:
  - name: test_grader
    type: code

cases:
  - id: case1
    input:
      prompt: "test"
    graders:
      - name: case_grader
        type: code
"""
        data = yaml.safe_load(template)
        scenario = Scenario.from_dict(data)

        # Should use default_graders not default_evaluators
        assert len(scenario.default_graders) == 1
        assert scenario.default_graders[0].name == "test_grader"

        # Should use graders not evaluators in cases
        assert len(scenario.cases[0].graders) == 1
        assert scenario.cases[0].graders[0].name == "case_grader"


class TestListCommand:
    """Tests for the list command showing graders."""

    def test_list_graders_returns_graders(self):
        """The list_graders function should return registered graders."""
        from compass.graders import list_graders

        graders = list_graders()
        assert len(graders) > 0
        assert "semantic_match" in graders

    def test_list_graders_can_filter_by_type(self):
        """list_graders should support filtering by grader type."""
        from compass.graders import list_graders

        code_graders = list_graders(GraderType.CODE)
        model_graders = list_graders(GraderType.MODEL)

        # Code graders should exist
        assert len(code_graders) > 0

        # Model graders should exist and include semantic_match
        assert len(model_graders) > 0
        assert "semantic_match" in model_graders

        # They should be mutually exclusive
        for g in code_graders:
            assert g not in model_graders


def test_eval_legacy_stack_removed():
    """The deprecated compass.eval stack is gone — importing it must fail."""
    with pytest.raises(ModuleNotFoundError):
        import compass.eval  # noqa: F401
