"""Unit tests for FilterAniX Neural Model Registry & Budget Policy."""
from __future__ import annotations

import pytest
from src.neural.registry import (
    ModelRegistry,
    MODEL_SPECS,
    MAX_TOTAL_MODEL_BUDGET_MB,
    RECOMMENDED_DEFAULT_BUNDLE_MB,
)


def test_model_budget_compliance():
    """Asserts that all registered models together do not exceed the 1024 MB budget."""
    total_mb = ModelRegistry.get_total_registered_size_mb()
    assert total_mb <= MAX_TOTAL_MODEL_BUDGET_MB
    assert total_mb <= RECOMMENDED_DEFAULT_BUNDLE_MB  # Recommended bundle <= 250 MB
    assert ModelRegistry.validate_budget_compliance() is True


def test_model_licensing_compliance():
    """Asserts that every model has an approved open-source license."""
    assert ModelRegistry.validate_licensing_compliance() is True
    for key, spec in MODEL_SPECS.items():
        assert spec.license.lower() in {"apache-2.0", "bsd-3-clause", "mit"}


def test_model_spec_attributes():
    """Verifies that all model specs have valid tasks, dimensions, and intervals."""
    assert "u2netp" in MODEL_SPECS
    assert "modnet" in MODEL_SPECS
    assert "micron_flow" in MODEL_SPECS
    assert "depth_anything_v2_small_int8" in MODEL_SPECS

    u2 = MODEL_SPECS["u2netp"]
    assert u2.task == "segmentation"
    assert u2.size_mb < 10.0

    mod = MODEL_SPECS["modnet"]
    assert mod.task == "portrait_matting"
    assert mod.size_mb < 35.0

    flow = MODEL_SPECS["micron_flow"]
    assert flow.task == "optical_flow"
    assert flow.size_mb < 5.0

    depth = MODEL_SPECS["depth_anything_v2_small_int8"]
    assert depth.task == "depth_estimation"
    assert depth.size_mb < 50.0
