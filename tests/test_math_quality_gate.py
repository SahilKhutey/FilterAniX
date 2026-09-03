"""Unit tests for FilterAniX Quality Gate validation (validate_render)."""
from __future__ import annotations

import numpy as np
import pytest

from src.art.mathematical import validate_render, RenderValidationError


def test_validate_render_success_on_valid_frame():
    input_rgb = np.full((120, 160, 3), 128, dtype=np.uint8)
    output_rgb = np.full((120, 160, 3), 135, dtype=np.uint8)

    assert validate_render(input_rgb, output_rgb) is True


def test_validate_render_rejects_non_uint8():
    input_rgb = np.full((120, 160, 3), 128, dtype=np.uint8)
    output_float = np.full((120, 160, 3), 0.5, dtype=np.float32)

    with pytest.raises(RenderValidationError, match="uint8"):
        validate_render(input_rgb, output_float)


def test_validate_render_rejects_dimension_mismatch():
    input_rgb = np.full((120, 160, 3), 128, dtype=np.uint8)
    output_wrong_shape = np.full((100, 160, 3), 128, dtype=np.uint8)

    with pytest.raises(RenderValidationError, match="dimensions"):
        validate_render(input_rgb, output_wrong_shape)


def test_validate_render_rejects_crushed_black():
    input_rgb = np.full((120, 160, 3), 120, dtype=np.uint8)
    output_crushed = np.full((120, 160, 3), 5, dtype=np.uint8)

    with pytest.raises(RenderValidationError, match="black"):
        validate_render(input_rgb, output_crushed)


def test_validate_render_rejects_blown_out_white():
    input_rgb = np.full((120, 160, 3), 120, dtype=np.uint8)
    output_blown = np.full((120, 160, 3), 250, dtype=np.uint8)

    with pytest.raises(RenderValidationError, match="white"):
        validate_render(input_rgb, output_blown)


def test_validate_render_rejects_flat_collapse():
    # Input has texture (varied pixel values)
    input_textured = np.random.randint(40, 200, (120, 160, 3), dtype=np.uint8)
    # Output is completely flat (std == 0)
    output_flat = np.full((120, 160, 3), 128, dtype=np.uint8)

    with pytest.raises(RenderValidationError, match="collapsed"):
        validate_render(input_textured, output_flat)
