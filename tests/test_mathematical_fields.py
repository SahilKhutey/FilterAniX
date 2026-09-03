"""Unit Tests for FilterAniX Mathematical Anime Engine Fields and Operators."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.art.mathematical.config import DEFAULT_ANIME_PALETTE, MathematicalAnimeStyle
from src.art.mathematical.color_field import compute_color_field
from src.art.mathematical.tone_field import compute_tone_field
from src.art.mathematical.palette_field import compute_palette_projection
from src.art.mathematical.shadow_field import compute_shadow_field
from src.art.mathematical.highlight_field import compute_highlight_field
from src.art.mathematical.edge_field import compute_edge_field
from src.art.mathematical.geometry_field import compute_surface_normals
from src.art.mathematical.face_field import compute_face_mask, apply_face_modulation
from src.art.mathematical.texture_field import compute_foreground_mask, apply_background_simplification
from src.art.mathematical.lighting_field import compute_lighting_field
from src.art.mathematical.temporal_field import TemporalOpticalFlowField
from src.art.mathematical.engine import MathematicalAnimeEngine


def test_config_defaults_and_immutability():
    style = MathematicalAnimeStyle()
    assert style.contrast == 1.08
    assert style.gamma == 0.96
    assert style.tone_strength == 0.82
    assert style.palette_mix == 0.60
    assert style.ink_color == (32, 27, 34)
    assert DEFAULT_ANIME_PALETTE.shape[1] == 3 and DEFAULT_ANIME_PALETTE.shape[0] >= 7

    # Immutability
    with pytest.raises((AttributeError, TypeError)):
        style.contrast = 1.20  # type: ignore


def test_color_field_decomposition():
    style = MathematicalAnimeStyle()
    dummy = np.random.randint(50, 200, (120, 160, 3), dtype=np.uint8)

    color_field, smooth_base = compute_color_field(dummy, style)
    assert color_field.shape == (120, 160, 3)
    assert smooth_base.shape == (120, 160, 3)
    assert color_field.dtype == np.float32
    assert 0.0 <= color_field.min() <= color_field.max() <= 1.0


def test_tone_field_s_curve_and_gamma():
    style = MathematicalAnimeStyle()
    grad = np.linspace(0.0, 1.0, 100, dtype=np.float32).reshape(100, 1, 1)
    grad_field = np.tile(grad, (1, 100, 3))

    Y_orig, Y_anime, toned = compute_tone_field(grad_field, style)
    assert Y_orig.shape == (100, 100)
    assert Y_anime.shape == (100, 100)
    assert toned.shape == (100, 100, 3)
    assert 0.0 <= toned.min() <= toned.max() <= 1.0


def test_palette_projection_soft_quantization():
    style = MathematicalAnimeStyle(palette_mix=0.75, palette_temperature=0.70)
    dummy_toned = np.full((64, 64, 3), 0.5, dtype=np.float32)

    projected = compute_palette_projection(dummy_toned, style)
    assert projected.shape == (64, 64, 3)
    assert projected.dtype == np.float32
    assert 0.0 <= projected.min() <= projected.max() <= 1.0


def test_shadow_and_highlight_fields():
    style = MathematicalAnimeStyle()
    field = np.full((50, 50, 3), 0.5, dtype=np.float32)
    lum = np.full((50, 50), 0.2, dtype=np.float32)  # Low luminance -> strong shadow

    shaded, shadow_mask = compute_shadow_field(field, lum, style)
    assert shaded.shape == (50, 50, 3)
    assert shadow_mask.shape == (50, 50, 1)
    assert np.mean(shadow_mask) > 0.8  # Strong shadow activation

    high_lum = np.full((50, 50), 0.9, dtype=np.float32)  # High luminance -> highlight
    lit, highlight_mask = compute_highlight_field(shaded, high_lum, style)
    assert lit.shape == (50, 50, 3)
    assert highlight_mask.shape == (50, 50, 1)
    assert np.mean(highlight_mask) > 0.8  # Strong highlight activation


def test_edge_field_sobel_laplacian():
    style = MathematicalAnimeStyle(edge_strength=0.80)
    field = np.full((80, 80, 3), 0.5, dtype=np.float32)
    # Create high-contrast step edge
    lum = np.zeros((80, 80), dtype=np.float32)
    lum[:, 40:] = 1.0

    inked, ink_intensity, raw_edges = compute_edge_field(field, lum, style)
    assert inked.shape == (80, 80, 3)
    assert ink_intensity.shape == (80, 80, 1)
    # Edge intensity should be elevated along the step boundary at x=40
    assert ink_intensity[40, 40, 0] > ink_intensity[40, 10, 0]


def test_geometry_and_lighting_fields():
    style = MathematicalAnimeStyle(warm_light_strength=0.20)
    field = np.full((60, 60, 3), 0.4, dtype=np.float32)
    lum = np.linspace(0.0, 1.0, 60, dtype=np.float32).reshape(60, 1)
    lum = np.tile(lum, (1, 60))

    normals = compute_surface_normals(lum)
    assert normals.shape == (60, 60, 3)
    # Normals should be normalized unit vectors
    lengths = np.sqrt(np.sum(normals ** 2, axis=-1))
    np.testing.assert_allclose(lengths, 1.0, atol=1e-3)

    lit, key_light = compute_lighting_field(field, lum, style)
    assert lit.shape == (60, 60, 3)
    assert key_light.shape == (60, 60, 1)


def test_face_field_and_eye_emphasis():
    style = MathematicalAnimeStyle()
    h, w = 120, 120

    class MockBBox:
        x, y, width, height = 0.3, 0.2, 0.4, 0.5

    class MockFace:
        bbox = MockBBox()
        landmarks = []

    face_mask, hair_mask, eye_mask = compute_face_mask(h, w, MockFace())
    assert face_mask.shape == (h, w, 1)
    assert hair_mask.shape == (h, w, 1)
    assert eye_mask.shape == (h, w, 1)
    # Face mask should peak near center of bbox
    cx, cy = int((0.3 + 0.2) * w), int((0.2 + 0.25) * h)
    assert face_mask[cy, cx, 0] > 0.5

    current_art = np.full((h, w, 3), 0.6, dtype=np.float32)
    orig_f = current_art.copy()
    modulated = apply_face_modulation(current_art, orig_f, face_mask, eye_mask, hair_mask, style)
    assert modulated.shape == (h, w, 3)


def test_texture_field_background_simplification():
    style = MathematicalAnimeStyle(background_simplification=0.70)
    h, w = 80, 80
    fg_mask = compute_foreground_mask(h, w, None)
    assert fg_mask.shape == (h, w, 1)

    art = np.random.uniform(0.2, 0.8, (h, w, 3)).astype(np.float32)
    simplified = apply_background_simplification(art, fg_mask, style)
    assert simplified.shape == (h, w, 3)


def test_optical_flow_temporal_field():
    style = MathematicalAnimeStyle(temporal_strength=0.20, use_optical_flow=True)
    temp_field = TemporalOpticalFlowField(style)

    f1 = np.full((64, 64, 3), 0.3, dtype=np.float32)
    lum1 = np.full((64, 64), 0.3, dtype=np.float32)
    src1 = np.full((64, 64), 80, dtype=np.uint8)

    # Frame 1
    out1, m1 = temp_field.stabilize_frame(f1, lum1, src1)
    assert np.allclose(out1, f1)
    assert m1 == 0.0

    # Frame 2 with slight motion
    f2 = np.full((64, 64, 3), 0.4, dtype=np.float32)
    lum2 = np.full((64, 64), 0.4, dtype=np.float32)
    src2 = np.full((64, 64), 100, dtype=np.uint8)

    out2, m2 = temp_field.stabilize_frame(f2, lum2, src2)
    assert m2 > 0.0
    # Blend should be between f2 (0.4) and f1 (0.3)
    assert 0.30 <= float(out2[0, 0, 0]) < 0.40

    # Scene cut reset
    out3, m3 = temp_field.stabilize_frame(f2, lum2, src2, scene_cut=True)
    assert np.allclose(out3, f2)


def test_mathematical_anime_engine_determinism_and_types():
    engine = MathematicalAnimeEngine()
    dummy = np.full((128, 128, 3), 120, dtype=np.uint8)
    cv2.circle(dummy, (64, 64), 30, (230, 190, 160), -1)

    out1 = engine.render(dummy, stabilize=False)
    out2 = engine.render(dummy, stabilize=False)

    assert out1.shape == (128, 128, 3)
    assert out1.dtype == np.uint8
    np.testing.assert_array_equal(out1, out2)

    summary = engine.diagnostics.summarize()
    assert summary.total_frames == 2
    assert summary.average_fps > 0
