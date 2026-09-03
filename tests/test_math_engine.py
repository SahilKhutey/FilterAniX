import numpy as np
from src.art.math_engine import MathematicalStyleEngine
from src.art.types import StyleConfig


def test_math_engine_preserves_shape_and_dtype():
    engine = MathematicalStyleEngine()
    dummy = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)

    out = engine.render(dummy)
    assert out.shape == (240, 320, 3)
    assert out.dtype == np.uint8


def test_math_engine_is_deterministic():
    engine = MathematicalStyleEngine()
    dummy = np.full((128, 128, 3), 150, dtype=np.uint8)
    dummy[30:90, 30:90] = [220, 180, 140]

    out1 = engine.render(dummy, stabilize=False)
    out2 = engine.render(dummy, stabilize=False)

    np.testing.assert_array_equal(out1, out2)


def test_math_engine_palette_projection_and_cel_shading():
    config = StyleConfig(
        color_palette_mix=0.80,
        tone_contrast=1.20,
        edge_strength=0.80,
    )
    engine = MathematicalStyleEngine(config)

    gradient = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (256, 1))
    rgb_grad = np.stack([gradient, gradient, gradient], axis=-1)

    out = engine.render(rgb_grad, stabilize=False)
    assert out.shape == (256, 256, 3)
    assert not np.array_equal(out, rgb_grad)


def test_math_engine_temporal_stabilization_and_scene_cut():
    engine = MathematicalStyleEngine()
    frame_a = np.full((100, 100, 3), 100, dtype=np.uint8)
    frame_b = np.full((100, 100, 3), 120, dtype=np.uint8)

    # Frame 1
    res_a = engine.render(frame_a, stabilize=True)

    # Frame 2 (Stabilized against Frame 1)
    res_b = engine.render(frame_b, stabilize=True)

    # Frame 3 with Scene Cut (hard reset)
    res_c = engine.render(frame_b, scene_cut=True, stabilize=True)

    assert not np.array_equal(res_b, res_c)
