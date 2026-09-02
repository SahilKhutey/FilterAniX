"""Tests for DiffusionRenderer ControlNet and Identity Reference Conditioning."""
from unittest.mock import MagicMock
import numpy as np
from PIL import Image
import pytest

from src.art.types import RenderConfig, RendererBackend, StylePreset
from src.art.preprocess import ControlMap
from src.art.diffusion_renderer import DiffusionRenderer
from src.art.style_engine import StyleEngine


def test_render_config_identity_defaults():
    """Verifies that RenderConfig explicitly exposes ControlNet and Identity Adapter configurations."""
    config = RenderConfig()
    assert config.controlnet_model_id is None
    assert config.controlnet_conditioning_scale == 0.80
    assert config.identity_adapter_model_id is None
    assert config.identity_conditioning_scale == 0.70

    custom_config = RenderConfig(
        model_id="runwayml/stable-diffusion-v1-5",
        controlnet_model_id="lllyasviel/control_v11p_sd15_lineart",
        controlnet_conditioning_scale=0.85,
        identity_adapter_model_id="h94/IP-Adapter",
        identity_conditioning_scale=0.75,
    )
    assert custom_config.controlnet_model_id == "lllyasviel/control_v11p_sd15_lineart"
    assert custom_config.controlnet_conditioning_scale == 0.85
    assert custom_config.identity_adapter_model_id == "h94/IP-Adapter"
    assert custom_config.identity_conditioning_scale == 0.75


def test_diffusion_renderer_cpu_fallback():
    """Verifies clean procedural fallback when diffusion model_id is not set."""
    renderer = DiffusionRenderer(RenderConfig(model_id=None))
    assert renderer.pipeline is None
    assert renderer.has_controlnet is False
    assert renderer.has_ip_adapter is False

    raw_frame = np.full((120, 160, 3), 180, dtype=np.uint8)
    control_map = ControlMap(edge_map=np.zeros((120, 160, 3), dtype=np.uint8))
    output = renderer.render(raw_frame, control_map)

    assert output.shape == (120, 160, 3)
    assert output.dtype == np.uint8


def test_diffusion_renderer_identity_conditioning_forwarded():
    """Regression test: verify reference_rgb is converted and passed as ip_adapter_image to diffusion pipeline."""
    config = RenderConfig(
        model_id="mock_model",
        identity_adapter_model_id="mock_ip_adapter",
        identity_conditioning_scale=0.65,
    )
    renderer = DiffusionRenderer(config)

    # Inject mock pipeline
    mock_pipeline = MagicMock()
    mock_out_pil = Image.fromarray(np.full((120, 160, 3), 200, dtype=np.uint8))
    mock_result = MagicMock()
    mock_result.images = [mock_out_pil]
    mock_pipeline.return_value = mock_result
    mock_pipeline.set_ip_adapter_scale = MagicMock()

    renderer.pipeline = mock_pipeline
    renderer.has_ip_adapter = True

    raw_frame = np.full((120, 160, 3), 150, dtype=np.uint8)
    ref_rgb = np.full((120, 160, 3), 220, dtype=np.uint8)
    control_map = ControlMap(edge_map=np.zeros((120, 160, 3), dtype=np.uint8))

    output = renderer.render(raw_frame, control_map, reference_rgb=ref_rgb)

    # Verify pipeline was called
    assert mock_pipeline.called
    call_kwargs = mock_pipeline.call_args[1]

    # ASSERT identity conditioning was actually supplied
    assert "ip_adapter_image" in call_kwargs
    assert isinstance(call_kwargs["ip_adapter_image"], Image.Image)
    assert call_kwargs["ip_adapter_image"].size == (160, 120)

    # Verify adapter scale was set
    mock_pipeline.set_ip_adapter_scale.assert_called_with(0.65)

    assert output.shape == (120, 160, 3)


def test_diffusion_renderer_controlnet_conditioning_forwarded():
    """Regression test: verify control_map is converted and passed as control_image with conditioning scale."""
    config = RenderConfig(
        model_id="mock_model",
        controlnet_model_id="mock_controlnet",
        controlnet_conditioning_scale=0.90,
    )
    renderer = DiffusionRenderer(config)

    mock_pipeline = MagicMock()
    mock_out_pil = Image.fromarray(np.full((100, 100, 3), 210, dtype=np.uint8))
    mock_result = MagicMock()
    mock_result.images = [mock_out_pil]
    mock_pipeline.return_value = mock_result

    renderer.pipeline = mock_pipeline
    renderer.has_controlnet = True

    raw_frame = np.full((100, 100, 3), 120, dtype=np.uint8)
    control_map = ControlMap(
        edge_map=np.full((100, 100, 3), 50, dtype=np.uint8),
        combined_control=np.full((100, 100, 3), 255, dtype=np.uint8),
    )

    output = renderer.render(raw_frame, control_map)

    assert mock_pipeline.called
    call_kwargs = mock_pipeline.call_args[1]

    # ASSERT ControlNet conditioning was actually supplied
    assert "control_image" in call_kwargs
    assert isinstance(call_kwargs["control_image"], Image.Image)
    assert call_kwargs["controlnet_conditioning_scale"] == 0.90
    assert output.shape == (100, 100, 3)


def test_diffusion_renderer_controlnet_and_identity_coexist():
    """Regression test: verify ControlNet and IP-Adapter identity conditioning function simultaneously."""
    config = RenderConfig(
        model_id="mock_model",
        controlnet_model_id="mock_controlnet",
        controlnet_conditioning_scale=0.85,
        identity_adapter_model_id="mock_ip_adapter",
        identity_conditioning_scale=0.75,
    )
    renderer = DiffusionRenderer(config)

    mock_pipeline = MagicMock()
    mock_out_pil = Image.fromarray(np.full((100, 100, 3), 230, dtype=np.uint8))
    mock_result = MagicMock()
    mock_result.images = [mock_out_pil]
    mock_pipeline.return_value = mock_result
    mock_pipeline.set_ip_adapter_scale = MagicMock()

    renderer.pipeline = mock_pipeline
    renderer.has_controlnet = True
    renderer.has_ip_adapter = True

    raw_frame = np.full((100, 100, 3), 100, dtype=np.uint8)
    ref_rgb = np.full((100, 100, 3), 250, dtype=np.uint8)
    control_map = ControlMap(
        edge_map=np.full((100, 100, 3), 30, dtype=np.uint8),
        combined_control=np.full((100, 100, 3), 255, dtype=np.uint8),
    )

    output = renderer.render(raw_frame, control_map, reference_rgb=ref_rgb)

    call_kwargs = mock_pipeline.call_args[1]
    assert "control_image" in call_kwargs
    assert call_kwargs["controlnet_conditioning_scale"] == 0.85
    assert "ip_adapter_image" in call_kwargs
    mock_pipeline.set_ip_adapter_scale.assert_called_with(0.75)
    assert output.shape == (100, 100, 3)


def test_diffusion_renderer_inference_exception_falls_back():
    """Verifies that an inference crash in the diffusion pipeline gracefully falls back to procedural rendering."""
    config = RenderConfig(model_id="mock_model")
    renderer = DiffusionRenderer(config)

    mock_pipeline = MagicMock()
    mock_pipeline.side_effect = RuntimeError("CUDA Out of Memory in Mock Pipeline")
    renderer.pipeline = mock_pipeline

    raw_frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    control_map = ControlMap(edge_map=np.zeros((100, 100, 3), dtype=np.uint8))

    output = renderer.render(raw_frame, control_map)

    # Should not raise, falls back to OpenCV renderer cleanly
    assert output.shape == (100, 100, 3)
    assert output.dtype == np.uint8


def test_style_engine_passes_reference_to_diffusion():
    """Verifies that StyleEngine routes reference_rgb to DiffusionRenderer."""
    config = RenderConfig(
        backend=RendererBackend.DIFFUSERS,
        model_id="mock_model",
        identity_adapter_model_id="mock_ip_adapter",
    )
    engine = StyleEngine(config)

    mock_pipeline = MagicMock()
    mock_out_pil = Image.fromarray(np.full((80, 80, 3), 190, dtype=np.uint8))
    mock_result = MagicMock()
    mock_result.images = [mock_out_pil]
    mock_pipeline.return_value = mock_result

    engine.renderer.pipeline = mock_pipeline
    engine.renderer.has_ip_adapter = True

    raw_frame = np.full((80, 80, 3), 100, dtype=np.uint8)
    ref_rgb = np.full((80, 80, 3), 200, dtype=np.uint8)

    art = engine.render_frame(raw_frame, reference_rgb=ref_rgb, stabilize=False)

    assert mock_pipeline.called
    call_kwargs = mock_pipeline.call_args[1]
    assert "ip_adapter_image" in call_kwargs
    assert art.shape == (80, 80, 3)
