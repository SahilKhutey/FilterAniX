from src.art.types import StyleConfig


def test_diffusion_parameters_are_configurable():
    config = StyleConfig()

    assert config.denoise_strength > 0
    assert config.guidance_scale > 0
    assert config.inference_steps > 0
    assert config.controlnet_conditioning_scale >= 0
    assert config.identity_conditioning_scale >= 0
