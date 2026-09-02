"""Pluggable Diffusion & Generative Model Renderer Interface."""
from typing import Optional
import numpy as np

from src.art.types import StylePreset, RenderConfig
from src.art.preprocess import ControlMap
from src.art.opencv_renderer import OpenCVIllustrationRenderer
from src.vision.models import FrameVisionData


class DiffusionRenderer:
    """Generative Diffusers/PyTorch renderer interface with automatic CPU/GPU model loading and fallback."""

    def __init__(self, config: Optional[RenderConfig] = None):
        self.config = config or RenderConfig()
        self.pipeline = None
        self._fallback = OpenCVIllustrationRenderer(self.config.style)
        self._init_diffusers()

    def _init_diffusers(self):
        """Attempts to initialize torch and diffusers if configured and available."""
        if not self.config.model_id:
            return

        try:
            import torch
            from diffusers import StableDiffusionImg2ImgPipeline
            device = "cuda" if torch.cuda.is_available() and self.config.device != "cpu" else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            
            self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.config.model_id,
                torch_dtype=dtype,
            ).to(device)
        except Exception:
            self.pipeline = None

    def render(
        self,
        rgb: np.ndarray,
        control_map: ControlMap,
        vision_data: Optional[FrameVisionData] = None,
        reference_rgb: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Executes diffusion inference or falls back to procedural illustration."""
        if self.pipeline is not None:
            try:
                from PIL import Image
                init_image = Image.fromarray(rgb)
                prompt = self.config.style.prompt
                negative_prompt = self.config.style.negative_prompt
                
                result = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=init_image,
                    strength=0.55,
                    guidance_scale=7.5,
                ).images[0]
                return np.array(result)
            except Exception:
                pass

        # Fallback to high-precision procedural renderer
        return self._fallback.render(rgb, control_map, vision_data, reference_rgb)
