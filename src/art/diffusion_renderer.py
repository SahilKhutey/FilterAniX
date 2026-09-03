"""Pluggable Diffusion & Generative Model Renderer Interface with ControlNet and Identity Conditioning."""
import logging
from typing import Optional
import numpy as np

from src.art.types import StylePreset, RenderConfig
from src.art.preprocess import ControlMap
from src.art.opencv_renderer import OpenCVIllustrationRenderer
from src.vision.models import FrameVisionData

logger = logging.getLogger("filteranix.art.diffusion")


class DiffusionRenderer:
    """Generative Diffusers/PyTorch renderer interface with ControlNet structural guidance,
    IP-Adapter identity reference conditioning, and automatic procedural fallback."""

    def __init__(self, config: Optional[RenderConfig] = None):
        self.config = config or RenderConfig()
        self.pipeline = None
        self.has_controlnet: bool = False
        self.has_ip_adapter: bool = False
        self._fallback = OpenCVIllustrationRenderer(self.config.style)
        self._init_diffusers()

    def _init_diffusers(self):
        """Attempts to initialize torch, diffusers, ControlNet, and IP-Adapter if configured and available."""
        if not self.config.model_id:
            return

        try:
            import torch
            from PIL import Image

            device = "cuda" if torch.cuda.is_available() and self.config.device != "cpu" else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32

            # 1. Initialize Base Pipeline (with or without ControlNet)
            if self.config.controlnet_model_id:
                try:
                    from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
                    controlnet = ControlNetModel.from_pretrained(
                        self.config.controlnet_model_id,
                        torch_dtype=dtype,
                    )
                    self.pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                        self.config.model_id,
                        controlnet=controlnet,
                        torch_dtype=dtype,
                    ).to(device)
                    self.has_controlnet = True
                except Exception as e:
                    logger.warning(
                        f"Failed to load ControlNet model '{self.config.controlnet_model_id}': {e}. "
                        "Falling back to standard img2img."
                    )
                    from diffusers import StableDiffusionImg2ImgPipeline
                    self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                        self.config.model_id,
                        torch_dtype=dtype,
                    ).to(device)
                    self.has_controlnet = False
            else:
                from diffusers import StableDiffusionImg2ImgPipeline
                self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                    self.config.model_id,
                    torch_dtype=dtype,
                ).to(device)
                self.has_controlnet = False

            # 2. Attach Identity Reference Adapter (IP-Adapter)
            if self.config.identity_adapter_model_id and self.pipeline is not None:
                try:
                    self.pipeline.load_ip_adapter(self.config.identity_adapter_model_id)
                    if hasattr(self.pipeline, "set_ip_adapter_scale"):
                        self.pipeline.set_ip_adapter_scale(self.config.identity_conditioning_scale)
                    self.has_ip_adapter = True
                except Exception as e:
                    logger.warning(f"Failed to load Identity Adapter '{self.config.identity_adapter_model_id}': {e}.")
                    self.has_ip_adapter = False
            else:
                self.has_ip_adapter = False

        except Exception as e:
            logger.warning(f"Failed to initialize Diffusers pipeline for '{self.config.model_id}': {e}")
            self.pipeline = None
            self.has_controlnet = False
            self.has_ip_adapter = False

    def render(
        self,
        rgb: np.ndarray,
        control_map: ControlMap,
        vision_data: Optional[FrameVisionData] = None,
        reference_rgb: Optional[np.ndarray] = None,
        reference_strength: Optional[float] = None,
        denoise_strength: Optional[float] = None,
    ) -> np.ndarray:
        """Executes diffusion inference with structural & identity conditioning or falls back to procedural illustration."""
        if self.pipeline is not None:
            try:
                from PIL import Image

                effective_reference_strength = (
                    getattr(self.config, "identity_conditioning_scale", 0.70)
                    if reference_strength is None
                    else float(reference_strength)
                )

                effective_denoise_strength = (
                    getattr(self.config, "denoise_strength", 0.35)
                    if denoise_strength is None
                    else float(denoise_strength)
                )

                init_image = Image.fromarray(rgb)
                prompt = self.config.style.prompt if hasattr(self.config, "style") and hasattr(self.config.style, "prompt") else self.config.positive_prompt
                negative_prompt = self.config.style.negative_prompt if hasattr(self.config, "style") and hasattr(self.config.style, "negative_prompt") else self.config.negative_prompt

                kwargs = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "image": init_image,
                    "strength": effective_denoise_strength,
                    "guidance_scale": getattr(self.config, "guidance_scale", 6.5),
                    "num_inference_steps": getattr(self.config, "inference_steps", 20),
                }

                # 1. Structural Conditioning (ControlNet) - only if actually using ControlNet pipeline
                use_controlnet = self.has_controlnet and getattr(self.config, "use_controlnet", True)
                control_img_arr = None
                if control_map is not None:
                    if control_map.combined_control is not None:
                        control_img_arr = control_map.combined_control
                    elif control_map.edge_map is not None:
                        control_img_arr = control_map.edge_map

                if use_controlnet and control_img_arr is not None:
                    kwargs["control_image"] = Image.fromarray(control_img_arr)
                    kwargs["controlnet_conditioning_scale"] = getattr(self.config, "controlnet_conditioning_scale", 0.8)

                # 2. Identity Reference Conditioning (IP-Adapter)
                use_identity = self.has_ip_adapter or (getattr(self.config, "identity_adapter_model_id", None) is not None)
                ref_pil = None
                if reference_rgb is not None:
                    ref_pil = Image.fromarray(reference_rgb)
                elif getattr(self.config, "reference_image_path", None):
                    try:
                        ref_pil = Image.open(self.config.reference_image_path).convert("RGB")
                    except Exception:
                        ref_pil = None

                if use_identity and ref_pil is not None:
                    kwargs["ip_adapter_image"] = ref_pil
                    if hasattr(self.pipeline, "set_ip_adapter_scale"):
                        self.pipeline.set_ip_adapter_scale(effective_reference_strength)

                result = self.pipeline(**kwargs).images[0]
                return np.array(result)
            except Exception as e:
                logger.warning(f"Diffusion rendering step failed ({e}), falling back to procedural engine.")

        # Fallback to high-precision procedural renderer
        return self._fallback.render(rgb, control_map, vision_data, reference_rgb)
