"""Offline High-Quality Video Stylization Pipeline."""
from pathlib import Path
from typing import Callable, Optional
import cv2
import numpy as np
from tqdm import tqdm

from filteranix.core.config import FilterAniXConfig
from filteranix.core.frame import FrameData
from filteranix.vision.segmenter import VideoSegmenter, BackgroundPlateBuilder
from filteranix.vision.pose_tracker import PoseTracker
from filteranix.vision.depth_estimator import DepthEstimator
from filteranix.stylization.ai_renderer import StyleRenderer
from filteranix.temporal.anchor_manager import AnchorManager
from filteranix.temporal.warp_blender import TemporalWarpBlender
from filteranix.temporal.deflicker import TemporalDeflicker
from filteranix.compositor.compositor import FrameCompositor


class OfflineVideoPipeline:
    """End-to-end multi-pass video stylization pipeline optimized for consistency, quality, and zero-flicker."""

    def __init__(self, config: FilterAniXConfig):
        self.config = config
        
        # Vision Subsystem
        self.segmenter = VideoSegmenter(use_mediapipe=config.pipeline.vision.enable_segmentation)
        self.pose_tracker = PoseTracker(
            enable_face=config.pipeline.vision.enable_face_mesh,
            enable_pose=config.pipeline.vision.enable_pose_tracking,
            enable_hands=config.pipeline.vision.enable_hand_tracking,
        )
        self.depth_estimator = DepthEstimator()

        # Stylization & Rendering Subsystem
        self.style_renderer = StyleRenderer(config.style, config.character)

        # Temporal Coherence Subsystem
        self.anchor_manager = AnchorManager(self.style_renderer, config)
        self.warp_blender = TemporalWarpBlender(config.pipeline.temporal)
        self.deflicker = TemporalDeflicker(config.pipeline.temporal)

        # Compositing Subsystem
        self.compositor = FrameCompositor(config.pipeline.compositor)

    def process_video(
        self,
        input_path: str | Path,
        output_path: str | Path,
        side_by_side: bool = False,
        max_frames: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ):
        """Processes entire video through the multi-pass stylization pipeline."""
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open input video: {input_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames is not None and max_frames > 0:
            total_frames = min(total_frames, max_frames)

        fps = cap.get(cv2.CAP_PROP_FPS) or self.config.pipeline.target_fps
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        target_w = self.config.pipeline.target_width or src_w
        target_h = self.config.pipeline.target_height or src_h

        out_w = target_w * 2 if side_by_side else target_w
        out_h = target_h

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))

        # ==========================================
        # PASS 1: Accumulate Static Background Plate
        # ==========================================
        print(f"[*] Pass 1: Analyzing scene & accumulating static background plate...")
        plate_builder = BackgroundPlateBuilder(
            target_frames=self.config.pipeline.vision.background_accumulation_frames
        )

        acc_frames = min(self.config.pipeline.vision.background_accumulation_frames, total_frames)
        first_frame = None

        for idx in range(acc_frames):
            ret, bgr = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rgb.shape[1] != target_w or rgb.shape[0] != target_h:
                rgb = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)

            if first_frame is None:
                first_frame = rgb.copy()

            person_mask, bg_mask = self.segmenter.segment_frame(rgb)
            plate_builder.add_frame(rgb, bg_mask)

        raw_bg_plate = plate_builder.build(fallback_frame=first_frame)
        self.anchor_manager.initialize_static_background(raw_bg_plate)
        stylized_bg = self.anchor_manager.get_stylized_background()
        print(f"[+] Static Background Anchor locked (0% room drift across entire video).")

        # ==========================================
        # PASS 2: Temporal Stylization & Compositing
        # ==========================================
        print(f"[*] Pass 2: Rendering stylized video with temporal coherence...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        prev_stylized_fg = None
        prev_raw_rgb = None

        pbar = tqdm(total=total_frames, desc="Stylizing Frames", unit="frame")

        for frame_idx in range(total_frames):
            ret, bgr = cap.read()
            if not ret:
                break

            raw_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if raw_rgb.shape[1] != target_w or raw_rgb.shape[0] != target_h:
                raw_rgb = cv2.resize(raw_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)

            # 1. Vision & Landmark Analysis
            person_mask, bg_mask = self.segmenter.segment_frame(raw_rgb)
            landmarks = self.pose_tracker.process(raw_rgb)
            depth_map = self.depth_estimator.estimate_depth(raw_rgb, person_mask=person_mask)

            # 2. Render Stylized Foreground Layer
            fresh_stylized_fg = self.style_renderer.render_layer(
                raw_rgb, mask=person_mask, depth_map=depth_map, is_foreground=True
            )

            # 3. Temporal Warp Blending
            if prev_stylized_fg is not None and self.config.pipeline.temporal.enable_temporal_warping:
                stylized_fg, _ = self.warp_blender.blend_with_prior(
                    curr_stylized_rgb=fresh_stylized_fg,
                    prev_stylized_rgb=prev_stylized_fg,
                    curr_raw_rgb=raw_rgb,
                    prev_raw_rgb=prev_raw_rgb,
                    person_mask=person_mask,
                )
            else:
                stylized_fg = fresh_stylized_fg

            prev_stylized_fg = stylized_fg.copy()
            prev_raw_rgb = raw_rgb.copy()

            # 4. Extract Dynamic Line Art (XDoG)
            line_art = self.style_renderer.extract_lines(raw_rgb, depth_map=depth_map)

            # 5. Composite Layers
            composite = self.compositor.composite(
                foreground_rgb=stylized_fg,
                background_rgb=stylized_bg,
                person_mask=person_mask,
                line_art_rgb=line_art,
            )

            # 6. Temporal Deflicker Post-Processing
            final_frame = self.deflicker.process(composite)

            # 7. Write output
            if side_by_side:
                combined_rgb = np.hstack([raw_rgb, final_frame])
                writer.write(cv2.cvtColor(combined_rgb, cv2.COLOR_RGB2BGR))
            else:
                writer.write(cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR))

            pbar.update(1)
            if progress_callback:
                progress_callback(frame_idx + 1, total_frames)

        pbar.close()
        cap.release()
        writer.release()
        print(f"[SUCCESS] Exported stylized video to: {output_path}")
