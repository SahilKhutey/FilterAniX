import json
import cv2
import numpy as np

from src.art.opencv_renderer import OpenCVArtRenderer
from src.art.style_controller import StyleController
from src.art.temporal import TemporalStabilizer
from src.art.types import StyleConfig


def test_opencv_renderer_preserves_dimensions():
    frame = np.zeros(
        (240, 320, 3),
        dtype=np.uint8,
    )
    frame[:, :, 1] = 180

    renderer = OpenCVArtRenderer()
    result = renderer.render(frame)

    assert result.shape == frame.shape
    assert result.dtype == np.uint8


def test_style_controller_builds_edge_control():
    frame = np.zeros(
        (240, 320, 3),
        dtype=np.uint8,
    )
    cv2.rectangle(
        frame,
        (50, 50),
        (200, 180),
        (255, 255, 255),
        3,
    )

    controller = StyleController(
        StyleConfig()
    )
    control = controller.build_control_map(
        frame,
        {},
    )

    assert control.edge_map is not None
    assert control.combined_control is not None
    assert (
        control.combined_control.shape
        == frame.shape[:2]
    )


def test_temporal_reset_on_scene_change():
    stabilizer = TemporalStabilizer(
        blend_strength=0.5
    )

    first = np.zeros(
        (20, 20, 3),
        dtype=np.uint8,
    )
    second = np.full(
        (20, 20, 3),
        255,
        dtype=np.uint8,
    )

    stabilizer.apply(
        first,
        scene_id=0,
    )
    result = stabilizer.apply(
        second,
        scene_id=1,
        scene_cut=True,
    )

    assert np.array_equal(
        result,
        second,
    )


def test_temporal_blending():
    stabilizer = TemporalStabilizer(
        blend_strength=0.5
    )

    first = np.zeros(
        (20, 20, 3),
        dtype=np.uint8,
    )
    second = np.full(
        (20, 20, 3),
        255,
        dtype=np.uint8,
    )

    stabilizer.apply(
        first,
        scene_id=0,
    )
    result = stabilizer.apply(
        second,
        scene_id=0,
    )

    assert 0 < int(result[0, 0, 0]) < 255
