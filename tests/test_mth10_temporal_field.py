from __future__ import annotations

import numpy as np
import pytest

from src.art.mathematical import (
    MathematicalAnimeStyle,
    MathematicalTemporalField,
    TemporalObservation,
)


def make_style():
    return MathematicalAnimeStyle.creator_anime()


def make_frame(
    height: int = 64,
    width: int = 96,
    value: int = 128,
) -> np.ndarray:
    frame = np.full(
        (height, width, 3),
        value,
        dtype=np.uint8,
    )

    return frame


def make_gradient(
    height: int = 64,
    width: int = 96,
) -> np.ndarray:
    x = np.linspace(
        0,
        255,
        width,
        dtype=np.uint8,
    )

    row = np.repeat(
        x[:, None],
        3,
        axis=1,
    )

    frame = np.repeat(
        row[None, :, :],
        height,
        axis=0,
    )

    return frame


def make_flow(
    height: int = 64,
    width: int = 96,
    dx: float = 0.0,
    dy: float = 0.0,
) -> np.ndarray:
    flow = np.zeros(
        (height, width, 2),
        dtype=np.float32,
    )

    flow[..., 0] = dx
    flow[..., 1] = dy

    return flow


def test_output_shape():
    engine = MathematicalTemporalField(
        make_style()
    )

    frame = make_frame()

    result = engine.transform(frame)

    assert result.output_rgb.shape == frame.shape


def test_output_range():
    engine = MathematicalTemporalField(
        make_style()
    )

    frame = make_gradient()

    result = engine.transform(frame)

    assert np.all(result.output_rgb >= 0.0)
    assert np.all(result.output_rgb <= 1.0)


def test_render_returns_uint8():
    engine = MathematicalTemporalField(
        make_style()
    )

    frame = make_frame()

    output = engine.render(frame)

    assert output.dtype == np.uint8
    assert output.shape == frame.shape


def test_first_frame_has_no_temporal_blend():
    engine = MathematicalTemporalField(
        make_style()
    )

    frame = make_gradient()

    result = engine.transform(frame)

    np.testing.assert_allclose(
        result.output_rgb,
        result.current_rgb,
        atol=1e-6,
    )


def test_second_identical_frame_is_stable():
    engine = MathematicalTemporalField(
        make_style()
    )

    frame = make_gradient()

    engine.transform(frame)

    result = engine.transform(
        frame,
        TemporalObservation(
            optical_flow=make_flow(),
        ),
    )

    difference = np.mean(
        np.abs(
            result.output_rgb -
            result.current_rgb
        )
    )

    assert difference < 1e-5


def test_motion_field_is_zero_for_zero_flow():
    engine = MathematicalTemporalField(
        make_style()
    )

    flow = make_flow(
        dx=0.0,
        dy=0.0,
    )

    result = engine.transform(
        make_frame(),
        TemporalObservation(
            optical_flow=flow,
        ),
    )

    assert np.allclose(
        result.flow_magnitude,
        0.0,
    )


def test_motion_stability_is_one_for_zero_flow():
    engine = MathematicalTemporalField(
        make_style()
    )

    flow = make_flow()

    result = engine.transform(
        make_frame(),
        TemporalObservation(
            optical_flow=flow,
        ),
    )

    assert np.allclose(
        result.motion_stability,
        1.0,
    )


def test_motion_stability_decreases_with_motion():
    engine = MathematicalTemporalField(
        make_style()
    )

    flow = make_flow(
        dx=1.0,
    )

    result = engine.transform(
        make_frame(),
        TemporalObservation(
            optical_flow=flow,
        ),
    )

    assert float(
        np.mean(result.motion_stability)
    ) < 1.0


def test_temporal_strength_is_bounded():
    engine = MathematicalTemporalField(
        make_style()
    )

    frame = make_gradient()

    engine.transform(frame)

    flow = make_flow()

    result = engine.transform(
        frame,
        TemporalObservation(
            optical_flow=flow,
        ),
    )

    assert np.all(
        result.temporal_strength >= 0.0
    )

    assert np.all(
        result.temporal_strength <= 1.0
    )


def test_scene_cut_disables_temporal_blend():
    engine = MathematicalTemporalField(
        make_style()
    )

    frame_a = make_frame(
        value=80,
    )

    frame_b = make_frame(
        value=220,
    )

    engine.transform(frame_a)

    result = engine.transform(
        frame_b,
        TemporalObservation(
            optical_flow=make_flow(),
            scene_cut=True,
        ),
    )

    np.testing.assert_allclose(
        result.output_rgb,
        result.current_rgb,
        atol=1e-6,
    )

    assert np.allclose(
        result.temporal_strength,
        0.0,
    )


def test_scene_cut_preserves_new_frame_as_history():
    engine = MathematicalTemporalField(
        make_style()
    )

    frame_a = make_frame(
        value=50,
    )

    frame_b = make_frame(
        value=200,
    )

    engine.transform(frame_a)

    engine.transform(
        frame_b,
        TemporalObservation(
            scene_cut=True,
        ),
    )

    assert engine.has_previous_frame


def test_reset_clears_history():
    engine = MathematicalTemporalField(
        make_style()
    )

    frame = make_frame()

    engine.transform(frame)

    assert engine.has_previous_frame

    engine.reset()

    assert not engine.has_previous_frame


def test_float_input_supported():
    engine = MathematicalTemporalField(
        make_style()
    )

    frame = np.full(
        (32, 48, 3),
        0.5,
        dtype=np.float32,
    )

    result = engine.transform(frame)

    assert result.output_rgb.dtype == np.float32
    assert result.output_rgb.shape == frame.shape


def test_invalid_frame_rejected():
    engine = MathematicalTemporalField(
        make_style()
    )

    bad = np.zeros(
        (64, 64),
        dtype=np.uint8,
    )

    with pytest.raises(ValueError):
        engine.transform(bad)


def test_invalid_flow_shape_rejected():
    engine = MathematicalTemporalField(
        make_style()
    )

    frame = make_frame()

    bad_flow = np.zeros(
        (32, 32, 2),
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        engine.transform(
            frame,
            TemporalObservation(
                optical_flow=bad_flow,
            ),
        )


def test_nonfinite_flow_rejected():
    engine = MathematicalTemporalField(
        make_style()
    )

    frame = make_frame()

    flow = make_flow()

    flow[10, 10, 0] = np.nan

    with pytest.raises(ValueError):
        engine.transform(
            frame,
            TemporalObservation(
                optical_flow=flow,
            ),
        )


def test_resolution_change_rejected():
    engine = MathematicalTemporalField(
        make_style()
    )

    engine.transform(
        make_frame(
            height=64,
            width=96,
        )
    )

    with pytest.raises(ValueError):
        engine.transform(
            make_frame(
                height=32,
                width=48,
            )
        )
