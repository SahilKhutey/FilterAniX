from src.art.mathematical.config import (
    MathematicalAnimeStyle,
)


def test_default_configuration_is_valid():

    style = MathematicalAnimeStyle.creator_anime()

    assert style.output_dtype == "uint8"

    assert style.contrast > 0

    assert style.gamma > 0

    assert len(style.palette) >= 2


def test_palette_is_rgb():

    style = MathematicalAnimeStyle.creator_anime()

    for color in style.palette:

        assert len(color) == 3

        for channel in color:

            assert 0 <= channel <= 255


def test_values_are_clamped():

    style = MathematicalAnimeStyle(
        contrast=100,
        gamma=-100,
        saturation=100,
        palette_mix=100,
        edge_strength=100,
        shadow_strength=100,
    ).validated()

    assert style.contrast == 1.8

    assert style.gamma == 0.5

    assert style.saturation == 2.0

    assert style.palette_mix == 1.0

    assert style.edge_strength == 1.0

    assert style.shadow_strength == 1.0


def test_invalid_palette_is_rejected():

    try:

        MathematicalAnimeStyle(
            palette=((255, 0),)
        ).validated()

        assert False

    except ValueError:

        assert True


def test_invalid_output_dtype_is_rejected():

    try:

        MathematicalAnimeStyle(
            output_dtype="float32"
        ).validated()

        assert False

    except ValueError:

        assert True


def test_serialization():

    style = MathematicalAnimeStyle.creator_anime()

    data = style.to_dict()

    assert isinstance(data, dict)

    assert "contrast" in data

    assert "palette" in data

    assert "temporal_strength" in data
