import json

from src.art.render_metrics import RenderMetrics


def test_metrics_save(tmp_path):

    metrics = RenderMetrics(
        total_frames=100,
        keyframes=10,
        propagated_frames=90,
        diffusion_frames=8,
        fallback_frames=2,
    )

    path = (
        tmp_path
        / "render_metrics.json"
    )

    metrics.save(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    assert data["total_frames"] == 100
    assert data["keyframes"] == 10
    assert data["keyframe_ratio"] == 0.1
    assert data["propagation_ratio"] == 0.9
