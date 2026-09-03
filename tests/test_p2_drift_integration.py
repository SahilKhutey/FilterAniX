import numpy as np

from src.consistency.identity_engine import IdentityEngine
from src.consistency.retry import IdentityRetryPolicy


def test_identity_engine_evaluates_and_retries():
    engine = IdentityEngine(
        warning_threshold=0.70,
        severe_threshold=0.50,
        max_retries=2,
        bank_size=4,
    )

    # Base reference image
    ref_img = np.zeros((128, 128, 3), dtype=np.uint8)
    ref_img[30:90, 30:90] = 200

    # Initial frame sets reference
    eval_0 = engine.evaluate(
        frame_index=0,
        scene_id=0,
        image_rgb=ref_img,
    )
    assert eval_0.metric.overall == 1.0
    assert not eval_0.metric.warning

    # Candidate frame with heavy corruption / drift
    corrupted_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    eval_1 = engine.evaluate(
        frame_index=1,
        scene_id=0,
        image_rgb=corrupted_img,
    )

    assert eval_1.metric.warning
    retry_dec = engine.retry_policy.decide(
        score=eval_1.metric.overall,
        attempt=0,
        base_reference_strength=0.70,
        base_denoise_strength=0.35,
    )
    assert retry_dec.retry
    assert retry_dec.attempt == 1
    assert retry_dec.reference_strength > 0.70
