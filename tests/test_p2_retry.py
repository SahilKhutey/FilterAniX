from src.consistency.retry import (
    IdentityRetryPolicy,
)


def test_identity_retry():

    policy = IdentityRetryPolicy(
        max_retries=2
    )

    result = policy.decide(
        score=0.40,
        attempt=0,
        base_reference_strength=0.70,
        base_denoise_strength=0.35,
    )

    assert result.retry
    assert result.attempt == 1
    assert result.reference_strength > 0.70
    assert result.denoise_strength < 0.35
