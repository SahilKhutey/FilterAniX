from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RetryDecision:

    retry: bool

    attempt: int

    reference_strength: float

    denoise_strength: float

    reason: str


class IdentityRetryPolicy:

    def __init__(
        self,
        max_retries: int = 2,
        warning_threshold: float = 0.62,
        severe_threshold: float = 0.48,
        threshold: Optional[float] = None,
        max_attempts: Optional[int] = None,
    ):

        if threshold is not None:
            warning_threshold = threshold

        if max_attempts is not None:
            max_retries = max_attempts

        self.max_retries = max(
            0,
            int(max_retries),
        )

        self.warning_threshold = (
            float(warning_threshold)
        )

        self.severe_threshold = (
            float(severe_threshold)
        )

    def should_retry(
        self,
        similarity: float,
        attempt: int,
    ) -> bool:

        return (
            float(similarity) < self.warning_threshold
            and int(attempt) < self.max_retries
        )

    def next_strength(
        self,
        strength: float,
        step: float = 0.10,
    ) -> float:

        return min(
            1.0,
            float(strength) + float(step),
        )

    def decide(
        self,
        score: float,
        attempt: int,
        base_reference_strength: float,
        base_denoise_strength: float,
    ) -> RetryDecision:

        if (
            score >= self.warning_threshold
            or attempt >= self.max_retries
        ):

            return RetryDecision(
                retry=False,
                attempt=attempt,
                reference_strength=base_reference_strength,
                denoise_strength=base_denoise_strength,
                reason="identity_accepted",
            )

        if score < self.severe_threshold:

            reference_strength = min(
                1.0,
                base_reference_strength
                + 0.15,
            )

            denoise_strength = max(
                0.15,
                base_denoise_strength
                - 0.10,
            )

            reason = (
                "severe_identity_drift"
            )

        else:

            reference_strength = min(
                1.0,
                base_reference_strength
                + 0.08,
            )

            denoise_strength = max(
                0.20,
                base_denoise_strength
                - 0.05,
            )

            reason = (
                "identity_warning"
            )

        return RetryDecision(
            retry=True,
            attempt=attempt + 1,
            reference_strength=reference_strength,
            denoise_strength=denoise_strength,
            reason=reason,
        )
