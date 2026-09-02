from __future__ import annotations


class IdentityRetryPolicy:
    """Policy governing automatic re-rendering attempts when generated frame identity drifts."""

    def __init__(
        self,
        threshold: float = 0.62,
        max_attempts: int = 2,
    ):
        self.threshold = threshold
        self.max_attempts = max_attempts

    def should_retry(
        self,
        similarity: float,
        attempt: int,
    ) -> bool:
        if attempt >= self.max_attempts:
            return False

        return similarity < self.threshold

    def next_strength(
        self,
        current_strength: float,
    ) -> float:
        return min(
            0.95,
            current_strength + 0.10,
        )
