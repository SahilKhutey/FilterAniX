from __future__ import annotations


def get_device(
    requested: str = "auto",
) -> str:

    requested = requested.lower()

    if requested == "cpu":
        return "cpu"

    if requested == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"

            raise RuntimeError(
                "CUDA requested but unavailable."
            )

        except ImportError as exc:
            raise RuntimeError(
                "CUDA requested but PyTorch is not installed."
            ) from exc

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"

    except Exception:
        pass

    return "cpu"
