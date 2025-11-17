import warnings


def get_device():
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"GPU check failed, defaulting to cpu: {exc}")
    return "cpu"


def maybe_cast_tensor(tensor):
    device = get_device()
    try:
        return tensor.to(device)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"Falling back to CPU tensor: {exc}")
        return tensor
