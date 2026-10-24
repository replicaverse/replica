"""Shared device resolution for the LM backends."""


def resolve_device(device: str | None = None) -> str:
    """Resolve a torch device string.

    Args:
        device: An explicit torch device such as ``"cuda"``, ``"cuda:1"`` or
            ``"cpu"``, returned unchanged. ``None`` selects ``"cuda"`` when a
            CUDA device is visible to torch, else ``"cpu"``.
    """
    if device is not None:
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


DTYPES = ("float32", "float16", "bfloat16")
DEVICE_MAPS = ("auto",)


def validate_dtype(dtype: str | None) -> str | None:
    """Return a supported parameter precision name, or ``None`` for the default.

    Args:
        dtype: ``float32``, ``float16``, ``bfloat16`` or ``None``.
    """
    if dtype is not None and dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {DTYPES}, got {dtype!r}")
    return dtype


def validate_device_map(device_map: str | None) -> str | None:
    """Return a supported device-map policy, or ``None`` for a single device.

    Args:
        device_map: ``auto`` or ``None``.
    """
    if device_map is not None and device_map not in DEVICE_MAPS:
        raise ValueError(f"device_map must be one of {DEVICE_MAPS}, got {device_map!r}")
    return device_map


def loading_options(dtype: str | None, device_map: str | None) -> dict:
    """Keyword arguments for ``from_pretrained`` that set precision and placement.

    Args:
        dtype: Validated precision name, or ``None`` to leave the default.
        device_map: Validated device-map policy, or ``None`` for a single device.
    """
    import torch

    options: dict = {}
    if dtype is not None:
        options["torch_dtype"] = getattr(torch, dtype)
    if device_map is not None:
        options["device_map"] = device_map
    return options
