from __future__ import annotations

import os


def configure_mitsuba_variant() -> str | None:
    """Select an explicit Mitsuba backend before importing :mod:`sionna.rt`.

    Sionna otherwise prefers CUDA whenever PyTorch exposes a CUDA device. CUDA
    visibility does not imply that the host provides the OptiX runtime required
    by Mitsuba. The environment override lets WSL run official Sionna RT kernels
    with LLVM while preserving CUDA for PyTorch training in the same process.
    """
    requested = os.environ.get("TCSM_MITSUBA_VARIANT")
    if not requested:
        return None
    import mitsuba as mi

    current = mi.variant()
    if current is None:
        mi.set_variant(requested)
    elif current != requested:
        raise RuntimeError(
            f"Mitsuba variant was already set to {current!r}; requested {requested!r}"
        )
    return mi.variant()
