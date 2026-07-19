from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_ARRAYS = (
    "query_xyz_m",
    "environment",
    "channel",
    "rss_db",
    "regime",
    "best_far_idx",
    "best_near_angle",
    "best_near_range",
    "far_rates",
    "near_rates",
    "oracle_rate_bps_hz",
)


@dataclass(frozen=True)
class SceneRecord:
    scene_id: str
    source: str
    split: str
    frequency_hz: float
    array_size: int
    path: Path
    metadata: dict[str, Any]


def validate_scene_arrays(arrays: dict[str, np.ndarray]) -> None:
    missing = [name for name in REQUIRED_ARRAYS if name not in arrays]
    if missing:
        raise ValueError(f"scene cache is missing arrays: {missing}")
    n_query = arrays["query_xyz_m"].shape[0]
    for name in REQUIRED_ARRAYS:
        if arrays[name].shape[0] != n_query:
            raise ValueError(f"{name} has {arrays[name].shape[0]} rows, expected {n_query}")
    if arrays["channel"].ndim != 2:
        raise ValueError("channel must have shape [query, tx_element]")
    if not np.iscomplexobj(arrays["channel"]):
        raise ValueError("channel must be complex")
    if arrays["environment"].ndim != 2:
        raise ValueError("environment must have shape [query, feature]")
    if "valid_query_mask" in arrays:
        valid_query_mask = np.asarray(arrays["valid_query_mask"])
        if valid_query_mask.shape != (n_query,):
            raise ValueError("valid_query_mask must contain one value per query")
        if np.sum(valid_query_mask.astype(bool)) < 2:
            raise ValueError("valid_query_mask must retain at least two query positions")
    if np.any(~np.isfinite(arrays["rss_db"])):
        raise ValueError("rss_db contains non-finite values")
    if np.any(arrays["oracle_rate_bps_hz"] < -1e-8):
        raise ValueError("oracle rate must be non-negative")


def save_scene(path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    validate_scene_arrays(arrays)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(destination)


def load_scene(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as cache:
        arrays = {key: cache[key] for key in cache.files}
    validate_scene_arrays(arrays)
    return arrays
