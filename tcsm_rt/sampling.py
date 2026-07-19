from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def scatter_indices(points: np.ndarray, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(len(points), size=count, replace=False)).astype(np.int64)


def _nearest_unique(points: np.ndarray, targets: np.ndarray, count: int) -> np.ndarray:
    tree = cKDTree(points)
    _, indices = tree.query(targets, k=1)
    ordered = list(dict.fromkeys(np.asarray(indices, dtype=np.int64).tolist()))
    if len(ordered) < count:
        missing = np.setdiff1d(np.arange(len(points)), np.asarray(ordered), assume_unique=False)
        ordered.extend(missing[: count - len(ordered)].tolist())
    return np.sort(np.asarray(ordered[:count], dtype=np.int64))


def trajectory_indices(points: np.ndarray, count: int, seed: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    rng = np.random.default_rng(seed)
    lower = points[:, :2].min(axis=0)
    upper = points[:, :2].max(axis=0)
    start = rng.uniform(lower, upper)
    end = rng.uniform(lower, upper)
    steps = np.linspace(0.0, 1.0, max(count * 4, 32))[:, None]
    targets = start[None, :] * (1.0 - steps) + end[None, :] * steps
    return _nearest_unique(points[:, :2], targets, count)


def coverage_trajectory_indices(points: np.ndarray, count: int, seed: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    rng = np.random.default_rng(seed)
    lower = points[:, :2].min(axis=0)
    upper = points[:, :2].max(axis=0)
    rows = max(2, int(np.ceil(np.sqrt(count))))
    y_values = np.linspace(lower[1], upper[1], rows)
    targets: list[np.ndarray] = []
    per_row = max(2, int(np.ceil(count * 2 / rows)))
    for row, y_value in enumerate(y_values):
        x_values = np.linspace(lower[0], upper[0], per_row)
        if row % 2:
            x_values = x_values[::-1]
        targets.append(np.column_stack([x_values, np.full_like(x_values, y_value)]))
    path = np.concatenate(targets, axis=0)
    path += rng.normal(scale=1e-6, size=path.shape)
    return _nearest_unique(points[:, :2], path, count)


def sample_indices(points: np.ndarray, count: int, mode: str, seed: int) -> np.ndarray:
    if not 1 <= count < len(points):
        raise ValueError(f"sample count {count} is invalid for {len(points)} points")
    if mode == "scatter":
        return scatter_indices(points, count, seed)
    if mode == "trajectory":
        return trajectory_indices(points, count, seed)
    if mode == "coverage_trajectory":
        return coverage_trajectory_indices(points, count, seed)
    raise ValueError(f"unknown sampling mode: {mode}")


def valid_query_indices(arrays: dict[str, np.ndarray]) -> np.ndarray:
    count = len(arrays["query_xyz_m"])
    mask = np.asarray(arrays.get("valid_query_mask", np.ones(count, dtype=bool)), dtype=bool)
    if mask.shape != (count,):
        raise ValueError(f"valid_query_mask must have shape ({count},), received {mask.shape}")
    indices = np.flatnonzero(mask)
    if len(indices) < 2:
        raise ValueError("a scene must contain at least two valid query positions")
    return indices.astype(np.int64)


def sample_scene_indices(
    arrays: dict[str, np.ndarray],
    count: int,
    mode: str,
    seed: int,
) -> np.ndarray:
    valid = valid_query_indices(arrays)
    local_count = min(int(count), len(valid) - 1)
    local = sample_indices(arrays["query_xyz_m"][valid], local_count, mode, seed)
    return valid[local]
