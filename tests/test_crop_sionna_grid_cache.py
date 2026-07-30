from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.crop_sionna_grid_cache import (
    center_crop_indices,
    crop_scene_arrays,
    crop_sionna_cache,
)
from tcsm_rt.provenance import sha256_file, write_json_atomic
from tcsm_rt.schema import load_scene, save_scene


def _scene(side: int = 5) -> dict[str, np.ndarray]:
    count = side * side
    scalar = np.arange(count, dtype=np.float32)
    return {
        "query_xyz_m": np.column_stack((scalar, scalar + 100, np.ones(count))).astype(np.float32),
        "environment": np.column_stack((scalar, scalar + 1)).astype(np.float32),
        "channel": (scalar[:, None] + 1j * scalar[:, None]).astype(np.complex64),
        "rss_db": scalar - 100,
        "regime": (np.arange(count) % 3).astype(np.int64),
        "best_far_idx": (np.arange(count) % 5).astype(np.int64),
        "best_near_angle": (np.arange(count) % 5).astype(np.int64),
        "best_near_range": (np.arange(count) % 2).astype(np.int64),
        "far_rates": np.column_stack((scalar, scalar + 1)).astype(np.float32),
        "near_rates": np.column_stack((scalar, scalar + 1)).astype(np.float32),
        "oracle_rate_bps_hz": scalar.astype(np.float32),
        "valid_query_mask": np.ones(count, dtype=bool),
        "far_angle_axis_deg": np.linspace(-60, 60, 5, dtype=np.float32),
    }


def test_center_crop_indices_preserve_row_major_geometry() -> None:
    assert center_crop_indices(5, 3).tolist() == [6, 7, 8, 11, 12, 13, 16, 17, 18]


def test_crop_scene_arrays_only_slices_query_aligned_arrays() -> None:
    cropped = crop_scene_arrays(_scene(), source_grid_size=5, target_grid_size=3)
    assert cropped["query_xyz_m"].shape == (9, 3)
    assert cropped["query_xyz_m"][:, 0].tolist() == [6, 7, 8, 11, 12, 13, 16, 17, 18]
    assert cropped["far_angle_axis_deg"].shape == (5,)


def test_crop_cache_records_parent_hash(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    destination = tmp_path / "derived" / "scene.npz"
    save_scene(source, _scene())
    write_json_atomic(
        source.with_suffix(".json"),
        {
            "config_id": "sionna_train_000",
            "query_count": 25,
            "cache_sha256": sha256_file(source),
        },
    )
    metadata = crop_sionna_cache(
        source,
        destination,
        source_grid_size=5,
        target_grid_size=3,
    )
    assert metadata["query_count"] == 9
    assert metadata["derived_cache"]["source_cache_sha256"] == sha256_file(source)
    assert metadata["derived_cache"]["axis_slice"] == [1, 4]
    assert len(load_scene(destination)["query_xyz_m"]) == 9
