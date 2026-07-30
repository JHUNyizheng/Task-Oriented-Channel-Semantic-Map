from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from tcsm_rt.provenance import sha256_file, write_json_atomic
from tcsm_rt.schema import load_scene, save_scene


def center_crop_indices(source_grid_size: int, target_grid_size: int) -> np.ndarray:
    source = int(source_grid_size)
    target = int(target_grid_size)
    if source <= 0 or target <= 0 or target > source:
        raise ValueError("grid sizes must satisfy 0 < target <= source")
    if (source - target) % 2:
        raise ValueError("source and target grids must have the same parity")
    offset = (source - target) // 2
    grid = np.arange(source * source, dtype=np.int64).reshape(source, source)
    return grid[offset : offset + target, offset : offset + target].reshape(-1)


def crop_scene_arrays(
    arrays: dict[str, np.ndarray],
    *,
    source_grid_size: int,
    target_grid_size: int,
) -> dict[str, np.ndarray]:
    source_count = int(source_grid_size) ** 2
    if len(arrays["query_xyz_m"]) != source_count:
        raise ValueError(f"cache has {len(arrays['query_xyz_m'])} queries, expected {source_count}")
    indices = center_crop_indices(source_grid_size, target_grid_size)
    return {
        name: np.asarray(value)[indices]
        if np.asarray(value).shape[:1] == (source_count,)
        else value
        for name, value in arrays.items()
    }


def crop_sionna_cache(
    source: Path,
    destination: Path,
    *,
    source_grid_size: int = 35,
    target_grid_size: int = 25,
) -> dict[str, Any]:
    source = source.resolve()
    destination = destination.resolve()
    if source == destination:
        raise ValueError("source and destination caches must be different")
    source_metadata_path = source.with_suffix(".json")
    if not source_metadata_path.exists():
        raise FileNotFoundError(f"missing source metadata: {source_metadata_path}")
    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    source_hash = sha256_file(source)
    declared_hash = source_metadata.get("cache_sha256")
    if declared_hash is not None and str(declared_hash) != source_hash:
        raise ValueError("source cache SHA-256 does not match its metadata")

    cropped = crop_scene_arrays(
        load_scene(source),
        source_grid_size=source_grid_size,
        target_grid_size=target_grid_size,
    )
    save_scene(destination, cropped)
    offset = (int(source_grid_size) - int(target_grid_size)) // 2
    metadata = {
        **source_metadata,
        "query_count": int(target_grid_size) ** 2,
        "cache_sha256": sha256_file(destination),
        "derived_cache": {
            "operation": "exact_center_grid_crop",
            "source_cache": str(source),
            "source_cache_sha256": source_hash,
            "source_grid_size": int(source_grid_size),
            "target_grid_size": int(target_grid_size),
            "axis_slice": [offset, offset + int(target_grid_size)],
        },
    }
    write_json_atomic(destination.with_suffix(".json"), metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--source-grid-size", type=int, default=35)
    parser.add_argument("--target-grid-size", type=int, default=25)
    args = parser.parse_args()
    metadata = crop_sionna_cache(
        args.source,
        args.destination,
        source_grid_size=args.source_grid_size,
        target_grid_size=args.target_grid_size,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
