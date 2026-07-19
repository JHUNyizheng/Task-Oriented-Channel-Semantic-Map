from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from tcsm_rt.provenance import sha256_file, write_json_atomic


def _load_rows(root: Path) -> list[dict[str, Any]]:
    index = root / "scene_index.json"
    if not index.exists():
        raise FileNotFoundError(index)
    rows = json.loads(index.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"scene index must contain a list: {index}")
    return rows


def stage_training_shard(source: Path, destination: Path, expected_count: int) -> dict[str, Any]:
    source_rows = [row for row in _load_rows(source) if row.get("split") == "train"]
    if len(source_rows) != expected_count:
        raise ValueError(
            f"expected {expected_count} Sionna training scenes, found {len(source_rows)}"
        )
    if any(not str(row.get("source", "")).startswith("sionna") for row in source_rows):
        raise ValueError("training shard contains a non-Sionna source")

    destination.mkdir(parents=True, exist_ok=True)
    destination_scenes = destination / "scenes"
    destination_scenes.mkdir(parents=True, exist_ok=True)
    destination_index = destination / "scene_index.json"
    existing_rows = (
        json.loads(destination_index.read_text(encoding="utf-8"))
        if destination_index.exists()
        else []
    )
    by_name = {Path(row["cache"]).name: row for row in existing_rows}
    staged: list[dict[str, Any]] = []
    for row in source_rows:
        source_cache = Path(row["cache"])
        if not source_cache.exists():
            source_cache = source / "scenes" / source_cache.name
        if not source_cache.exists():
            raise FileNotFoundError(source_cache)
        actual_hash = sha256_file(source_cache)
        if row.get("cache_sha256") != actual_hash:
            raise ValueError(f"declared SHA-256 mismatch: {source_cache}")
        destination_cache = destination_scenes / source_cache.name
        if destination_cache.exists() and sha256_file(destination_cache) != actual_hash:
            raise ValueError(f"conflicting destination cache: {destination_cache}")
        if not destination_cache.exists():
            shutil.copy2(source_cache, destination_cache)
        metadata = source_cache.with_suffix(".json")
        if metadata.exists():
            shutil.copy2(metadata, destination_cache.with_suffix(".json"))
        staged_row = {
            **row,
            "cache": str(destination_cache.resolve()),
            "cache_sha256": actual_hash,
            "delegated_training_only": True,
        }
        by_name[destination_cache.name] = staged_row
        staged.append(staged_row)

    # Existing DeepMIMO rows are preserved. Sionna ID/OOD rows are never imported here.
    merged = sorted(by_name.values(), key=lambda item: Path(item["cache"]).name)
    write_json_atomic(destination_index, merged)
    manifest = {
        "source": str(source.resolve()),
        "destination": str(destination.resolve()),
        "training_scene_count": len(staged),
        "cache_sha256": {Path(row["cache"]).name: row["cache_sha256"] for row in staged},
        "excluded_splits": ["id", "geometry_ood", "system_ood", "compound_ood"],
    }
    write_json_atomic(destination / "training_shard_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=32)
    args = parser.parse_args()
    result = stage_training_shard(
        args.source.resolve(),
        args.destination.resolve(),
        args.expected_count,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
