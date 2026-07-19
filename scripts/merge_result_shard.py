from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from tcsm_rt.provenance import sha256_file, write_json_atomic


def _rows(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def merge_scene_shard(source: Path, destination: Path) -> list[dict[str, Any]]:
    source_rows = _rows(source / "scene_index.json")
    destination_rows = _rows(destination / "scene_index.json")
    by_name = {Path(row["cache"]).name: row for row in destination_rows}
    destination_scenes = destination / "scenes"
    destination_scenes.mkdir(parents=True, exist_ok=True)
    for source_row in source_rows:
        source_cache = Path(source_row["cache"])
        if not source_cache.exists():
            source_cache = source / "scenes" / source_cache.name
        if not source_cache.exists():
            raise FileNotFoundError(source_cache)
        actual_hash = sha256_file(source_cache)
        declared_hash = source_row.get("cache_sha256")
        if declared_hash and actual_hash != declared_hash:
            raise ValueError(f"source hash mismatch: {source_cache}")
        destination_cache = destination_scenes / source_cache.name
        existing = by_name.get(source_cache.name)
        if destination_cache.exists() and sha256_file(destination_cache) != actual_hash:
            raise ValueError(f"conflicting cache content: {destination_cache}")
        if not destination_cache.exists():
            shutil.copy2(source_cache, destination_cache)
        source_metadata = source_cache.with_suffix(".json")
        if source_metadata.exists():
            shutil.copy2(source_metadata, destination_cache.with_suffix(".json"))
        merged = {**source_row, "cache": str(destination_cache.resolve()), "cache_sha256": actual_hash}
        by_name[source_cache.name] = {**(existing or {}), **merged}
    destination.mkdir(parents=True, exist_ok=True)
    rows = sorted(by_name.values(), key=lambda row: Path(row["cache"]).name)
    write_json_atomic(destination / "scene_index.json", rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    rows = merge_scene_shard(args.source.resolve(), args.destination.resolve())
    print(json.dumps({"merged_scene_count": len(rows), "destination": str(args.destination)}, indent=2))


if __name__ == "__main__":
    main()
