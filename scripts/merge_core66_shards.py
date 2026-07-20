from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.merge_result_shard import merge_scene_shard
from tcsm_rt.provenance import sha256_file, write_json_atomic


def merge_core66_shards(
    sources: list[Path],
    destination: Path,
    selection_path: Path,
) -> dict[str, object]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    expected_ids = set(selection["core_config_ids"])
    expected_count = int(selection["core_record_count"])
    if len(expected_ids) != expected_count:
        raise ValueError("Core-66 selection contains duplicate or missing configuration IDs")

    destination.mkdir(parents=True, exist_ok=True)
    for source in sources:
        merge_scene_shard(source.resolve(), destination.resolve(), expected_ids)

    rows = json.loads((destination / "scene_index.json").read_text(encoding="utf-8"))
    observed = {
        str(row.get("config_id") or Path(row["cache"]).stem)
        for row in rows
        if str(row.get("source", "")).startswith("sionna")
    }
    missing = sorted(expected_ids - observed)
    unexpected = sorted(observed - expected_ids)
    if missing or unexpected or len(observed) != expected_count:
        raise ValueError(
            f"Core-66 merge mismatch: missing={missing}, unexpected={unexpected}, "
            f"observed={len(observed)}"
        )

    copied_selection = destination / "core66_selection.json"
    write_json_atomic(copied_selection, selection)
    cache_hashes = {
        Path(row["cache"]).name: sha256_file(row["cache"])
        for row in rows
        if str(row.get("config_id") or Path(row["cache"]).stem) in expected_ids
    }
    report: dict[str, object] = {
        "passed": True,
        "protocol_id": selection["protocol_id"],
        "selection_sha256": sha256_file(copied_selection),
        "source_directories": [str(source.resolve()) for source in sources],
        "core_cache_count": len(cache_hashes),
        "core_config_ids": sorted(observed),
        "cache_sha256": cache_hashes,
        "reserve_cache_count": 0,
    }
    write_json_atomic(destination / "core66_merge_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("configs/core66_selection.json"),
    )
    args = parser.parse_args()
    result = merge_core66_shards(
        args.source,
        args.destination.resolve(),
        args.selection.resolve(),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
