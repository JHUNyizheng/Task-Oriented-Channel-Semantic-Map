from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from tcsm_rt.config import load_config
from tcsm_rt.data.common import sionna_configuration_manifest
from tcsm_rt.provenance import sha256_file, write_json_atomic


SPLITS = ("train", "id", "geometry_ood", "system_ood", "compound_ood")


def _manifest_digest(records: list[dict[str, Any]]) -> str:
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_manifest(path: Path | None, config_path: Path) -> list[dict[str, Any]]:
    if path is not None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("Sionna manifest must be a JSON list")
        return payload
    config = load_config(config_path)
    return [record.__dict__ for record in sionna_configuration_manifest(config)]


def _flatten(groups: dict[str, list[int]]) -> list[int]:
    missing = set(SPLITS) - set(groups)
    extra = set(groups) - set(SPLITS)
    if missing or extra:
        raise ValueError(f"invalid protocol split keys: missing={sorted(missing)}, extra={sorted(extra)}")
    return [int(index) for split in SPLITS for index in groups[split]]


def build_core66_selection(
    records: list[dict[str, Any]],
    protocol: dict[str, Any],
    protocol_sha256: str | None = None,
) -> dict[str, Any]:
    declared_count = int(protocol["declared_record_count"])
    if len(records) != declared_count:
        raise ValueError(f"manifest has {len(records)} records, expected {declared_count}")

    core_groups = protocol["core_records"]
    reserve_groups = protocol["reserve_records"]
    core = _flatten(core_groups)
    reserve = _flatten(reserve_groups)
    if len(core) != len(set(core)) or len(reserve) != len(set(reserve)):
        raise ValueError("core and reserve record IDs must be unique")
    if set(core) & set(reserve):
        raise ValueError("core and reserve record IDs overlap")
    expected_ids = set(range(declared_count))
    if set(core) | set(reserve) != expected_ids:
        missing = sorted(expected_ids - (set(core) | set(reserve)))
        extra = sorted((set(core) | set(reserve)) - expected_ids)
        raise ValueError(f"core/reserve partition is incomplete: missing={missing}, extra={extra}")
    if len(core) != int(protocol["core_record_count"]):
        raise ValueError("core record count does not match protocol")
    if len(reserve) != int(protocol["reserve_record_count"]):
        raise ValueError("reserve record count does not match protocol")

    required_counts = {
        split: int(count)
        for split, count in protocol["selection_requirements"]["core_split_counts"].items()
    }
    observed_counts = Counter(records[index]["split"] for index in core)
    if dict(observed_counts) != required_counts:
        raise ValueError(f"core split counts are {dict(observed_counts)}, expected {required_counts}")
    for split in SPLITS:
        if any(records[index]["split"] != split for index in core_groups[split]):
            raise ValueError(f"core record assigned to the wrong split: {split}")
        if any(records[index]["split"] != split for index in reserve_groups[split]):
            raise ValueError(f"reserve record assigned to the wrong split: {split}")

    for split in SPLITS[1:]:
        cells = [
            (
                records[index]["scene"],
                float(records[index]["frequency_hz"]),
                int(records[index]["array_size"]),
            )
            for index in core_groups[split]
        ]
        if len(cells) != len(set(cells)):
            raise ValueError(f"{split} does not contain exactly one record per design cell")

    train_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index in core_groups["train"]:
        train_by_scene[str(records[index]["scene"])].append(records[index])
    requirements = protocol["selection_requirements"]
    expected_scenes = int(requirements["train_scene_count"])
    expected_per_scene = int(requirements["train_records_per_scene"])
    expected_system_cells = int(requirements["train_system_cell_count_per_scene"])
    required_placements = {int(value) for value in requirements["train_required_placement_indices"]}
    if len(train_by_scene) != expected_scenes:
        raise ValueError(f"training core covers {len(train_by_scene)} scenes, expected {expected_scenes}")
    for scene, scene_records in train_by_scene.items():
        if len(scene_records) != expected_per_scene:
            raise ValueError(f"training scene {scene} has {len(scene_records)} records")
        system_cells = {
            (float(record["frequency_hz"]), int(record["array_size"]))
            for record in scene_records
        }
        if len(system_cells) != expected_system_cells:
            raise ValueError(f"training scene {scene} covers {len(system_cells)} system cells")
        placements = {int(record["placement_index"]) for record in scene_records}
        if not required_placements <= placements:
            raise ValueError(f"training scene {scene} lacks placements {sorted(required_placements)}")

    core_records = [{"record_index": index, **records[index]} for index in core]
    reserve_records = [{"record_index": index, **records[index]} for index in reserve]
    return {
        "schema_version": int(protocol["schema_version"]),
        "protocol_id": str(protocol["protocol_id"]),
        "protocol_sha256": protocol_sha256,
        "source_manifest_sha256": _manifest_digest(records),
        "declared_record_count": declared_count,
        "core_record_count": len(core),
        "reserve_record_count": len(reserve),
        "core_split_counts": required_counts,
        "core_record_indices": core,
        "reserve_record_indices": reserve,
        "core_config_ids": [record["config_id"] for record in core_records],
        "reserve_config_ids": [record["config_id"] for record in reserve_records],
        "core_records": core_records,
        "reserve_records": reserve_records,
        "statistics": protocol["statistics"],
        "reserve_activation_gates": protocol["reserve_activation_gates"],
        "reporting_rules": protocol["reporting_rules"],
        "validation": {
            "partition_complete": True,
            "split_counts_match": True,
            "evaluation_cells_unique": True,
            "training_scene_system_placement_coverage": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/full_rt_zhengyi.yaml"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--protocol", type=Path, default=Path("configs/core66_protocol.yaml"))
    parser.add_argument("--output", type=Path, default=Path("configs/core66_selection.json"))
    args = parser.parse_args()
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise ValueError("selection protocol must be a YAML mapping")
    records = _load_manifest(args.manifest, args.config)
    result = build_core66_selection(records, protocol, sha256_file(args.protocol))
    write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
