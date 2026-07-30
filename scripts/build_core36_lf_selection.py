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

PRIMARY_SPLITS = ("train", "id", "geometry_ood")


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
    if set(groups) != set(PRIMARY_SPLITS):
        raise ValueError(f"primary split keys must be {list(PRIMARY_SPLITS)}")
    return [int(index) for split in PRIMARY_SPLITS for index in groups[split]]


def build_core36_lf_selection(
    records: list[dict[str, Any]],
    protocol: dict[str, Any],
    protocol_sha256: str | None = None,
) -> dict[str, Any]:
    declared_count = int(protocol["declared_record_count"])
    if len(records) != declared_count:
        raise ValueError(f"manifest has {len(records)} records, expected {declared_count}")

    primary_groups = protocol["primary_records"]
    primary = _flatten(primary_groups)
    diagnostic = [int(index) for index in protocol["high_frequency_diagnostic_records"]]
    if len(primary) != len(set(primary)) or len(diagnostic) != len(set(diagnostic)):
        raise ValueError("primary and diagnostic record IDs must be unique")
    if set(primary) & set(diagnostic):
        raise ValueError("primary and diagnostic records overlap")
    invalid = [index for index in primary + diagnostic if index < 0 or index >= declared_count]
    if invalid:
        raise ValueError(f"record IDs out of range: {invalid}")
    if len(primary) != int(protocol["primary_record_count"]):
        raise ValueError("primary record count does not match protocol")

    requirements = protocol["selection_requirements"]
    expected_counts = {
        split: int(count) for split, count in requirements["primary_split_counts"].items()
    }
    observed_counts = Counter(records[index]["split"] for index in primary)
    if dict(observed_counts) != expected_counts:
        raise ValueError(
            f"primary split counts are {dict(observed_counts)}, expected {expected_counts}"
        )

    required_cells = {
        (float(frequency_hz), int(array_size))
        for frequency_hz, array_size in requirements["system_cells_per_scene"]
    }
    expected_scene_count = int(requirements["scene_count_per_split"])
    expected_records_per_scene = int(requirements["records_per_scene"])
    required_placements = {int(value) for value in requirements["placement_indices"]}
    scenes_by_split: dict[str, set[str]] = {}
    for split in PRIMARY_SPLITS:
        if any(records[index]["split"] != split for index in primary_groups[split]):
            raise ValueError(f"primary record assigned to wrong split: {split}")
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for index in primary_groups[split]:
            grouped[str(records[index]["scene"])].append(records[index])
        if len(grouped) != expected_scene_count:
            raise ValueError(
                f"{split} covers {len(grouped)} scenes, expected {expected_scene_count}"
            )
        for scene, scene_records in grouped.items():
            if len(scene_records) != expected_records_per_scene:
                raise ValueError(f"{split}/{scene} has {len(scene_records)} records")
            cells = {
                (float(record["frequency_hz"]), int(record["array_size"]))
                for record in scene_records
            }
            if cells != required_cells:
                raise ValueError(f"{split}/{scene} has system cells {sorted(cells)}")
            placements = {int(record["placement_index"]) for record in scene_records}
            if placements != required_placements:
                raise ValueError(f"{split}/{scene} has placements {sorted(placements)}")
        scenes_by_split[split] = set(grouped)

    if scenes_by_split["train"] != scenes_by_split["id"]:
        raise ValueError("training and ID must use the same scene families")
    if scenes_by_split["train"] & scenes_by_split["geometry_ood"]:
        raise ValueError("geometry-OOD scenes must be disjoint from training scenes")
    all_primary_scenes = set().union(*scenes_by_split.values())
    if len(all_primary_scenes) != int(requirements["official_scene_count"]):
        raise ValueError("primary selection does not cover all official scene families")

    diagnostic_records = [records[index] for index in diagnostic]
    if {str(record["scene"]) for record in diagnostic_records} != all_primary_scenes:
        raise ValueError("high-frequency diagnostics must cover all official scene families")
    if any(float(record["frequency_hz"]) < 60e9 for record in diagnostic_records):
        raise ValueError("high-frequency diagnostic contains a low-frequency record")

    primary_records = [{"record_index": index, **records[index]} for index in primary]
    diagnostic_payload = [{"record_index": index, **records[index]} for index in diagnostic]
    excluded = sorted(set(range(declared_count)) - set(primary) - set(diagnostic))
    return {
        "schema_version": int(protocol["schema_version"]),
        "protocol_id": str(protocol["protocol_id"]),
        "protocol_sha256": protocol_sha256,
        "source_manifest_sha256": _manifest_digest(records),
        "declared_record_count": declared_count,
        "primary_record_count": len(primary),
        "progress_denominator": int(protocol["progress_denominator"]),
        "primary_split_counts": expected_counts,
        "primary_record_indices": primary,
        "primary_config_ids": [record["config_id"] for record in primary_records],
        "primary_records": primary_records,
        "high_frequency_diagnostic_record_indices": diagnostic,
        "high_frequency_diagnostic_records": diagnostic_payload,
        "excluded_record_indices": excluded,
        "query_grid": protocol["query_grid"],
        "statistics": protocol["statistics"],
        "evidence_gates": protocol["evidence_gates"],
        "failure_policy": protocol["failure_policy"],
        "reporting_rules": protocol["reporting_rules"],
        "validation": {
            "primary_counts_match": True,
            "primary_cells_complete": True,
            "scene_families_complete_and_disjoint": True,
            "high_frequency_diagnostics_separate": True,
            "no_data_dependent_reserve": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/full_rt_core36_lf_zhengyi.yaml"),
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/core36_lf_protocol.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/core36_lf_selection.json"),
    )
    args = parser.parse_args()
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise TypeError("selection protocol must be a YAML mapping")
    records = _load_manifest(args.manifest, args.config)
    result = build_core36_lf_selection(records, protocol, sha256_file(args.protocol))
    write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
