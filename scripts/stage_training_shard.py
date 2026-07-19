from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from pathlib import PurePosixPath
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


def _load_evidence(source: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    candidates = {
        "audit": (source / "source_audit_report.json", source / "audit_report.json"),
        "material": (
            source / "source_material_frequency_audit.json",
            source / "material_frequency_audit.json",
        ),
        "budget": (
            source / "source_selected_ray_budget.json",
            source / "convergence" / "selected_ray_budget.json",
        ),
    }
    loaded: dict[str, dict[str, Any]] = {}
    for name, paths in candidates.items():
        path = next((candidate for candidate in paths if candidate.exists()), None)
        if path is None:
            raise FileNotFoundError(f"missing {name} evidence under {source}")
        loaded[name] = json.loads(path.read_text(encoding="utf-8"))
    if not loaded["audit"].get("passed", False):
        raise ValueError("source Sionna run audit does not pass")
    if not loaded["material"].get("passed", False):
        raise ValueError("source material-frequency audit does not pass")
    if int(loaded["material"].get("scene_metadata_count", -1)) != 96:
        raise ValueError("source material-frequency audit must cover all 96 Sionna scenes")
    if int(loaded["budget"].get("selected_samples_per_source", -1)) <= 0:
        raise ValueError("source ray-budget selection is invalid")
    return loaded["audit"], loaded["material"], loaded["budget"]


def stage_training_shard(source: Path, destination: Path, expected_count: int) -> dict[str, Any]:
    source_audit, material_audit, budget_report = _load_evidence(source)
    selected_budget = int(budget_report["selected_samples_per_source"])
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
        source_metadata = source_cache.with_suffix(".json")
        if not source_metadata.exists():
            source_metadata = source / "scenes" / source_metadata.name
        if not source_metadata.exists():
            raise FileNotFoundError(source_metadata)
        metadata_payload = json.loads(source_metadata.read_text(encoding="utf-8"))
        observed_budget = int(
            metadata_payload.get("solver", {}).get("samples_per_source", -1)
        )
        if observed_budget != selected_budget:
            raise ValueError(
                f"cache ray budget {observed_budget} does not match selected budget "
                f"{selected_budget}: {source_cache}"
            )
        if metadata_payload.get("solver", {}).get("element_channel") != "explicit_array":
            raise ValueError(f"delegated cache is not an explicit-array channel: {source_cache}")
        if not isinstance(metadata_payload.get("material_frequency"), dict):
            raise ValueError(f"delegated cache lacks material-frequency metadata: {source_cache}")
        destination_cache = destination_scenes / source_cache.name
        if destination_cache.exists() and sha256_file(destination_cache) != actual_hash:
            raise ValueError(f"conflicting destination cache: {destination_cache}")
        if not destination_cache.exists():
            shutil.copy2(source_cache, destination_cache)
        shutil.copy2(source_metadata, destination_cache.with_suffix(".json"))
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
        "source_audit_passed": bool(source_audit["passed"]),
        "source_material_scene_count": int(material_audit["scene_metadata_count"]),
        "selected_samples_per_source": selected_budget,
        "cache_sha256": {Path(row["cache"]).name: row["cache_sha256"] for row in staged},
        "excluded_splits": ["id", "geometry_ood", "system_ood", "compound_ood"],
    }
    write_json_atomic(destination / "training_shard_manifest.json", manifest)
    return manifest


def _validate_archive_member(member: tarfile.TarInfo) -> None:
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe path in training shard: {member.name}")
    if member.issym() or member.islnk():
        raise ValueError(f"links are not allowed in training shard: {member.name}")
    if not (member.isfile() or member.isdir()):
        raise ValueError(f"unsupported member in training shard: {member.name}")


def stage_training_input(source: Path, destination: Path, expected_count: int) -> dict[str, Any]:
    """Stage a verified shard directory or its packaged ``tar.gz`` archive."""
    if source.is_dir():
        return stage_training_shard(source, destination, expected_count)
    if not source.is_file():
        raise FileNotFoundError(source)

    archive_hash = sha256_file(source)
    with tempfile.TemporaryDirectory(prefix="tcsm_training_shard_") as temporary:
        extracted = Path(temporary)
        with tarfile.open(source, "r:*") as archive:
            members = archive.getmembers()
            for member in members:
                _validate_archive_member(member)
            archive.extractall(extracted, members=members, filter="data")
        manifest = stage_training_shard(extracted, destination, expected_count)

    manifest["source"] = str(source.resolve())
    manifest["source_archive_sha256"] = archive_hash
    write_json_atomic(destination / "training_shard_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=32)
    args = parser.parse_args()
    result = stage_training_input(
        args.source.resolve(),
        args.destination.resolve(),
        args.expected_count,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
