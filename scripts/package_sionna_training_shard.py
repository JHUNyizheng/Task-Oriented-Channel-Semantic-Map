from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path
from typing import Any

from tcsm_rt.provenance import sha256_file, write_json_atomic


def _json_member(archive: tarfile.TarFile, name: str, payload: Any) -> None:
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    member = tarfile.TarInfo(name)
    member.size = len(content)
    member.mode = 0o644
    archive.addfile(member, io.BytesIO(content))


def package_training_shard(run_dir: Path, output: Path, expected_count: int) -> dict[str, Any]:
    required = {
        "audit": run_dir / "audit_report.json",
        "material": run_dir / "material_frequency_audit.json",
        "budget": run_dir / "convergence" / "selected_ray_budget.json",
        "index": run_dir / "scene_index.json",
    }
    for path in required.values():
        if not path.exists():
            raise FileNotFoundError(path)
    audit = json.loads(required["audit"].read_text(encoding="utf-8"))
    material = json.loads(required["material"].read_text(encoding="utf-8"))
    budget = json.loads(required["budget"].read_text(encoding="utf-8"))
    if not audit.get("passed", False):
        raise ValueError("Sionna run audit does not pass")
    if not material.get("passed", False) or int(material.get("scene_metadata_count", -1)) != 96:
        raise ValueError("material-frequency audit must pass for all 96 scenes")
    selected_budget = int(budget.get("selected_samples_per_source", -1))
    if selected_budget <= 0:
        raise ValueError("selected ray budget is invalid")
    rows = json.loads(required["index"].read_text(encoding="utf-8"))
    train_rows = [row for row in rows if row.get("split") == "train"]
    if len(train_rows) != expected_count:
        raise ValueError(f"expected {expected_count} training scenes, found {len(train_rows)}")

    normalized_rows: list[dict[str, Any]] = []
    archive_files: list[tuple[Path, str]] = []
    cache_hashes: dict[str, str] = {}
    for row in train_rows:
        cache = Path(row["cache"])
        if not cache.exists():
            cache = run_dir / "scenes" / cache.name
        metadata = cache.with_suffix(".json")
        if not cache.exists() or not metadata.exists():
            raise FileNotFoundError(cache if not cache.exists() else metadata)
        actual_hash = sha256_file(cache)
        if row.get("cache_sha256") != actual_hash:
            raise ValueError(f"cache SHA-256 mismatch: {cache}")
        metadata_payload = json.loads(metadata.read_text(encoding="utf-8"))
        if int(metadata_payload.get("solver", {}).get("samples_per_source", -1)) != selected_budget:
            raise ValueError(f"cache does not use selected ray budget: {cache}")
        if metadata_payload.get("solver", {}).get("element_channel") != "explicit_array":
            raise ValueError(f"cache is not explicit-array: {cache}")
        normalized_rows.append(
            {
                **row,
                "cache": f"scenes/{cache.name}",
                "cache_sha256": actual_hash,
                "delegated_training_only": True,
            }
        )
        archive_files.extend(
            [(cache, f"scenes/{cache.name}"), (metadata, f"scenes/{metadata.name}")]
        )
        cache_hashes[cache.name] = actual_hash

    shard_manifest = {
        "training_scene_count": len(normalized_rows),
        "selected_samples_per_source": selected_budget,
        "source_audit_sha256": sha256_file(required["audit"]),
        "source_material_audit_sha256": sha256_file(required["material"]),
        "source_budget_sha256": sha256_file(required["budget"]),
        "cache_sha256": cache_hashes,
        "excluded_splits": ["id", "geometry_ood", "system_ood", "compound_ood"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for path, arcname in sorted(archive_files, key=lambda item: item[1]):
            archive.add(path, arcname=arcname)
        _json_member(archive, "scene_index.json", normalized_rows)
        _json_member(archive, "source_audit_report.json", audit)
        _json_member(archive, "source_material_frequency_audit.json", material)
        _json_member(archive, "source_selected_ray_budget.json", budget)
        _json_member(archive, "training_shard_manifest.json", shard_manifest)
    report = {
        **shard_manifest,
        "archive": str(output.resolve()),
        "archive_sha256": sha256_file(output),
        "archive_file_count": len(archive_files) + 5,
    }
    write_json_atomic(output.with_suffix(output.suffix + ".manifest.json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=32)
    args = parser.parse_args()
    print(
        json.dumps(
            package_training_shard(
                args.run_dir.resolve(),
                args.output.resolve(),
                args.expected_count,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
