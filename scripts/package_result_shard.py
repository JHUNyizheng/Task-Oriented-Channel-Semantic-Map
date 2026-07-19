from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

from tcsm_rt.provenance import sha256_file, write_json_atomic


INCLUDED_DIRECTORIES = (
    "checkpoints",
    "deepmimo_crosscity_checkpoints",
    "deepmimo_crosscity_checkpoints_v2",
    "metrics",
)
INCLUDED_FILES = (
    "scene_index.json",
    "environment_manifest.json",
    "audit_report.json",
    "training_summary.json",
    "deepmimo_crosscity_training_summary.json",
    "grid_training_summary.json",
    "training_shard_manifest.json",
)


def package_result_shard(run_dir: Path, output: Path) -> dict[str, object]:
    if not (run_dir / "audit_report.json").exists():
        raise FileNotFoundError(run_dir / "audit_report.json")
    audit = json.loads((run_dir / "audit_report.json").read_text(encoding="utf-8"))
    if not audit.get("passed", False):
        raise ValueError("result shard cannot be packaged before its audit passes")
    files: list[Path] = []
    for name in INCLUDED_FILES:
        candidate = run_dir / name
        if candidate.exists():
            files.append(candidate)
    for name in INCLUDED_DIRECTORIES:
        directory = run_dir / name
        if directory.exists():
            files.extend(path for path in directory.rglob("*") if path.is_file())
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for path in sorted(files):
            archive.add(path, arcname=path.relative_to(run_dir.parent))
    manifest = {
        "archive": str(output.resolve()),
        "archive_sha256": sha256_file(output),
        "file_count": len(files),
        "files": {
            str(path.relative_to(run_dir)): sha256_file(path)
            for path in sorted(files)
        },
    }
    write_json_atomic(output.with_suffix(output.suffix + ".manifest.json"), manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            package_result_shard(args.run_dir.resolve(), args.output.resolve()),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
