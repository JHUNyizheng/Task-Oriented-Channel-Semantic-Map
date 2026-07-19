#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sionna.rt import load_scene

from tcsm_rt.data.sionna_adapter import (
    _configure_itu_material_frequency,
    _scene_constant,
)
from tcsm_rt.provenance import write_json_atomic


def _core_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in report.items()
        if key not in {"application_stage", "generation_consistency"}
    }


def _expected_report(metadata: dict[str, Any]) -> dict[str, Any]:
    scene = load_scene(_scene_constant(str(metadata["scene"])), merge_shapes=False)
    return _configure_itu_material_frequency(
        scene,
        float(metadata["frequency_hz"]),
        "clamp_to_itu_range",
    )


def audit_run_directory(run_dir: Path, repair_missing: bool) -> dict[str, Any]:
    scene_dir = run_dir / "scenes"
    metadata_paths = sorted(scene_dir.glob("sionna_*.json"))
    repaired: list[str] = []
    validated: list[str] = []
    for metadata_path in metadata_paths:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = _expected_report(metadata)
        observed = metadata.get("material_frequency")
        if observed is None:
            if not repair_missing:
                raise RuntimeError(f"missing material metadata: {metadata_path}")
            if int(expected["clamped_material_count"]) != 0:
                raise RuntimeError(
                    "cannot backfill a cache that required boundary-held materials; "
                    f"regenerate {metadata_path} with the pre-trace material policy"
                )
            expected["application_stage"] = "post_generation_metadata_audit"
            expected["generation_consistency"] = (
                "all ITU materials were within their documented ranges; the official "
                "Sionna callback and the audited closed-form evaluation coincide"
            )
            metadata["material_frequency"] = expected
            write_json_atomic(metadata_path, metadata)
            repaired.append(metadata_path.name)
        elif _core_report(observed) != _core_report(expected):
            raise RuntimeError(f"material metadata mismatch: {metadata_path}")
        else:
            validated.append(metadata_path.name)
    summary = {
        "run_dir": str(run_dir.resolve()),
        "scene_metadata_count": len(metadata_paths),
        "repaired_count": len(repaired),
        "validated_count": len(validated),
        "repaired": repaired,
        "validated": validated,
        "passed": True,
    }
    write_json_atomic(run_dir / "material_frequency_audit.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--repair-missing", action="store_true")
    args = parser.parse_args()
    summaries = [
        audit_run_directory(Path(run_dir), args.repair_missing)
        for run_dir in args.run_dir
    ]
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
