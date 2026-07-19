from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from tcsm_rt.provenance import sha256_file, write_json_atomic


DEFAULT_MODELS = (
    "deepsets",
    "set_transformer",
    "storm",
    "radiounet",
    "fno",
    "wno",
    "gated_hlg",
    "gated_hlg_no_environment",
    "gated_hlg_no_global",
    "gated_hlg_no_local_attention",
    "gated_hlg_no_local_prior",
    "gated_hlg_fixed_gate",
)


def _merge_summary(source: Path, destination: Path) -> int:
    source_rows = json.loads(source.read_text(encoding="utf-8")) if source.exists() else []
    destination_rows = (
        json.loads(destination.read_text(encoding="utf-8")) if destination.exists() else []
    )
    by_key = {
        (str(row["model"]), int(row["seed"])): row
        for row in destination_rows
        if "model" in row and "seed" in row
    }
    for row in source_rows:
        normalized = dict(row)
        if "checkpoint" in normalized:
            normalized["checkpoint"] = str(
                (destination.parent / "checkpoints" / Path(normalized["checkpoint"]).name).resolve()
            )
        by_key[(str(normalized["model"]), int(normalized["seed"]))] = normalized
    merged = [by_key[key] for key in sorted(by_key)]
    write_json_atomic(destination, merged)
    return len(merged)


def merge_compute_artifacts(
    source: Path,
    destination: Path,
    models: tuple[str, ...],
    seeds: tuple[int, ...],
    expected_steps: int,
) -> dict[str, Any]:
    source_audit_path = source / "audit_report.json"
    if not source_audit_path.exists():
        raise FileNotFoundError(source_audit_path)
    source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    if not source_audit.get("passed", False):
        raise ValueError("delegated result audit does not pass")
    source_checkpoints = source / "checkpoints"
    destination_checkpoints = destination / "checkpoints"
    destination_checkpoints.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for model in models:
        for seed in seeds:
            stem = f"{model}_seed{seed}"
            checkpoint = source_checkpoints / f"{stem}.pt"
            history = source_checkpoints / f"{stem}.history.json"
            if not checkpoint.exists() or not history.exists():
                raise FileNotFoundError(checkpoint if not checkpoint.exists() else history)
            history_rows = json.loads(history.read_text(encoding="utf-8"))
            if not history_rows or int(history_rows[-1].get("step", -1)) != expected_steps:
                raise ValueError(f"delegated training is incomplete: {history}")
            for artifact in (checkpoint, history):
                target = destination_checkpoints / artifact.name
                artifact_hash = sha256_file(artifact)
                if target.exists() and sha256_file(target) != artifact_hash:
                    raise ValueError(f"conflicting delegated artifact: {target}")
                if not target.exists():
                    shutil.copy2(artifact, target)
                copied[artifact.name] = artifact_hash
    summary_counts = {
        name: _merge_summary(source / name, destination / name)
        for name in ("training_summary.json", "grid_training_summary.json")
    }
    report = {
        "source": str(source.resolve()),
        "destination": str(destination.resolve()),
        "models": list(models),
        "seeds": list(seeds),
        "expected_steps": expected_steps,
        "artifact_count": len(copied),
        "artifact_sha256": copied,
        "summary_row_counts": summary_counts,
        "source_audit_sha256": sha256_file(source_audit_path),
        "passed": True,
    }
    write_json_atomic(destination / "delegated_artifact_merge.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[53, 71])
    parser.add_argument("--expected-steps", type=int, default=8000)
    args = parser.parse_args()
    print(
        json.dumps(
            merge_compute_artifacts(
                args.source.resolve(),
                args.destination.resolve(),
                tuple(args.models),
                tuple(args.seeds),
                args.expected_steps,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
