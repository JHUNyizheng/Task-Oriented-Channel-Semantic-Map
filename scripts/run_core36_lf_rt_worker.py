from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

WORKERS = ("zhengyi_a", "zhengyi_b", "zhengyi4090")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True, choices=WORKERS)
    parser.add_argument(
        "--allocation",
        type=Path,
        default=Path("configs/compute_allocation_core36_no_mac.yaml"),
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("configs/core36_lf_selection.json"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/full_rt_core36_lf_zhengyi.yaml"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--samples-per-source", type=int, default=500_000)
    parser.add_argument(
        "--backend",
        choices=("llvm_ad_mono_polarized", "cuda_ad_mono_polarized"),
        default="llvm_ad_mono_polarized",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a mapping in {path}")
    return payload


def _git_head(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def _audit_cache(path: Path, expected_id: str, expected_queries: int) -> None:
    metadata_path = path.with_suffix(".json")
    if not metadata_path.exists():
        raise RuntimeError(f"missing metadata companion: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("config_id") != expected_id:
        raise RuntimeError(f"cache identity mismatch for {path}")
    if int(metadata.get("query_count", -1)) != expected_queries:
        raise RuntimeError(f"cache query count mismatch for {path}")
    with np.load(path, allow_pickle=False) as arrays:
        if not arrays.files:
            raise RuntimeError(f"empty cache: {path}")
        for name in arrays.files:
            value = arrays[name]
            if value.size == 0:
                raise RuntimeError(f"empty array {name} in {path}")
            if np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)):
                raise RuntimeError(f"non-finite array {name} in {path}")


def main() -> None:
    args = _arguments()
    root = Path(__file__).resolve().parents[1]
    allocation = _load_yaml(root / args.allocation)
    selection = json.loads((root / args.selection).read_text(encoding="utf-8"))
    worker = allocation["workers"][args.worker]
    assigned = [int(value) for value in worker["primary_record_indices"]]
    manifest = {
        int(record["record_index"]): str(record["config_id"])
        for record in selection["primary_records"]
    }
    if any(index not in manifest for index in assigned):
        raise RuntimeError(f"{args.worker} allocation contains a non-primary record")

    output_dir = (args.output_dir or Path(f"outputs/core36_lf_v2_{args.worker}")).resolve()
    scene_dir = output_dir / "scenes"
    scene_dir.mkdir(parents=True, exist_ok=True)
    expected_queries = int(selection["query_grid"]["query_count"])
    complete: list[int] = []
    remaining: list[int] = []
    for index in assigned:
        cache = scene_dir / f"{manifest[index]}.npz"
        if cache.exists():
            _audit_cache(cache, manifest[index], expected_queries)
            complete.append(index)
        else:
            remaining.append(index)

    launch = {
        "schema_version": 2,
        "created_at": datetime.now(UTC).isoformat(),
        "protocol_id": selection["protocol_id"],
        "protocol_sha256": selection["protocol_sha256"],
        "worker": args.worker,
        "git_commit": _git_head(root),
        "host": platform.node(),
        "output_dir": str(output_dir),
        "samples_per_source": int(args.samples_per_source),
        "assigned_record_indices": assigned,
        "verified_complete_record_indices": complete,
        "remaining_record_indices": remaining,
        "backend": args.backend,
        "dry_run": bool(args.dry_run),
    }
    print(json.dumps(launch, indent=2))
    if args.dry_run or not remaining:
        return

    lock = output_dir / ".core36_lf_worker.lock"
    log_path = output_dir / f"core36_lf_{args.worker}.jsonl"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"worker lock already exists: {lock}") from error
    try:
        os.write(descriptor, json.dumps(launch, indent=2).encode("utf-8"))
        os.close(descriptor)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": "start", **launch}) + "\n")
        os.environ["TCSM_OUTPUT_DIR"] = str(output_dir)
        os.environ["TCSM_SAMPLES_PER_SOURCE"] = str(args.samples_per_source)
        os.environ["TCSM_MITSUBA_VARIANT"] = args.backend
        from tcsm_rt.config import load_config
        from tcsm_rt.pipeline import prepare_sionna

        config = load_config(root / args.config)
        prepare_sionna(config, record_indices=remaining)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "event": "complete",
                        "completed_at": datetime.now(UTC).isoformat(),
                        "record_indices": remaining,
                    }
                )
                + "\n"
            )
    except Exception as error:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "event": "error",
                        "failed_at": datetime.now(UTC).isoformat(),
                        "error": repr(error),
                        "traceback": traceback.format_exc(),
                    }
                )
                + "\n"
            )
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
