from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


WORKERS = {
    "zhengyi-a": {
        "allocation": "zhengyi",
        "output": "outputs/zhengyi_sionna",
    },
    "zhengyi-b": {
        "allocation": "zhengyi_b",
        "output": "outputs/zhengyi_sionna_shard_b",
    },
    "mac-c1": {
        "allocation": "mac_studio",
        "output": "outputs/macstudio_sionna_shard_c1",
    },
    "mac-c2": {
        "allocation": "mac_studio_c2",
        "output": "outputs/macstudio_sionna_shard_c2",
    },
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True, choices=sorted(WORKERS))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--samples-per-source", type=int, default=500_000)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a mapping in {path}")
    return payload


def _git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def _configure_backend() -> dict[str, str]:
    os.environ.setdefault("TCSM_MITSUBA_VARIANT", "llvm_ad_mono_polarized")
    report = {"mitsuba_variant": os.environ["TCSM_MITSUBA_VARIANT"]}
    if platform.system() == "Darwin" and not os.environ.get("DRJIT_LIBLLVM_PATH"):
        stable = Path("/opt/homebrew/opt/llvm@20/lib/libLLVM.dylib")
        runtime = Path.home() / "Projects" / "Radio2026" / "runtime" / "llvm20-bottle"
        candidates = [
            path
            for path in [stable, *sorted(runtime.glob("**/libLLVM.dylib"))]
            if path.exists()
        ]
        selected = None
        failures: dict[str, str] = {}
        for candidate in candidates:
            try:
                ctypes.CDLL(str(candidate))
            except OSError as error:
                failures[str(candidate)] = str(error)
            else:
                selected = candidate
                break
        if selected is None:
            raise RuntimeError(
                "no loadable pinned LLVM 20 backend; run scripts/bootstrap_mac_llvm.py; "
                f"failures={failures}"
            )
        os.environ["DRJIT_LIBLLVM_PATH"] = str(selected)
    if os.environ.get("DRJIT_LIBLLVM_PATH"):
        report["drjit_libllvm_path"] = os.environ["DRJIT_LIBLLVM_PATH"]
    return report


def _audit_cache(path: Path, expected_id: str) -> None:
    metadata_path = path.with_suffix(".json")
    if not metadata_path.exists():
        raise RuntimeError(f"missing metadata companion: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("config_id") != expected_id:
        raise RuntimeError(f"cache identity mismatch for {path}")
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
    worker = WORKERS[args.worker]
    allocation = _load_yaml(root / "configs" / "compute_allocation.yaml")
    selection = json.loads(
        (root / "configs" / "core66_selection.json").read_text(encoding="utf-8")
    )
    manifest = {
        int(record["record_index"]): str(record["config_id"])
        for record in selection["core_records"]
    }
    assigned = [
        int(value)
        for value in allocation["workers"][worker["allocation"]]["core_record_indices"]
    ]
    if any(index not in manifest for index in assigned):
        raise RuntimeError(f"{args.worker} allocation contains a non-core record")

    output_dir = (args.output_dir or Path(worker["output"])).resolve()
    scene_dir = output_dir / "scenes"
    scene_dir.mkdir(parents=True, exist_ok=True)
    complete: list[int] = []
    remaining: list[int] = []
    for index in assigned:
        cache = scene_dir / f"{manifest[index]}.npz"
        if cache.exists():
            _audit_cache(cache, manifest[index])
            complete.append(index)
        else:
            remaining.append(index)

    backend = _configure_backend()
    launch = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "worker": args.worker,
        "git_commit": _git_head(root),
        "output_dir": str(output_dir),
        "samples_per_source": int(args.samples_per_source),
        "assigned_record_indices": assigned,
        "verified_complete_record_indices": complete,
        "remaining_record_indices": remaining,
        "backend": backend,
        "dry_run": bool(args.dry_run),
    }
    print(json.dumps(launch, indent=2))
    if args.dry_run or not remaining:
        return

    lock = output_dir / ".core66_worker.lock"
    log_path = output_dir / f"core66_{args.worker}.jsonl"
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
        from tcsm_rt.config import load_config
        from tcsm_rt.pipeline import prepare_sionna

        config = load_config(root / "configs" / "full_rt_zhengyi.yaml")
        prepare_sionna(config, record_indices=remaining)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "event": "complete",
                        "completed_at": datetime.now(timezone.utc).isoformat(),
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
                        "failed_at": datetime.now(timezone.utc).isoformat(),
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
