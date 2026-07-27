from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/full_rt_zhengyi.yaml"))
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument("--samples-per-source", type=int, default=500_000)
    parser.add_argument("--point-count", type=int, default=12)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _command(args: argparse.Namespace, batch_size: int, output: Path) -> list[str]:
    return [
        sys.executable,
        "scripts/run_sionna_sample_convergence.py",
        "--config",
        str(args.config),
        "--record-index",
        str(args.record_index),
        "--sample-counts",
        str(args.samples_per_source),
        "--point-count",
        str(args.point_count),
        "--batch-size",
        str(batch_size),
        "--output",
        str(output),
    ]


def main() -> None:
    args = _arguments()
    if any(batch_size <= 0 for batch_size in args.batch_sizes):
        raise ValueError("batch sizes must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["TCSM_MITSUBA_VARIANT"] = "cuda_ad_mono_polarized"
    rows = []
    for batch_size in args.batch_sizes:
        output = args.output / f"batch_{batch_size:02d}"
        command = _command(args, batch_size, output)
        if args.dry_run:
            rows.append({"batch_size": batch_size, "command": command, "status": "dry_run"})
            continue
        if output.exists():
            raise FileExistsError(
                f"refusing to overwrite an existing diagnostic directory: {output}"
            )
        started = time.perf_counter()
        process = subprocess.run(
            command,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        elapsed = time.perf_counter() - started
        output.mkdir(parents=True, exist_ok=True)
        (output / "stdout.log").write_text(process.stdout, encoding="utf-8")
        (output / "stderr.log").write_text(process.stderr, encoding="utf-8")
        rows.append(
            {
                "batch_size": batch_size,
                "returncode": process.returncode,
                "elapsed_seconds": elapsed,
                "status": "passed" if process.returncode == 0 else "failed",
                "command": command,
            }
        )
    summary = {
        "backend": environment["TCSM_MITSUBA_VARIANT"],
        "record_index": args.record_index,
        "samples_per_source": args.samples_per_source,
        "point_count": args.point_count,
        "results": rows,
    }
    (args.output / "batch_sweep.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
