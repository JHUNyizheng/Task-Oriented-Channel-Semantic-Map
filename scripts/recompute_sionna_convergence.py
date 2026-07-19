from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from run_sionna_sample_convergence import _comparison
from tcsm_rt.provenance import sha256_file, write_json_atomic


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path, nargs="+")
    return parser.parse_args()


def _load(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        channel = archive["channel"]
        labels = {
            key: archive[key]
            for key in (
                "rss_db",
                "oracle_rate_bps_hz",
                "regime",
                "best_far_idx",
                "best_near_angle",
                "best_near_range",
            )
        }
    return channel, labels


def recompute(directory: Path) -> dict[str, object]:
    directory = directory.resolve()
    files = sorted(directory.glob("samples_*.npz"))
    if not files:
        raise FileNotFoundError(f"no convergence samples found in {directory}")
    counts = [int(path.stem.rsplit("_", 1)[1]) for path in files]
    channels: dict[int, np.ndarray] = {}
    labels: dict[int, dict[str, np.ndarray]] = {}
    for count, path in zip(counts, files, strict=True):
        channels[count], labels[count] = _load(path)
    reference_count = max(counts)
    rows = [
        _comparison(
            count,
            float("nan"),
            channels[count],
            labels[count],
            channels[reference_count],
            labels[reference_count],
        )
        for count in counts
    ]
    output = directory / "convergence.csv"
    original = directory / "convergence_original_metric.csv"
    if output.exists() and not original.exists():
        output.replace(original)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "directory": str(directory),
        "metric_correction": (
            "channel correlation divides by the measured norm product; only exact zero norms "
            "are assigned zero correlation"
        ),
        "reference_samples_per_source": reference_count,
        "source_npz_sha256": {path.name: sha256_file(path) for path in files},
        "original_csv": str(original) if original.exists() else None,
        "recomputed_csv": str(output),
        "rows": rows,
    }
    write_json_atomic(directory / "convergence_recompute_manifest.json", report)
    return report


def main() -> None:
    reports = [recompute(directory) for directory in _arguments().directory]
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
