from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


THRESHOLDS = {
    "median_channel_correlation": 0.95,
    "p10_channel_correlation": 0.90,
    "rss_rmse_db_max": 1.0,
    "oracle_rate_mae_bps_hz_max": 0.15,
    "regime_agreement": 0.90,
    "far_label_agreement": 0.90,
    "near_angle_agreement": 0.90,
    "near_range_agreement": 0.90,
}


def _passes(row: dict[str, float]) -> bool:
    return bool(
        row["median_channel_correlation"] >= THRESHOLDS["median_channel_correlation"]
        and row["p10_channel_correlation"] >= THRESHOLDS["p10_channel_correlation"]
        and row["rss_rmse_db"] <= THRESHOLDS["rss_rmse_db_max"]
        and row["oracle_rate_mae_bps_hz"] <= THRESHOLDS["oracle_rate_mae_bps_hz_max"]
        and row["regime_agreement"] >= THRESHOLDS["regime_agreement"]
        and row["far_label_agreement"] >= THRESHOLDS["far_label_agreement"]
        and row["near_angle_agreement"] >= THRESHOLDS["near_angle_agreement"]
        and row["near_range_agreement"] >= THRESHOLDS["near_range_agreement"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    per_file: dict[str, dict[int, dict[str, float]]] = {}
    common: set[int] | None = None
    for path in args.csv:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = {
                int(row["samples_per_source"]): {
                    key: float(value)
                    for key, value in row.items()
                    if key not in {"samples_per_source", "point_count", "active_point_count"}
                }
                for row in csv.DictReader(handle)
            }
        per_file[str(path)] = rows
        common = set(rows) if common is None else common & set(rows)
    if not common:
        raise ValueError("convergence files do not share any sample budget")
    selected = None
    decisions: dict[int, bool] = {}
    for count in sorted(common):
        passed = all(_passes(per_file[str(path)][count]) for path in args.csv)
        decisions[count] = passed
        if passed and selected is None:
            selected = count
    if selected is None:
        raise RuntimeError("no ray budget satisfies the declared convergence thresholds")
    report = {
        "selected_samples_per_source": selected,
        "thresholds": THRESHOLDS,
        "passes_all_scenes": decisions,
        "inputs": [str(path.resolve()) for path in args.csv],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
