from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from tcsm_rt.config import load_config
from tcsm_rt.data.common import grid_xyz, sionna_configuration_manifest
from tcsm_rt.data.sionna_adapter import (
    _object_footprints,
    _placement,
    _scene_constant,
    _to_numpy,
    _trace_explicit_channel_batch,
)
from tcsm_rt.physics import channel_correlation, make_task_labels
from tcsm_rt.provenance import write_json_atomic


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument(
        "--sample-counts",
        type=int,
        nargs="+",
        default=[20_000, 50_000, 100_000, 250_000, 500_000, 1_000_000],
    )
    parser.add_argument("--point-count", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _labels(
    channel: np.ndarray,
    query_xyz: np.ndarray,
    transmitter_xyz: np.ndarray,
    record: object,
    config: dict,
) -> dict[str, np.ndarray]:
    return make_task_labels(
        channel,
        record.frequency_hz,
        int(config["model"]["far_beams"]),
        int(config["model"]["near_angles"]),
        int(config["model"]["near_ranges"]),
        distance_m=np.linalg.norm(query_xyz - transmitter_xyz[None, :], axis=1),
        tx_power_dbm=float(config["system"]["tx_power_dbm"]),
        bandwidth_hz=float(config["system"]["bandwidth_hz"]),
        noise_figure_db=float(config["system"]["noise_figure_db"]),
        low_margin_bps_hz=float(config["system"]["regime_low_margin_bps_hz"]),
        high_margin_bps_hz=float(config["system"]["regime_high_margin_bps_hz"]),
    )


def _comparison(
    sample_count: int,
    runtime_s: float,
    channel: np.ndarray,
    labels: dict[str, np.ndarray],
    reference_channel: np.ndarray,
    reference_labels: dict[str, np.ndarray],
) -> dict[str, float | int]:
    reference_norm = np.linalg.norm(reference_channel, axis=1)
    estimate_norm = np.linalg.norm(channel, axis=1)
    active = (reference_norm > 1e-15) & (estimate_norm > 1e-15)
    correlation = channel_correlation(reference_channel[active], channel[active])
    rss_error = labels["rss_db"].astype(np.float64) - reference_labels["rss_db"].astype(np.float64)
    return {
        "samples_per_source": sample_count,
        "runtime_s": runtime_s,
        "point_count": len(channel),
        "active_point_count": int(np.sum(active)),
        "median_channel_correlation": float(np.median(correlation)) if correlation.size else float("nan"),
        "p10_channel_correlation": float(np.quantile(correlation, 0.1)) if correlation.size else float("nan"),
        "rss_rmse_db": float(np.sqrt(np.mean(rss_error**2))),
        "oracle_rate_mae_bps_hz": float(
            np.mean(
                np.abs(
                    labels["oracle_rate_bps_hz"].astype(np.float64)
                    - reference_labels["oracle_rate_bps_hz"].astype(np.float64)
                )
            )
        ),
        "regime_agreement": float(np.mean(labels["regime"] == reference_labels["regime"])),
        "far_label_agreement": float(
            np.mean(labels["best_far_idx"] == reference_labels["best_far_idx"])
        ),
        "near_angle_agreement": float(
            np.mean(labels["best_near_angle"] == reference_labels["best_near_angle"])
        ),
        "near_range_agreement": float(
            np.mean(labels["best_near_range"] == reference_labels["best_near_range"])
        ),
    }


def main() -> None:
    from sionna.rt import PlanarArray, Transmitter, load_scene

    args = _arguments()
    config = load_config(args.config)
    records = sionna_configuration_manifest(config)
    record = records[args.record_index]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scene = load_scene(_scene_constant(record.scene), merge_shapes=False)
    scene.frequency = float(record.frequency_hz)
    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    scene.rx_array = scene.tx_array
    footprints = _object_footprints(scene)
    transmitter_xyz, grid_center = _placement(
        scene,
        record,
        int(config["data"]["grid_size"]),
        float(config["data"]["cell_size_m"]),
        footprints,
    )
    scene.add(
        Transmitter(
            name="bs",
            position=transmitter_xyz.tolist(),
            orientation=[0.0, 0.0, 0.0],
        )
    )
    full_grid = grid_xyz(
        grid_center,
        int(config["data"]["grid_size"]),
        float(config["data"]["cell_size_m"]),
    )
    point_indices = np.unique(
        np.linspace(0, len(full_grid) - 1, min(args.point_count, len(full_grid))).round().astype(int)
    )
    query_xyz = full_grid[point_indices]
    channels: dict[int, np.ndarray] = {}
    labels: dict[int, dict[str, np.ndarray]] = {}
    runtimes: dict[int, float] = {}
    for sample_count in sorted(set(args.sample_counts)):
        settings = dict(config["data"]["sionna"])
        settings["samples_per_source"] = int(sample_count)
        started = time.perf_counter()
        batches = []
        for start in range(0, len(query_xyz), args.batch_size):
            stop = min(start + args.batch_size, len(query_xyz))
            batches.append(
                _trace_explicit_channel_batch(
                    scene,
                    query_xyz[start:stop],
                    record,
                    settings,
                    record.seed + start,
                )
            )
        channel = np.concatenate(batches, axis=0)
        runtimes[sample_count] = time.perf_counter() - started
        channels[sample_count] = channel
        labels[sample_count] = _labels(channel, query_xyz, transmitter_xyz, record, config)
        np.savez_compressed(
            output / f"samples_{sample_count:07d}.npz",
            channel=channel,
            query_xyz_m=query_xyz,
            point_indices=point_indices,
            rss_db=labels[sample_count]["rss_db"],
            regime=labels[sample_count]["regime"],
            best_far_idx=labels[sample_count]["best_far_idx"],
            best_near_angle=labels[sample_count]["best_near_angle"],
            best_near_range=labels[sample_count]["best_near_range"],
            oracle_rate_bps_hz=labels[sample_count]["oracle_rate_bps_hz"],
        )
    reference_count = max(channels)
    rows = [
        _comparison(
            count,
            runtimes[count],
            channels[count],
            labels[count],
            channels[reference_count],
            labels[reference_count],
        )
        for count in sorted(channels)
    ]
    with (output / "convergence.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json_atomic(
        output / "manifest.json",
        {
            "record": record.__dict__,
            "transmitter_xyz_m": transmitter_xyz.tolist(),
            "grid_center_xyz_m": grid_center.tolist(),
            "point_indices": point_indices.tolist(),
            "reference_samples_per_source": reference_count,
            "rows": rows,
            "scene_bbox": [
                _to_numpy(scene.mi_scene.bbox().min).tolist(),
                _to_numpy(scene.mi_scene.bbox().max).tolist(),
            ],
        },
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
