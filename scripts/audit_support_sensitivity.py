#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from tcsm_rt.config import load_config
from tcsm_rt.evaluation import (
    _load_point_model,
    _point_prediction,
    _softmax,
    evaluation_partitions,
)
from tcsm_rt.learning import resolve_device
from tcsm_rt.provenance import write_json_atomic
from tcsm_rt.schema import load_scene


def _support_hash(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def _branch_change(
    first: dict[str, np.ndarray],
    second: dict[str, np.ndarray],
    key: str,
) -> dict[str, float]:
    first_probability = _softmax(first[key])
    second_probability = _softmax(second[key])
    return {
        "mean_l1_probability_change": float(
            np.mean(np.sum(np.abs(first_probability - second_probability), axis=1))
        ),
        "argmax_change_fraction": float(
            np.mean(np.argmax(first_probability, axis=1) != np.argmax(second_probability, axis=1))
        ),
    }


def audit_support_sensitivity(
    config_path: Path,
    checkpoint_path: Path,
    lower_count: int,
    upper_count: int,
    sampling_mode: str,
    eval_seed: int,
    partition_suffix: str,
) -> dict[str, Any]:
    config = load_config(config_path)
    run_dir = Path(config["run"]["output_dir"])
    if not run_dir.is_absolute():
        run_dir = config_path.resolve().parent.parent / run_dir
    rows = json.loads((run_dir / "scene_index.json").read_text(encoding="utf-8"))
    selected_splits = set(config.get("evaluation", {}).get("scene_splits", []))
    scene_row = next(
        row
        for row in rows
        if not selected_splits or str(row.get("split")) in selected_splits
    )
    arrays = load_scene(scene_row["cache"])
    device = resolve_device(config["run"]["device"])
    name, model, checkpoint_config, train_seed = _load_point_model(checkpoint_path, device)
    predictions: dict[int, dict[str, np.ndarray]] = {}
    supports: dict[int, np.ndarray] = {}
    queries: dict[int, np.ndarray] = {}
    for count in (lower_count, upper_count):
        partitions = evaluation_partitions(
            arrays,
            scene_row,
            count,
            sampling_mode,
            eval_seed + int(scene_row.get("seed", 0)),
        )
        _, _, support, query = next(
            partition for partition in partitions if partition[0].endswith(partition_suffix)
        )
        predictions[count] = _point_prediction(
            name,
            model,
            arrays,
            support,
            query,
            checkpoint_config,
            device,
        )
        supports[count] = support
        queries[count] = query
    np.testing.assert_array_equal(queries[lower_count], queries[upper_count])
    first = predictions[lower_count]
    second = predictions[upper_count]
    branch_keys = {
        "fused": "far_logits",
        "neural": "neural_far_logits",
        "local_prior": "local_prior_far_logits",
    }
    report = {
        "config": str(config_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "model": name,
        "train_seed": train_seed,
        "scene_id": Path(scene_row["cache"]).stem,
        "partition_suffix": partition_suffix,
        "sampling_mode": sampling_mode,
        "eval_seed": eval_seed,
        "lower_support_count": len(supports[lower_count]),
        "upper_support_count": len(supports[upper_count]),
        "lower_support_sha256": _support_hash(supports[lower_count]),
        "upper_support_sha256": _support_hash(supports[upper_count]),
        "query_count": len(queries[lower_count]),
        "branches": {
            branch: _branch_change(first, second, key)
            for branch, key in branch_keys.items()
            if key in first and key in second
        },
        "mean_absolute_rss_change_db": float(
            np.mean(np.abs(first["rss_db"] - second["rss_db"]))
        ),
        "gate_far_mean": {
            str(lower_count): float(np.mean(first["gates"]["far"])),
            str(upper_count): float(np.mean(second["gates"]["far"])),
        },
    }
    report["passed"] = bool(
        report["lower_support_sha256"] != report["upper_support_sha256"]
        and report["branches"]["local_prior"]["mean_l1_probability_change"] > 0.0
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lower-count", type=int, default=1)
    parser.add_argument("--upper-count", type=int, default=245)
    parser.add_argument("--sampling-mode", default="trajectory")
    parser.add_argument("--eval-seed", type=int, default=101)
    parser.add_argument("--partition-suffix", default="spatial_holdout")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_support_sensitivity(
        args.config,
        args.checkpoint,
        args.lower_count,
        args.upper_count,
        args.sampling_mode,
        args.eval_seed,
        args.partition_suffix,
    )
    if args.output:
        write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
