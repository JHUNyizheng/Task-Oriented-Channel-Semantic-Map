from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"configuration must be a mapping: {config_path}")
    sample_override = os.environ.get("TCSM_SAMPLES_PER_SOURCE")
    if sample_override:
        config["data"]["sionna"]["samples_per_source"] = int(sample_override)
    config["_config_path"] = str(config_path)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = ("run", "data", "model", "system", "statistics")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"missing configuration sections: {missing}")
    grid_size = int(config["data"]["grid_size"])
    if grid_size < 3:
        raise ValueError("grid_size must be at least 3")
    point_count = grid_size * grid_size
    support_counts = [int(value) for value in config["data"]["support_counts"]]
    if support_counts != sorted(set(support_counts)):
        raise ValueError("support_counts must be sorted and unique")
    if support_counts[0] < 1 or support_counts[-1] >= point_count:
        raise ValueError("support counts must lie in [1, grid_size^2 - 1]")
    if int(config["model"]["far_beams"]) % 2 == 0:
        raise ValueError("far_beams must be odd so that broadside is represented")
    if int(config["model"]["near_angles"]) % 2 == 0:
        raise ValueError("near_angles must be odd so that broadside is represented")
    system = config["system"]
    if float(system["bandwidth_hz"]) <= 0:
        raise ValueError("system.bandwidth_hz must be positive")
    low = float(system["regime_low_margin_bps_hz"])
    high = float(system["regime_high_margin_bps_hz"])
    if not 0.0 <= low < high:
        raise ValueError("regime margins must satisfy 0 <= low < high")
    thresholds = config.get("thresholds")
    if thresholds:
        values = [float(value) for value in thresholds.get("fixed_bps_hz", [])]
        if not values or values != sorted(set(values)) or values[0] <= 0:
            raise ValueError("thresholds.fixed_bps_hz must be positive, sorted, and unique")
        ratio = float(thresholds.get("low_to_high_ratio", low / high))
        if not 0.0 <= ratio < 1.0:
            raise ValueError("thresholds.low_to_high_ratio must lie in [0, 1)")
    robustness = config.get("robustness")
    if robustness:
        for key, values in robustness.items():
            if key in {"models", "train_seeds", "support_count", "sampling_mode"}:
                continue
            numeric = [float(value) for value in values]
            if numeric != sorted(set(numeric)) or numeric[0] < 0:
                raise ValueError(f"robustness.{key} must be non-negative, sorted, and unique")


def canonical_json(config: dict[str, Any]) -> str:
    public = {key: value for key, value in config.items() if not key.startswith("_")}
    return json.dumps(public, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
