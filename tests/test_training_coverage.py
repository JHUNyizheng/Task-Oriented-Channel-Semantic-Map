from __future__ import annotations

import json

import numpy as np

from tcsm_rt.audit import audit_training_label_coverage
from tcsm_rt.schema import save_scene


def _scene(regime: np.ndarray) -> dict[str, np.ndarray]:
    count = len(regime)
    channel = np.ones((count, 4), dtype=np.complex64) * 1e-6
    return {
        "query_xyz_m": np.column_stack([np.arange(count), np.zeros(count), np.ones(count)]),
        "environment": np.column_stack([np.zeros((count, 9)), np.ones(count)]),
        "valid_query_mask": np.ones(count, dtype=bool),
        "channel": channel,
        "rss_db": np.full(count, -80.0, dtype=np.float32),
        "regime": regime.astype(np.int8),
        "best_far_idx": np.arange(count, dtype=np.int16) % 3,
        "best_near_angle": np.arange(count, dtype=np.int16) % 3,
        "best_near_range": np.arange(count, dtype=np.int16) % 2,
        "far_rates": np.ones((count, 3), dtype=np.float32),
        "near_rates": np.ones((count, 6), dtype=np.float32),
        "oracle_rate_bps_hz": np.ones(count, dtype=np.float32),
    }


def test_training_coverage_accepts_all_regimes(tmp_path):
    cache = tmp_path / "train.npz"
    save_scene(cache, _scene(np.repeat([0, 1, 2], 4)))
    (tmp_path / "scene_index.json").write_text(
        json.dumps([{"split": "train", "cache": str(cache)}]),
        encoding="utf-8",
    )
    config = {
        "data": {"sionna": {"configs_per_split": {"train": 1}}},
        "model": {"far_beams": 3, "near_angles": 3, "near_ranges": 2},
        "quality_gates": {"min_regime_fraction": 0.01},
    }
    report = audit_training_label_coverage(tmp_path, config)
    assert report["passed"]
    assert report["valid_training_points"] == 12


def test_training_coverage_rejects_collapsed_regime(tmp_path):
    cache = tmp_path / "train.npz"
    save_scene(cache, _scene(np.full(12, 2)))
    (tmp_path / "scene_index.json").write_text(
        json.dumps([{"split": "train", "cache": str(cache)}]),
        encoding="utf-8",
    )
    config = {
        "data": {"sionna": {"configs_per_split": {"train": 1}}},
        "model": {"far_beams": 3, "near_angles": 3, "near_ranges": 2},
        "quality_gates": {"min_regime_fraction": 0.01},
    }
    report = audit_training_label_coverage(tmp_path, config)
    assert not report["passed"]
    assert any("near fraction" in error for error in report["errors"])
