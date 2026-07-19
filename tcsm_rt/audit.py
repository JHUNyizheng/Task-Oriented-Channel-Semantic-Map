from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import policy_gap, rmse_db
from .physics import classify_near_cross_far
from .provenance import sha256_file, write_json_atomic
from .schema import load_scene


def audit_training_label_coverage(
    run_dir: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    root = Path(run_dir)
    rows = json.loads((root / "scene_index.json").read_text(encoding="utf-8"))
    train_rows = [row for row in rows if row.get("split") == "train"]
    regime_counts = np.zeros(3, dtype=np.int64)
    far_counts = np.zeros(int(config["model"]["far_beams"]), dtype=np.int64)
    angle_counts = np.zeros(int(config["model"]["near_angles"]), dtype=np.int64)
    range_counts = np.zeros(int(config["model"]["near_ranges"]), dtype=np.int64)
    rss_values: list[np.ndarray] = []
    rate_values: list[np.ndarray] = []
    valid_points = 0
    errors: list[str] = []
    for row in train_rows:
        arrays = load_scene(row["cache"])
        valid = np.asarray(
            arrays.get("valid_query_mask", np.ones(len(arrays["query_xyz_m"]), dtype=bool)),
            dtype=bool,
        )
        valid_points += int(np.sum(valid))
        regime_counts += np.bincount(arrays["regime"][valid], minlength=3)
        far_counts += np.bincount(
            arrays["best_far_idx"][valid],
            minlength=len(far_counts),
        )
        angle_counts += np.bincount(
            arrays["best_near_angle"][valid],
            minlength=len(angle_counts),
        )
        range_counts += np.bincount(
            arrays["best_near_range"][valid],
            minlength=len(range_counts),
        )
        rss_values.append(np.asarray(arrays["rss_db"][valid], dtype=np.float64))
        rate_values.append(np.asarray(arrays["oracle_rate_bps_hz"][valid], dtype=np.float64))
        if arrays["environment"].shape[1] != 10:
            errors.append(f"{Path(row['cache']).stem}: environment feature count is not 10")
        if not np.all(arrays["environment"][valid, 9] > 0.5):
            errors.append(f"{Path(row['cache']).stem}: Sionna environment modality is unavailable")
    expected_scenes = int(config["data"]["sionna"]["configs_per_split"]["train"])
    if len(train_rows) != expected_scenes:
        errors.append(f"training scene count is {len(train_rows)}, expected {expected_scenes}")
    if valid_points == 0:
        errors.append("training split contains no valid query points")
        fractions = np.zeros(3, dtype=np.float64)
    else:
        fractions = regime_counts / valid_points
    minimum_fraction = float(config.get("quality_gates", {}).get("min_regime_fraction", 0.01))
    for label, name in enumerate(("near", "cross", "far")):
        if fractions[label] < minimum_fraction:
            errors.append(
                f"aggregate {name} fraction is {fractions[label]:.6f}, "
                f"below {minimum_fraction:.6f}"
            )
    rss = np.concatenate(rss_values) if rss_values else np.array([], dtype=np.float64)
    rate = np.concatenate(rate_values) if rate_values else np.array([], dtype=np.float64)
    report = {
        "passed": not errors,
        "training_scene_count": len(train_rows),
        "valid_training_points": valid_points,
        "minimum_regime_fraction": minimum_fraction,
        "regime_counts": {
            name: int(regime_counts[label])
            for label, name in enumerate(("near", "cross", "far"))
        },
        "regime_fractions": {
            name: float(fractions[label])
            for label, name in enumerate(("near", "cross", "far"))
        },
        "far_beam_classes_present": int(np.sum(far_counts > 0)),
        "near_angle_classes_present": int(np.sum(angle_counts > 0)),
        "near_range_classes_present": int(np.sum(range_counts > 0)),
        "rss_db_quantiles": np.quantile(rss, [0.05, 0.5, 0.95]).tolist() if rss.size else [],
        "oracle_rate_quantiles_bps_hz": (
            np.quantile(rate, [0.05, 0.5, 0.95]).tolist() if rate.size else []
        ),
        "errors": errors,
    }
    write_json_atomic(root / "training_label_coverage.json", report)
    return report


def audit_run(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    index_path = root / "scene_index.json"
    if not index_path.exists():
        report = {"passed": False, "errors": ["scene_index.json is missing"], "warnings": []}
        write_json_atomic(root / "audit_report.json", report)
        return report
    rows = json.loads(index_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    warnings: list[str] = []
    sources: set[str] = set()
    splits: set[str] = set()
    explicit_points = 0
    reconstruction_rejections = 0
    for row in rows:
        cache = Path(row["cache"])
        if not cache.exists():
            errors.append(f"missing cache: {cache}")
            continue
        expected_hash = row.get("cache_sha256")
        if expected_hash and sha256_file(cache) != expected_hash:
            errors.append(f"hash mismatch: {cache}")
        arrays = load_scene(cache)
        valid_query = np.asarray(
            arrays.get("valid_query_mask", np.ones(len(arrays["query_xyz_m"]), dtype=bool)),
            dtype=bool,
        )
        if valid_query.shape != (len(arrays["query_xyz_m"]),):
            errors.append(f"invalid valid_query_mask shape: {cache}")
        elif np.sum(valid_query) < 2:
            errors.append(f"fewer than two valid query positions: {cache}")
        elif str(row["source"]).startswith("sionna") and np.all(valid_query):
            warnings.append(f"Sionna scene contains no occupied grid cells: {cache}")
        sources.add(str(row["source"]))
        splits.add(str(row.get("split", "external")))
        rss_recomputed = 10.0 * np.log10(
            np.maximum(np.sum(np.abs(arrays["channel"]) ** 2, axis=1), 1e-30)
        )
        rss_recompute_rmse = rmse_db(arrays["rss_db"], rss_recomputed)
        if rss_recompute_rmse > 1e-4:
            errors.append(f"RSS independent recomputation RMSE={rss_recompute_rmse:.3e} dB: {cache}")
        oracle_recomputed = np.maximum(
            np.max(arrays["far_rates"], axis=1),
            np.max(arrays["near_rates"], axis=1),
        )
        oracle_max_error = float(
            np.max(np.abs(arrays["oracle_rate_bps_hz"].astype(np.float64) - oracle_recomputed))
        )
        if oracle_max_error > 1e-5:
            errors.append(f"oracle-rate independent recomputation max error={oracle_max_error:.3e}: {cache}")
        if "far_codebook_loss_bps_hz" in arrays:
            far_loss_recomputed = oracle_recomputed - np.max(arrays["far_rates"], axis=1)
            far_loss_error = float(
                np.max(
                    np.abs(
                        arrays["far_codebook_loss_bps_hz"].astype(np.float64)
                        - far_loss_recomputed.astype(np.float64)
                    )
                )
            )
            if far_loss_error > 1e-5:
                errors.append(f"far-codebook-loss max error={far_loss_error:.3e}: {cache}")
        regime = np.asarray(arrays["regime"], dtype=np.int64)
        far_default = np.asarray(arrays["far_rates"][:, 0], dtype=np.float64)
        near_default = np.asarray(arrays["near_rates"][:, 0], dtype=np.float64)
        # Regime 2 executes the far-field codebook. Near and cross-field labels use the
        # near-field focusing codebook for this direction-only sanity check.
        selected = np.where(regime == 2, far_default, near_default)
        try:
            gap = policy_gap(arrays["oracle_rate_bps_hz"], selected)
            if not np.all(gap >= 0.0):
                errors.append(f"policy gap direction failed: {cache}")
        except ValueError as error:
            errors.append(str(error))
        classes, counts = np.unique(arrays["regime"], return_counts=True)
        fractions = {int(label): int(count) / len(arrays["regime"]) for label, count in zip(classes, counts)}
        missing_classes = sorted({0, 1, 2} - set(fractions))
        if missing_classes:
            warnings.append(f"missing regime classes {missing_classes}: {cache}")
        elif min(fractions.values()) < 0.005:
            warnings.append(f"regime class below 0.5% ({fractions}): {cache}")
        if str(row["source"]).startswith("sionna"):
            solver = row.get("solver", {})
            if solver.get("element_channel") != "explicit_array":
                errors.append(f"Sionna production cache is not explicit-array: {cache}")
            validation = row.get("explicit_array_validation") or {}
            explicit_points += int(validation.get("points", 0))
            if validation and not validation.get("reconstruction_accepted", False):
                reconstruction_rejections += 1
            system = row.get("system", {})
            if system and "far_codebook_loss_bps_hz" in arrays:
                low = float(system["regime_low_margin_bps_hz"])
                high = float(system["regime_high_margin_bps_hz"])
                expected_regime = classify_near_cross_far(
                    arrays["distance_m"],
                    arrays["rayleigh_distance_m"],
                    arrays["far_codebook_loss_bps_hz"],
                    low,
                    high,
                )
                mismatch = float(np.mean(expected_regime != arrays["regime"]))
                if mismatch > 0.0:
                    errors.append(f"regime-label recomputation mismatch={mismatch:.3%}: {cache}")
    if not any(source.startswith("sionna") for source in sources):
        errors.append("no official Sionna cache is present")
    if not any(source.startswith("deepmimo") for source in sources):
        warnings.append("no DeepMIMO cache is present")
    if "train" in splits and len(splits) < 3:
        warnings.append("fewer than three evaluation split types are present")
    report = {
        "passed": not errors,
        "scene_count": len(rows),
        "sources": sorted(sources),
        "splits": sorted(splits),
        "errors": errors,
        "warnings": warnings,
        "explicit_validation_points": explicit_points,
        "reconstruction_rejections": reconstruction_rejections,
    }
    write_json_atomic(root / "audit_report.json", report)
    return report
