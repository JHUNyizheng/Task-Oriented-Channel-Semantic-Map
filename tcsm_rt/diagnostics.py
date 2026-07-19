from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import numpy as np
import torch
from scipy.spatial import cKDTree

from .evaluation import (
    _baseline_prediction,
    _grid_prediction,
    _load_grid_model,
    _load_point_model,
    _point_prediction,
    _prediction_metrics,
    _write_rows,
)
from .grid_learning import GRID_MODELS
from .learning import accelerator_memory_mb, reset_peak_memory, resolve_device, synchronize_device
from .metrics import clustered_bootstrap_ci
from .provenance import write_json_atomic
from .sampling import sample_scene_indices, valid_query_indices
from .schema import load_scene


Predictor = Callable[[dict[str, np.ndarray], np.ndarray, np.ndarray], dict[str, np.ndarray]]


def regime_labels_for_margins(
    arrays: dict[str, np.ndarray],
    low_margin_bps_hz: float | np.ndarray,
    high_margin_bps_hz: float | np.ndarray,
) -> np.ndarray:
    """Re-label a cached scene without recomputing channels or codebook rates."""
    loss = np.asarray(arrays["far_codebook_loss_bps_hz"], dtype=np.float64)
    distance = np.asarray(arrays["distance_m"], dtype=np.float64)
    rayleigh = np.asarray(arrays["rayleigh_distance_m"], dtype=np.float64)
    low = np.broadcast_to(np.asarray(low_margin_bps_hz, dtype=np.float64), loss.shape)
    high = np.broadcast_to(np.asarray(high_margin_bps_hz, dtype=np.float64), loss.shape)
    if np.any(low < 0) or np.any(low >= high):
        raise ValueError("all regime margins must satisfy 0 <= low < high")
    regime = np.full(loss.shape, 1, dtype=np.int8)
    regime[(distance <= rayleigh) & (loss >= high)] = 0
    regime[(distance > rayleigh) & (loss <= low)] = 2
    return regime


def mcs_adaptive_margins(
    arrays: dict[str, np.ndarray],
    efficiencies_bps_hz: Iterable[float],
    low_to_high_ratio: float,
    lower_clip_bps_hz: float,
    upper_clip_bps_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the rate gain needed to cross the next configured MCS efficiency."""
    efficiencies = np.sort(np.unique(np.asarray(list(efficiencies_bps_hz), dtype=np.float64)))
    if len(efficiencies) < 2 or np.any(efficiencies <= 0):
        raise ValueError("at least two positive MCS spectral efficiencies are required")
    far_rate = np.max(np.asarray(arrays["far_rates"], dtype=np.float64), axis=1)
    insertion = np.searchsorted(efficiencies, far_rate, side="right")
    insertion = np.minimum(insertion, len(efficiencies) - 1)
    raw = efficiencies[insertion] - far_rate
    terminal_spacing = efficiencies[-1] - efficiencies[-2]
    raw[insertion == len(efficiencies) - 1] = np.maximum(
        raw[insertion == len(efficiencies) - 1],
        terminal_spacing,
    )
    high = np.clip(raw, lower_clip_bps_hz, upper_clip_bps_hz)
    return high * low_to_high_ratio, high


def _scene_rows(run_dir: Path, sionna_only: bool = True) -> list[dict[str, Any]]:
    rows = json.loads((run_dir / "scene_index.json").read_text(encoding="utf-8"))
    selected = [row for row in rows if row.get("split") != "train"]
    if sionna_only:
        selected = [row for row in selected if str(row.get("source", "")).startswith("sionna")]
    return selected


def _predictors(
    config: dict[str, Any],
    run_dir: Path,
    requested_models: set[str],
    requested_seeds: set[int] | None,
    device: torch.device,
) -> Iterator[tuple[str, int, Predictor, bool]]:
    for baseline in ("knn", "idw"):
        if baseline in requested_models:
            yield (
                baseline,
                -1,
                lambda arrays, support, query, method=baseline: _baseline_prediction(
                    arrays, support, query, config, method
                ),
                False,
            )
    for checkpoint in sorted((run_dir / "checkpoints").glob("*.pt")):
        stem = checkpoint.stem.rsplit("_seed", 1)[0]
        if stem not in requested_models:
            continue
        if stem in GRID_MODELS:
            name, model, checkpoint_config, seed = _load_grid_model(checkpoint, device)
            if requested_seeds is not None and seed not in requested_seeds:
                continue
            yield (
                name,
                seed,
                lambda arrays, support, query, m=model, c=checkpoint_config: _grid_prediction(
                    m, arrays, support, query, c, device
                ),
                True,
            )
        else:
            name, model, checkpoint_config, seed = _load_point_model(checkpoint, device)
            if requested_seeds is not None and seed not in requested_seeds:
                continue
            yield (
                name,
                seed,
                lambda arrays, support, query, n=name, m=model, c=checkpoint_config: _point_prediction(
                    n, m, arrays, support, query, c, device
                ),
                False,
            )


def _summarize_diagnostic(
    rows: list[dict[str, Any]],
    group_keys: tuple[str, ...],
    metric_names: Iterable[str],
    config: dict[str, Any],
    output_path: Path,
) -> None:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in group_keys), []).append(row)
    summary: list[dict[str, Any]] = []
    for key, group in grouped.items():
        cluster_ids = np.asarray([row["scene_id"] for row in group])
        result: dict[str, Any] = {
            **dict(zip(group_keys, key, strict=True)),
            "n_rows": len(group),
            "n_scenes": len(np.unique(cluster_ids)),
        }
        for metric in metric_names:
            values = np.asarray([row[metric] for row in group], dtype=np.float64)
            mean, low, high = clustered_bootstrap_ci(
                values,
                cluster_ids,
                int(config["statistics"]["bootstrap_samples"]),
                float(config["statistics"]["confidence"]),
                int(config["run"]["seed"]),
            )
            result[f"{metric}_mean"] = mean
            result[f"{metric}_ci_low"] = low
            result[f"{metric}_ci_high"] = high
        summary.append(result)
    _write_rows(output_path, summary)


def run_threshold_sensitivity(config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    settings = config.get("thresholds")
    if not settings:
        return {"status": "skipped", "reason": "thresholds section is absent"}
    rows = _scene_rows(run_dir)
    device = resolve_device(config["run"]["device"])
    models = set(settings.get("models", ["storm", "gated_hlg"]))
    seeds = set(int(value) for value in settings.get("sensitivity_train_seeds", [])) or None
    support_count = int(settings.get("support_count", 24))
    sampling_mode = str(settings.get("sampling_mode", "trajectory"))
    eval_seeds = [int(value) for value in settings.get("eval_seeds", config["run"]["eval_seeds"])]
    original_low = float(config["system"]["regime_low_margin_bps_hz"])
    original_high = float(config["system"]["regime_high_margin_bps_hz"])
    ratio = float(settings.get("low_to_high_ratio", original_low / original_high))
    conditions: list[tuple[str, float | None, float | None]] = [
        ("fixed", ratio * float(high), float(high))
        for high in settings["fixed_bps_hz"]
    ]
    conditions.append(("mcs_adaptive", None, None))
    raw_rows: list[dict[str, Any]] = []
    for model_name, train_seed, predictor, regular_only in _predictors(
        config, run_dir, models, seeds, device
    ):
        for scene_row in rows:
            arrays = load_scene(scene_row["cache"])
            count = len(arrays["query_xyz_m"])
            side = int(round(np.sqrt(count)))
            if regular_only and side * side != count:
                continue
            valid = valid_query_indices(arrays)
            for eval_seed in eval_seeds:
                support = sample_scene_indices(
                    arrays,
                    support_count,
                    sampling_mode,
                    eval_seed + int(scene_row.get("seed", 0)),
                )
                query = np.setdiff1d(valid, support)
                for threshold_type, low, high in conditions:
                    if threshold_type == "mcs_adaptive":
                        low_values, high_values = mcs_adaptive_margins(
                            arrays,
                            settings["mcs_spectral_efficiency_bps_hz"],
                            ratio,
                            float(settings["adaptive_clip_bps_hz"][0]),
                            float(settings["adaptive_clip_bps_hz"][1]),
                        )
                        regime = regime_labels_for_margins(arrays, low_values, high_values)
                        low_report = float(np.mean(low_values))
                        high_report = float(np.mean(high_values))
                    else:
                        regime = regime_labels_for_margins(arrays, float(low), float(high))
                        low_report = float(low)
                        high_report = float(high)
                    relabeled_input = dict(arrays)
                    relabeled_input["regime"] = regime
                    prediction = predictor(relabeled_input, support, query)
                    relabeled_truth = dict(arrays)
                    relabeled_truth["regime"] = regime
                    metric = _prediction_metrics(relabeled_truth, query, prediction, config)
                    raw_rows.append(
                        {
                            "source": scene_row["source"],
                            "split": scene_row["split"],
                            "scene_id": Path(scene_row["cache"]).stem,
                            "model": model_name,
                            "train_seed": train_seed,
                            "eval_seed": eval_seed,
                            "support_count": len(support),
                            "sampling_mode": sampling_mode,
                            "threshold_type": threshold_type,
                            "low_margin_bps_hz": low_report,
                            "high_margin_bps_hz": high_report,
                            "near_fraction": float(np.mean(regime == 0)),
                            "cross_fraction": float(np.mean(regime == 1)),
                            "far_fraction": float(np.mean(regime == 2)),
                            **metric,
                        }
                    )
    raw_path = run_dir / "metrics" / "threshold_sensitivity_raw.csv"
    _write_rows(raw_path, raw_rows)
    summary_path = run_dir / "metrics" / "threshold_sensitivity_summary.csv"
    _summarize_diagnostic(
        raw_rows,
        ("model", "split", "threshold_type", "high_margin_bps_hz"),
        (
            "mean_policy_gap",
            "p90_policy_gap",
            "rate95",
            "regime_macro_f1",
            "regime_ece",
            "near_fraction",
            "cross_fraction",
            "far_fraction",
        ),
        config,
        summary_path,
    )
    manifest = {
        "status": "complete",
        "protocol": "fixed checkpoints; support and query regime labels are redefined together",
        "raw_rows": len(raw_rows),
        "models": sorted(models),
        "train_seeds": sorted(seeds) if seeds else "all",
        "raw": str(raw_path),
        "summary": str(summary_path),
    }
    write_json_atomic(run_dir / "metrics" / "threshold_sensitivity_manifest.json", manifest)
    return manifest


def _corrupt_categorical(
    values: np.ndarray,
    indices: np.ndarray,
    fraction: float,
    class_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    result = np.asarray(values).copy()
    count = int(round(fraction * len(indices)))
    if count == 0:
        return result
    selected = rng.choice(indices, size=min(count, len(indices)), replace=False)
    offsets = rng.integers(1, class_count, size=len(selected))
    result[selected] = (result[selected].astype(np.int64) + offsets) % class_count
    return result


def _jitter_support(
    arrays: dict[str, np.ndarray],
    support: np.ndarray,
    fraction: float,
    cell_size_m: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if fraction <= 0:
        return support.copy()
    valid = valid_query_indices(arrays)
    points = arrays["query_xyz_m"][valid]
    tree = cKDTree(points[:, :2])
    result = support.copy()
    selected = rng.choice(
        len(result),
        size=min(len(result), int(round(fraction * len(result)))),
        replace=False,
    )
    targets = arrays["query_xyz_m"][result[selected], :2] + rng.normal(
        scale=cell_size_m,
        size=(len(selected), 2),
    )
    _, nearest = tree.query(targets, k=min(8, len(valid)))
    nearest = np.atleast_2d(nearest)
    if nearest.shape[0] != len(selected):
        nearest = nearest.T
    occupied = set(result.tolist())
    for row, position in enumerate(selected):
        occupied.discard(int(result[position]))
        for candidate_local in np.asarray(nearest[row]).reshape(-1):
            candidate = int(valid[int(candidate_local)])
            if candidate not in occupied:
                result[position] = candidate
                occupied.add(candidate)
                break
        else:
            occupied.add(int(result[position]))
    return np.sort(np.unique(result)).astype(np.int64)


def perturb_observations(
    arrays: dict[str, np.ndarray],
    support: np.ndarray,
    impairment: str,
    level: float,
    seed: int,
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Corrupt only evidence available to the estimator; labels in arrays remain truth."""
    rng = np.random.default_rng(seed)
    observed = dict(arrays)
    perturbed_support = np.asarray(support, dtype=np.int64).copy()
    if impairment == "position_error_m":
        points = np.asarray(arrays["query_xyz_m"], dtype=np.float32)
        noisy = points.copy()
        noisy[:, :2] += rng.normal(scale=level, size=(len(points), 2)).astype(np.float32)
        nearest = cKDTree(points[:, :2]).query(noisy[:, :2], k=1)[1]
        observed["query_xyz_m"] = noisy
        observed["environment"] = np.asarray(arrays["environment"])[nearest].copy()
    elif impairment == "power_noise_db":
        rss = np.asarray(arrays["rss_db"], dtype=np.float32).copy()
        rss[perturbed_support] += rng.normal(scale=level, size=len(perturbed_support)).astype(np.float32)
        observed["rss_db"] = rss
    elif impairment == "label_error_fraction":
        specifications = (
            ("regime", 3),
            ("best_far_idx", int(config["model"]["far_beams"])),
            ("best_near_angle", int(config["model"]["near_angles"])),
            ("best_near_range", int(config["model"]["near_ranges"])),
        )
        for name, class_count in specifications:
            observed[name] = _corrupt_categorical(
                arrays[name],
                perturbed_support,
                level,
                class_count,
                rng,
            )
    elif impairment == "support_missing_fraction":
        retain = max(1, int(round((1.0 - level) * len(perturbed_support))))
        perturbed_support = np.sort(
            rng.choice(perturbed_support, size=retain, replace=False)
        ).astype(np.int64)
    elif impairment == "trajectory_jitter_fraction":
        perturbed_support = _jitter_support(
            arrays,
            perturbed_support,
            level,
            float(config["data"]["cell_size_m"]),
            rng,
        )
    else:
        raise ValueError(f"unknown impairment: {impairment}")
    return observed, perturbed_support


def run_robustness(config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    settings = config.get("robustness")
    if not settings:
        return {"status": "skipped", "reason": "robustness section is absent"}
    scene_rows = _scene_rows(run_dir)
    device = resolve_device(config["run"]["device"])
    models = set(settings.get("models", ["set_transformer", "storm", "radiounet", "gated_hlg"]))
    seeds = set(int(value) for value in settings.get("train_seeds", [])) or None
    support_count = int(settings.get("support_count", 24))
    sampling_mode = str(settings.get("sampling_mode", "trajectory"))
    eval_seeds = [int(value) for value in settings.get("eval_seeds", config["run"]["eval_seeds"])]
    impairment_levels = {
        key: [float(value) for value in values]
        for key, values in settings.items()
        if key
        in {
            "position_error_m",
            "power_noise_db",
            "label_error_fraction",
            "support_missing_fraction",
            "trajectory_jitter_fraction",
        }
    }
    raw_rows: list[dict[str, Any]] = []
    for model_name, train_seed, predictor, regular_only in _predictors(
        config, run_dir, models, seeds, device
    ):
        for scene_row in scene_rows:
            arrays = load_scene(scene_row["cache"])
            side = int(round(np.sqrt(len(arrays["query_xyz_m"]))))
            if regular_only and side * side != len(arrays["query_xyz_m"]):
                continue
            valid = valid_query_indices(arrays)
            for eval_seed in eval_seeds:
                base_support = sample_scene_indices(
                    arrays,
                    support_count,
                    sampling_mode,
                    eval_seed + int(scene_row.get("seed", 0)),
                )
                for impairment, levels in impairment_levels.items():
                    for level in levels:
                        perturbation_seed = (
                            eval_seed
                            + int(scene_row.get("seed", 0)) * 1009
                            + int(round(level * 1000)) * 9176
                        )
                        observed, support = perturb_observations(
                            arrays,
                            base_support,
                            impairment,
                            level,
                            perturbation_seed,
                            config,
                        )
                        query = np.setdiff1d(valid, support)
                        prediction = predictor(observed, support, query)
                        metric = _prediction_metrics(arrays, query, prediction, config)
                        raw_rows.append(
                            {
                                "source": scene_row["source"],
                                "split": scene_row["split"],
                                "scene_id": Path(scene_row["cache"]).stem,
                                "model": model_name,
                                "train_seed": train_seed,
                                "eval_seed": eval_seed,
                                "impairment": impairment,
                                "level": level,
                                "nominal_support_count": len(base_support),
                                "effective_support_count": len(support),
                                "sampling_mode": sampling_mode,
                                **metric,
                            }
                        )
    raw_path = run_dir / "metrics" / "robustness_raw.csv"
    _write_rows(raw_path, raw_rows)
    summary_path = run_dir / "metrics" / "robustness_summary.csv"
    _summarize_diagnostic(
        raw_rows,
        ("model", "split", "impairment", "level"),
        (
            "mean_policy_gap",
            "p90_policy_gap",
            "rate95",
            "regime_macro_f1",
            "regime_ece",
            "far_top1",
            "near_angle_mae_deg",
            "near_range_mae_m",
            "rss_rmse_db",
        ),
        config,
        summary_path,
    )
    manifest = {
        "status": "complete",
        "protocol": "truth-preserving corruption of estimator-visible evidence",
        "raw_rows": len(raw_rows),
        "models": sorted(models),
        "train_seeds": sorted(seeds) if seeds else "all",
        "raw": str(raw_path),
        "summary": str(summary_path),
    }
    write_json_atomic(run_dir / "metrics" / "robustness_manifest.json", manifest)
    return manifest


def _profile_latency(
    predictor: Predictor,
    arrays: dict[str, np.ndarray],
    support: np.ndarray,
    query: np.ndarray,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[np.ndarray, float, str]:
    for _ in range(warmup):
        predictor(arrays, support, query)
    synchronize_device(device)
    reset_peak_memory(device)
    durations = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        started = time.perf_counter()
        predictor(arrays, support, query)
        synchronize_device(device)
        durations[index] = time.perf_counter() - started
    memory_mb, measurement = accelerator_memory_mb(device)
    return durations, memory_mb, measurement


def profile_models(config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    settings = config.get("deployment", {})
    scene_rows = _scene_rows(run_dir)
    if not scene_rows:
        raise RuntimeError("deployment profiling requires at least one non-training Sionna scene")
    arrays = load_scene(scene_rows[0]["cache"])
    support_count = int(settings.get("support_count", 24))
    support = sample_scene_indices(
        arrays,
        support_count,
        str(settings.get("sampling_mode", "trajectory")),
        int(config["run"]["seed"]),
    )
    query = np.setdiff1d(valid_query_indices(arrays), support)
    warmup = int(settings.get("warmup", 5))
    repeats = int(settings.get("repeats", 30))
    requested = set(settings.get("models", config["model"]["baselines"]))
    requested_seeds = set(int(value) for value in settings.get("train_seeds", [])) or None
    device_requests = list(settings.get("devices", ["auto", "cpu"]))
    result_rows: list[dict[str, Any]] = []
    for requested_device in device_requests:
        device = resolve_device(str(requested_device))
        seen_architectures: set[str] = set()
        for name, seed, predictor, _ in _predictors(
            config, run_dir, requested, requested_seeds, device
        ):
            if name in seen_architectures:
                continue
            seen_architectures.add(name)
            checkpoint = next(
                iter(sorted((run_dir / "checkpoints").glob(f"{name}_seed*.pt"))),
                None,
            )
            if checkpoint is None:
                parameters = 0
                checkpoint_mb = 0.0
            else:
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                parameters = int(sum(value.numel() for value in payload["model"].values()))
                checkpoint_mb = checkpoint.stat().st_size / 2**20
            durations, memory_mb, memory_measurement = _profile_latency(
                predictor,
                arrays,
                support,
                query,
                device,
                warmup,
                repeats,
            )
            result_rows.append(
                {
                    "model": name,
                    "profile_seed": seed,
                    "device": str(device),
                    "support_count": len(support),
                    "query_count": len(query),
                    "parameters": parameters,
                    "checkpoint_mb": checkpoint_mb,
                    "latency_median_ms": float(np.median(durations) * 1000.0),
                    "latency_p95_ms": float(np.quantile(durations, 0.95) * 1000.0),
                    "throughput_query_s": float(len(query) / np.mean(durations)),
                    "accelerator_memory_mb": memory_mb,
                    "memory_measurement": memory_measurement,
                    "warmup": warmup,
                    "repeats": repeats,
                    "includes_preprocessing": True,
                }
            )
    output_path = run_dir / "metrics" / "deployment_profile.csv"
    _write_rows(output_path, result_rows)
    manifest = {
        "status": "complete",
        "rows": len(result_rows),
        "scene_id": Path(scene_rows[0]["cache"]).stem,
        "output": str(output_path),
    }
    write_json_atomic(run_dir / "metrics" / "deployment_profile_manifest.json", manifest)
    return manifest
