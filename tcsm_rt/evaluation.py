from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

from .grid_learning import GRID_MODELS, build_grid_batch, decode_grid_output, new_grid_model
from .learning import PointBatch, build_point_batch, resolve_device
from .metrics import classification_metrics, clustered_bootstrap_ci, executed_rate, holm_adjust, paired_wilcoxon, policy_gap, rmse_db
from .models import DeepSetsOperator, GatedHLG, SetTransformerOperator, StormRMEOperator
from .provenance import write_json_atomic
from .sampling import sample_indices, sample_scene_indices, valid_query_indices
from .schema import load_scene


POINT_MODELS = ("deepsets", "set_transformer", "storm", "gated_hlg")
CLASS_TASKS = ("regime", "far", "near_angle", "near_range")
BASE_FIELDS = (
    "source",
    "split",
    "scene_id",
    "model",
    "train_seed",
    "eval_seed",
    "support_count",
    "support_ratio",
    "sampling_mode",
)
METRIC_FIELDS = (
    "mean_policy_gap",
    "p90_policy_gap",
    "rate95",
    "mean_far_rate_gap",
    "p90_far_rate_gap",
    "far_rate95",
    "regime_accuracy",
    "regime_macro_f1",
    "regime_ece",
    "regime_nll",
    "regime_brier",
    "regime_confidence",
    "regime_entropy",
    "far_top1",
    "far_top3",
    "far_index_mae",
    "far_confidence",
    "far_entropy",
    "near_angle_top1",
    "near_angle_top3",
    "near_range_top1",
    "near_angle_mae_deg",
    "near_range_mae_m",
    "rss_rmse_db",
    "near_count",
    "near_policy_gap",
    "cross_count",
    "cross_policy_gap",
    "far_count",
    "far_policy_gap",
)
GATE_FIELDS = tuple(
    f"gate_{task}_{stat}"
    for task in ("rss", *CLASS_TASKS)
    for stat in ("mean", "p10", "p90")
)
RAW_FIELDS = (*BASE_FIELDS, *METRIC_FIELDS, *GATE_FIELDS)


def evaluation_partitions(
    arrays: dict[str, np.ndarray],
    scene_row: dict[str, Any],
    support_count: int,
    mode: str,
    seed: int,
) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    scene_id = Path(scene_row["cache"]).stem
    source = str(scene_row.get("source", ""))
    valid = valid_query_indices(arrays)
    if source.startswith("deepmimo"):
        if "spatial_split" not in arrays:
            raise ValueError(f"DeepMIMO cache lacks spatial_split: {scene_row['cache']}")
        spatial = np.asarray(arrays["spatial_split"], dtype=np.int8)
        support_candidates = np.intersect1d(valid, np.flatnonzero(spatial == 0))
        local_count = min(int(support_count), len(support_candidates) - 1)
        local = sample_indices(arrays["query_xyz_m"][support_candidates], local_count, mode, seed)
        support = support_candidates[local]
        partitions: list[tuple[str, str, np.ndarray, np.ndarray]] = []
        base_split = str(scene_row.get("split", "deepmimo_external"))
        for partition_id, partition_name in ((1, "spatial_id"), (2, "spatial_holdout")):
            query = np.intersect1d(valid, np.flatnonzero(spatial == partition_id))
            if len(query):
                partitions.append(
                    (
                        f"{scene_id}__{partition_name}",
                        f"{base_split}_{partition_name}",
                        support,
                        query,
                    )
                )
        return partitions
    support = sample_scene_indices(arrays, support_count, mode, seed)
    query = np.setdiff1d(valid, support)
    return [(scene_id, str(scene_row.get("split", "external")), support, query)]


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(np.sum(exp, axis=-1, keepdims=True), 1e-12)


def _expected_calibration_error(probability: np.ndarray, truth: np.ndarray, bins: int = 15) -> float:
    confidence = np.max(probability, axis=1)
    prediction = np.argmax(probability, axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(truth)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        selected = (confidence > lower) & (confidence <= upper)
        if np.any(selected):
            accuracy = np.mean(prediction[selected] == truth[selected])
            error += np.sum(selected) / total * abs(float(accuracy) - float(np.mean(confidence[selected])))
    return float(error)


def _topk_accuracy(logits: np.ndarray, truth: np.ndarray, k: int) -> float:
    k = min(k, logits.shape[1])
    top = np.argpartition(logits, -k, axis=1)[:, -k:]
    return float(np.mean(np.any(top == truth[:, None], axis=1)))


def _baseline_prediction(
    arrays: dict[str, np.ndarray],
    support: np.ndarray,
    query: np.ndarray,
    config: dict[str, Any],
    method: str,
) -> dict[str, np.ndarray]:
    points = arrays["query_xyz_m"]
    if method == "knn":
        neighbour_count = min(3, len(support))
    elif method == "idw":
        neighbour_count = min(int(config["model"]["local_neighbors"]), len(support))
    else:
        raise ValueError(method)
    distances, local = cKDTree(points[support]).query(points[query], k=neighbour_count)
    if neighbour_count == 1:
        distances = distances[:, None]
        local = local[:, None]
    if method == "knn":
        weights = np.ones_like(distances, dtype=np.float64)
    else:
        weights = 1.0 / np.maximum(distances, 1e-3) ** 2
    weights /= np.sum(weights, axis=1, keepdims=True)
    result: dict[str, np.ndarray] = {
        "rss_db": np.sum(arrays["rss_db"][support][local] * weights, axis=1),
    }
    task_specs = {
        "regime": ("regime", 3),
        "far": ("best_far_idx", int(config["model"]["far_beams"])),
        "near_angle": ("best_near_angle", int(config["model"]["near_angles"])),
        "near_range": ("best_near_range", int(config["model"]["near_ranges"])),
    }
    for output_name, (array_name, count) in task_specs.items():
        distribution = np.zeros((len(query), count), dtype=np.float64)
        neighbour_values = arrays[array_name][support][local]
        for column in range(neighbour_count):
            distribution[np.arange(len(query)), neighbour_values[:, column]] += weights[:, column]
        result[output_name + "_logits"] = np.log(np.maximum(distribution, 1e-9))
    return result


def _load_point_model(checkpoint_path: Path, device: torch.device) -> tuple[str, torch.nn.Module, dict[str, Any], int]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model_config = config["model"]
    counts = (
        int(model_config["far_beams"]),
        int(model_config["near_angles"]),
        int(model_config["near_ranges"]),
    )
    name = str(checkpoint["model_name"])
    if name == "gated_hlg" or name.startswith("gated_hlg_"):
        ablation = name.removeprefix("gated_hlg_") if name != "gated_hlg" else None
        model = GatedHLG(
            checkpoint["support_dim"],
            checkpoint["query_dim"],
            int(model_config["hidden"]),
            *counts,
            ablation=ablation,
        )
    elif name == "deepsets":
        model = DeepSetsOperator(checkpoint["support_dim"], checkpoint["query_dim"], int(model_config["hidden"]), counts)
    elif name == "set_transformer":
        model = SetTransformerOperator(checkpoint["support_dim"], checkpoint["query_dim"], int(model_config["hidden"]), counts)
    elif name == "storm":
        model = StormRMEOperator(
            checkpoint["support_dim"],
            checkpoint["query_dim"],
            int(model_config["hidden"]),
            counts,
        )
    else:
        raise ValueError(name)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return name, model, config, int(checkpoint["seed"])


def _point_prediction(
    name: str,
    model: torch.nn.Module,
    arrays: dict[str, np.ndarray],
    support: np.ndarray,
    query: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, np.ndarray]:
    chunks: dict[str, list[np.ndarray]] = {"rss_db": []}
    for task in CLASS_TASKS:
        chunks[task + "_logits"] = []
    gate_chunks: dict[str, list[np.ndarray]] = {task: [] for task in ("rss", *CLASS_TASKS)}
    query_batch = int(config["model"].get("query_batch", 768))
    with torch.inference_mode():
        for start in range(0, len(query), query_batch):
            indices = query[start : start + query_batch]
            batch: PointBatch = build_point_batch(arrays, support, indices, config["model"], device)
            if name == "gated_hlg" or name.startswith("gated_hlg_"):
                output = model(batch.support, batch.query, batch.local_indices, batch.local_prior)
            else:
                output = model(batch.support, batch.query)
            chunks["rss_db"].append((output["rss"][0].detach().cpu().numpy() * 20.0 - 100.0))
            for task in CLASS_TASKS:
                chunks[task + "_logits"].append(output[task][0].detach().cpu().numpy())
            if name == "gated_hlg" or name.startswith("gated_hlg_"):
                for task in gate_chunks:
                    gate_chunks[task].append(output["gates"][task][0].detach().cpu().numpy())
    result = {key: np.concatenate(value, axis=0) for key, value in chunks.items()}
    if name == "gated_hlg":
        result["gates"] = {key: np.concatenate(value, axis=0) for key, value in gate_chunks.items()}
    return result


def _load_grid_model(checkpoint_path: Path, device: torch.device) -> tuple[str, torch.nn.Module, dict[str, Any], int]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    name = str(checkpoint["model_name"])
    model = new_grid_model(name, int(checkpoint["in_channels"]), config)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return name, model, config, int(checkpoint["seed"])


def _grid_prediction(
    model: torch.nn.Module,
    arrays: dict[str, np.ndarray],
    support: np.ndarray,
    query: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, np.ndarray]:
    batch = build_grid_batch(arrays, support, config["model"], device)
    with torch.inference_mode():
        output = decode_grid_output(model(batch.inputs), config["model"])
    result = {"rss_db": output["rss"][0, query].detach().cpu().numpy() * 20.0 - 100.0}
    for task in CLASS_TASKS:
        result[task + "_logits"] = output[task][0, query].detach().cpu().numpy()
    return result


def _prediction_metrics(
    arrays: dict[str, np.ndarray],
    query: np.ndarray,
    prediction: dict[str, np.ndarray],
    config: dict[str, Any],
    task_scope: set[str] | None = None,
) -> dict[str, float]:
    task_scope = task_scope or {"rss", "regime", "far_beam", "near_focus", "rate_decision"}
    truth_regime = arrays["regime"][query]
    regime_logits = prediction["regime_logits"]
    regime_probability = _softmax(regime_logits)
    regime_prediction = np.argmax(regime_logits, axis=1)
    far_prediction = np.argmax(prediction["far_logits"], axis=1)
    far_probability = _softmax(prediction["far_logits"])
    angle_prediction = np.argmax(prediction["near_angle_logits"], axis=1)
    range_prediction = np.argmax(prediction["near_range_logits"], axis=1)
    far_rates = arrays["far_rates"][query]
    far_selected_rate = np.take_along_axis(far_rates, far_prediction[:, None], axis=1)[:, 0]
    far_oracle_rate = np.max(far_rates, axis=1)
    far_gap = policy_gap(far_oracle_rate, far_selected_rate)
    full_task = {"regime", "near_focus", "rate_decision"}.issubset(task_scope)
    if full_task:
        selected_rate = executed_rate(
            regime_prediction,
            far_prediction,
            angle_prediction,
            range_prediction,
            far_rates,
            arrays["near_rates"][query],
            int(config["model"]["near_angles"]),
        )
        oracle = arrays["oracle_rate_bps_hz"][query]
        gap = policy_gap(oracle, selected_rate)
        regime_metrics = classification_metrics(truth_regime, regime_prediction)
    else:
        selected_rate = np.full(len(query), np.nan)
        oracle = np.full(len(query), np.nan)
        gap = np.full(len(query), np.nan)
        regime_metrics = {"accuracy": float("nan"), "macro_f1": float("nan")}
    if "near_angles_axis_rad" in arrays:
        angle_axis_deg = np.rad2deg(arrays["near_angles_axis_rad"])
    else:
        angle_axis_deg = np.linspace(-70.0, 70.0, int(config["model"]["near_angles"]))
    if "near_ranges_axis_m" in arrays:
        range_axis = arrays["near_ranges_axis_m"]
    else:
        range_axis = np.geomspace(4.0, 180.0, int(config["model"]["near_ranges"]))
    metrics = {
        "mean_policy_gap": float(np.mean(gap)) if full_task else float("nan"),
        "p90_policy_gap": float(np.quantile(gap, 0.9)) if full_task else float("nan"),
        "rate95": (
            float(np.mean(selected_rate + 1e-7 >= 0.95 * oracle))
            if full_task
            else float("nan")
        ),
        "mean_far_rate_gap": float(np.mean(far_gap)),
        "p90_far_rate_gap": float(np.quantile(far_gap, 0.9)),
        "far_rate95": float(np.mean(far_selected_rate + 1e-7 >= 0.95 * far_oracle_rate)),
        "regime_accuracy": regime_metrics["accuracy"],
        "regime_macro_f1": regime_metrics["macro_f1"],
        "regime_ece": (
            _expected_calibration_error(regime_probability, truth_regime)
            if full_task
            else float("nan")
        ),
        "regime_nll": (
            float(-np.mean(np.log(np.maximum(regime_probability[np.arange(len(query)), truth_regime], 1e-12))))
            if full_task
            else float("nan")
        ),
        "regime_brier": (
            float(
                np.mean(
                    np.sum(
                        (regime_probability - np.eye(3, dtype=np.float64)[truth_regime]) ** 2,
                        axis=1,
                    )
                )
            )
            if full_task
            else float("nan")
        ),
        "regime_confidence": (
            float(np.mean(np.max(regime_probability, axis=1))) if full_task else float("nan")
        ),
        "regime_entropy": (
            float(
                np.mean(
                    -np.sum(
                        regime_probability * np.log(np.maximum(regime_probability, 1e-12)),
                        axis=1,
                    )
                    / np.log(regime_probability.shape[1])
                )
            )
            if full_task
            else float("nan")
        ),
        "far_top1": float(np.mean(far_prediction == arrays["best_far_idx"][query])),
        "far_top3": _topk_accuracy(prediction["far_logits"], arrays["best_far_idx"][query], 3),
        "far_index_mae": float(np.mean(np.abs(far_prediction - arrays["best_far_idx"][query]))),
        "far_confidence": float(np.mean(np.max(far_probability, axis=1))),
        "far_entropy": float(
            np.mean(
                -np.sum(far_probability * np.log(np.maximum(far_probability, 1e-12)), axis=1)
                / np.log(far_probability.shape[1])
            )
        ),
        "near_angle_top1": (
            float(np.mean(angle_prediction == arrays["best_near_angle"][query]))
            if full_task
            else float("nan")
        ),
        "near_angle_top3": (
            _topk_accuracy(prediction["near_angle_logits"], arrays["best_near_angle"][query], 3)
            if full_task
            else float("nan")
        ),
        "near_range_top1": (
            float(np.mean(range_prediction == arrays["best_near_range"][query]))
            if full_task
            else float("nan")
        ),
        "near_angle_mae_deg": (
            float(
                np.mean(
                    np.abs(
                        angle_axis_deg[angle_prediction]
                        - angle_axis_deg[arrays["best_near_angle"][query]]
                    )
                )
            )
            if full_task
            else float("nan")
        ),
        "near_range_mae_m": (
            float(
                np.mean(
                    np.abs(
                        range_axis[range_prediction]
                        - range_axis[arrays["best_near_range"][query]]
                    )
                )
            )
            if full_task
            else float("nan")
        ),
        "rss_rmse_db": rmse_db(arrays["rss_db"][query], prediction["rss_db"]),
    }
    for label, region in ((0, "near"), (1, "cross"), (2, "far")):
        selected = truth_regime == label
        metrics[f"{region}_count"] = float(np.sum(selected))
        metrics[f"{region}_policy_gap"] = (
            float(np.mean(gap[selected])) if full_task and np.any(selected) else float("nan")
        )
    gates = prediction.get("gates")
    if isinstance(gates, dict):
        for task, values in gates.items():
            metrics[f"gate_{task}_mean"] = float(np.mean(values))
            metrics[f"gate_{task}_p10"] = float(np.quantile(values, 0.1))
            metrics[f"gate_{task}_p90"] = float(np.quantile(values, 0.9))
    return metrics


def _write_rows(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fieldnames or sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _evaluation_conditions(config: dict[str, Any]) -> list[tuple[int, str, int]]:
    return [
        (int(count), str(mode), int(seed))
        for count in config["data"]["support_counts"]
        for mode in config["data"]["sampling_modes"]
        for seed in config["run"]["eval_seeds"]
    ]


def evaluate_models(config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    index = json.loads((run_dir / "scene_index.json").read_text(encoding="utf-8"))
    scene_rows = [row for row in index if row.get("split") != "train"]
    checkpoints = sorted((run_dir / "checkpoints").glob("*.pt"))
    device = resolve_device(config["run"]["device"])
    conditions = _evaluation_conditions(config)
    raw_path = run_dir / "metrics" / "evaluation_raw.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[tuple[str, str, int, int, int, str]] = set()
    if raw_path.exists() and config["run"].get("resume", True):
        with raw_path.open(newline="", encoding="utf-8") as existing_handle:
            for row in csv.DictReader(existing_handle):
                seen.add(
                    (
                        row["scene_id"],
                        row["model"],
                        int(float(row["train_seed"])),
                        int(float(row["eval_seed"])),
                        int(float(row["support_count"])),
                        row["sampling_mode"],
                    )
                )
    append_handle = raw_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(append_handle, fieldnames=list(RAW_FIELDS), extrasaction="ignore")
    if raw_path.stat().st_size == 0:
        writer.writeheader()
        append_handle.flush()

    def evaluate_one(model_name: str, train_seed: int, predictor: Any, regular_only: bool = False) -> None:
        for scene_row in scene_rows:
            arrays = load_scene(scene_row["cache"])
            count_points = len(arrays["query_xyz_m"])
            valid = valid_query_indices(arrays)
            side = int(round(np.sqrt(count_points)))
            if regular_only and side * side != count_points:
                continue
            for support_count, mode, eval_seed in conditions:
                if support_count >= len(valid):
                    continue
                partitions = evaluation_partitions(
                    arrays,
                    scene_row,
                    support_count,
                    mode,
                    eval_seed + int(scene_row.get("seed", 0)),
                )
                for scene_id, split, support, query in partitions:
                    row_key = (
                        scene_id,
                        model_name,
                        train_seed,
                        eval_seed,
                        support_count,
                        mode,
                    )
                    if row_key in seen:
                        continue
                    prediction = predictor(arrays, support, query)
                    task_scope = set(
                        scene_row.get(
                            "external_task_scope",
                            ["rss", "regime", "far_beam", "near_focus", "rate_decision"],
                        )
                    )
                    metric = _prediction_metrics(arrays, query, prediction, config, task_scope)
                    writer.writerow(
                        {
                            "source": scene_row["source"],
                            "split": split,
                            "scene_id": scene_id,
                            "model": model_name,
                            "train_seed": train_seed,
                            "eval_seed": eval_seed,
                            "support_count": len(support),
                            "support_ratio": len(support) / len(valid),
                            "sampling_mode": mode,
                            **metric,
                        }
                    )
                    append_handle.flush()
                    seen.add(row_key)

    for baseline in ("knn", "idw"):
        evaluate_one(
            baseline,
            -1,
            lambda arrays, support, query, method=baseline: _baseline_prediction(
                arrays, support, query, config, method
            ),
        )
    for checkpoint in checkpoints:
        name = checkpoint.stem.rsplit("_seed", 1)[0]
        if name in POINT_MODELS or name.startswith("gated_hlg_"):
            loaded_name, model, checkpoint_config, train_seed = _load_point_model(checkpoint, device)
            evaluate_one(
                loaded_name,
                train_seed,
                lambda arrays, support, query, n=loaded_name, m=model, c=checkpoint_config: _point_prediction(
                    n, m, arrays, support, query, c, device
                ),
            )
        elif name in GRID_MODELS:
            loaded_name, model, checkpoint_config, train_seed = _load_grid_model(checkpoint, device)
            evaluate_one(
                loaded_name,
                train_seed,
                lambda arrays, support, query, m=model, c=checkpoint_config: _grid_prediction(
                    m, arrays, support, query, c, device
                ),
                regular_only=True,
            )
    append_handle.close()
    frame = pd.read_csv(raw_path)
    rows = frame.to_dict(orient="records")
    summary = summarize_evaluation(rows, config)
    return {"raw": str(raw_path), **summary, "row_count": len(rows)}


def summarize_evaluation(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    metric_names = (
        "mean_policy_gap",
        "p90_policy_gap",
        "rate95",
        "mean_far_rate_gap",
        "p90_far_rate_gap",
        "far_rate95",
        "far_top1",
        "far_top3",
        "far_index_mae",
        "regime_macro_f1",
        "regime_ece",
        "regime_nll",
        "regime_brier",
        "regime_confidence",
        "regime_entropy",
        "far_confidence",
        "far_entropy",
        "near_angle_top1",
        "near_angle_top3",
        "near_range_top1",
        "near_angle_mae_deg",
        "near_range_mae_m",
        "near_policy_gap",
        "cross_policy_gap",
        "far_policy_gap",
        "rss_rmse_db",
    )
    group_keys = ("model", "source", "split", "support_count", "sampling_mode")
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in group_keys), []).append(row)
    summary_rows: list[dict[str, Any]] = []
    for key, group in grouped.items():
        base = dict(zip(group_keys, key, strict=True))
        cluster_ids = np.array([row["scene_id"] for row in group])
        summary_row: dict[str, Any] = {**base, "n_rows": len(group), "n_scenes": len(np.unique(cluster_ids))}
        for metric in metric_names:
            values = np.array([row[metric] for row in group], dtype=np.float64)
            mean, lower, upper = clustered_bootstrap_ci(
                values,
                cluster_ids,
                int(config["statistics"]["bootstrap_samples"]),
                float(config["statistics"]["confidence"]),
                int(config["run"]["seed"]),
            )
            summary_row[metric + "_mean"] = mean
            summary_row[metric + "_ci_low"] = lower
            summary_row[metric + "_ci_high"] = upper
        summary_rows.append(summary_row)
    output_root = Path(config["_config_path"]).parent.parent / config["run"]["output_dir"]
    summary_path = output_root / "metrics" / "evaluation_summary.csv"
    _write_rows(summary_path, summary_rows)

    comparison_rows: list[dict[str, Any]] = []
    strata = ("source", "split", "support_count", "sampling_mode")
    for stratum in sorted({tuple(row[key] for key in strata) for row in rows}):
        selected = [row for row in rows if tuple(row[key] for key in strata) == stratum]
        comparison_metric = "mean_far_rate_gap" if str(stratum[0]).startswith("deepmimo") else "mean_policy_gap"
        scene_models: dict[tuple[str, str], list[float]] = {}
        for row in selected:
            value = float(row[comparison_metric])
            if np.isfinite(value):
                scene_models.setdefault((row["scene_id"], row["model"]), []).append(value)
        ours_scenes = sorted({scene for scene, model in scene_models if model == "gated_hlg"})
        baselines = sorted({model for _, model in scene_models if model != "gated_hlg"})
        pending: list[dict[str, Any]] = []
        p_values: list[float] = []
        for baseline in baselines:
            common = [scene for scene in ours_scenes if (scene, baseline) in scene_models]
            if len(common) < 2:
                continue
            ours = np.array([np.mean(scene_models[(scene, "gated_hlg")]) for scene in common])
            other = np.array([np.mean(scene_models[(scene, baseline)]) for scene in common])
            p_value = paired_wilcoxon(other, ours)
            pending.append(
                {
                    **dict(zip(strata, stratum, strict=True)),
                    "baseline": baseline,
                    "metric": comparison_metric,
                    "n_scenes": len(common),
                    "ours_mean_gap": float(np.mean(ours)),
                    "baseline_mean_gap": float(np.mean(other)),
                    "paired_delta_baseline_minus_ours": float(np.mean(other - ours)),
                    "p_value": p_value,
                }
            )
            p_values.append(p_value)
        if pending:
            for row, adjusted in zip(pending, holm_adjust(p_values), strict=True):
                row["holm_p_value"] = adjusted
            comparison_rows.extend(pending)
    significance_path = output_root / "metrics" / "paired_significance.csv"
    _write_rows(significance_path, comparison_rows)
    write_json_atomic(
        output_root / "metrics" / "evaluation_manifest.json",
        {"raw_rows": len(rows), "summary_rows": len(summary_rows), "comparison_rows": len(comparison_rows)},
    )
    return {"summary": str(summary_path), "significance": str(significance_path)}
