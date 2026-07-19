from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree

from .models import DeepSetsOperator, GatedHLG, SetTransformerOperator, StormRMEOperator
from .provenance import write_json_atomic
from .sampling import sample_scene_indices, valid_query_indices
from .schema import load_scene
from .training_state import load_training_state, save_training_state, training_state_path


@dataclass(frozen=True)
class PointBatch:
    support: torch.Tensor
    query: torch.Tensor
    local_indices: torch.Tensor
    local_prior: dict[str, torch.Tensor]
    targets: dict[str, torch.Tensor]
    availability: torch.Tensor


def _one_hot(values: np.ndarray, count: int) -> np.ndarray:
    result = np.zeros((len(values), count), dtype=np.float32)
    result[np.arange(len(values)), np.asarray(values, dtype=np.int64)] = 1.0
    return result


def _coordinates(points: np.ndarray) -> np.ndarray:
    center = points.mean(axis=0, keepdims=True)
    scale = np.maximum(points.std(axis=0, keepdims=True), 1.0)
    return ((points - center) / scale).astype(np.float32)


def build_point_batch(
    arrays: dict[str, np.ndarray],
    support_indices: np.ndarray,
    query_indices: np.ndarray,
    model_config: dict[str, Any],
    device: torch.device,
) -> PointBatch:
    points = arrays["query_xyz_m"].astype(np.float32)
    coordinates = _coordinates(points)
    environment = arrays["environment"].astype(np.float32)
    env_scale = np.maximum(np.nanstd(environment, axis=0, keepdims=True), 1.0)
    env_center = np.nanmean(environment, axis=0, keepdims=True)
    environment = np.nan_to_num((environment - env_center) / env_scale).astype(np.float32)
    far_count = int(model_config["far_beams"])
    angle_count = int(model_config["near_angles"])
    range_count = int(model_config["near_ranges"])
    availability = arrays.get(
        "task_availability",
        np.ones((len(points), 5), dtype=np.float32),
    ).astype(np.float32)
    support_availability = availability[support_indices]
    rss_normalized = ((arrays["rss_db"] + 100.0) / 20.0).astype(np.float32)
    support = np.column_stack(
        [
            coordinates[support_indices],
            environment[support_indices],
            rss_normalized[support_indices, None] * support_availability[:, 0:1],
            _one_hot(arrays["regime"][support_indices], 3) * support_availability[:, 1:2],
            _one_hot(arrays["best_far_idx"][support_indices], far_count)
            * support_availability[:, 2:3],
            _one_hot(arrays["best_near_angle"][support_indices], angle_count)
            * support_availability[:, 3:4],
            _one_hot(arrays["best_near_range"][support_indices], range_count)
            * support_availability[:, 4:5],
            support_availability,
        ]
    ).astype(np.float32)
    support_ratio = np.full((len(query_indices), 1), len(support_indices) / len(points), dtype=np.float32)
    query = np.column_stack([coordinates[query_indices], environment[query_indices], support_ratio]).astype(np.float32)
    neighbour_count = min(int(model_config["local_neighbors"]), len(support_indices))
    tree = cKDTree(points[support_indices])
    distances, local_indices = tree.query(points[query_indices], k=neighbour_count)
    if neighbour_count == 1:
        distances = distances[:, None]
        local_indices = local_indices[:, None]
    weights = 1.0 / np.maximum(distances, 1e-3) ** 2
    weights /= np.sum(weights, axis=1, keepdims=True)

    def available_weights(column: int) -> tuple[np.ndarray, np.ndarray]:
        local_available = support_availability[local_indices, column]
        selected_weights = weights * local_available
        denominator = np.sum(selected_weights, axis=1, keepdims=True)
        normalized = np.zeros_like(selected_weights, dtype=np.float64)
        np.divide(selected_weights, denominator, out=normalized, where=denominator > 0.0)
        return normalized, denominator[:, 0] > 0.0

    def prior_distribution(values: np.ndarray, count: int, column: int) -> np.ndarray:
        neighbours = values[support_indices][local_indices]
        distribution = np.zeros((len(query_indices), count), dtype=np.float32)
        task_weights, task_available = available_weights(column)
        for neighbour_column in range(neighbour_count):
            distribution[
                np.arange(len(query_indices)),
                neighbours[:, neighbour_column],
            ] += task_weights[:, neighbour_column]
        pseudocount = float(model_config.get("prior_pseudocount", 0.0))
        if pseudocount < 0.0:
            raise ValueError("model.prior_pseudocount must be non-negative")
        if pseudocount:
            distribution[task_available] += pseudocount / count
            distribution[task_available] /= np.sum(
                distribution[task_available],
                axis=1,
                keepdims=True,
            )
        logits = np.zeros_like(distribution, dtype=np.float32)
        logits[task_available] = np.log(
            np.maximum(distribution[task_available], 1e-6)
        ).astype(np.float32)
        return logits

    rss_weights, _ = available_weights(0)
    prior = {
        "rss": np.sum(
            rss_normalized[support_indices][local_indices]
            * rss_weights,
            axis=1,
        ).astype(np.float32),
        "regime": prior_distribution(arrays["regime"], 3, 1),
        "far": prior_distribution(arrays["best_far_idx"], far_count, 2),
        "near_angle": prior_distribution(arrays["best_near_angle"], angle_count, 3),
        "near_range": prior_distribution(arrays["best_near_range"], range_count, 4),
    }
    targets = {
        "rss": rss_normalized[query_indices],
        "regime": arrays["regime"][query_indices].astype(np.int64),
        "far": arrays["best_far_idx"][query_indices].astype(np.int64),
        "near_angle": arrays["best_near_angle"][query_indices].astype(np.int64),
        "near_range": arrays["best_near_range"][query_indices].astype(np.int64),
    }
    return PointBatch(
        support=torch.from_numpy(support[None]).to(device),
        query=torch.from_numpy(query[None]).to(device),
        local_indices=torch.from_numpy(local_indices[None].astype(np.int64)).to(device),
        local_prior={key: torch.from_numpy(value[None]).to(device) for key, value in prior.items()},
        targets={key: torch.from_numpy(value[None]).to(device) for key, value in targets.items()},
        availability=torch.from_numpy(availability[query_indices][None]).to(device),
    )


def multitask_loss(
    prediction: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    class_weights: dict[str, torch.Tensor] | None = None,
    availability: torch.Tensor | None = None,
) -> torch.Tensor:
    class_weights = class_weights or {}

    if availability is None:
        availability = torch.ones(
            (*targets["rss"].shape, 5),
            dtype=prediction["rss"].dtype,
            device=prediction["rss"].device,
        )

    def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(dtype=values.dtype).reshape_as(values)
        denominator = torch.sum(mask)
        if not bool(denominator.detach().cpu() > 0):
            return torch.sum(values) * 0.0
        return torch.sum(values * mask) / denominator

    def masked_cross_entropy(task: str, column: int) -> torch.Tensor:
        logits = prediction[task].flatten(0, 1)
        truth = targets[task].flatten()
        mask = availability[..., column].flatten()
        if not bool(torch.any(mask > 0).detach().cpu()):
            return torch.sum(logits) * 0.0
        values = torch.nn.functional.cross_entropy(
            logits,
            truth,
            weight=class_weights.get(task),
            reduction="none",
        )
        return masked_mean(values, mask)

    rss = masked_mean(
        torch.square(prediction["rss"] - targets["rss"]),
        availability[..., 0],
    )
    regime = masked_cross_entropy("regime", 1)
    far = masked_cross_entropy("far", 2)
    near_angle = masked_cross_entropy("near_angle", 3)
    near_range = masked_cross_entropy("near_range", 4)
    return 0.15 * rss + 0.65 * regime + 0.60 * far + 0.55 * near_angle + 0.55 * near_range


def compute_class_weights(
    scenes: list[dict[str, np.ndarray]],
    model_config: dict[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    specifications = {
        "regime": ("regime", 3),
        "far": ("best_far_idx", int(model_config["far_beams"])),
        "near_angle": ("best_near_angle", int(model_config["near_angles"])),
        "near_range": ("best_near_range", int(model_config["near_ranges"])),
    }
    result: dict[str, torch.Tensor] = {}
    for task, (array_name, class_count) in specifications.items():
        counts = np.zeros(class_count, dtype=np.float64)
        availability_column = {
            "regime": 1,
            "far": 2,
            "near_angle": 3,
            "near_range": 4,
        }[task]
        for arrays in scenes:
            valid = valid_query_indices(arrays)
            task_availability = arrays.get("task_availability")
            if task_availability is not None:
                valid = valid[np.asarray(task_availability)[valid, availability_column] > 0]
            counts += np.bincount(arrays[array_name][valid], minlength=class_count)
        present = counts > 0
        if not np.any(present):
            continue
        weights = np.zeros(class_count, dtype=np.float32)
        weights[present] = 1.0 / np.sqrt(counts[present])
        weights[present] /= np.mean(weights[present])
        weights[present] = np.clip(weights[present], 0.25, 4.0)
        result[task] = torch.from_numpy(weights).to(device)
    return result


def restrict_training_partition(
    arrays: dict[str, np.ndarray],
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Return a training view restricted to an explicitly declared spatial partition."""
    settings = config.get("external_training", {})
    partition = settings.get("spatial_split")
    if partition is None:
        return arrays
    if "spatial_split" not in arrays:
        raise ValueError("external_training.spatial_split requires a spatial_split array")
    result = dict(arrays)
    base = np.asarray(
        arrays.get("valid_query_mask", np.ones(len(arrays["query_xyz_m"]), dtype=bool)),
        dtype=bool,
    )
    result["valid_query_mask"] = base & (np.asarray(arrays["spatial_split"]) == int(partition))
    if np.sum(result["valid_query_mask"]) < 2:
        raise ValueError(f"training spatial partition {partition} retains fewer than two queries")
    return result


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def accelerator_memory_mb(device: torch.device) -> tuple[float, str]:
    if device.type == "cuda":
        return float(torch.cuda.max_memory_allocated(device) / 2**20), "cuda_peak_allocated"
    if device.type == "mps":
        return float(torch.mps.current_allocated_memory() / 2**20), "mps_allocated_at_end"
    return 0.0, "not_applicable"


def _new_model(name: str, batch: PointBatch, config: dict[str, Any]) -> torch.nn.Module:
    model_config = config["model"]
    support_dim = batch.support.shape[-1]
    query_dim = batch.query.shape[-1]
    hidden = int(model_config["hidden"])
    counts = (
        int(model_config["far_beams"]),
        int(model_config["near_angles"]),
        int(model_config["near_ranges"]),
    )
    if name == "gated_hlg" or name.startswith("gated_hlg_"):
        ablation = name.removeprefix("gated_hlg_") if name != "gated_hlg" else None
        return GatedHLG(
            support_dim,
            query_dim,
            hidden,
            *counts,
            ablation=ablation,
            gate_evidence_features=bool(model_config.get("gate_evidence_features", False)),
        )
    if name == "deepsets":
        return DeepSetsOperator(support_dim, query_dim, hidden, counts)
    if name == "set_transformer":
        return SetTransformerOperator(support_dim, query_dim, hidden, counts)
    if name == "storm":
        return StormRMEOperator(support_dim, query_dim, hidden, counts)
    raise ValueError(f"unsupported point model: {name}")


def _forward(name: str, model: torch.nn.Module, batch: PointBatch) -> dict[str, torch.Tensor]:
    if name == "gated_hlg" or name.startswith("gated_hlg_"):
        return model(batch.support, batch.query, batch.local_indices, batch.local_prior)
    return model(batch.support, batch.query)


def train_point_model(
    name: str,
    scene_paths: list[Path],
    config: dict[str, Any],
    seed: int,
    output_path: Path,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = resolve_device(config["run"]["device"])
    scenes = [restrict_training_partition(load_scene(path), config) for path in scene_paths]
    class_weights = compute_class_weights(scenes, config["model"], device)
    first = scenes[0]
    valid_first = valid_query_indices(first)
    probe_support = sample_scene_indices(first, min(8, len(valid_first) - 1), "scatter", seed)
    probe_query = np.setdiff1d(valid_first, probe_support)[:16]
    probe = build_point_batch(first, probe_support, probe_query, config["model"], device)
    model = _new_model(name, probe, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["model"].get("learning_rate", 2e-3)),
        weight_decay=float(config["model"].get("weight_decay", 1e-4)),
    )
    steps = int(config["run"]["train_steps"])
    support_counts = list(config["data"]["support_counts"])
    sampling_modes = list(config["data"]["sampling_modes"])
    query_batch = int(config["model"].get("query_batch", 768))
    steps_per_checkpoint = int(config["run"].get("training_checkpoint_interval", 400))
    if steps_per_checkpoint <= 0:
        raise ValueError("training_checkpoint_interval must be positive")
    state_path = training_state_path(output_path)
    start_step, history, rng, elapsed_before = load_training_state(
        state_path,
        model=model,
        optimizer=optimizer,
        model_name=name,
        seed=seed,
        target_steps=steps,
        device=device,
    )
    reset_peak_memory(device)
    synchronize_device(device)
    training_started = time.perf_counter()
    model.train()
    for step in range(start_step, steps):
        scene_index = int(rng.integers(0, len(scenes)))
        arrays = scenes[scene_index]
        count = int(support_counts[int(rng.integers(0, len(support_counts)))])
        mode = sampling_modes[int(rng.integers(0, len(sampling_modes)))]
        support = sample_scene_indices(arrays, count, mode, seed + step * 7919)
        available = np.setdiff1d(valid_query_indices(arrays), support)
        query_count = min(query_batch, len(available))
        query = rng.choice(available, size=query_count, replace=False)
        batch = build_point_batch(arrays, support, query, config["model"], device)
        optimizer.zero_grad(set_to_none=True)
        prediction = _forward(name, model, batch)
        loss = multitask_loss(prediction, batch.targets, class_weights, batch.availability)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        if step == 0 or (step + 1) % max(1, steps // 40) == 0:
            history.append({"step": step + 1, "loss": float(loss.detach().cpu())})
        if (step + 1) % steps_per_checkpoint == 0:
            synchronize_device(device)
            save_training_state(
                state_path,
                model=model,
                optimizer=optimizer,
                model_name=name,
                seed=seed,
                step=step + 1,
                target_steps=steps,
                history=history,
                rng=rng,
                elapsed_seconds=elapsed_before + time.perf_counter() - training_started,
                device=device,
            )
    synchronize_device(device)
    training_seconds = elapsed_before + time.perf_counter() - training_started
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "model_name": name,
            "seed": seed,
            "support_dim": probe.support.shape[-1],
            "query_dim": probe.query.shape[-1],
            "class_weights": {
                task: weight.detach().cpu().numpy().tolist()
                for task, weight in class_weights.items()
            },
            "config": {key: value for key, value in config.items() if not key.startswith("_")},
        },
        output_path,
    )
    write_json_atomic(output_path.with_suffix(".history.json"), history)
    state_path.unlink(missing_ok=True)
    memory_mb, memory_measurement = accelerator_memory_mb(device)
    return {
        "model": name,
        "seed": seed,
        "checkpoint": str(output_path),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "final_loss": history[-1]["loss"],
        "training_seconds": training_seconds,
        "steps_per_second": steps / max(training_seconds, 1e-12),
        "checkpoint_mb": output_path.stat().st_size / 2**20,
        "accelerator_memory_mb": memory_mb,
        "memory_measurement": memory_measurement,
        "device": str(device),
    }
