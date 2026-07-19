from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .learning import (
    _coordinates,
    _one_hot,
    accelerator_memory_mb,
    compute_class_weights,
    multitask_loss,
    reset_peak_memory,
    resolve_device,
    synchronize_device,
)
from .models import FNOOperator, RadioUNet, WNOOperator
from .provenance import write_json_atomic
from .sampling import sample_scene_indices, valid_query_indices
from .schema import load_scene
from .training_state import load_training_state, save_training_state, training_state_path


GRID_MODELS = ("radiounet", "fno", "wno")


@dataclass(frozen=True)
class GridBatch:
    inputs: torch.Tensor
    targets: dict[str, torch.Tensor]
    query_mask: torch.Tensor
    height: int
    width: int


def _grid_shape(arrays: dict[str, np.ndarray]) -> tuple[int, int]:
    count = len(arrays["query_xyz_m"])
    side = int(round(np.sqrt(count)))
    if side * side != count:
        raise ValueError(f"grid baselines require a square raster, received {count} points")
    return side, side


def _task_channel_count(model_config: dict[str, Any]) -> int:
    return (
        1
        + 3
        + int(model_config["far_beams"])
        + int(model_config["near_angles"])
        + int(model_config["near_ranges"])
    )


def build_grid_batch(
    arrays: dict[str, np.ndarray],
    support_indices: np.ndarray,
    model_config: dict[str, Any],
    device: torch.device,
) -> GridBatch:
    height, width = _grid_shape(arrays)
    count = height * width
    support_indices = np.asarray(support_indices, dtype=np.int64)
    query_mask = np.zeros(count, dtype=bool)
    query_mask[valid_query_indices(arrays)] = True
    query_mask[support_indices] = False
    coordinates = _coordinates(arrays["query_xyz_m"].astype(np.float32))
    environment = arrays["environment"].astype(np.float32)
    env_scale = np.maximum(np.nanstd(environment, axis=0, keepdims=True), 1.0)
    env_center = np.nanmean(environment, axis=0, keepdims=True)
    environment = np.nan_to_num((environment - env_center) / env_scale).astype(np.float32)
    far_count = int(model_config["far_beams"])
    angle_count = int(model_config["near_angles"])
    range_count = int(model_config["near_ranges"])
    rss_normalized = ((arrays["rss_db"] + 100.0) / 20.0).astype(np.float32)
    mask = np.zeros((count, 1), dtype=np.float32)
    mask[support_indices] = 1.0

    observed = np.zeros((count, 1 + 3 + far_count + angle_count + range_count), dtype=np.float32)
    observed[support_indices] = np.column_stack(
        [
            rss_normalized[support_indices, None],
            _one_hot(arrays["regime"][support_indices], 3),
            _one_hot(arrays["best_far_idx"][support_indices], far_count),
            _one_hot(arrays["best_near_angle"][support_indices], angle_count),
            _one_hot(arrays["best_near_range"][support_indices], range_count),
        ]
    )
    features = np.column_stack([coordinates, environment, mask, observed]).astype(np.float32)
    inputs = torch.from_numpy(features.T.reshape(1, features.shape[1], height, width)).to(device)
    targets = {
        "rss": torch.from_numpy(rss_normalized[None]).to(device),
        "regime": torch.from_numpy(arrays["regime"].astype(np.int64)[None]).to(device),
        "far": torch.from_numpy(arrays["best_far_idx"].astype(np.int64)[None]).to(device),
        "near_angle": torch.from_numpy(arrays["best_near_angle"].astype(np.int64)[None]).to(device),
        "near_range": torch.from_numpy(arrays["best_near_range"].astype(np.int64)[None]).to(device),
    }
    return GridBatch(
        inputs=inputs,
        targets=targets,
        query_mask=torch.from_numpy(query_mask[None]).to(device),
        height=height,
        width=width,
    )


def decode_grid_output(output: torch.Tensor, model_config: dict[str, Any]) -> dict[str, torch.Tensor]:
    flat = output.flatten(2).transpose(1, 2)
    far_count = int(model_config["far_beams"])
    angle_count = int(model_config["near_angles"])
    range_count = int(model_config["near_ranges"])
    cursor = 0
    result: dict[str, torch.Tensor] = {"rss": flat[..., cursor]}
    cursor += 1
    result["regime"] = flat[..., cursor : cursor + 3]
    cursor += 3
    result["far"] = flat[..., cursor : cursor + far_count]
    cursor += far_count
    result["near_angle"] = flat[..., cursor : cursor + angle_count]
    cursor += angle_count
    result["near_range"] = flat[..., cursor : cursor + range_count]
    return result


def new_grid_model(name: str, in_channels: int, config: dict[str, Any]) -> torch.nn.Module:
    hidden = int(config["model"]["hidden"])
    output_channels = _task_channel_count(config["model"])
    if name == "radiounet":
        return RadioUNet(in_channels, hidden, output_channels)
    if name == "fno":
        return FNOOperator(in_channels, hidden, output_channels)
    if name == "wno":
        return WNOOperator(in_channels, hidden, output_channels)
    raise ValueError(f"unsupported grid model: {name}")


def _masked_loss(
    prediction: dict[str, torch.Tensor],
    batch: GridBatch,
    class_weights: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    mask = batch.query_mask
    selected_prediction = {
        name: value[mask] if value.ndim == 2 else value[mask]
        for name, value in prediction.items()
    }
    selected_targets = {name: value[mask] for name, value in batch.targets.items()}
    selected_prediction = {
        name: value[None] if value.ndim == 1 else value[None]
        for name, value in selected_prediction.items()
    }
    selected_targets = {name: value[None] for name, value in selected_targets.items()}
    return multitask_loss(selected_prediction, selected_targets, class_weights)


def train_grid_model(
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
    scenes = [load_scene(path) for path in scene_paths]
    class_weights = compute_class_weights(scenes, config["model"], device)
    valid_first = valid_query_indices(scenes[0])
    probe_support = sample_scene_indices(scenes[0], min(8, len(valid_first) - 1), "scatter", seed)
    probe = build_grid_batch(scenes[0], probe_support, config["model"], device)
    model = new_grid_model(name, probe.inputs.shape[1], config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["model"].get("learning_rate", 2e-3)),
        weight_decay=float(config["model"].get("weight_decay", 1e-4)),
    )
    steps = int(config["run"]["train_steps"])
    support_counts = list(config["data"]["support_counts"])
    sampling_modes = list(config["data"]["sampling_modes"])
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
        arrays = scenes[int(rng.integers(0, len(scenes)))]
        count = int(rng.choice(support_counts))
        mode = str(rng.choice(sampling_modes))
        support = sample_scene_indices(arrays, count, mode, seed + step * 7919)
        batch = build_grid_batch(arrays, support, config["model"], device)
        optimizer.zero_grad(set_to_none=True)
        prediction = decode_grid_output(model(batch.inputs), config["model"])
        loss = _masked_loss(prediction, batch, class_weights)
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
            "in_channels": probe.inputs.shape[1],
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
