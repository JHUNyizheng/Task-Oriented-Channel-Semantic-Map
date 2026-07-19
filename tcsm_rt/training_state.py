from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def training_state_path(checkpoint: Path) -> Path:
    return checkpoint.with_suffix(".train_state.pt")


def save_training_state(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    model_name: str,
    seed: int,
    step: int,
    target_steps: int,
    history: list[dict[str, float | int]],
    rng: np.random.Generator,
    elapsed_seconds: float,
    device: torch.device,
) -> None:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "model_name": model_name,
        "seed": int(seed),
        "step": int(step),
        "target_steps": int(target_steps),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "history": history,
        "numpy_rng_state": rng.bit_generator.state,
        "numpy_global_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "elapsed_seconds": float(elapsed_seconds),
    }
    if device.type == "cuda":
        payload["accelerator_rng_state"] = torch.cuda.get_rng_state_all()
    elif device.type == "mps" and hasattr(torch.mps, "get_rng_state"):
        payload["accelerator_rng_state"] = torch.mps.get_rng_state()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_training_state(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    model_name: str,
    seed: int,
    target_steps: int,
    device: torch.device,
) -> tuple[int, list[dict[str, float | int]], np.random.Generator, float]:
    rng = np.random.default_rng(seed)
    if not path.exists():
        return 0, [], rng, 0.0
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("model_name") != model_name or int(payload.get("seed", -1)) != seed:
        raise ValueError(f"training-state identity mismatch: {path}")
    if int(payload.get("target_steps", -1)) != target_steps:
        raise ValueError(f"training-state target-step mismatch: {path}")
    step = int(payload.get("step", -1))
    if step < 0 or step > target_steps:
        raise ValueError(f"invalid saved training step {step}: {path}")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    rng.bit_generator.state = payload["numpy_rng_state"]
    np.random.set_state(payload["numpy_global_rng_state"])
    random.setstate(payload["python_rng_state"])
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    accelerator_state = payload.get("accelerator_rng_state")
    if accelerator_state is not None and device.type == "cuda":
        torch.cuda.set_rng_state_all(accelerator_state)
    elif (
        accelerator_state is not None
        and device.type == "mps"
        and hasattr(torch.mps, "set_rng_state")
    ):
        torch.mps.set_rng_state(accelerator_state.cpu())
    history = list(payload.get("history", []))
    return step, history, rng, float(payload.get("elapsed_seconds", 0.0))
