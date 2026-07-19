from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from tcsm_rt.pipeline import _training_artifact_complete
from tcsm_rt.training_state import (
    load_training_state,
    save_training_state,
    training_state_path,
)


def test_training_state_restores_parameters_optimizer_and_rng(tmp_path: Path) -> None:
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model(torch.ones(2, 3)).sum()
    loss.backward()
    optimizer.step()
    rng = np.random.default_rng(17)
    rng.integers(0, 1000, size=5)
    path = tmp_path / "model.train_state.pt"
    history = [{"step": 40, "loss": 0.25}]
    save_training_state(
        path,
        model=model,
        optimizer=optimizer,
        model_name="model",
        seed=17,
        step=40,
        target_steps=80,
        history=history,
        rng=rng,
        elapsed_seconds=12.5,
        device=torch.device("cpu"),
    )
    expected_numpy = rng.integers(0, 1000, size=8)
    expected_numpy_global = np.random.randint(0, 1000, size=8)
    expected_python = random.random()
    expected_torch = torch.rand(4)
    expected_parameters = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }

    restored_model = torch.nn.Linear(3, 2)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    step, restored_history, restored_rng, elapsed = load_training_state(
        path,
        model=restored_model,
        optimizer=restored_optimizer,
        model_name="model",
        seed=17,
        target_steps=80,
        device=torch.device("cpu"),
    )

    assert step == 40
    assert restored_history == history
    assert elapsed == pytest.approx(12.5)
    for key, value in restored_model.state_dict().items():
        torch.testing.assert_close(value, expected_parameters[key])
    np.testing.assert_array_equal(restored_rng.integers(0, 1000, size=8), expected_numpy)
    np.testing.assert_array_equal(np.random.randint(0, 1000, size=8), expected_numpy_global)
    assert random.random() == pytest.approx(expected_python)
    torch.testing.assert_close(torch.rand(4), expected_torch)
    for candidate, candidate_optimizer in (
        (model, optimizer),
        (restored_model, restored_optimizer),
    ):
        candidate_optimizer.zero_grad(set_to_none=True)
        continuation_loss = candidate(torch.full((2, 3), 0.25)).square().sum()
        continuation_loss.backward()
        candidate_optimizer.step()
    for key, value in restored_model.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[key])


def test_complete_training_artifact_requires_final_history_step(tmp_path: Path) -> None:
    checkpoint = tmp_path / "gated_hlg_seed53.pt"
    checkpoint.write_bytes(b"checkpoint")
    assert not _training_artifact_complete(checkpoint, 8000)
    checkpoint.with_suffix(".history.json").write_text('[{"step": 7600}]')
    assert not _training_artifact_complete(checkpoint, 8000)
    checkpoint.with_suffix(".history.json").write_text('[{"step": 8000}]')
    assert _training_artifact_complete(checkpoint, 8000)


def test_training_state_path_does_not_collide_with_final_checkpoint() -> None:
    checkpoint = Path("checkpoints/gated_hlg_seed53.pt")
    assert training_state_path(checkpoint).name == "gated_hlg_seed53.train_state.pt"
