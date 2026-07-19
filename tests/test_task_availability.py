from __future__ import annotations

import numpy as np
import torch

from tcsm_rt.learning import multitask_loss, restrict_training_partition


def _prediction() -> dict[str, torch.Tensor]:
    return {
        "rss": torch.tensor([[0.1, -0.2]], requires_grad=True),
        "regime": torch.zeros((1, 2, 3), requires_grad=True),
        "far": torch.zeros((1, 2, 4), requires_grad=True),
        "near_angle": torch.zeros((1, 2, 5), requires_grad=True),
        "near_range": torch.zeros((1, 2, 3), requires_grad=True),
    }


def test_multitask_loss_ignores_unavailable_deepmimo_targets() -> None:
    targets = {
        "rss": torch.tensor([[0.0, 0.0]]),
        "regime": torch.tensor([[0, 2]]),
        "far": torch.tensor([[1, 3]]),
        "near_angle": torch.tensor([[0, 4]]),
        "near_range": torch.tensor([[1, 2]]),
    }
    availability = torch.tensor([[[1.0, 0.0, 1.0, 0.0, 0.0]]]).expand(1, 2, 5)
    first = _prediction()
    second = _prediction()
    second["regime"].data.fill_(100.0)
    second["near_angle"].data.fill_(-100.0)
    second["near_range"].data.fill_(50.0)
    first_loss = multitask_loss(first, targets, availability=availability)
    second_loss = multitask_loss(second, targets, availability=availability)
    torch.testing.assert_close(first_loss, second_loss)
    first_loss.backward()
    assert torch.count_nonzero(first["regime"].grad) == 0
    assert torch.count_nonzero(first["near_angle"].grad) == 0
    assert torch.count_nonzero(first["near_range"].grad) == 0
    assert torch.count_nonzero(first["far"].grad) > 0


def test_external_training_view_uses_only_declared_spatial_partition() -> None:
    arrays = {
        "query_xyz_m": np.column_stack([np.arange(8), np.zeros(8), np.ones(8)]),
        "valid_query_mask": np.array([True, True, False, True, True, True, True, True]),
        "spatial_split": np.array([0, 0, 0, 1, 1, 2, 2, 2]),
    }
    restricted = restrict_training_partition(
        arrays,
        {"external_training": {"spatial_split": 0}},
    )
    np.testing.assert_array_equal(
        restricted["valid_query_mask"],
        np.array([True, True, False, False, False, False, False, False]),
    )
    assert restricted is not arrays

