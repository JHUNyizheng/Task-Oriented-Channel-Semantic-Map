from __future__ import annotations

import torch


TASK_NAMES = ("regime", "far", "near_angle", "near_range")


class MultiTaskHeads(torch.nn.Module):
    def __init__(self, hidden: int, far_count: int, near_angle_count: int, near_range_count: int):
        super().__init__()
        self.rss = torch.nn.Linear(hidden, 1)
        self.regime = torch.nn.Linear(hidden, 3)
        self.far = torch.nn.Linear(hidden, far_count)
        self.near_angle = torch.nn.Linear(hidden, near_angle_count)
        self.near_range = torch.nn.Linear(hidden, near_range_count)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "rss": self.rss(features).squeeze(-1),
            "regime": self.regime(features),
            "far": self.far(features),
            "near_angle": self.near_angle(features),
            "near_range": self.near_range(features),
        }

