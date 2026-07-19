from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class RTConfiguration:
    config_id: str
    source: str
    scene: str
    split: str
    frequency_hz: float
    array_size: int
    placement_index: int
    seed: int


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def grid_xyz(
    center_xyz_m: np.ndarray,
    grid_size: int,
    cell_size_m: float,
    receiver_height_m: float = 1.5,
) -> np.ndarray:
    coordinate = (np.arange(grid_size, dtype=np.float64) - (grid_size - 1) / 2.0) * cell_size_m
    xx, yy = np.meshgrid(coordinate, coordinate, indexing="xy")
    points = np.column_stack(
        [
            center_xyz_m[0] + xx.ravel(),
            center_xyz_m[1] + yy.ravel(),
            np.full(xx.size, receiver_height_m),
        ]
    )
    return points.astype(np.float32)


def allocate_counts(total: int, labels: Iterable[str]) -> dict[str, int]:
    names = list(labels)
    quotient, remainder = divmod(total, len(names))
    return {name: quotient + int(index < remainder) for index, name in enumerate(names)}


def sionna_configuration_manifest(config: dict) -> list[RTConfiguration]:
    settings = config["data"]["sionna"]
    scenes = list(settings["scenes"])
    if config["run"]["name"] == "full_rt" and len(scenes) != 6:
        raise ValueError("the full benchmark requires exactly six declared Sionna scenes")
    if not scenes:
        raise ValueError("at least one Sionna scene must be declared")
    pivot = max(1, len(scenes) // 2)
    in_domain = scenes[:pivot]
    held_out = scenes[pivot:] or in_domain
    counts = settings["configs_per_split"]
    train_system = [
        (float(frequency) * 1e9, int(array_size))
        for frequency in settings["train_frequencies_ghz"]
        for array_size in settings["train_arrays"]
    ]
    ood_system = [
        (float(frequency) * 1e9, int(array_size))
        for frequency in settings["ood_frequencies_ghz"]
        for array_size in settings["ood_arrays"]
    ]
    split_spec = {
        "train": (in_domain, train_system),
        "id": (in_domain, train_system),
        "geometry_ood": (held_out, train_system),
        "system_ood": (in_domain, ood_system),
        "compound_ood": (held_out, ood_system),
    }
    records: list[RTConfiguration] = []
    for split, (split_scenes, systems) in split_spec.items():
        count = int(counts[split])
        for index in range(count):
            scene = split_scenes[index % len(split_scenes)]
            frequency_hz, array_size = systems[(index // len(split_scenes)) % len(systems)]
            placement_index = index // (len(split_scenes) * len(systems))
            seed = stable_seed(config["run"]["seed"], split, scene, index)
            records.append(
                RTConfiguration(
                    config_id=f"sionna_{split}_{index:03d}",
                    source="sionna_rt_2.0.1",
                    scene=scene,
                    split=split,
                    frequency_hz=frequency_hz,
                    array_size=array_size,
                    placement_index=placement_index,
                    seed=seed,
                )
            )
    expected = sum(int(value) for value in counts.values())
    if len(records) != expected:
        raise AssertionError(f"generated {len(records)} records, expected {expected}")
    return records


def spatial_split_ids(xy: np.ndarray, fractions: tuple[float, float, float] = (0.6, 0.2, 0.2)) -> np.ndarray:
    """Assign contiguous x-stripes to train, ID and held-out sets."""
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("split fractions must sum to one")
    x = np.asarray(xy, dtype=np.float64)[:, 0]
    first = np.quantile(x, fractions[0])
    second = np.quantile(x, fractions[0] + fractions[1])
    result = np.full(x.shape[0], 2, dtype=np.int8)
    result[x <= second] = 1
    result[x <= first] = 0
    return result
