from __future__ import annotations

import math
from collections import Counter
from typing import Any


def estimate_sionna_workload(
    records: list[dict[str, Any]],
    *,
    grid_size: int,
    samples_per_source: int,
    path_batch_size: int,
    explicit_batch_size: int,
) -> dict[str, Any]:
    if not records:
        raise ValueError("records must not be empty")
    for name, value in (
        ("grid_size", grid_size),
        ("samples_per_source", samples_per_source),
        ("path_batch_size", path_batch_size),
        ("explicit_batch_size", explicit_batch_size),
    ):
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")

    query_count = int(grid_size) ** 2
    path_calls = math.ceil(query_count / int(path_batch_size))
    explicit_calls = math.ceil(query_count / int(explicit_batch_size))
    cell_counts = Counter(
        (int(record["frequency_hz"]), int(record["array_size"]))
        for record in records
    )
    record_count = len(records)
    solver_calls = record_count * (path_calls + explicit_calls)
    scalar_sample_budget = solver_calls * int(samples_per_source)

    # This proxy weights a non-synthetic explicit-array trace by its element
    # count. It is not an exact ray count, but it exposes the array-size scaling
    # hidden by a scalar samples_per_source setting.
    ray_element_work_proxy = sum(
        count
        * (
            path_calls * int(samples_per_source)
            + explicit_calls * int(samples_per_source) * array_size
        )
        for (_, array_size), count in cell_counts.items()
    )
    return {
        "record_count": record_count,
        "query_count_per_record": query_count,
        "path_solver_calls_per_record": path_calls,
        "explicit_solver_calls_per_record": explicit_calls,
        "total_solver_calls_per_record": path_calls + explicit_calls,
        "total_solver_calls": solver_calls,
        "scalar_sample_budget": scalar_sample_budget,
        "ray_element_work_proxy": ray_element_work_proxy,
        "frequency_array_cells": [
            {
                "frequency_hz": frequency_hz,
                "array_size": array_size,
                "record_count": count,
            }
            for (frequency_hz, array_size), count in sorted(cell_counts.items())
        ],
    }
