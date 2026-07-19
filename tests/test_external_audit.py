from __future__ import annotations

import numpy as np

from tcsm_rt.external_audit import summarize_deepmimo_scene


def test_deepmimo_external_audit_enforces_task_scope_and_spatial_splits():
    count = 12
    far_rates = np.zeros((count, 4), dtype=np.float32)
    best_far = np.arange(count) % 4
    far_rates[np.arange(count), best_far] = 1.0
    arrays = {
        "query_xyz_m": np.column_stack(
            [np.arange(count), np.arange(count) % 3, np.ones(count)]
        ),
        "environment": np.column_stack(
            [np.zeros((count, 5)), np.arange(count) % 2, np.zeros((count, 4))]
        ),
        "channel": np.ones((count, 8), dtype=np.complex64),
        "rss_db": np.linspace(-100.0, -60.0, count),
        "far_rates": far_rates,
        "best_far_idx": best_far,
        "task_availability": np.tile([1.0, 0.0, 1.0, 0.0, 0.0], (count, 1)),
        "spatial_split": np.repeat([0, 1, 2], 4),
    }
    metadata = {
        "scenario": "city_0_newyork_28",
        "dataset_index": 0,
        "external_task_scope": ["rss", "far_beam"],
        "near_field_evidence": "unsupported_by_standard_synthetic_array_dataset",
        "frequency_hz": 28e9,
        "array_size": 8,
    }
    summary, errors = summarize_deepmimo_scene(arrays, metadata)
    assert not errors
    assert summary["query_count"] == count
    assert summary["far_label_argmax_agreement"] == 1.0


def test_deepmimo_external_audit_rejects_near_field_scope():
    count = 6
    arrays = {
        "query_xyz_m": np.column_stack([np.arange(count), np.zeros(count), np.ones(count)]),
        "environment": np.zeros((count, 10)),
        "channel": np.ones((count, 4), dtype=np.complex64),
        "rss_db": np.full(count, -80.0),
        "far_rates": np.eye(count, 3, dtype=np.float32),
        "best_far_idx": np.argmax(np.eye(count, 3), axis=1),
        "task_availability": np.tile([1.0, 1.0, 1.0, 0.0, 0.0], (count, 1)),
        "spatial_split": np.repeat([0, 1, 2], 2),
    }
    metadata = {
        "external_task_scope": ["rss", "regime", "far_beam"],
        "near_field_evidence": "claimed",
    }
    _, errors = summarize_deepmimo_scene(arrays, metadata)
    assert any("task scope" in error for error in errors)
    assert any("near-field evidence boundary" in error for error in errors)
