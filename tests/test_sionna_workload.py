import json
from pathlib import Path

import numpy as np

from tcsm_rt.data.sionna_adapter import (
    _load_explicit_checkpoint,
    _save_explicit_checkpoint,
)
from tcsm_rt.workload import estimate_sionna_workload


def test_core_workload_exposes_explicit_array_scaling() -> None:
    records = [
        {"frequency_hz": 28e9, "array_size": 64},
        {"frequency_hz": 39e9, "array_size": 128},
    ]
    report = estimate_sionna_workload(
        records,
        grid_size=35,
        samples_per_source=500_000,
        path_batch_size=64,
        explicit_batch_size=16,
    )
    assert report["path_solver_calls_per_record"] == 20
    assert report["explicit_solver_calls_per_record"] == 77
    assert report["total_solver_calls"] == 194
    assert report["ray_element_work_proxy"] > report["scalar_sample_budget"]


def test_explicit_batch_checkpoint_rejects_stale_signature(tmp_path: Path) -> None:
    path = tmp_path / "explicit_0000_0002.npz"
    signature = {"config_id": "record-0", "start": 0, "stop": 2}
    channel = np.ones((2, 4), dtype=np.complex64)
    _save_explicit_checkpoint(path, signature, channel)

    loaded = _load_explicit_checkpoint(path, signature, channel.shape)
    np.testing.assert_array_equal(loaded, channel)
    assert _load_explicit_checkpoint(
        path,
        {**signature, "stop": 3},
        channel.shape,
    ) is None
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["signature"] == signature
