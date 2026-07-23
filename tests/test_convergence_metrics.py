from __future__ import annotations

import numpy as np

from scripts.run_sionna_sample_convergence import _comparison


def _labels(rss_db: list[float], regime: list[int]) -> dict[str, np.ndarray]:
    values = np.asarray(regime, dtype=np.int16)
    return {
        "rss_db": np.asarray(rss_db, dtype=np.float32),
        "oracle_rate_bps_hz": np.asarray([1.0, 0.0, 2.0], dtype=np.float32),
        "regime": values,
        "best_far_idx": values,
        "best_near_angle": values,
        "best_near_range": values,
    }


def test_convergence_metrics_separate_path_detection_from_active_error() -> None:
    reference_channel = np.asarray(
        [[1.0, 0.0], [0.0, 0.0], [1.0, 0.0]],
        dtype=np.complex64,
    )
    estimate_channel = np.asarray(
        [[1.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        dtype=np.complex64,
    )
    reference_labels = _labels([-60.0, -200.0, -70.0], [0, 2, 1])
    estimate_labels = _labels([-61.0, -40.0, -72.0], [0, 0, 1])

    row = _comparison(
        500_000,
        1.0,
        estimate_channel,
        estimate_labels,
        reference_channel,
        reference_labels,
    )

    assert row["reference_active_point_count"] == 2
    assert row["estimate_active_point_count"] == 3
    assert row["active_point_count"] == 2
    assert row["path_detection_agreement"] == 2.0 / 3.0
    assert row["spurious_active_point_count"] == 1
    assert row["missed_active_point_count"] == 0
    assert row["rss_rmse_db"] == np.sqrt(2.5)
    assert row["rss_rmse_all_db"] > 90.0
    assert row["regime_agreement"] == 1.0
