from __future__ import annotations

import numpy as np

from tcsm_rt.diagnostics import (
    mcs_adaptive_margins,
    perturb_observations,
    regime_labels_for_margins,
)


def _arrays() -> dict[str, np.ndarray]:
    x, y = np.meshgrid(np.arange(5), np.arange(5), indexing="xy")
    points = np.column_stack([x.ravel(), y.ravel(), np.ones(x.size)]).astype(np.float32)
    count = len(points)
    return {
        "query_xyz_m": points,
        "environment": np.column_stack([x.ravel(), y.ravel()]).astype(np.float32),
        "rss_db": np.linspace(-100.0, -60.0, count, dtype=np.float32),
        "regime": np.ones(count, dtype=np.int8),
        "best_far_idx": np.arange(count, dtype=np.int16) % 5,
        "best_near_angle": np.arange(count, dtype=np.int16) % 5,
        "best_near_range": np.arange(count, dtype=np.int16) % 3,
        "far_rates": np.tile(np.linspace(0.1, 2.0, 5, dtype=np.float32), (count, 1)),
        "far_codebook_loss_bps_hz": np.linspace(0.0, 1.2, count, dtype=np.float32),
        "distance_m": np.linspace(2.0, 50.0, count, dtype=np.float32),
        "rayleigh_distance_m": np.full(count, 20.0, dtype=np.float32),
        "valid_query_mask": np.ones(count, dtype=bool),
    }


def _config() -> dict:
    return {
        "model": {"far_beams": 5, "near_angles": 5, "near_ranges": 3},
        "data": {"cell_size_m": 1.0},
    }


def test_regime_sweep_uses_geometry_and_rate_margin() -> None:
    arrays = _arrays()
    labels = regime_labels_for_margins(arrays, 0.2, 0.75)
    near_geometry = arrays["distance_m"] <= arrays["rayleigh_distance_m"]
    far_geometry = ~near_geometry
    assert np.all(labels[(near_geometry) & (arrays["far_codebook_loss_bps_hz"] >= 0.75)] == 0)
    assert np.all(labels[(far_geometry) & (arrays["far_codebook_loss_bps_hz"] <= 0.2)] == 2)
    assert set(np.unique(labels)) <= {0, 1, 2}


def test_adaptive_mcs_margin_is_bounded_and_has_hysteresis() -> None:
    low, high = mcs_adaptive_margins(_arrays(), [0.25, 0.5, 1.0, 2.0, 3.0], 0.25, 0.1, 1.0)
    assert np.all((high >= 0.1) & (high <= 1.0))
    np.testing.assert_allclose(low, 0.25 * high)


def test_power_and_label_corruption_change_support_only() -> None:
    arrays = _arrays()
    support = np.array([0, 6, 12, 18, 24])
    query = np.setdiff1d(np.arange(25), support)
    power, retained = perturb_observations(arrays, support, "power_noise_db", 3.0, 9, _config())
    np.testing.assert_array_equal(retained, support)
    np.testing.assert_array_equal(power["rss_db"][query], arrays["rss_db"][query])
    assert np.any(power["rss_db"][support] != arrays["rss_db"][support])
    labels, _ = perturb_observations(arrays, support, "label_error_fraction", 1.0, 11, _config())
    np.testing.assert_array_equal(labels["regime"][query], arrays["regime"][query])
    assert np.all(labels["regime"][support] != arrays["regime"][support])


def test_missing_support_never_removes_all_measurements() -> None:
    arrays = _arrays()
    support = np.array([0, 6, 12, 18, 24])
    _, retained = perturb_observations(
        arrays,
        support,
        "support_missing_fraction",
        1.0,
        3,
        _config(),
    )
    assert len(retained) == 1
