import numpy as np
import pytest

from tcsm_rt.metrics import clustered_bootstrap_ci, holm_adjust, policy_gap, rmse_db


def test_policy_gap_direction():
    gap = policy_gap(np.array([4.0, 3.0]), np.array([3.5, 3.0]))
    np.testing.assert_allclose(gap, [0.5, 0.0])


def test_policy_gap_rejects_rate_above_oracle():
    with pytest.raises(ValueError):
        policy_gap(np.array([2.0]), np.array([2.1]))


def test_rmse_is_computed_in_db_domain():
    assert rmse_db(np.array([-80.0, -70.0]), np.array([-79.0, -71.0])) == pytest.approx(1.0)


def test_holm_adjustment_is_monotone_in_sorted_order():
    raw = [0.01, 0.04, 0.03]
    adjusted = holm_adjust(raw)
    order = np.argsort(raw)
    assert np.all(np.diff(np.asarray(adjusted)[order]) >= 0)


def test_clustered_bootstrap_ignores_not_applicable_metrics():
    mean, low, high = clustered_bootstrap_ci(
        np.array([np.nan, 1.0, 3.0]),
        np.array(["unused", "a", "b"]),
        200,
        0.95,
        7,
    )
    assert mean == pytest.approx(2.0)
    assert low <= mean <= high
