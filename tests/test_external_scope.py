import numpy as np
import torch

from tcsm_rt.evaluation import _baseline_prediction, _prediction_metrics, evaluation_partitions
from tcsm_rt.learning import build_point_batch
from tcsm_rt.physics import make_task_labels


def _external_scene() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(31)
    points = np.column_stack(
        [
            np.arange(9, dtype=np.float32),
            np.zeros(9, dtype=np.float32),
            np.ones(9, dtype=np.float32),
        ]
    )
    channel = (
        rng.normal(size=(9, 16)) + 1j * rng.normal(size=(9, 16))
    ).astype(np.complex64) * 1e-6
    labels = make_task_labels(
        channel,
        28e9,
        17,
        17,
        5,
        distance_m=np.linspace(10.0, 90.0, 9),
    )
    return {
        "query_xyz_m": points,
        "environment": rng.normal(size=(9, 10)).astype(np.float32),
        "channel": channel,
        "task_availability": np.tile(
            np.array([1.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            (9, 1),
        ),
        **{
            key: value
            for key, value in labels.items()
            if key not in {"near_ranges_m", "near_angles_rad"}
        },
        "near_ranges_axis_m": labels["near_ranges_m"],
        "near_angles_axis_rad": labels["near_angles_rad"],
    }


def test_external_support_zeros_unavailable_task_modalities():
    arrays = _external_scene()
    model_config = {"far_beams": 17, "near_angles": 17, "near_ranges": 5, "local_neighbors": 2}
    batch = build_point_batch(
        arrays,
        np.array([0, 2, 4, 6]),
        np.array([1, 3, 5, 7]),
        model_config,
        torch.device("cpu"),
    )
    support = batch.support[0].numpy()
    np.testing.assert_allclose(support[:, 14:17], 0.0)
    np.testing.assert_allclose(support[:, 34:56], 0.0)
    np.testing.assert_allclose(
        support[:, 56:61],
        np.tile([1.0, 0.0, 1.0, 0.0, 0.0], (len(support), 1)),
    )
    np.testing.assert_allclose(batch.local_prior["regime"].numpy(), 0.0)
    np.testing.assert_allclose(batch.local_prior["near_angle"].numpy(), 0.0)


def test_dirichlet_smoothed_prior_retains_unobserved_far_classes() -> None:
    arrays = _external_scene()
    model_config = {
        "far_beams": 17,
        "near_angles": 17,
        "near_ranges": 5,
        "local_neighbors": 2,
        "prior_pseudocount": 0.1,
    }
    batch = build_point_batch(
        arrays,
        np.array([0]),
        np.array([1, 3, 5, 7]),
        model_config,
        torch.device("cpu"),
    )
    probability = torch.softmax(batch.local_prior["far"], dim=-1)
    torch.testing.assert_close(probability.sum(dim=-1), torch.ones((1, 4)))
    assert torch.all(probability > 0)


def test_gated_local_prior_matches_standalone_idw_decision() -> None:
    arrays = _external_scene()
    support = np.array([0, 2, 4, 6])
    query = np.array([1, 3, 5, 7])
    model_config = {
        "far_beams": 17,
        "near_angles": 17,
        "near_ranges": 5,
        "local_neighbors": 3,
    }
    config = {"model": model_config}
    standalone = _baseline_prediction(arrays, support, query, config, "idw")
    batch = build_point_batch(arrays, support, query, model_config, torch.device("cpu"))

    np.testing.assert_array_equal(
        np.argmax(standalone["far_logits"], axis=1),
        np.argmax(batch.local_prior["far"][0].numpy(), axis=1),
    )
    np.testing.assert_allclose(
        standalone["rss_db"],
        batch.local_prior["rss"][0].numpy() * 20.0 - 100.0,
        rtol=1e-5,
        atol=1e-5,
    )


def test_external_metrics_do_not_report_near_field_or_policy_results():
    arrays = _external_scene()
    query = np.arange(9)

    def perfect_logits(values: np.ndarray, count: int) -> np.ndarray:
        logits = np.full((len(values), count), -20.0, dtype=np.float32)
        logits[np.arange(len(values)), values] = 20.0
        return logits

    prediction = {
        "rss_db": arrays["rss_db"].copy(),
        "regime_logits": perfect_logits(arrays["regime"], 3),
        "far_logits": perfect_logits(arrays["best_far_idx"], 17),
        "near_angle_logits": perfect_logits(arrays["best_near_angle"], 17),
        "near_range_logits": perfect_logits(arrays["best_near_range"], 5),
    }
    config = {"model": {"near_angles": 17, "near_ranges": 5}}
    metrics = _prediction_metrics(arrays, query, prediction, config, {"rss", "far_beam"})
    assert metrics["rss_rmse_db"] == 0.0
    assert metrics["mean_far_rate_gap"] == 0.0
    assert metrics["far_top1"] == 1.0
    assert np.isnan(metrics["mean_policy_gap"])
    assert np.isnan(metrics["regime_macro_f1"])
    assert np.isnan(metrics["near_angle_top1"])


def test_deepmimo_evaluation_uses_contiguous_support_and_query_stripes():
    arrays = _external_scene()
    arrays["spatial_split"] = np.repeat([0, 1, 2], 3).astype(np.int8)
    row = {
        "cache": "/tmp/deepmimo_city_tx00.npz",
        "source": "deepmimo_v4",
        "split": "deepmimo_newyork",
    }
    partitions = evaluation_partitions(arrays, row, 2, "trajectory", 17)
    assert [partition[0] for partition in partitions] == [
        "deepmimo_city_tx00__spatial_id",
        "deepmimo_city_tx00__spatial_holdout",
    ]
    for _, _, support, _ in partitions:
        assert set(support) <= {0, 1, 2}
    np.testing.assert_array_equal(partitions[0][3], [3, 4, 5])
    np.testing.assert_array_equal(partitions[1][3], [6, 7, 8])
