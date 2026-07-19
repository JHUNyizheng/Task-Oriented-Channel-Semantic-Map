from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, f1_score


def executed_rate(
    regime_prediction: np.ndarray,
    far_prediction: np.ndarray,
    near_angle_prediction: np.ndarray,
    near_range_prediction: np.ndarray,
    far_rates: np.ndarray,
    near_rates: np.ndarray,
    near_angle_count: int,
) -> np.ndarray:
    regime = np.asarray(regime_prediction)
    far_index = np.asarray(far_prediction, dtype=np.int64)
    near_index = (
        np.asarray(near_range_prediction, dtype=np.int64) * near_angle_count
        + np.asarray(near_angle_prediction, dtype=np.int64)
    )
    far_selected = np.take_along_axis(far_rates, far_index[:, None], axis=1)[:, 0]
    near_selected = np.take_along_axis(near_rates, near_index[:, None], axis=1)[:, 0]
    return np.where(regime == 2, far_selected, near_selected)


def policy_gap(oracle_rate: np.ndarray, selected_rate: np.ndarray) -> np.ndarray:
    gap = np.asarray(oracle_rate, dtype=np.float64) - np.asarray(selected_rate, dtype=np.float64)
    if np.min(gap) < -1e-5:
        raise ValueError(f"policy gap has a negative value: {np.min(gap)}")
    return np.maximum(gap, 0.0)


def rmse_db(truth_db: np.ndarray, prediction_db: np.ndarray) -> float:
    error = np.asarray(prediction_db, dtype=np.float64) - np.asarray(truth_db, dtype=np.float64)
    return float(np.sqrt(np.mean(error**2)))


def classification_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
    }


def clustered_bootstrap_ci(
    values: np.ndarray,
    cluster_ids: np.ndarray,
    samples: int,
    confidence: float,
    seed: int,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    clusters = np.asarray(cluster_ids)
    finite = np.isfinite(values)
    values = values[finite]
    clusters = clusters[finite]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    unique = np.unique(clusters)
    cluster_values = np.array([np.mean(values[clusters == cluster]) for cluster in unique])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(unique), size=(samples, len(unique)))
    bootstrap = np.mean(cluster_values[indices], axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.mean(cluster_values)),
        float(np.quantile(bootstrap, alpha)),
        float(np.quantile(bootstrap, 1.0 - alpha)),
    )


def paired_wilcoxon(reference: np.ndarray, candidate: np.ndarray) -> float:
    delta = np.asarray(reference, dtype=np.float64) - np.asarray(candidate, dtype=np.float64)
    if np.allclose(delta, 0.0):
        return 1.0
    return float(wilcoxon(delta, alternative="two-sided", zero_method="wilcox").pvalue)


def holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()
