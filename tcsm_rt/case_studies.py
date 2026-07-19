from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .diagnostics import _predictors
from .learning import resolve_device
from .metrics import executed_rate, policy_gap
from .provenance import write_json_atomic
from .sampling import sample_scene_indices_ordered, valid_query_indices
from .schema import load_scene


def _select_baseline(frame: pd.DataFrame, candidates: list[str]) -> str:
    selected = frame[frame["model"].isin(candidates)]
    grouped = selected.groupby("model", as_index=False)["mean_policy_gap"].mean()
    if grouped.empty:
        raise RuntimeError(f"none of the case baseline candidates are available: {candidates}")
    return str(grouped.sort_values("mean_policy_gap").iloc[0]["model"])


def _case_candidates(frame: pd.DataFrame, baseline: str) -> pd.DataFrame:
    grouped = (
        frame[frame["model"].isin(["gated_hlg", baseline])]
        .groupby(["scene_id", "split", "model"], as_index=False)["mean_policy_gap"]
        .mean()
    )
    pivot = grouped.pivot(index=["scene_id", "split"], columns="model", values="mean_policy_gap")
    pivot = pivot.dropna(subset=["gated_hlg", baseline]).reset_index()
    pivot["delta_baseline_minus_ours"] = pivot[baseline] - pivot["gated_hlg"]
    return pivot


def _nearest_quantile(frame: pd.DataFrame, quantile: float) -> pd.Series:
    if frame.empty:
        raise RuntimeError("no scene is available for the requested case stratum")
    target = frame["delta_baseline_minus_ours"].quantile(quantile)
    index = (frame["delta_baseline_minus_ours"] - target).abs().idxmin()
    return frame.loc[index]


def _select_cases(frame: pd.DataFrame) -> list[dict[str, Any]]:
    definitions = (
        ("id_representative", frame[frame["split"] == "id"], 0.5),
        ("ood_advantage", frame[frame["split"].str.endswith("ood")], 0.8),
        ("ood_boundary", frame[frame["split"].str.endswith("ood")], 0.0),
    )
    result: list[dict[str, Any]] = []
    used: set[str] = set()
    for name, subset, quantile in definitions:
        available = subset[~subset["scene_id"].isin(used)]
        if available.empty:
            continue
        row = _nearest_quantile(available, quantile)
        used.add(str(row["scene_id"]))
        result.append(
            {
                "case_name": name,
                "selection_quantile": quantile,
                "scene_id": str(row["scene_id"]),
                "split": str(row["split"]),
                "delta_baseline_minus_ours": float(row["delta_baseline_minus_ours"]),
            }
        )
    return result


def _prediction_maps(
    arrays: dict[str, np.ndarray],
    support: np.ndarray,
    query: np.ndarray,
    prediction: dict[str, np.ndarray],
    near_angle_count: int,
) -> dict[str, np.ndarray]:
    count = len(arrays["query_xyz_m"])
    regime = np.full(count, -1, dtype=np.int16)
    far = np.full(count, -1, dtype=np.int16)
    angle = np.full(count, -1, dtype=np.int16)
    range_index = np.full(count, -1, dtype=np.int16)
    rss = np.full(count, np.nan, dtype=np.float32)
    gap = np.full(count, np.nan, dtype=np.float32)
    regime[query] = np.argmax(prediction["regime_logits"], axis=1)
    far[query] = np.argmax(prediction["far_logits"], axis=1)
    angle[query] = np.argmax(prediction["near_angle_logits"], axis=1)
    range_index[query] = np.argmax(prediction["near_range_logits"], axis=1)
    rss[query] = prediction["rss_db"]
    selected_rate = executed_rate(
        regime[query],
        far[query],
        angle[query],
        range_index[query],
        arrays["far_rates"][query],
        arrays["near_rates"][query],
        near_angle_count,
    )
    gap[query] = policy_gap(arrays["oracle_rate_bps_hz"][query], selected_rate)
    # Support locations are observations, not predictions. They are filled with truth
    # solely to make a complete map and remain identified by the support mask.
    regime[support] = arrays["regime"][support]
    far[support] = arrays["best_far_idx"][support]
    angle[support] = arrays["best_near_angle"][support]
    range_index[support] = arrays["best_near_range"][support]
    rss[support] = arrays["rss_db"][support]
    gap[support] = 0.0
    return {
        "rss_db": rss,
        "regime": regime,
        "far": far,
        "near_angle": angle,
        "near_range": range_index,
        "focus": range_index.astype(np.int32) * near_angle_count + angle.astype(np.int32),
        "policy_gap": gap,
    }


def _reshape(values: np.ndarray, side: int, valid: np.ndarray) -> np.ma.MaskedArray:
    mask = ~valid.reshape(side, side) | ~np.isfinite(values.reshape(side, side))
    return np.ma.array(values.reshape(side, side), mask=mask)


def _grid_extent(points: np.ndarray) -> tuple[float, float, float, float]:
    x_values = np.unique(points[:, 0])
    y_values = np.unique(points[:, 1])
    x_step = float(np.median(np.diff(x_values))) if len(x_values) > 1 else 1.0
    y_step = float(np.median(np.diff(y_values))) if len(y_values) > 1 else 1.0
    return (
        float(x_values.min() - x_step / 2),
        float(x_values.max() + x_step / 2),
        float(y_values.min() - y_step / 2),
        float(y_values.max() + y_step / 2),
    )


def _overlay_bs(
    axis: plt.Axes,
    transmitter_xyz_m: np.ndarray,
    extent: tuple[float, float, float, float],
) -> None:
    x_min, x_max, y_min, y_max = extent
    tx_x, tx_y = float(transmitter_xyz_m[0]), float(transmitter_xyz_m[1])
    inside = x_min <= tx_x <= x_max and y_min <= tx_y <= y_max
    marker_x = float(np.clip(tx_x, x_min, x_max))
    marker_y = float(np.clip(tx_y, y_min, y_max))
    axis.scatter(
        marker_x,
        marker_y,
        marker="*" if inside else ">",
        s=95,
        c="#C62828",
        edgecolors="white",
        linewidths=0.7,
        zorder=5,
        clip_on=True,
    )
    axis.annotate(
        "BS" if inside else "BS direction",
        (marker_x, marker_y),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=7,
        color="#8E1B1B",
        weight="bold",
    )


def _draw_case(
    output: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    support_route: np.ndarray,
    ours: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    baseline_name: str,
    case: dict[str, Any],
    near_angle_count: int,
) -> None:
    count = len(arrays["query_xyz_m"])
    side = int(round(np.sqrt(count)))
    if side * side != count:
        raise ValueError("case figures require a regular square query grid")
    valid = np.asarray(arrays.get("valid_query_mask", np.ones(count, dtype=bool)), dtype=bool)
    environment = arrays["environment"]
    extent = _grid_extent(arrays["query_xyz_m"])
    transmitter = np.asarray(metadata["transmitter_xyz_m"], dtype=np.float64)
    truth_focus = (
        arrays["best_near_range"].astype(np.int32) * near_angle_count
        + arrays["best_near_angle"].astype(np.int32)
    )
    truth = {
        "regime": arrays["regime"],
        "far": arrays["best_far_idx"],
        "focus": truth_focus,
        "oracle_rate": arrays["oracle_rate_bps_hz"],
    }
    fig, axes = plt.subplots(4, 4, figsize=(13.6, 10.6), constrained_layout=True)
    input_specs = (
        (environment[:, 0], "Building occupancy", "Greys", 0.0, 1.0),
        (environment[:, 5], "LOS state", "cividis", 0.0, 1.0),
        (environment[:, 7], "Excess loss (dB)", "magma", None, None),
    )
    for column, (values, title, cmap, lower, upper) in enumerate(input_specs):
        image = axes[0, column].imshow(
            _reshape(values, side, np.ones(count, dtype=bool)),
            origin="lower",
            cmap=cmap,
            vmin=lower,
            vmax=upper,
            extent=extent,
            aspect="equal",
        )
        fig.colorbar(image, ax=axes[0, column], fraction=0.046, pad=0.02)
        axes[0, column].set_title(title)
        _overlay_bs(axes[0, column], transmitter, extent)
    support_mask = np.zeros(count, dtype=np.float32)
    support_mask[support_route] = 1.0
    axes[0, 3].imshow(
        _reshape(support_mask, side, np.ones(count, dtype=bool)),
        origin="lower",
        cmap="Blues",
        vmin=0,
        vmax=1,
        extent=extent,
        aspect="equal",
    )
    route_xy = arrays["query_xyz_m"][support_route, :2]
    axes[0, 3].plot(route_xy[:, 0], route_xy[:, 1], color="#C23B22", linewidth=1.2)
    axes[0, 3].scatter(
        route_xy[:, 0],
        route_xy[:, 1],
        s=14,
        c="#C23B22",
        edgecolors="white",
        linewidths=0.3,
    )
    axes[0, 3].set_title(f"Observed trajectory ($N={len(support_route)}$)")
    _overlay_bs(axes[0, 3], transmitter, extent)
    row_specs = (
        (truth, "Ground truth"),
        (ours, "Gated-HLG (Ours)"),
        (baseline, baseline_name),
    )
    task_specs = (
        ("regime", "Near/cross/far", "viridis", 0, 2),
        ("far", "Far-beam index", "turbo", 0, arrays["far_rates"].shape[1] - 1),
        (
            "focus",
            "Near-focus index",
            "plasma",
            0,
            arrays["near_rates"].shape[1] - 1,
        ),
        ("oracle_rate", "Oracle rate (bit/s/Hz)", "cividis", 0, None),
    )
    shared_gap_max = float(
        np.nanpercentile(
            np.concatenate([ours["policy_gap"][valid], baseline["policy_gap"][valid]]),
            99,
        )
    )
    for row, (maps, row_label) in enumerate(row_specs, start=1):
        for column, (key, title, cmap, lower, upper) in enumerate(task_specs):
            plot_key = key
            display_title = title
            if row > 1 and key == "oracle_rate":
                plot_key = "policy_gap"
                display_title = "Policy gap (bit/s/Hz)"
            image = axes[row, column].imshow(
                _reshape(np.asarray(maps[plot_key]), side, valid),
                origin="lower",
                cmap=cmap if plot_key != "policy_gap" else "inferno",
                vmin=lower,
                vmax=shared_gap_max if plot_key == "policy_gap" else upper,
                extent=extent,
                aspect="equal",
            )
            fig.colorbar(image, ax=axes[row, column], fraction=0.046, pad=0.02)
            axes[row, column].set_title(display_title)
            if column == 0:
                axes[row, column].set_ylabel(row_label)
            if row > 1:
                route_xy = arrays["query_xyz_m"][support_route, :2]
                axes[row, column].scatter(
                    route_xy[:, 0],
                    route_xy[:, 1],
                    s=4,
                    facecolors="none",
                    edgecolors="white",
                    linewidths=0.25,
                    alpha=0.8,
                )
    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(
        f"{case['case_name']}: {case['split']} | baseline - Ours gap = "
        f"{case['delta_baseline_minus_ours']:.3f} bit/s/Hz",
        fontsize=12,
    )
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_case_studies(config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    settings = config.get("case_gallery", {})
    raw_path = run_dir / "metrics" / "evaluation_raw.csv"
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    support_count = int(settings.get("support_count", 24))
    sampling_mode = str(settings.get("sampling_mode", "trajectory"))
    eval_seed = int(settings.get("eval_seed", config["run"]["eval_seeds"][0]))
    train_seed = int(settings.get("train_seed", config["run"]["train_seeds"][0]))
    candidates = list(settings.get("baseline_candidates", ["storm", "set_transformer", "radiounet"]))
    raw = pd.read_csv(raw_path)
    frame = raw[
        raw["source"].astype(str).str.startswith("sionna")
        & (raw["support_count"] == support_count)
        & (raw["sampling_mode"] == sampling_mode)
        & (raw["eval_seed"] == eval_seed)
    ]
    baseline_name = _select_baseline(frame, candidates)
    selected_cases = _select_cases(_case_candidates(frame, baseline_name))
    index = json.loads((run_dir / "scene_index.json").read_text(encoding="utf-8"))
    rows_by_scene = {Path(row["cache"]).stem: row for row in index}
    device = resolve_device(config["run"]["device"])
    predictors = {
        name: predictor
        for name, seed, predictor, _ in _predictors(
            config,
            run_dir,
            {"gated_hlg", baseline_name},
            {train_seed},
            device,
        )
    }
    if set(predictors) != {"gated_hlg", baseline_name}:
        raise RuntimeError(
            f"case predictors are incomplete: found {sorted(predictors)}, expected Gated-HLG and {baseline_name}"
        )
    output_dir = run_dir / "cases"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_cases: list[dict[str, Any]] = []
    for case in selected_cases:
        scene_row = rows_by_scene[case["scene_id"]]
        arrays = load_scene(scene_row["cache"])
        support_route = sample_scene_indices_ordered(
            arrays,
            support_count,
            sampling_mode,
            eval_seed + int(scene_row.get("seed", 0)),
        )
        support = np.sort(support_route)
        query = np.setdiff1d(valid_query_indices(arrays), support)
        ours_prediction = predictors["gated_hlg"](arrays, support, query)
        baseline_prediction = predictors[baseline_name](arrays, support, query)
        ours_maps = _prediction_maps(
            arrays,
            support,
            query,
            ours_prediction,
            int(config["model"]["near_angles"]),
        )
        baseline_maps = _prediction_maps(
            arrays,
            support,
            query,
            baseline_prediction,
            int(config["model"]["near_angles"]),
        )
        metadata_path = Path(scene_row["cache"]).with_suffix(".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        stem = output_dir / case["case_name"]
        np.savez_compressed(
            stem.with_suffix(".npz"),
            query_xyz_m=arrays["query_xyz_m"],
            environment=arrays["environment"],
            valid_query_mask=arrays.get(
                "valid_query_mask",
                np.ones(len(arrays["query_xyz_m"]), dtype=bool),
            ),
            support_indices=support,
            support_route_indices=support_route,
            truth_regime=arrays["regime"],
            truth_far=arrays["best_far_idx"],
            truth_near_angle=arrays["best_near_angle"],
            truth_near_range=arrays["best_near_range"],
            truth_rss_db=arrays["rss_db"],
            truth_oracle_rate_bps_hz=arrays["oracle_rate_bps_hz"],
            **{f"ours_{key}": value for key, value in ours_maps.items()},
            **{f"baseline_{key}": value for key, value in baseline_maps.items()},
        )
        _draw_case(
            stem,
            arrays,
            metadata,
            support_route,
            ours_maps,
            baseline_maps,
            baseline_name,
            case,
            int(config["model"]["near_angles"]),
        )
        manifest_cases.append(
            {
                **case,
                "baseline": baseline_name,
                "train_seed": train_seed,
                "eval_seed": eval_seed,
                "support_count": len(support),
                "sampling_mode": sampling_mode,
                "sampling_seed": eval_seed + int(scene_row.get("seed", 0)),
                "npz": str(stem.with_suffix(".npz")),
                "pdf": str(stem.with_suffix(".pdf")),
                "svg": str(stem.with_suffix(".svg")),
                "png": str(stem.with_suffix(".png")),
            }
        )
    manifest = {
        "status": "complete",
        "selection_rule": "ID median, OOD 80th-percentile advantage, and OOD minimum advantage",
        "baseline_selection": "lowest global mean policy gap among configured published candidates",
        "baseline": baseline_name,
        "cases": manifest_cases,
    }
    write_json_atomic(output_dir / "case_manifest.json", manifest)
    return manifest
