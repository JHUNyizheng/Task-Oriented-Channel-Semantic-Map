#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tcsm_rt.metrics import holm_adjust, paired_wilcoxon
from tcsm_rt.provenance import sha256_file, write_json_atomic


KEYS = (
    "source",
    "split",
    "scene_id",
    "train_seed",
    "eval_seed",
    "support_count",
    "sampling_mode",
)
METRICS = ("mean_far_rate_gap", "far_top1", "far_top3", "rss_rmse_db")


def _prepare(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    selected = frame.loc[frame["model"] == "gated_hlg", [*KEYS, *METRICS]].copy()
    selected["version"] = label
    return selected


def compare_versions(old_path: Path, new_path: Path, output_dir: Path) -> dict[str, object]:
    old_frame = pd.read_csv(old_path)
    new_frame = pd.read_csv(new_path)
    old_ours = _prepare(old_frame, "original_gate")
    new_ours = _prepare(new_frame, "evidence_conditioned_gate")
    merged = old_ours.merge(new_ours, on=list(KEYS), suffixes=("_old", "_new"), validate="one_to_one")
    expected = len(old_ours)
    if len(merged) != expected or len(new_ours) != expected:
        raise RuntimeError(
            f"gate-version rows do not align: old={len(old_ours)}, new={len(new_ours)}, matched={len(merged)}"
        )

    summary_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    p_values: list[float] = []
    for split, group in merged.groupby("split"):
        row: dict[str, object] = {"split": split, "n_rows": len(group)}
        for metric in METRICS:
            old_values = group[f"{metric}_old"].to_numpy(dtype=np.float64)
            new_values = group[f"{metric}_new"].to_numpy(dtype=np.float64)
            row[f"{metric}_old"] = float(np.mean(old_values))
            row[f"{metric}_new"] = float(np.mean(new_values))
            row[f"{metric}_new_minus_old"] = float(np.mean(new_values - old_values))
        summary_rows.append(row)

        scene_old = group.groupby("scene_id")["mean_far_rate_gap_old"].mean()
        scene_new = group.groupby("scene_id")["mean_far_rate_gap_new"].mean()
        common = scene_old.index.intersection(scene_new.index)
        p_value = paired_wilcoxon(
            scene_old.loc[common].to_numpy(),
            scene_new.loc[common].to_numpy(),
        )
        paired_rows.append(
            {
                "split": split,
                "n_scenes": len(common),
                "old_mean_far_rate_gap": float(scene_old.loc[common].mean()),
                "new_mean_far_rate_gap": float(scene_new.loc[common].mean()),
                "old_minus_new": float(
                    np.mean(scene_old.loc[common].to_numpy() - scene_new.loc[common].to_numpy())
                ),
                "p_value": p_value,
            }
        )
        p_values.append(p_value)
    for row, adjusted in zip(paired_rows, holm_adjust(p_values), strict=True):
        row["holm_p_value"] = adjusted

    support_rows: list[dict[str, object]] = []
    for support_count, group in merged.groupby("support_count"):
        support_rows.append(
            {
                "support_count": int(support_count),
                "old_mean_far_rate_gap": float(group["mean_far_rate_gap_old"].mean()),
                "new_mean_far_rate_gap": float(group["mean_far_rate_gap_new"].mean()),
                "old_minus_new": float(
                    np.mean(group["mean_far_rate_gap_old"] - group["mean_far_rate_gap_new"])
                ),
                "old_far_top1": float(group["far_top1_old"].mean()),
                "new_far_top1": float(group["far_top1_new"].mean()),
            }
        )

    new_gate = new_frame[new_frame["model"] == "gated_hlg"].copy()
    new_gate["far_fusion_gain_vs_best_component"] = (
        new_gate[["neural_mean_far_rate_gap", "local_prior_mean_far_rate_gap"]].min(axis=1)
        - new_gate["mean_far_rate_gap"]
    )
    gate_rows = (
        new_gate.groupby(["split", "support_count"])[
            [
                "gate_far_mean",
                "gate_far_p10",
                "gate_far_p90",
                "neural_mean_far_rate_gap",
                "local_prior_mean_far_rate_gap",
                "mean_far_rate_gap",
                "far_fusion_gain_vs_best_component",
            ]
        ]
        .mean()
        .reset_index()
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "deepmimo_gate_revision_summary.csv"
    paired_path = output_dir / "deepmimo_gate_revision_paired.csv"
    support_path = output_dir / "deepmimo_gate_revision_support.csv"
    gate_path = output_dir / "deepmimo_gate_revision_diagnostics.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(paired_rows).to_csv(paired_path, index=False)
    pd.DataFrame(support_rows).to_csv(support_path, index=False)
    gate_rows.to_csv(gate_path, index=False)
    report = {
        "old_raw": str(old_path.resolve()),
        "new_raw": str(new_path.resolve()),
        "old_sha256": sha256_file(old_path),
        "new_sha256": sha256_file(new_path),
        "matched_rows": len(merged),
        "summary": str(summary_path.resolve()),
        "paired": str(paired_path.resolve()),
        "support": str(support_path.resolve()),
        "gate_diagnostics": str(gate_path.resolve()),
        "passed": True,
    }
    write_json_atomic(output_dir / "deepmimo_gate_revision_manifest.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-raw", type=Path, required=True)
    parser.add_argument("--new-raw", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare_versions(args.old_raw, args.new_raw, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
