from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .provenance import sha256_file, write_json_atomic
from .schema import load_scene


EXPECTED_EXTERNAL_SCOPE = {"rss", "far_beam"}
EXPECTED_AVAILABILITY = np.array([1.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)


def summarize_deepmimo_scene(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    count = len(arrays["query_xyz_m"])
    scope = set(metadata.get("external_task_scope", []))
    if scope != EXPECTED_EXTERNAL_SCOPE:
        errors.append(f"external task scope is {sorted(scope)}, expected {sorted(EXPECTED_EXTERNAL_SCOPE)}")
    if metadata.get("near_field_evidence") != "unsupported_by_standard_synthetic_array_dataset":
        errors.append("near-field evidence boundary is missing or inconsistent")
    availability = np.asarray(arrays.get("task_availability"))
    if availability.shape != (count, len(EXPECTED_AVAILABILITY)):
        errors.append(f"task_availability has shape {availability.shape}")
    elif not np.allclose(availability, EXPECTED_AVAILABILITY[None, :]):
        errors.append("task_availability exposes an unsupported DeepMIMO task")
    spatial_split = np.asarray(arrays.get("spatial_split"))
    if spatial_split.shape != (count,):
        errors.append(f"spatial_split has shape {spatial_split.shape}")
        split_counts = {label: 0 for label in (0, 1, 2)}
    else:
        split_counts = {label: int(np.sum(spatial_split == label)) for label in (0, 1, 2)}
        if any(value == 0 for value in split_counts.values()):
            errors.append(f"one or more contiguous spatial splits are empty: {split_counts}")
    far_consistency = float(np.mean(np.argmax(arrays["far_rates"], axis=1) == arrays["best_far_idx"]))
    if far_consistency < 1.0:
        errors.append(f"far-beam argmax agreement is {far_consistency:.6f}")
    los = np.asarray(arrays["environment"][:, 5], dtype=np.float64)
    rss = np.asarray(arrays["rss_db"], dtype=np.float64)
    labels, label_counts = np.unique(arrays["best_far_idx"], return_counts=True)
    probabilities = label_counts / np.sum(label_counts)
    far_entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    xy = np.asarray(arrays["query_xyz_m"][:, :2], dtype=np.float64)
    return (
        {
            "scenario": metadata.get("scenario"),
            "dataset_index": int(metadata.get("dataset_index", -1)),
            "query_count": count,
            "raw_receiver_count": int(metadata.get("raw_receiver_count", count)),
            "discarded_no_path_count": int(metadata.get("discarded_no_path_count", 0)),
            "frequency_ghz": float(metadata.get("frequency_hz", np.nan)) / 1e9,
            "array_size": int(metadata.get("array_size", arrays["channel"].shape[1])),
            "spatial_train_count": split_counts[0],
            "spatial_id_count": split_counts[1],
            "spatial_holdout_count": split_counts[2],
            "los_fraction": float(np.mean(los)),
            "nlos_fraction": float(1.0 - np.mean(los)),
            "rss_mean_db": float(np.mean(rss)),
            "rss_p05_db": float(np.quantile(rss, 0.05)),
            "rss_p95_db": float(np.quantile(rss, 0.95)),
            "far_beam_classes_present": int(len(labels)),
            "far_beam_entropy_bits": far_entropy,
            "far_label_argmax_agreement": far_consistency,
            "x_span_m": float(np.ptp(xy[:, 0])),
            "y_span_m": float(np.ptp(xy[:, 1])),
        },
        errors,
    )


def audit_deepmimo_external(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    index_path = root / "scene_index.json"
    rows = json.loads(index_path.read_text(encoding="utf-8"))
    selected = [row for row in rows if str(row.get("source", "")).startswith("deepmimo")]
    errors: list[str] = []
    summaries: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for row in selected:
        cache = Path(row["cache"])
        arrays = load_scene(cache)
        summary, scene_errors = summarize_deepmimo_scene(arrays, row)
        summary["scene_id"] = cache.stem
        summary["split"] = row.get("split")
        summary["cache_sha256"] = sha256_file(cache)
        summaries.append(summary)
        hashes[cache.stem] = summary["cache_sha256"]
        errors.extend(f"{cache.stem}: {error}" for error in scene_errors)
    scenarios = {str(row.get("scenario")) for row in selected}
    if len(selected) != 6:
        errors.append(f"found {len(selected)} DeepMIMO BS caches, expected six")
    if scenarios != {"city_0_newyork_28", "city_17_seattle_28"}:
        errors.append(f"DeepMIMO scenario coverage is {sorted(scenarios)}")
    metrics_dir = root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    csv_path = metrics_dir / "deepmimo_external_audit.csv"
    pd.DataFrame(summaries).to_csv(csv_path, index=False)
    report = {
        "passed": not errors,
        "evidence_scope": "public real-world ray-tracing data; RSS and far-beam tasks only",
        "near_field_claim": "not supported by standard DeepMIMO synthetic-array data",
        "scene_count": len(selected),
        "scenario_count": len(scenarios),
        "total_evaluated_receivers": int(sum(row["query_count"] for row in summaries)),
        "cache_sha256": hashes,
        "csv": str(csv_path),
        "errors": errors,
    }
    write_json_atomic(metrics_dir / "deepmimo_external_audit.json", report)
    return report
