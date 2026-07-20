from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Sequence
from typing import Any

from .audit import audit_run, audit_training_label_coverage
from .case_studies import generate_case_studies
from .data.common import sionna_configuration_manifest
from .data.deepmimo_adapter import generate_deepmimo_scenario
from .data.sionna_adapter import generate_sionna_scene
from .evaluation import evaluate_models
from .diagnostics import profile_models, run_robustness, run_threshold_sensitivity
from .external_audit import audit_deepmimo_external
from .grid_learning import GRID_MODELS, train_grid_model
from .learning import train_point_model
from .provenance import environment_manifest, sha256_file, write_json_atomic


def _output_root(config: dict[str, Any]) -> Path:
    config_root = Path(config["_config_path"]).parent.parent
    output = Path(config["run"]["output_dir"])
    return output if output.is_absolute() else config_root / output


def _load_index(root: Path) -> list[dict[str, Any]]:
    path = root / "scene_index.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def _write_index(root: Path, rows: list[dict[str, Any]]) -> None:
    unique = {str(row["cache"]): row for row in rows}
    write_json_atomic(root / "scene_index.json", list(unique.values()))


def _training_artifact_complete(checkpoint: Path, expected_steps: int) -> bool:
    history_path = checkpoint.with_suffix(".history.json")
    if not checkpoint.exists() or not history_path.exists():
        return False
    try:
        history = json.loads(history_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return bool(history) and int(history[-1].get("step", -1)) == expected_steps


def write_manifests(config: dict[str, Any]) -> Path:
    root = _output_root(config)
    root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(root / "environment_manifest.json", environment_manifest(config))
    records = [record.__dict__ for record in sionna_configuration_manifest(config)]
    write_json_atomic(root / "sionna_configuration_manifest.json", records)
    selection_name = config["data"].get("sionna", {}).get("selection_manifest")
    if selection_name:
        selection_path = Path(config["_config_path"]).parent / str(selection_name)
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        write_json_atomic(root / "core66_selection.json", selection)
    return root


def _select_sionna_records(
    records: Sequence[Any],
    record_start: int | None = None,
    record_stop: int | None = None,
    record_indices: Sequence[int] | None = None,
) -> list[Any]:
    if record_indices is not None:
        if record_start is not None or record_stop is not None:
            raise ValueError("record indices cannot be combined with a record interval")
        indices = [int(index) for index in record_indices]
        if not indices:
            raise ValueError("record indices must not be empty")
        if len(indices) != len(set(indices)):
            raise ValueError("record indices must be unique")
        invalid = [index for index in indices if index < 0 or index >= len(records)]
        if invalid:
            raise ValueError(f"Sionna record indices out of range: {invalid}")
        return [records[index] for index in indices]

    start = 0 if record_start is None else int(record_start)
    stop = len(records) if record_stop is None else int(record_stop)
    if start < 0 or stop > len(records) or start >= stop:
        raise ValueError(
            f"invalid Sionna record interval [{start}, {stop}) for {len(records)} records"
        )
    return list(records[start:stop])


def prepare_sionna(
    config: dict[str, Any],
    limit: int | None = None,
    record_start: int | None = None,
    record_stop: int | None = None,
    record_indices: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    root = write_manifests(config)
    cache_dir = root / "scenes"
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_index(root)
    existing = {str(row["cache"]): row for row in rows}
    records = _select_sionna_records(
        sionna_configuration_manifest(config),
        record_start=record_start,
        record_stop=record_stop,
        record_indices=record_indices,
    )
    if limit is not None:
        records = records[:limit]
    completion: list[dict[str, Any]] = []
    for record in records:
        cache = (cache_dir / f"{record.config_id}.npz").resolve()
        if cache.exists() and config["run"].get("resume", True):
            metadata_path = cache.with_suffix(".json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        else:
            metadata = generate_sionna_scene(record, config, cache)
        metadata.update(
            {
                "cache": str(cache),
                "cache_sha256": sha256_file(cache),
                "source": record.source,
                "split": record.split,
            }
        )
        existing[str(cache)] = metadata
        completion.append({"stage": "sionna", "item": record.config_id, "status": "complete"})
        _write_index(root, list(existing.values()))
        write_json_atomic(root / "completion_matrix.json", completion)
    return list(existing.values())


def prepare_deepmimo(config: dict[str, Any], limit_scenarios: int | None = None) -> list[dict[str, Any]]:
    root = write_manifests(config)
    settings = config["data"]["deepmimo"]
    scenarios = list(settings["scenarios"])
    if limit_scenarios is not None:
        scenarios = scenarios[:limit_scenarios]
    rows = _load_index(root)
    raw_dir = root / "deepmimo_scenarios"
    cache_dir = root / "scenes"
    for scenario in scenarios:
        scenario_rows = generate_deepmimo_scenario(scenario, config, raw_dir, cache_dir)
        for row in scenario_rows:
            row["split"] = "deepmimo_newyork" if "newyork" in scenario.lower() else "deepmimo_seattle_ood"
            row["cache"] = str(Path(row["cache"]).resolve())
        rows.extend(scenario_rows)
        _write_index(root, rows)
    return _load_index(root)


def train_point_models(config: dict[str, Any], smoke: bool = False) -> list[dict[str, Any]]:
    root = _output_root(config)
    coverage = audit_training_label_coverage(root, config)
    if not coverage["passed"] and not smoke:
        raise RuntimeError(f"training label coverage failed: {coverage['errors']}")
    rows = _load_index(root)
    train_paths = [Path(row["cache"]) for row in rows if row.get("split") == "train"]
    if not train_paths:
        raise RuntimeError("no training scene caches are available")
    requested = set(config["model"]["baselines"])
    models = [name for name in ("deepsets", "set_transformer", "storm", "gated_hlg") if name in requested]
    if not smoke:
        models.extend(
            f"gated_hlg_{name}"
            for name in config["model"].get("ablations", [])
        )
    seeds = list(config["run"]["train_seeds"])
    if smoke:
        seeds = seeds[:1]
    summaries: list[dict[str, Any]] = []
    expected_steps = int(config["run"]["train_steps"])
    for model_name in models:
        for seed in seeds:
            checkpoint = root / "checkpoints" / f"{model_name}_seed{seed}.pt"
            if (
                config["run"].get("resume", True)
                and _training_artifact_complete(checkpoint, expected_steps)
            ):
                summaries.append({"model": model_name, "seed": seed, "checkpoint": str(checkpoint), "resumed": True})
                continue
            summaries.append(train_point_model(model_name, train_paths, config, int(seed), checkpoint))
            write_json_atomic(root / "training_summary.json", summaries)
    write_json_atomic(root / "training_summary.json", summaries)
    return summaries


def train_deepmimo_crosscity_models(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Train RSS/far-beam point operators on a spatially isolated DeepMIMO city split."""
    root = _output_root(config)
    external_report = audit_deepmimo_external(root)
    if not external_report["passed"]:
        raise RuntimeError(f"DeepMIMO external audit failed: {external_report['errors']}")
    settings = config.get("external_training", {})
    source_split = str(settings.get("source_split", "deepmimo_newyork"))
    rows = _load_index(root)
    train_paths = [Path(row["cache"]) for row in rows if row.get("split") == source_split]
    if not train_paths:
        raise RuntimeError(f"no DeepMIMO caches are available for source split {source_split!r}")
    requested = set(config["model"]["baselines"])
    models = [
        name
        for name in ("deepsets", "set_transformer", "storm", "gated_hlg")
        if name in requested
    ]
    checkpoint_subdir = str(settings.get("checkpoint_subdir", "deepmimo_crosscity_checkpoints"))
    summaries: list[dict[str, Any]] = []
    expected_steps = int(config["run"]["train_steps"])
    for model_name in models:
        for seed in config["run"]["train_seeds"]:
            checkpoint = root / checkpoint_subdir / f"{model_name}_seed{seed}.pt"
            if (
                config["run"].get("resume", True)
                and _training_artifact_complete(checkpoint, expected_steps)
            ):
                summaries.append(
                    {
                        "model": model_name,
                        "seed": int(seed),
                        "checkpoint": str(checkpoint),
                        "resumed": True,
                    }
                )
                continue
            summary = train_point_model(model_name, train_paths, config, int(seed), checkpoint)
            summary.update(
                {
                    "evidence_scope": ["rss", "far_beam"],
                    "source_split": source_split,
                    "training_spatial_split": int(settings.get("spatial_split", 0)),
                }
            )
            summaries.append(summary)
            write_json_atomic(root / "deepmimo_crosscity_training_summary.json", summaries)
    write_json_atomic(root / "deepmimo_crosscity_training_summary.json", summaries)
    return summaries


def train_grid_models(config: dict[str, Any], smoke: bool = False) -> list[dict[str, Any]]:
    root = _output_root(config)
    coverage = audit_training_label_coverage(root, config)
    if not coverage["passed"] and not smoke:
        raise RuntimeError(f"training label coverage failed: {coverage['errors']}")
    rows = _load_index(root)
    train_paths = [Path(row["cache"]) for row in rows if row.get("split") == "train"]
    if not train_paths:
        raise RuntimeError("no training scene caches are available")
    requested = set(config["model"]["baselines"])
    models = [name for name in GRID_MODELS if name in requested]
    seeds = list(config["run"]["train_seeds"])
    if smoke:
        seeds = seeds[:1]
    summaries: list[dict[str, Any]] = []
    expected_steps = int(config["run"]["train_steps"])
    for model_name in models:
        for seed in seeds:
            checkpoint = root / "checkpoints" / f"{model_name}_seed{seed}.pt"
            if (
                config["run"].get("resume", True)
                and _training_artifact_complete(checkpoint, expected_steps)
            ):
                summaries.append({"model": model_name, "seed": seed, "checkpoint": str(checkpoint), "resumed": True})
                continue
            summaries.append(train_grid_model(model_name, train_paths, config, int(seed), checkpoint))
            write_json_atomic(root / "grid_training_summary.json", summaries)
    write_json_atomic(root / "grid_training_summary.json", summaries)
    return summaries


def run_smoke(config: dict[str, Any]) -> dict[str, Any]:
    prepare_sionna(config, limit=2)
    training = train_point_models(config, smoke=True)
    evaluation = evaluate_models(config, _output_root(config))
    report = audit_run(_output_root(config))
    return {"training": training, "evaluation": evaluation, "audit": report}


def run_full(config: dict[str, Any]) -> dict[str, Any]:
    prepare_sionna(config)
    if config["data"]["deepmimo"].get("enabled", False):
        prepare_deepmimo(config)
    point_training = train_point_models(config)
    grid_training = train_grid_models(config)
    evaluation = evaluate_models(config, _output_root(config))
    threshold_sensitivity = run_threshold_sensitivity(config, _output_root(config))
    robustness = run_robustness(config, _output_root(config))
    deployment = profile_models(config, _output_root(config)) if config.get("deployment") else {
        "status": "skipped",
        "reason": "deployment section is absent",
    }
    cases = generate_case_studies(config, _output_root(config)) if config.get("case_gallery") else {
        "status": "skipped",
        "reason": "case_gallery section is absent",
    }
    report = audit_run(_output_root(config))
    return {
        "point_training": point_training,
        "grid_training": grid_training,
        "evaluation": evaluation,
        "threshold_sensitivity": threshold_sensitivity,
        "robustness": robustness,
        "deployment": deployment,
        "cases": cases,
        "audit": report,
    }
