from __future__ import annotations

import json
import tarfile
from pathlib import Path

import numpy as np
import yaml

from scripts.merge_compute_artifacts import merge_compute_artifacts
from scripts.package_sionna_training_shard import package_training_shard
from scripts.stage_training_shard import stage_training_shard
from tcsm_rt.data.common import sionna_configuration_manifest
from tcsm_rt.provenance import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_verified_training_shard_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "full_run"
    scenes = source / "scenes"
    scenes.mkdir(parents=True)
    cache = scenes / "sionna_train_000.npz"
    np.savez_compressed(cache, channel=np.ones((2, 2), dtype=np.complex64))
    metadata = {
        "solver": {"samples_per_source": 500_000, "element_channel": "explicit_array"},
        "material_frequency": {"policy": "clamp_to_itu_range"},
    }
    _write_json(cache.with_suffix(".json"), metadata)
    _write_json(
        source / "scene_index.json",
        [
            {
                "cache": str(cache),
                "cache_sha256": sha256_file(cache),
                "source": "sionna_rt_2.0.1",
                "split": "train",
            }
        ],
    )
    _write_json(source / "audit_report.json", {"passed": True})
    _write_json(
        source / "material_frequency_audit.json",
        {"passed": True, "scene_metadata_count": 96},
    )
    _write_json(
        source / "convergence" / "selected_ray_budget.json",
        {"selected_samples_per_source": 500_000},
    )

    archive = tmp_path / "training.tar.gz"
    package = package_training_shard(source, archive, expected_count=1)
    assert package["archive_file_count"] == 7
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(extracted, filter="data")
    destination = tmp_path / "mac_run"
    staged = stage_training_shard(extracted, destination, expected_count=1)
    assert staged["training_scene_count"] == 1
    assert staged["selected_samples_per_source"] == 500_000
    staged_cache = next((destination / "scenes").glob("*.npz"))
    assert sha256_file(staged_cache) == sha256_file(cache)


def test_delegated_checkpoint_merge_requires_complete_history(tmp_path: Path) -> None:
    source = tmp_path / "mac_run"
    destination = tmp_path / "main_run"
    checkpoints = source / "checkpoints"
    checkpoints.mkdir(parents=True)
    _write_json(source / "audit_report.json", {"passed": True})
    checkpoint = checkpoints / "gated_hlg_seed53.pt"
    checkpoint.write_bytes(b"checkpoint")
    _write_json(checkpoints / "gated_hlg_seed53.history.json", [{"step": 8, "loss": 1.0}])
    _write_json(
        source / "training_summary.json",
        [
            {
                "model": "gated_hlg",
                "seed": 53,
                "checkpoint": str(checkpoint),
            }
        ],
    )
    _write_json(source / "grid_training_summary.json", [])

    report = merge_compute_artifacts(
        source,
        destination,
        models=("gated_hlg",),
        seeds=(53,),
        expected_steps=8,
    )
    assert report["passed"]
    assert report["artifact_count"] == 2
    merged_checkpoint = destination / "checkpoints" / checkpoint.name
    assert merged_checkpoint.read_bytes() == b"checkpoint"
    summary = json.loads((destination / "training_summary.json").read_text())
    assert Path(summary[0]["checkpoint"]) == merged_checkpoint.resolve()


def test_mac_delegated_training_manifest_matches_zhengyi_train_split() -> None:
    zhengyi = yaml.safe_load((ROOT / "configs/full_rt_zhengyi.yaml").read_text())
    mac = yaml.safe_load((ROOT / "configs/full_rt_macstudio.yaml").read_text())
    zhengyi_train = [
        record for record in sionna_configuration_manifest(zhengyi) if record.split == "train"
    ]
    mac_train = [record for record in sionna_configuration_manifest(mac) if record.split == "train"]
    assert mac_train == zhengyi_train
