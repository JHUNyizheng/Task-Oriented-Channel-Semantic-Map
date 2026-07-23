from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.run_core66_rt_worker import WORKERS
from scripts.build_core66_selection import build_core66_selection
from tcsm_rt.config import load_config
from tcsm_rt.data.common import sionna_configuration_manifest
from tcsm_rt.pipeline import _select_sionna_records


ROOT = Path(__file__).resolve().parents[1]


def _records() -> list[dict]:
    config = load_config(ROOT / "configs/full_rt_zhengyi.yaml")
    return [record.__dict__ for record in sionna_configuration_manifest(config)]


def _protocol() -> dict:
    return yaml.safe_load((ROOT / "configs/core66_protocol.yaml").read_text(encoding="utf-8"))


def test_core66_protocol_is_a_complete_stratified_partition() -> None:
    selection = build_core66_selection(_records(), _protocol())
    assert selection["core_record_count"] == 66
    assert selection["reserve_record_count"] == 30
    assert selection["core_split_counts"] == {
        "train": 18,
        "id": 12,
        "geometry_ood": 12,
        "system_ood": 12,
        "compound_ood": 12,
    }
    assert set(selection["core_record_indices"]).isdisjoint(
        selection["reserve_record_indices"]
    )
    assert set(selection["core_record_indices"]) | set(
        selection["reserve_record_indices"]
    ) == set(range(96))
    assert all(selection["validation"].values())


def test_checked_in_selection_matches_protocol() -> None:
    checked_in = json.loads(
        (ROOT / "configs/core66_selection.json").read_text(encoding="utf-8")
    )
    rebuilt = build_core66_selection(_records(), _protocol())
    for key in (
        "source_manifest_sha256",
        "core_record_indices",
        "reserve_record_indices",
        "core_config_ids",
        "reserve_config_ids",
        "validation",
    ):
        assert checked_in[key] == rebuilt[key]


def test_rt_worker_allocations_cover_core_once_or_are_pending() -> None:
    allocation = yaml.safe_load((ROOT / "configs/compute_allocation.yaml").read_text())
    assigned: list[int] = []
    for worker in WORKERS.values():
        worker_ids = allocation["workers"][worker["allocation"]]["core_record_indices"]
        assert len(worker_ids) == len(set(worker_ids))
        assigned.extend(worker_ids)

    checked_in = json.loads(
        (ROOT / "configs/core66_selection.json").read_text(encoding="utf-8")
    )
    assert len(assigned) == len(set(assigned))
    assert set(assigned) == set(checked_in["core_record_indices"])


def test_sparse_record_selection_preserves_declared_order() -> None:
    records = list(range(10))
    assert _select_sionna_records(records, record_indices=[7, 2, 5]) == [7, 2, 5]


@pytest.mark.parametrize("indices", [[], [1, 1], [-1], [10]])
def test_sparse_record_selection_rejects_invalid_indices(indices: list[int]) -> None:
    with pytest.raises(ValueError):
        _select_sionna_records(list(range(10)), record_indices=indices)


def test_sparse_record_selection_rejects_interval_combination() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        _select_sionna_records(list(range(10)), record_start=0, record_indices=[1, 2])
