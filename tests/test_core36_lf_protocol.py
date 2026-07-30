from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.build_core36_lf_selection import build_core36_lf_selection
from tcsm_rt.config import load_config
from tcsm_rt.data.common import sionna_configuration_manifest

ROOT = Path(__file__).resolve().parents[1]


def _records() -> list[dict]:
    config = load_config(ROOT / "configs/full_rt_core36_lf_zhengyi.yaml")
    return [record.__dict__ for record in sionna_configuration_manifest(config)]


def _protocol() -> dict:
    return yaml.safe_load((ROOT / "configs/core36_lf_protocol.yaml").read_text(encoding="utf-8"))


def test_core36_lf_protocol_is_complete_and_predeclared() -> None:
    selection = build_core36_lf_selection(_records(), _protocol())
    assert selection["primary_record_count"] == 36
    assert selection["progress_denominator"] == 36
    assert selection["primary_split_counts"] == {
        "train": 12,
        "id": 12,
        "geometry_ood": 12,
    }
    assert selection["primary_record_indices"] == [
        *range(12),
        *range(32, 44),
        *range(48, 60),
    ]
    assert selection["high_frequency_diagnostic_record_indices"] == [
        64,
        68,
        72,
        83,
        87,
        91,
    ]
    assert not (
        set(selection["primary_record_indices"])
        & set(selection["high_frequency_diagnostic_record_indices"])
    )
    assert all(selection["validation"].values())


def test_checked_in_core36_selection_matches_protocol() -> None:
    checked_in = json.loads((ROOT / "configs/core36_lf_selection.json").read_text(encoding="utf-8"))
    rebuilt = build_core36_lf_selection(_records(), _protocol())
    for key in (
        "source_manifest_sha256",
        "primary_record_indices",
        "primary_config_ids",
        "high_frequency_diagnostic_record_indices",
        "excluded_record_indices",
        "query_grid",
        "validation",
    ):
        assert checked_in[key] == rebuilt[key]


def test_core36_compute_queues_are_disjoint_and_complete() -> None:
    allocation = yaml.safe_load(
        (ROOT / "configs/compute_allocation_core36_no_mac.yaml").read_text(encoding="utf-8")
    )
    accepted = set(allocation["selection"]["accepted_legacy_candidates"])
    queues = [
        set(allocation["workers"][worker]["primary_record_indices"])
        for worker in ("zhengyi_a", "zhengyi_b", "zhengyi4090")
    ]
    assert all(not (left & right) for i, left in enumerate(queues) for right in queues[i + 1 :])
    assert not any(accepted & queue for queue in queues)
    assert accepted | set().union(*queues) == {
        *range(12),
        *range(32, 44),
        *range(48, 60),
    }
    assert allocation["workers"]["mac_studio"]["required"] is False
    assert allocation["workers"]["mac_studio"]["primary_record_indices"] == []
