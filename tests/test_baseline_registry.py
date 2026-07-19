from __future__ import annotations

from pathlib import Path

import yaml

from tcsm_rt.grid_learning import GRID_MODELS


ROOT = Path(__file__).resolve().parents[1]


def test_every_configured_baseline_has_a_published_reference() -> None:
    registry = yaml.safe_load((ROOT / "configs/baseline_registry.yaml").read_text())
    entries = registry["baselines"]
    configured: set[str] = set()
    for path in (ROOT / "configs").glob("*.yaml"):
        config = yaml.safe_load(path.read_text())
        if isinstance(config, dict) and "model" in config:
            configured.update(config["model"].get("baselines", []))
    assert configured <= entries.keys()
    for name in configured - {"gated_hlg"}:
        entry = entries[name]
        assert entry["paper"]
        assert entry["venue"]
        assert int(entry["year"]) <= 2026
        assert str(entry["url"]).startswith("https://")
        assert entry["adaptation"]


def test_registry_covers_all_trainable_comparison_models() -> None:
    registry = yaml.safe_load((ROOT / "configs/baseline_registry.yaml").read_text())
    expected = {"deepsets", "set_transformer", "storm", *GRID_MODELS}
    assert expected <= registry["baselines"].keys()


def test_distributed_training_seeds_are_disjoint_and_complete() -> None:
    allocation = yaml.safe_load((ROOT / "configs/compute_allocation.yaml").read_text())
    zhengyi = set(allocation["workers"]["zhengyi"]["train_seeds"])
    mac_studio = set(allocation["workers"]["mac_studio"]["train_seeds"])
    assert zhengyi.isdisjoint(mac_studio)
    assert sorted(zhengyi | mac_studio) == allocation["merge_policy"]["combined_train_seeds"]
