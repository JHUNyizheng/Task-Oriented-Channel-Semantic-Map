from __future__ import annotations

import os
import subprocess
import sys

import mitsuba as mi
import numpy as np
import pytest

from tcsm_rt.data.common import RTConfiguration, grid_xyz
from tcsm_rt.data.sionna_adapter import (
    _configure_itu_material_frequency,
    _geometry_features,
    _object_footprints,
    _placement,
)


def test_explicit_platform_variant_is_selected_before_sionna_import() -> None:
    preferred = (
        "llvm_ad_mono_polarized",
        "cuda_ad_mono_polarized",
        "llvm_ad_rgb",
        "scalar_rgb",
    )
    requested = next((name for name in preferred if name in mi.variants()), None)
    assert requested is not None, "Mitsuba exposes no supported test variant"

    environment = dict(os.environ)
    environment["TCSM_MITSUBA_VARIANT"] = requested
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from tcsm_rt.sionna_backend import configure_mitsuba_variant; "
                f"assert configure_mitsuba_variant() == {requested!r}; "
                "import sionna.rt, mitsuba as mi; "
                f"assert mi.variant() == {requested!r}"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0


def test_itu_materials_use_documented_boundary_at_73_ghz() -> None:
    import drjit as dr
    import sionna.rt
    from sionna.rt import load_scene

    scene = load_scene(sionna.rt.scene.simple_street_canyon, merge_shapes=False)
    report = _configure_itu_material_frequency(scene, 73e9, "clamp_to_itu_range")
    records = {record["itu_type"]: record for record in report["materials"]}

    assert records["brick"]["evaluation_frequency_ghz"] == 40.0
    assert records["marble"]["evaluation_frequency_ghz"] == 60.0
    assert records["concrete"]["evaluation_frequency_ghz"] == 73.0
    assert records["brick"]["clamped"] is True
    assert records["marble"]["clamped"] is True
    assert records["concrete"]["clamped"] is False
    assert scene.radio_materials["brick"].frequency_update_callback is None
    assert scene.radio_materials["marble"].frequency_update_callback is None
    assert all(
        material.frequency_update_callback is None
        for material in scene.radio_materials.values()
        if getattr(material, "itu_type", None) is not None
    )
    assert dr.slice(scene.frequency) == pytest.approx(73e9, rel=1e-6)


def test_itu_material_strict_policy_rejects_unsupported_frequency() -> None:
    import sionna.rt
    from sionna.rt import load_scene

    scene = load_scene(sionna.rt.scene.simple_street_canyon, merge_shapes=False)
    with pytest.raises(ValueError, match="marble|brick"):
        _configure_itu_material_frequency(scene, 73e9, "strict")


def test_itu_material_boundary_is_evaluated_without_scene_callback_error() -> None:
    import sionna.rt
    from sionna.rt import load_scene

    scene = load_scene(sionna.rt.scene.simple_street_canyon, merge_shapes=False)
    report = _configure_itu_material_frequency(scene, 60e9, "clamp_to_itu_range")
    records = {record["itu_type"]: record for record in report["materials"]}

    assert records["marble"]["evaluation_frequency_ghz"] == 60.0
    assert records["marble"]["clamped"] is False
    assert records["brick"]["evaluation_frequency_ghz"] == 40.0
    assert records["brick"]["clamped"] is True


def test_san_francisco_terrain_is_not_treated_as_building_occupancy() -> None:
    import sionna.rt
    from sionna.rt import load_scene

    scene = load_scene(sionna.rt.scene.san_francisco, merge_shapes=False)
    footprints = _object_footprints(scene)
    record = RTConfiguration(
        config_id="sionna_compound_ood_002",
        source="sionna_rt_2.0.1",
        scene="san_francisco",
        split="compound_ood",
        frequency_hz=60e9,
        array_size=96,
        placement_index=0,
        seed=1272473381,
    )
    _, center = _placement(scene, record, grid_size=35, cell_size_m=2.0, footprints=footprints)
    query_xyz = grid_xyz(center, grid_size=35, cell_size_m=2.0)
    occupied, _ = _geometry_features(query_xyz, footprints)

    assert len(footprints) > 1000
    assert int(np.sum(occupied < 0.5)) >= 2
