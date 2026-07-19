from __future__ import annotations

import os
import subprocess
import sys

import mitsuba as mi


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
