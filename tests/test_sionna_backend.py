from __future__ import annotations

import os
import subprocess
import sys


def test_explicit_llvm_variant_is_selected_before_sionna_import() -> None:
    environment = dict(os.environ)
    environment["TCSM_MITSUBA_VARIANT"] = "llvm_ad_mono_polarized"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from tcsm_rt.sionna_backend import configure_mitsuba_variant; "
                "assert configure_mitsuba_variant() == 'llvm_ad_mono_polarized'; "
                "import sionna.rt, mitsuba as mi; "
                "assert mi.variant() == 'llvm_ad_mono_polarized'"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0
