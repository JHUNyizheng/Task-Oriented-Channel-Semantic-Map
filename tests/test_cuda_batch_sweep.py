import argparse
from pathlib import Path

from scripts.run_cuda_batch_sweep import _command


def test_cuda_batch_sweep_uses_mutually_exclusive_batch_directory() -> None:
    args = argparse.Namespace(
        config=Path("configs/full_rt_zhengyi.yaml"),
        record_index=3,
        samples_per_source=500_000,
        point_count=12,
    )
    command = _command(args, 2, Path("outputs/cuda-sweep/batch_02"))
    assert command[-4:] == [
        "--batch-size",
        "2",
        "--output",
        "outputs/cuda-sweep/batch_02",
    ]
    assert "500000" in command
