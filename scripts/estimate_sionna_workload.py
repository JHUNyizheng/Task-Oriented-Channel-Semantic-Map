from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from tcsm_rt.workload import estimate_sionna_workload


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/full_rt_core36_lf_zhengyi.yaml"),
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("configs/core36_lf_selection.json"),
    )
    parser.add_argument("--samples-per-source", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    settings = config["data"]["sionna"]
    selected_records = selection.get("primary_records", selection.get("core_records"))
    if not isinstance(selected_records, list):
        raise TypeError("selection must contain primary_records or core_records")
    report = estimate_sionna_workload(
        selected_records,
        grid_size=int(config["data"]["grid_size"]),
        samples_per_source=int(
            args.samples_per_source or settings["samples_per_source"]
        ),
        path_batch_size=int(settings.get("path_batch_size", 64)),
        explicit_batch_size=int(settings.get("explicit_batch_size", 16)),
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
