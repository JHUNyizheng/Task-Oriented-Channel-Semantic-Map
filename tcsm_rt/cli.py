from __future__ import annotations

import argparse
import importlib
import json
import sys
from importlib import metadata
from pathlib import Path

from .audit import audit_run
from .case_studies import generate_case_studies
from .config import load_config
from .evaluation import evaluate_models
from .diagnostics import profile_models, run_robustness, run_threshold_sensitivity
from .pipeline import (
    _output_root,
    prepare_deepmimo,
    prepare_sionna,
    run_full,
    run_smoke,
    train_grid_models,
    train_point_models,
    write_manifests,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tcsm-rt")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in (
        "doctor",
        "manifest",
        "smoke",
        "prepare-sionna",
        "prepare-deepmimo",
        "train-point",
        "train-grid",
        "evaluate",
        "threshold-sensitivity",
        "robustness",
        "profile",
        "cases",
        "run",
        "audit",
    ):
        command = subparsers.add_parser(name)
        if name == "audit":
            command.add_argument("--run-dir", required=True)
        else:
            command.add_argument("--config", required=True)
        if name in {"prepare-sionna", "prepare-deepmimo"}:
            command.add_argument("--limit", type=int)
        if name == "run":
            command.add_argument("--resume", action="store_true")
    return parser


def doctor(config_path: str) -> dict:
    config = load_config(config_path)
    packages = ("torch", "sionna.rt", "deepmimo", "mitsuba", "drjit")
    imports: dict[str, dict[str, str | bool | None]] = {}
    for name in packages:
        try:
            module = importlib.import_module(name)
            package_name = "sionna-rt" if name == "sionna.rt" else name.split(".")[0]
            try:
                version = metadata.version(package_name)
            except metadata.PackageNotFoundError:
                version = getattr(module, "__version__", None)
            imports[name] = {"available": True, "version": version}
        except Exception as error:  # package import failures are part of the doctor output
            imports[name] = {"available": False, "error": repr(error)}
    torch_module = importlib.import_module("torch") if imports["torch"]["available"] else None
    mitsuba_module = importlib.import_module("mitsuba") if imports["mitsuba"]["available"] else None
    result = {
        "python": sys.version,
        "config": str(Path(config_path).resolve()),
        "imports": imports,
        "cuda_available": bool(torch_module and torch_module.cuda.is_available()),
        "cuda_device": torch_module.cuda.get_device_name(0) if torch_module and torch_module.cuda.is_available() else None,
        "mitsuba_variant": mitsuba_module.variant() if mitsuba_module else None,
        "valid": all(value["available"] for value in imports.values()),
        "run_name": config["run"]["name"],
    }
    return result


def main() -> None:
    args = _parser().parse_args()
    if args.command == "audit":
        result = audit_run(args.run_dir)
    else:
        config = load_config(args.config)
        if args.command == "doctor":
            result = doctor(args.config)
        elif args.command == "manifest":
            result = {"output_dir": str(write_manifests(config))}
        elif args.command == "smoke":
            result = run_smoke(config)
        elif args.command == "prepare-sionna":
            result = prepare_sionna(config, limit=args.limit)
        elif args.command == "prepare-deepmimo":
            result = prepare_deepmimo(config, limit_scenarios=args.limit)
        elif args.command == "train-point":
            result = train_point_models(config)
        elif args.command == "train-grid":
            result = train_grid_models(config)
        elif args.command == "evaluate":
            result = evaluate_models(config, _output_root(config))
        elif args.command == "threshold-sensitivity":
            result = run_threshold_sensitivity(config, _output_root(config))
        elif args.command == "robustness":
            result = run_robustness(config, _output_root(config))
        elif args.command == "profile":
            result = profile_models(config, _output_root(config))
        elif args.command == "cases":
            result = generate_case_studies(config, _output_root(config))
        elif args.command == "run":
            if args.resume:
                config["run"]["resume"] = True
            result = run_full(config)
        else:
            raise AssertionError(args.command)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
