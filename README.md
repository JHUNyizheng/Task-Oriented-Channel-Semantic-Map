# Task-Oriented Channel Semantic Map

This repository contains the auditable experiment pipeline accompanying the manuscript
**Task-Oriented Channel Semantic Maps for Near-/Far-Field Beam Management**. It constructs
task-oriented channel semantic maps (T-CSMs) from sparse channel observations and explicit
propagation-environment inputs, then evaluates near/cross/far regime classification, far-field
beam selection, near-field focusing, RSS estimation, and rate decisions.

## Evidence scope

- Sionna RT 2.0.1 and DeepMIMO v4 provide reproducible ray-tracing evidence. They are not
  over-the-air measurements.
- Sionna channels used for the main near-/far-field study are generated with
  `synthetic_array=False`, so each transmit element participates in ray tracing.
- DeepMIMO validation is restricted to quantities supported by the released data: RSS and
  far-field beam decisions. Near-field regime and focus metrics are marked not applicable.
- A center-ray spherical reconstruction is retained only as a diagnostic. Its output cannot enter
  manuscript tables unless it passes the correlation and label-agreement gates.
- A result is manuscript-eligible only when `audit_report.json` records `"passed": true`.

## Reproducible environment

The supported interpreter is Python 3.12. The paper run fixes Sionna RT 2.0.1, DeepMIMO 4.0.3,
Mitsuba 3.8.0, Dr.Jit 1.3.1, PyTorch 2.9.1, and NumPy 2.2.6 in `pyproject.toml`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
python -m pytest
tcsm-rt doctor --config configs/full_rt.yaml
tcsm-rt smoke --config configs/smoke.yaml
```

The full pipeline is resumable:

```bash
tcsm-rt run --config configs/full_rt.yaml --resume
tcsm-rt audit --run-dir outputs/full_rt
```

The ray-budget convergence study is run before full Sionna generation. The selected sample count
must satisfy the declared channel and task-label convergence tolerances.

## Data

The pipeline downloads DeepMIMO scenarios through the official package interfaces. Raw scenario
archives are not redistributed. Sionna scenes and any OpenStreetMap-derived geometry remain under
their original licenses; see [THIRD_PARTY.md](THIRD_PARTY.md).

The main benchmark uses six official Sionna scenes and the complete valid receiver sets from
DeepMIMO `city_0_newyork_28` and `city_17_seattle_28`. Configuration files record spatially
disjoint training, ID, geometry-OOD, system-OOD, and compound-OOD splits.

## Baselines and adaptations

Every baseline is linked to an existing publication in
[`configs/baseline_registry.yaml`](configs/baseline_registry.yaml). The registry distinguishes
preserved architectural components from task-head adaptations for KNN, IDW, DeepSets, Set
Transformer, RadioUNet, STORM, FNO, and WNO. The repository does not present adapted baselines as
unchanged reproductions of their source papers.

## Distributed execution

`configs/compute_allocation.yaml` assigns Sionna RT generation, convergence checks, and seeds
11/23/37 to the CUDA worker. The Apple-silicon worker runs seeds 53/71 and complete DeepMIMO
external validation after importing only the 32 verified Sionna training caches. The staging and
merge scripts verify SHA-256 digests and reject evaluation-split leakage.

No machine password, access token, raw proprietary data, or private path is stored in this
repository.

## Citation

The manuscript citation will be updated after public release. Software metadata is provided in
[`CITATION.cff`](CITATION.cff).

## License

Code is released under the Apache License 2.0. Third-party data and scene licenses remain in
force.
