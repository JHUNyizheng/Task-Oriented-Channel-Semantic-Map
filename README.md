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

Paper-facing diagnostics are explicit stages rather than configuration-only claims:

```bash
tcsm-rt threshold-sensitivity --config configs/full_rt.yaml
tcsm-rt robustness --config configs/full_rt.yaml
tcsm-rt profile --config configs/full_rt.yaml
tcsm-rt cases --config configs/full_rt.yaml
tcsm-rt deepmimo-audit --config configs/full_rt.yaml
```

The threshold stage keeps the declared low/high hysteresis ratio while sweeping the high margin
from 0.1 to 1.0 bit/s/Hz and also reports an MCS-aware margin from 3GPP spectral efficiencies. The
robustness stage corrupts estimator-visible support evidence while retaining independent query
truth. The deployment stage reports end-to-end latency, throughput, checkpoint size, parameter
count, and device-memory measurements.

The ray-budget convergence study is run before full Sionna generation. The selected sample count
must satisfy the declared channel and task-label convergence tolerances.

Before full training, `training_label_coverage.json` must confirm that all 32 declared Sionna
training configurations are present, the aggregate near/cross/far fractions each exceed the
configured minimum, every environment modality is populated, and all task codebooks have usable
label support. The full pipeline stops when this gate fails; it does not rebalance a collapsed
label distribution silently.

The `cases` stage selects auditable ID and OOD examples from evaluation records, preserves the
actual sampled trajectory order, and renders environment inputs, task truth, predictions, policy
loss, and five task-gate maps from saved arrays. For every gate, zero means exclusive use of the
local prior and one means exclusive use of the neural branch.

## Data

The pipeline downloads DeepMIMO scenarios through the official package interfaces. Raw scenario
archives are not redistributed. Sionna scenes and any OpenStreetMap-derived geometry remain under
their original licenses; see [THIRD_PARTY.md](THIRD_PARTY.md).

The main benchmark uses six official Sionna scenes and the complete valid receiver sets from
DeepMIMO `city_0_newyork_28` and `city_17_seattle_28`. Configuration files record spatially
disjoint training, ID, geometry-OOD, system-OOD, and compound-OOD splits.

DeepMIMO evaluation uses contiguous coordinate stripes rather than random receiver splits. The
nearest 60% of each transmitter's valid receivers are support candidates, the next 20% form a
spatial-ID query region, and the furthest 20% form a spatial holdout. The external audit covers
110,280 valid receiver--transmitter samples across the six released transmitter views and reports
discarded no-path receivers, split counts, task availability, and cache hashes. It permits RSS and
far-beam claims only.

The Apple-silicon worker also runs a separate cross-city protocol:

```bash
tcsm-rt train-deepmimo-crosscity --config configs/deepmimo_crosscity_macstudio.yaml
tcsm-rt evaluate-deepmimo-crosscity --config configs/deepmimo_crosscity_macstudio.yaml
```

Only the 60% contiguous New York training stripes contribute gradients. New York spatial-ID and
holdout stripes, together with both Seattle query stripes, remain evaluation-only. The loss reads
the released `task_availability` mask and supervises RSS and far-beam heads only; the corresponding
checkpoints and metrics use the `deepmimo_crosscity` prefix and cannot support near-field claims.

## Baselines and adaptations

Every baseline is linked to an existing publication in
[`configs/baseline_registry.yaml`](configs/baseline_registry.yaml). The registry distinguishes
preserved architectural components from task-head adaptations for KNN, IDW, DeepSets, Set
Transformer, RadioUNet, STORM, FNO, and WNO. The repository does not present adapted baselines as
unchanged reproductions of their source papers.

STORM is identified as an arXiv preprint baseline. RadioUNet, STORM, FNO, and WNO retain the
published structural bias named in the registry while using common T-CSM task heads; their results
therefore test adapted model families under a shared interface rather than claim byte-identical
reproduction of the original application.

## Distributed execution

`configs/compute_allocation.yaml` assigns Sionna RT generation, convergence checks, and seeds
11/23/37 to the CUDA worker. On the current WSL host, official Sionna RT kernels use Mitsuba's LLVM
backend because the NVIDIA OptiX runtime is unavailable; PyTorch training remains on CUDA. The
Apple-silicon worker runs seeds 53/71 and complete DeepMIMO
external validation after importing only the 32 verified Sionna training caches. The staging and
merge scripts verify SHA-256 digests and reject evaluation-split leakage.

`scripts/run_zhengyi_sharded_full.sh` divides the 96 CPU ray-tracing records into three disjoint
half-open intervals. Every worker writes a separate output directory. The coordinator verifies
cache hashes during merge, requires all 96 configurations, and starts training only after the
merged training-label coverage gate passes.

No machine password, access token, raw proprietary data, or private path is stored in this
repository.

When metric columns change, evaluation archives the prior `evaluation_raw.csv` under a numbered
schema-backup name and writes `evaluation_schema_migration.json` before recomputation. This avoids
combining rows produced by incompatible metric definitions. Result shards are accepted only after
cache hashes, seed ownership, absolute-path rewriting, and split leakage checks pass.

## Citation

The manuscript citation will be updated after public release. Software metadata is provided in
[`CITATION.cff`](CITATION.cff).

## License

Code is released under the Apache License 2.0. Third-party data and scene licenses remain in
force.
