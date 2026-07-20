# T-CSM Full Ray-Tracing Benchmark

This directory is the auditable experiment source for the T-CSM manuscript. It uses the
official Sionna RT and DeepMIMO packages. The earlier lightweight circular-blocker simulator
is not used as Sionna or DeepMIMO evidence.

## Evidence boundary

- Sionna RT and DeepMIMO provide ray-tracing evidence, not over-the-air measurements.
- Full-grid Sionna channels are generated with `synthetic_array=False`; every transmitting
  element therefore participates in ray tracing. Center-ray spherical reconstruction is retained
  only as an audited diagnostic and cannot enter the manuscript after failing its correlation gate.
- DeepMIMO uses published path matrices and interaction locations. Its external validation is
  code-enforced to RSS and far-field beam decisions; regime and near-field focus metrics are
  reported as not applicable.
- Path extraction, channel reconstruction, label construction and metric computation use
  separate files and separate manifests.

## Commands

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[test]'
tcsm-rt doctor --config configs/full_rt.yaml
tcsm-rt smoke --config configs/smoke.yaml
tcsm-rt run --config configs/full_rt.yaml --resume
tcsm-rt audit --run-dir outputs/full_rt
tcsm-rt deepmimo-audit --config configs/full_rt.yaml
tcsm-rt cases --config configs/full_rt.yaml
```

The immutable generator still declares 96 configurations so completed cache IDs and hashes remain
valid. The submission protocol uses a preregistered 66-configuration core: 18 train and 12 each for
ID, geometry OOD, system OOD and compound OOD. The other 30 records are a conditional reserve and
cannot enter the primary analysis unless a gate in `configs/core66_protocol.yaml` is triggered.
Build and verify the machine-readable selection before launching a sparse queue:

```bash
python scripts/build_core66_selection.py
tcsm-rt prepare-sionna --config configs/full_rt_zhengyi.yaml \
  --record-index-file configs/core66_selection.json
```

`--record-indices 2,3,4` may be used for a worker-specific queue. Explicit indices and the legacy
half-open interval flags are mutually exclusive. Full training starts only after
`training_label_coverage.json` confirms all 18 core training configurations, non-collapsed
near/cross/far labels, populated environment modalities and usable task-codebook coverage.

The distributed allocation is recorded in `configs/compute_allocation.yaml`. `ZHENGYI` runs
Sionna explicit-array generation and training seeds 11, 23 and 37. Mac Studio handles both full
DeepMIMO cities and delegated training seeds 53 and 71 after importing the 18 core Sionna training
caches through `scripts/stage_training_shard.py`. It accepts the packaged `.tar.gz` directly;
the importer safely extracts the archive, rejects links/path traversal, and verifies every SHA-256
digest and excludes all Sionna evaluation splits from the Mac worker. The two workers therefore
produce disjoint training seeds. `scripts/merge_result_shard.py` verifies declared hashes and
rewrites remote absolute paths before a shard enters the combined evidence directory.

Point and grid training persist an atomic recovery state every 400 optimization steps. The state
contains model and optimizer parameters, NumPy/Python/PyTorch random states, the completed step,
loss history, and accumulated training time. A final checkpoint is treated as complete only when
its companion history reaches step 8000. After the Mac worker finishes, the compute-artifact
merger requires all 12 model configurations for seeds 53 and 71 and verifies the SHA-256 digest of
48 checkpoint/history files before the five-seed evaluation can start on ZHENGYI.
The Mac worker skips the four Sionna-backend tests only when its host lacks a usable LLVM/CUDA RT
backend; ZHENGYI and GitHub CI continue to run the complete test set.

`scripts/run_zhengyi_sharded_full.sh` is retained as the legacy 96-record launcher and is not the
submission protocol. Core-66 workers use the explicit queues in `configs/compute_allocation.yaml`.
Each worker writes an independent output directory; completed legacy-interval workers stop at a
cache boundary before the explicit queue starts. The merger accepts exactly the 66 selected cache
IDs, verifies every SHA-256 digest and rejects any undeclared reserve cache from the primary run.
The full $35\times35$ query grid, 500,000-ray budget, six scene templates, all frequency-array
cells, five training/evaluation seeds and published baselines remain unchanged.

If a shard is restarted after a worker-specific failure, its process, log and existing cache hashes
are recorded before a non-overlapping explicit queue is resumed. The metadata auditor may repair
missing records only when every material was inside its documented validity range during
generation. A cache that required a boundary-held material must have applied the policy before ray
tracing and cannot be repaired retrospectively.

DeepMIMO evaluation uses contiguous coordinate stripes. The nearest 60% of valid receivers are
support candidates, the next 20% form a spatial-ID region, and the furthest 20% form a spatial
holdout. The six transmitter views contain 110,280 valid receiver--transmitter samples in total;
the external audit records the split counts, discarded no-path receivers, available tasks and
cache hashes. It authorizes RSS and far-beam evidence only.

On Mac Studio, the full worker command is:

```bash
TCSM_SIONNA_TRAIN_SHARD=/path/to/zhengyi_sionna_train_shard \
  bash scripts/run_mac_studio.sh
```

Before the Sionna training shard arrives, the Mac worker independently runs the declared
DeepMIMO cross-city protocol:

```bash
tcsm-rt train-deepmimo-crosscity --config configs/deepmimo_crosscity_macstudio.yaml
tcsm-rt evaluate-deepmimo-crosscity --config configs/deepmimo_crosscity_macstudio.yaml
```

This protocol trains only on the 60% contiguous New York spatial-training stripes and evaluates
the two disjoint New York query stripes plus both Seattle query stripes. The loss reads
`task_availability`, so standard DeepMIMO caches supervise RSS and far-beam outputs only. Its
checkpoints and metric tables use a separate `deepmimo_crosscity` prefix and cannot be merged with
the Sionna near/far task evidence.

No remote-machine password, token or private path is read from project configuration.

The full run writes a completion matrix, source hashes, raw per-scene metrics and manuscript
tables. A run is not manuscript-ready until `audit_report.json` has `"passed": true`.
If the metric schema changes, the evaluator archives the preceding raw table and writes a migration
manifest before recomputation. The `cases` stage renders real environment arrays, ordered support
trajectories, task truth, predictions, policy loss, and five task-gate maps; gate value zero denotes
the local prior and one denotes the neural branch.
