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
valid. The active submission protocol is the preregistered `Core-36-LF v2`: 12 train, 12 spatial-ID,
and 12 geometry-OOD configurations. Each split retains the complete 28/39-GHz and
64/128-element design over three scene families. The primary grid is a centered `25x25` grid at
the unchanged 2-m spacing. Six completed `35x35` caches may be reused only through an exact,
hash-tracked center crop. The previous Core-66 protocol is retained as historical evidence and is
not the active completion denominator. Build and verify the active selection before launching a
sparse queue:

```bash
PYTHONPATH=. python scripts/build_core36_lf_selection.py
tcsm-rt prepare-sionna --config configs/full_rt_core36_lf_zhengyi.yaml \
  --record-index-file configs/core36_lf_selection.json
```

`--record-indices 2,3,4` may be used for a worker-specific queue. Explicit indices and the legacy
half-open interval flags are mutually exclusive. Full training starts only after
`training_label_coverage.json` confirms all 12 primary training configurations, non-collapsed
near/cross/far labels, populated environment modalities and usable task-codebook coverage.

The no-Mac-Studio allocation is recorded in
`configs/compute_allocation_core36_no_mac.yaml`. ZHENGYI and ZHENGYI4090 receive disjoint RT
queues. ZHENGYI4090 subsequently runs training seeds 11, 23, 37, 53 and 71; independent seeds do
not require separate physical hosts. The local Mac performs audit, statistics, figures and
manuscript builds. Mac Studio is absent from all completion and merge gates.

Point and grid training persist an atomic recovery state every 400 optimization steps. The state
contains model and optimizer parameters, NumPy/Python/PyTorch random states, the completed step,
loss history, and accumulated training time. A final checkpoint is treated as complete only when
its companion history reaches step 8000. The compute-artifact merger requires all declared model
and seed combinations and verifies every checkpoint/history SHA-256 before evaluation.

`scripts/run_zhengyi_sharded_full.sh` is retained as the legacy 96-record launcher and is not the
submission protocol. Core-36-LF workers use the explicit queues in
`configs/compute_allocation_core36_no_mac.yaml`.
Each worker writes an independent output directory; completed legacy-interval workers stop at a
cache boundary before the explicit queue starts. The merger accepts exactly the 36 primary cache
IDs, verifies every SHA-256 digest and rejects high-frequency diagnostics from the primary run.
The 500,000-ray low-frequency budget, six scene templates, five training/evaluation seeds and
published baselines remain unchanged.

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

On ZHENGYI4090, the declared DeepMIMO cross-city protocol is:

```bash
tcsm-rt train-deepmimo-crosscity --config configs/deepmimo_crosscity_zhengyi4090.yaml
tcsm-rt evaluate-deepmimo-crosscity --config configs/deepmimo_crosscity_zhengyi4090.yaml
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
