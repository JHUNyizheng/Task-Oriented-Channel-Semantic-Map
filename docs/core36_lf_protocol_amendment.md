# Core-36-LF v2 Protocol Amendment

Date frozen: 2026-07-30

## Decision

The primary Sionna RT estimand uses 36 low-frequency configurations:

- train: immutable record IDs 0--11;
- spatial ID: immutable record IDs 32--43;
- geometry OOD: immutable record IDs 48--59.

Each split contains the complete Cartesian design of three scene families,
28/39 GHz, and 64/128 transmit elements. ID and geometry OOD therefore retain
12 configuration clusters each. The three geometry-OOD scene families are
disjoint from training.

The active query grid is the centered 25x25 subset at the unchanged 2-m
spacing. Previously completed 35x35 caches may enter only through the exact
center-crop utility, which records the parent SHA-256 and the derived-cache
SHA-256. No interpolation or regenerated value enters a crop.

At the frozen 500k-ray budget and existing batch sizes, Core-66/35x35 requires
6,402 solver calls and has a 274.164-billion ray-element work proxy.
Core-36-LF/25x25 requires 1,800 solver calls and has a 69.300-billion proxy,
or 25.3% of the original explicit-array workload. This proxy is used only for
compute planning; it is not a propagation metric.

## Reason for amendment

The amendment was declared before inspection of official model-comparison
results. Mac Studio became unavailable, and the 60/73-GHz 500k-ray caches did
not satisfy the predeclared convergence gate. Continuing Core-66 unchanged
would couple the primary inference to an unresolved high-frequency ray budget
and an unavailable compute node.

Core-36-LF preserves the paper's primary scientific question: whether a
task-oriented map improves near/cross/far regime, far-beam, near-focus, and
policy-rate decisions under sparse support. Cross-frequency and compound
frequency--geometry transfer are no longer primary claims.

## Separated evidence

Six fixed high-frequency IDs (64, 68, 72, 83, 87, and 91) cover all official
scene families and span the high-frequency system extrema. They remain a
separate physical-boundary diagnostic. They are regenerated only after a
frequency-specific ray-budget gate and cannot be pooled into Core-36-LF.

DeepMIMO New York-to-Seattle evaluation remains external RSS/far-beam evidence.
It does not supply near-field regime or focus labels.

## Compute allocation

- ZHENGYI runs one LLVM RT worker by default. A second worker is enabled only
  when measured aggregate throughput is at least 1.6 times the single-worker
  rate.
- ZHENGYI4090 runs the CUDA batch sweep, a disjoint RT queue, all five Sionna
  training seeds, and the three-seed DeepMIMO cross-city protocol.
- The local Mac performs cache audit, statistics, figures, LaTeX builds, and
  release packaging.
- Mac Studio is not required by any merge or completion gate.

## Admission gates

Every primary cache must pass immutable identity, schema, finite-value,
SHA-256, spatial-isolation, regime-coverage, and independent metric
recomputation checks. The complete 36-configuration table is frozen before
high-frequency diagnostics are interpreted. A failed record is regenerated
under the same ID in a mutually exclusive output directory; no record is
substituted based on model results.
