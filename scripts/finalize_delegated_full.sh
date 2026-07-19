#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:?repository root is required}"
MAC_RUN="${2:?unpacked Mac Studio run directory is required}"
cd "$ROOT"

VENV="${TCSM_VENV:-/home/zheng/.venvs/tcsm-rt}"
PYTHON="$VENV/bin/python"
CONFIG="configs/full_rt_zhengyi.yaml"
DESTINATION="outputs/zhengyi_sionna"

"$PYTHON" scripts/merge_compute_artifacts.py \
  --source "$MAC_RUN" \
  --destination "$DESTINATION" \
  --seeds 53 71 \
  --expected-steps 8000 \
  | tee logs/merge_macstudio_compute_artifacts.json

# Evaluation enumerates all checkpoint files, so this pass replaces the earlier
# three-seed table with the complete five-seed evidence matrix.
"$PYTHON" -m tcsm_rt.cli evaluate --config "$CONFIG" \
  | tee logs/evaluate_five_seed_zhengyi.json
"$PYTHON" -m tcsm_rt.cli audit --run-dir "$DESTINATION" \
  | tee logs/audit_five_seed_zhengyi.json
