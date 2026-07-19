#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:?repository root is required}"
PID_A="${2:?shard A PID is required}"
PID_B="${3:?shard B PID is required}"
PID_C="${4:?shard C PID is required}"
cd "$ROOT"

export TCSM_MITSUBA_VARIANT="${TCSM_MITSUBA_VARIANT:-llvm_ad_mono_polarized}"
export TCSM_SAMPLES_PER_SOURCE="${TCSM_SAMPLES_PER_SOURCE:-500000}"
VENV="${TCSM_VENV:-/home/zheng/.venvs/tcsm-rt}"
PYTHON="$VENV/bin/python"
CONFIG="configs/full_rt_zhengyi.yaml"

while kill -0 "$PID_A" 2>/dev/null || kill -0 "$PID_B" 2>/dev/null || kill -0 "$PID_C" 2>/dev/null; do
  sleep 120
done

declare -a SHARDS=(
  "outputs/zhengyi_sionna"
  "outputs/zhengyi_sionna_shard_b"
  "outputs/zhengyi_sionna_shard_c"
)
for shard in "${SHARDS[@]}"; do
  count="$(find "$shard/scenes" -name 'sionna_*.npz' | wc -l | tr -d ' ')"
  if [[ "$count" -ne 32 ]]; then
    printf 'expected 32 caches in %s, found %s\n' "$shard" "$count" >&2
    exit 1
  fi
done

"$PYTHON" scripts/audit_sionna_material_metadata.py --repair-missing \
  --run-dir outputs/zhengyi_sionna \
  --run-dir outputs/zhengyi_sionna_shard_b \
  --run-dir outputs/zhengyi_sionna_shard_c \
  | tee logs/material_frequency_audit_shards.json

"$PYTHON" scripts/merge_result_shard.py \
  --source outputs/zhengyi_sionna_shard_b \
  --destination outputs/zhengyi_sionna \
  | tee logs/merge_sionna_shard_b.json
"$PYTHON" scripts/merge_result_shard.py \
  --source outputs/zhengyi_sionna_shard_c \
  --destination outputs/zhengyi_sionna \
  | tee logs/merge_sionna_shard_c.json

scene_count="$(find outputs/zhengyi_sionna/scenes -name 'sionna_*.npz' | wc -l | tr -d ' ')"
if [[ "$scene_count" -ne 96 ]]; then
  printf 'expected 96 merged Sionna caches, found %s\n' "$scene_count" >&2
  exit 1
fi
"$PYTHON" scripts/audit_sionna_material_metadata.py \
  --run-dir outputs/zhengyi_sionna \
  | tee logs/material_frequency_audit_merged.json

"$PYTHON" -m tcsm_rt.cli train-point --config "$CONFIG" | tee logs/train_point_zhengyi.json
"$PYTHON" -m tcsm_rt.cli train-grid --config "$CONFIG" | tee logs/train_grid_zhengyi.json
"$PYTHON" -m tcsm_rt.cli evaluate --config "$CONFIG" | tee logs/evaluate_zhengyi.json
"$PYTHON" -m tcsm_rt.cli threshold-sensitivity --config "$CONFIG" | tee logs/threshold_sensitivity_zhengyi.json
"$PYTHON" -m tcsm_rt.cli robustness --config "$CONFIG" | tee logs/robustness_zhengyi.json
"$PYTHON" -m tcsm_rt.cli profile --config "$CONFIG" | tee logs/deployment_profile_zhengyi.json
"$PYTHON" -m tcsm_rt.cli cases --config "$CONFIG" | tee logs/case_gallery_zhengyi.json
"$PYTHON" -m tcsm_rt.cli audit --run-dir outputs/zhengyi_sionna | tee logs/audit_zhengyi.json
