#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/mnt/d/Projects/Radio2026/tcsm_rt_full/sionna_deepmimo_full}"
cd "$ROOT"
mkdir -p logs
export TCSM_MITSUBA_VARIANT="${TCSM_MITSUBA_VARIANT:-llvm_ad_mono_polarized}"
VENV="${TCSM_VENV:-/home/zheng/.venvs/tcsm-rt}"
PYTHON="$VENV/bin/python"
CONFIG="configs/full_rt_zhengyi.yaml"
SELECTED="outputs/zhengyi_sionna/convergence/selected_ray_budget.json"

if [[ ! -x "$PYTHON" ]]; then
  printf 'missing Python environment: %s\n' "$PYTHON" >&2
  exit 1
fi
if [[ ! -f "$SELECTED" ]]; then
  printf 'missing selected ray budget: %s\n' "$SELECTED" >&2
  exit 1
fi

RAYS="$($PYTHON -c 'import json; print(json.load(open("outputs/zhengyi_sionna/convergence/selected_ray_budget.json"))["selected_samples_per_source"])')"

run_shard() {
  local output_dir="$1"
  local start="$2"
  local stop="$3"
  local label="$4"
  TCSM_OUTPUT_DIR="$output_dir" TCSM_SAMPLES_PER_SOURCE="$RAYS" \
    "$PYTHON" -m tcsm_rt.cli prepare-sionna --config "$CONFIG" \
      --record-start "$start" --record-stop "$stop" \
      > "logs/prepare_sionna_${label}.stdout.log" \
      2> "logs/prepare_sionna_${label}.stderr.log"
}

run_shard outputs/zhengyi_sionna 0 32 shard_a &
pid_a=$!
run_shard outputs/zhengyi_sionna_shard_b 32 64 shard_b &
pid_b=$!
run_shard outputs/zhengyi_sionna_shard_c 64 96 shard_c &
pid_c=$!
printf '%s\n' "$pid_a" "$pid_b" "$pid_c" > logs/sionna_shard_pids.txt

status=0
wait "$pid_a" || status=1
wait "$pid_b" || status=1
wait "$pid_c" || status=1
if [[ "$status" -ne 0 ]]; then
  printf 'one or more Sionna shards failed; inspect logs/prepare_sionna_shard_*.stderr.log\n' >&2
  exit 1
fi

"$PYTHON" scripts/merge_result_shard.py \
  --source outputs/zhengyi_sionna_shard_b \
  --destination outputs/zhengyi_sionna \
  | tee logs/merge_sionna_shard_b.json
"$PYTHON" scripts/merge_result_shard.py \
  --source outputs/zhengyi_sionna_shard_c \
  --destination outputs/zhengyi_sionna \
  | tee logs/merge_sionna_shard_c.json

scene_count="$(find outputs/zhengyi_sionna/scenes -name '*.npz' | wc -l | tr -d ' ')"
if [[ "$scene_count" -ne 96 ]]; then
  printf 'expected 96 merged Sionna caches, found %s\n' "$scene_count" >&2
  exit 1
fi

"$PYTHON" -m tcsm_rt.cli train-point --config "$CONFIG" \
  | tee logs/train_point_zhengyi.json
"$PYTHON" -m tcsm_rt.cli train-grid --config "$CONFIG" \
  | tee logs/train_grid_zhengyi.json
"$PYTHON" -m tcsm_rt.cli evaluate --config "$CONFIG" \
  | tee logs/evaluate_zhengyi.json
"$PYTHON" -m tcsm_rt.cli threshold-sensitivity --config "$CONFIG" \
  | tee logs/threshold_sensitivity_zhengyi.json
"$PYTHON" -m tcsm_rt.cli robustness --config "$CONFIG" \
  | tee logs/robustness_zhengyi.json
"$PYTHON" -m tcsm_rt.cli profile --config "$CONFIG" \
  | tee logs/deployment_profile_zhengyi.json
"$PYTHON" -m tcsm_rt.cli cases --config "$CONFIG" \
  | tee logs/case_gallery_zhengyi.json
"$PYTHON" -m tcsm_rt.cli audit --run-dir outputs/zhengyi_sionna \
  | tee logs/audit_zhengyi.json

