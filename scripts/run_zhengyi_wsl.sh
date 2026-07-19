#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/mnt/d/Projects/Radio2026/tcsm_rt_full/sionna_deepmimo_full}"
cd "$ROOT"
mkdir -p logs

VENV="${TCSM_VENV:-/home/zheng/.venvs/tcsm-rt}"
if [[ ! -x "$VENV/bin/python" ]]; then
  mkdir -p "$(dirname "$VENV")"
  python3 -m venv "$VENV"
fi
PYTHON="$VENV/bin/python"
"$PYTHON" -m pip install --upgrade pip wheel
"$PYTHON" -m pip install -e '.[test]'
"$PYTHON" -m pip freeze > logs/environment_zhengyi.lock.txt
nvidia-smi -q > logs/nvidia_smi_zhengyi.txt
"$PYTHON" -m tcsm_rt.cli doctor --config configs/full_rt_zhengyi.yaml \
  | tee logs/doctor_zhengyi.json
"$PYTHON" -m pytest | tee logs/tests_zhengyi.txt

for record in 0 48; do
  convergence="outputs/zhengyi_sionna/convergence/record_${record}"
  if [[ ! -f "$convergence/convergence.csv" ]]; then
    "$PYTHON" scripts/run_sionna_sample_convergence.py \
      --config configs/full_rt_zhengyi.yaml \
      --record-index "$record" \
      --sample-counts 20000 50000 100000 250000 500000 1000000 \
      --point-count 12 \
      --batch-size 4 \
      --output "$convergence" \
      | tee "logs/convergence_record_${record}.json"
  fi
done
"$PYTHON" scripts/select_ray_budget.py \
  outputs/zhengyi_sionna/convergence/record_0/convergence.csv \
  outputs/zhengyi_sionna/convergence/record_48/convergence.csv \
  --output outputs/zhengyi_sionna/convergence/selected_ray_budget.json \
  | tee logs/selected_ray_budget.json
export TCSM_SAMPLES_PER_SOURCE
TCSM_SAMPLES_PER_SOURCE="$($PYTHON -c 'import json; print(json.load(open("outputs/zhengyi_sionna/convergence/selected_ray_budget.json"))["selected_samples_per_source"])')"

"$PYTHON" -m tcsm_rt.cli prepare-sionna --config configs/full_rt_zhengyi.yaml \
  | tee logs/prepare_sionna_zhengyi.json
"$PYTHON" -m tcsm_rt.cli train-point --config configs/full_rt_zhengyi.yaml \
  | tee logs/train_point_zhengyi.json
"$PYTHON" -m tcsm_rt.cli train-grid --config configs/full_rt_zhengyi.yaml \
  | tee logs/train_grid_zhengyi.json
"$PYTHON" -m tcsm_rt.cli evaluate --config configs/full_rt_zhengyi.yaml \
  | tee logs/evaluate_zhengyi.json
"$PYTHON" -m tcsm_rt.cli audit --run-dir outputs/zhengyi_sionna \
  | tee logs/audit_zhengyi.json
