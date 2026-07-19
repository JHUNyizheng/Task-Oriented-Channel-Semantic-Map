#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${1:-}" ]]; then
  ROOT="$1"
elif [[ -n "${TCSM_ROOT:-}" ]]; then
  ROOT="$TCSM_ROOT"
elif [[ -d /mnt/d/Projects/Radio2026/tcsm_rt_full/sourcev11 ]]; then
  ROOT=/mnt/d/Projects/Radio2026/tcsm_rt_full/sourcev11
else
  ROOT=/mnt/d/Projects/Radio2026/tcsm_rt_full/sionna_deepmimo_full
fi
if [[ ! -f "$ROOT/pyproject.toml" ]]; then
  printf 'unable to locate T-CSM repository: %s\n' "$ROOT" >&2
  exit 1
fi
cd "$ROOT"
mkdir -p logs

# WSL exposes CUDA to PyTorch but this host does not provide libnvoptix.so.1.
# Sionna RT therefore uses its official LLVM variant; neural training remains on CUDA.
export TCSM_MITSUBA_VARIANT="${TCSM_MITSUBA_VARIANT:-llvm_ad_mono_polarized}"

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
"$PYTHON" scripts/recompute_sionna_convergence.py \
  outputs/zhengyi_sionna/convergence/record_0 \
  outputs/zhengyi_sionna/convergence/record_48 \
  | tee logs/recompute_convergence.json
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
"$PYTHON" -m tcsm_rt.cli threshold-sensitivity --config configs/full_rt_zhengyi.yaml \
  | tee logs/threshold_sensitivity_zhengyi.json
"$PYTHON" -m tcsm_rt.cli robustness --config configs/full_rt_zhengyi.yaml \
  | tee logs/robustness_zhengyi.json
"$PYTHON" -m tcsm_rt.cli profile --config configs/full_rt_zhengyi.yaml \
  | tee logs/deployment_profile_zhengyi.json
"$PYTHON" -m tcsm_rt.cli cases --config configs/full_rt_zhengyi.yaml \
  | tee logs/case_gallery_zhengyi.json
"$PYTHON" -m tcsm_rt.cli audit --run-dir outputs/zhengyi_sionna \
  | tee logs/audit_zhengyi.json
