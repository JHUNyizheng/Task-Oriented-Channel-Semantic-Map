#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${1:-}" ]]; then
  ROOT="$1"
elif [[ -n "${TCSM_ROOT:-}" ]]; then
  ROOT="$TCSM_ROOT"
elif [[ -d "$HOME/Projects/Radio2026/Task-Oriented-Channel-Semantic-Map" ]]; then
  ROOT="$HOME/Projects/Radio2026/Task-Oriented-Channel-Semantic-Map"
else
  ROOT="$HOME/Projects/Radio2026/sionna_deepmimo_full"
fi
SIONNA_TRAIN_SHARD="${2:-${TCSM_SIONNA_TRAIN_SHARD:-}}"
if [[ ! -f "$ROOT/pyproject.toml" ]]; then
  printf 'unable to locate T-CSM repository: %s\n' "$ROOT" >&2
  exit 1
fi
cd "$ROOT"
mkdir -p logs

if [[ ! -x .venv/bin/python ]]; then
  if command -v python3.12 >/dev/null 2>&1; then
    python3.12 -m venv .venv
  else
    UV_BOOTSTRAP="${TCSM_UV_BOOTSTRAP:-$HOME/Projects/Radio2026/runtime/uv-bootstrap}"
    mkdir -p "$(dirname "$UV_BOOTSTRAP")"
    if [[ ! -x "$UV_BOOTSTRAP/bin/uv" ]]; then
      python3 -m venv "$UV_BOOTSTRAP"
      "$UV_BOOTSTRAP/bin/python" -m pip install --upgrade pip uv
    fi
    "$UV_BOOTSTRAP/bin/uv" python install 3.12.13
    "$UV_BOOTSTRAP/bin/uv" venv --python 3.12.13 .venv
  fi
fi
.venv/bin/python -m pip install --upgrade pip wheel
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -VV > logs/python_runtime_macstudio.txt
.venv/bin/python -m pip freeze > logs/environment_macstudio.lock.txt
system_profiler SPHardwareDataType > logs/hardware_macstudio.txt
.venv/bin/python -m tcsm_rt.cli doctor --config configs/full_rt_macstudio.yaml \
  | tee logs/doctor_macstudio.json
if .venv/bin/python -c 'import sionna.rt' >/dev/null 2>&1; then
  .venv/bin/python -m pytest | tee logs/tests_macstudio.txt
else
  printf '%s\n' 'Sionna RT backend unavailable; RT-only tests remain assigned to ZHENGYI/CI.' \
    | tee logs/sionna_rt_tests_skipped_macstudio.txt
  .venv/bin/python -m pytest --ignore=tests/test_sionna_backend.py \
    | tee logs/tests_macstudio.txt
fi
.venv/bin/python -m tcsm_rt.cli prepare-deepmimo --config configs/full_rt_macstudio.yaml \
  | tee logs/prepare_deepmimo_macstudio.json
.venv/bin/python -m tcsm_rt.cli deepmimo-audit --config configs/full_rt_macstudio.yaml \
  | tee logs/deepmimo_external_audit_macstudio.json
.venv/bin/python -m tcsm_rt.cli train-deepmimo-crosscity \
  --config configs/deepmimo_crosscity_macstudio.yaml \
  | tee logs/train_deepmimo_crosscity_macstudio.json
.venv/bin/python -m tcsm_rt.cli evaluate-deepmimo-crosscity \
  --config configs/deepmimo_crosscity_macstudio.yaml \
  | tee logs/evaluate_deepmimo_crosscity_macstudio.json

if [[ -z "$SIONNA_TRAIN_SHARD" ]]; then
  printf '%s\n' \
    'DeepMIMO preparation is complete. Set TCSM_SIONNA_TRAIN_SHARD to a transferred' \
    'ZHENGYI run shard before starting delegated seeds 53 and 71.' \
    | tee logs/waiting_for_sionna_train_shard.txt
  exit 0
fi

.venv/bin/python scripts/stage_training_shard.py \
  --source "$SIONNA_TRAIN_SHARD" \
  --destination outputs/macstudio_deepmimo \
  --expected-count 32 \
  | tee logs/stage_training_shard_macstudio.json
.venv/bin/python -m tcsm_rt.cli train-point --config configs/full_rt_macstudio.yaml \
  | tee logs/train_point_macstudio.json
.venv/bin/python -m tcsm_rt.cli train-grid --config configs/full_rt_macstudio.yaml \
  | tee logs/train_grid_macstudio.json
.venv/bin/python -m tcsm_rt.cli evaluate --config configs/full_rt_macstudio.yaml \
  | tee logs/evaluate_deepmimo_macstudio.json
.venv/bin/python -m tcsm_rt.cli audit --run-dir outputs/macstudio_deepmimo \
  | tee logs/audit_macstudio.json
.venv/bin/python scripts/package_result_shard.py \
  --run-dir outputs/macstudio_deepmimo \
  --output outputs/macstudio_deepmimo_result_shard.tar.gz \
  | tee logs/package_macstudio.json
