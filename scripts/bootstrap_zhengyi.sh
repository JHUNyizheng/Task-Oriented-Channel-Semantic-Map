#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/d/Projects/Radio2026/tcsm_rt_full
SOURCE="$(cd "$(dirname "$0")/.." && pwd)"
VENV=/home/zheng/.venvs/tcsm-rt
LOGS="$ROOT/logs"

mkdir -p /home/zheng/.venvs "$LOGS"
if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e "$SOURCE[test]"

cd "$SOURCE"
python -m pytest -q
python -m tcsm_rt.cli doctor --config configs/full_rt_zhengyi.yaml
python -m tcsm_rt.cli manifest --config configs/full_rt_zhengyi.yaml

date -u +%Y-%m-%dT%H:%M:%SZ > "$LOGS/bootstrap.complete"
TCSM_VENV="$VENV" nohup "$SOURCE/scripts/run_zhengyi_wsl.sh" "$SOURCE" \
  > "$LOGS/full_run.stdout.log" 2> "$LOGS/full_run.stderr.log" &
echo "$!" > "$LOGS/full_run.pid"
