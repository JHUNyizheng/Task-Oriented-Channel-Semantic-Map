@echo off
setlocal
set ROOT=D:\Projects\Radio2026\tcsm_rt_full
if not exist "%ROOT%\logs" mkdir "%ROOT%\logs"
set SOURCE=source
if exist "%ROOT%\sourcev11\pyproject.toml" set SOURCE=sourcev11
wsl.exe -d Ubuntu -- bash -lc "set -e; cd /mnt/d/Projects/Radio2026/tcsm_rt_full; if [ -d %SOURCE%/.git ]; then git -C %SOURCE% pull --ff-only; elif [ ! -f %SOURCE%/pyproject.toml ]; then git clone https://github.com/JHUNyizheng/Task-Oriented-Channel-Semantic-Map.git %SOURCE%; fi"
start "TCSM RT Bootstrap" /min cmd /c "wsl.exe -d Ubuntu -- bash /mnt/d/Projects/Radio2026/tcsm_rt_full/%SOURCE%/scripts/bootstrap_zhengyi.sh 1^>D:\Projects\Radio2026\tcsm_rt_full\logs\bootstrap.stdout.log 2^>D:\Projects\Radio2026\tcsm_rt_full\logs\bootstrap.stderr.log"
endlocal
