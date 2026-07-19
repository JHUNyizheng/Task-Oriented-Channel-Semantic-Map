@echo off
setlocal
set ROOT=D:\Projects\Radio2026\tcsm_rt_full
if not exist "%ROOT%\logs" mkdir "%ROOT%\logs"
set SOURCE=%ROOT%\source
wsl.exe -d Ubuntu -- bash -lc "set -e; cd /mnt/d/Projects/Radio2026/tcsm_rt_full; if [ -d source/.git ]; then git -C source pull --ff-only; else git clone https://github.com/JHUNyizheng/Task-Oriented-Channel-Semantic-Map.git source; fi"
start "TCSM RT Bootstrap" /min cmd /c "wsl.exe -d Ubuntu -- bash /mnt/d/Projects/Radio2026/tcsm_rt_full/source/scripts/bootstrap_zhengyi.sh 1^>D:\Projects\Radio2026\tcsm_rt_full\logs\bootstrap.stdout.log 2^>D:\Projects\Radio2026\tcsm_rt_full\logs\bootstrap.stderr.log"
endlocal
