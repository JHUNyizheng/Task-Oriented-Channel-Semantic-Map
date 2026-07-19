@echo off
setlocal
set ROOT=D:\Projects\Radio2026\tcsm_rt_full
if not exist "%ROOT%\logs" mkdir "%ROOT%\logs"
set SOURCE=%ROOT%\source_v10
if not exist "%SOURCE%" mkdir "%SOURCE%"
powershell -NoProfile -Command "Expand-Archive -LiteralPath '%ROOT%\tcsm_rt_full_source_v10.zip' -DestinationPath '%SOURCE%' -Force"
start "TCSM RT Bootstrap" /min cmd /c "wsl.exe -d Ubuntu -- bash /mnt/d/Projects/Radio2026/tcsm_rt_full/source_v10/experiments/sionna_deepmimo_full/scripts/bootstrap_zhengyi.sh 1^>D:\Projects\Radio2026\tcsm_rt_full\logs\bootstrap.stdout.log 2^>D:\Projects\Radio2026\tcsm_rt_full\logs\bootstrap.stderr.log"
endlocal
