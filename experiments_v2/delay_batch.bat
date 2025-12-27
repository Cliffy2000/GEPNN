@echo off

echo T-XOR Experiments
echo Started: %date% %time%
echo.

set delay=1
set gen=2000

for %%h in (2 3 4) do (
    for %%m in (0.15 0.2 0.25 0.3 0.35) do (
        echo ============================================================
        echo Running: HEAD=%%h, MUTATION=%%m, DELAY=%delay%, GENLIMIT=%gen%
        python delay.py --head %%h --mutation %%m --delay %delay% --max_gen %gen% --quiet
        echo.
        echo.
    )
)

set delay=2
set gen=3000

for %%h in (3 4 5) do (
    for %%m in (0.15 0.2 0.25 0.3 0.35) do (
        echo ============================================================
        echo Running: HEAD=%%h, MUTATION=%%m, DELAY=%delay%, GENLIMIT=%gen%
        python delay.py --head %%h --mutation %%m --delay %delay% --max_gen %gen%  --quiet
        echo.
        echo.
    )
)

echo.
echo.
echo ============================================================
echo.
python ..\reports_v2\delay_results.py --delay %delay%
echo.
echo ============================================================
echo.

echo Finished: %date% %time%
pause