@echo off

echo XOR Experiments
echo Started: %date% %time%
echo.

for %%h in (2) do (
    for %%m in (0.4) do (
        echo ============================================================
        echo.
        echo Running: HEAD=%%h, MUTATION=%%m
        python xor_sync.py --head %%h --mutation %%m --quiet
        echo.
        echo.
    )
)

echo.
echo.
echo ============================================================
echo.
python ..\reports_v2\xor_results.py
echo.
echo ============================================================
echo.

echo Finished: %date% %time%
pause