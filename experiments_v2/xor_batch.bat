@echo off

echo XOR Experiments
echo Started: %date% %time%
echo.

for %%h in (3 4 5 6 7) do (
    for %%m in (0.4) do (
        echo ============================================================
        echo.
        echo Running: HEAD=%%h, MUTATION=%%m
        python xor.py --head %%h --mutation %%m --quiet
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