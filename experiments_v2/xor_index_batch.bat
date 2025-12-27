@echo off

echo XOR Experiments
echo Started: %date% %time%
echo.

for %%h in (3 4 5 6) do (
    for %%m in (0.2 0.25 0.3 0.35 0.4) do (
        echo ============================================================
        echo.
        echo Running: HEAD=%%h, MUTATION=%%m
        python xor_index.py --head %%h --mutation %%m --quiet
        echo.
        echo.
    )
)

echo.
echo.
echo ============================================================
echo.
python ..\reports_v2\xor_index_results.py
echo.
echo ============================================================
echo.

echo Finished: %date% %time%
pause