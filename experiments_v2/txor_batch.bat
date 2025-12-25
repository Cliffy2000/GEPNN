@echo off

echo T-XOR Experiments
echo Started: %date% %time%
echo.

set T1=0
set T2=-2

for %%h in (7 8 9) do (
    for %%m in (0.25 0.3 0.35 0.4) do (
        echo ============================================================
        echo Running: HEAD=%%h, MUTATION=%%m, T1=%T1%, T2=%T2%
        python txor.py --head %%h --mutation %%m --t1 %T1% --t2 %T2% --quiet
        echo.
        echo.
    )
)

echo.
echo.
echo ============================================================
echo.
python ..\reports_v2\txor_results.py --t1 %T1% --t2 %T2%
echo.
echo ============================================================
echo.

echo Finished: %date% %time%
pause