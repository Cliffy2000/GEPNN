@echo off

echo T-XOR Experiments
echo Started: %date% %time%
echo.

set T1=0
set T2=-1

for %%h in (2 3) do (
    for %%m in (0.15 0.2 0.25 0.3) do (
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