@echo off
setlocal enabledelayedexpansion

set SCRIPT_PATH=trial_xor_multiple.py
set LOG_FILE=xor_multiple_results.txt

REM Clear log file and write header
echo XOR Experiment Results > %LOG_FILE%
echo ====================== >> %LOG_FILE%
echo Started: %date% %time% >> %LOG_FILE%
echo. >> %LOG_FILE%
echo HEAD_LENGTH, MUTATION_RATE, PERFECT_COUNT >> %LOG_FILE%

for %%h in (3 4 5 6) do (
    for %%m in (0.15 0.2 0.25 0.3 0.35) do (
        echo ========================================
        echo Running: HEAD_LENGTH=%%h, MUTATION_RATE=%%m
        echo ========================================

        REM Modify HEAD_LENGTH
        powershell -Command "(Get-Content '%SCRIPT_PATH%') -replace '^HEAD_LENGTH = \d+', 'HEAD_LENGTH = %%h' | Set-Content '%SCRIPT_PATH%'"

        REM Modify MUTATION_RATE
        powershell -Command "(Get-Content '%SCRIPT_PATH%') -replace '^MUTATION_RATE = [\d.]+', 'MUTATION_RATE = %%m' | Set-Content '%SCRIPT_PATH%'"

        REM Run the experiment and capture output
        for /f "tokens=*" %%o in ('python %SCRIPT_PATH% ^| findstr /C:"perfect solutions"') do (
            set OUTPUT=%%o
        )

        REM Extract perfect count from output like "Completed: 95/100 perfect solutions"
        for /f "tokens=2 delims=:/" %%p in ("!OUTPUT!") do (
            set PERFECT=%%p
            set PERFECT=!PERFECT: =!
        )

        REM Log result
        echo %%h, %%m, !PERFECT! >> %LOG_FILE%
        echo Result: HEAD_LENGTH=%%h, MUTATION_RATE=%%m, PERFECT=!PERFECT!
        echo.
    )
)

echo. >> %LOG_FILE%
echo Completed: %date% %time% >> %LOG_FILE%

echo ========================================
echo All experiments completed!
echo Results saved to %LOG_FILE%
echo ========================================
type %LOG_FILE%
pause