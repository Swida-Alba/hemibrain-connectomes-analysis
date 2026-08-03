@echo off
REM Self-healing DROCAT UI launcher for Windows.
setlocal EnableDelayedExpansion
set "PYTHONNOUSERSITE=1"
set "SCRIPT_DIR=%~dp0"
set "CONDA_BIN="

where conda >nul 2>nul && set "CONDA_BIN=conda"
if not defined CONDA_BIN if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" set "CONDA_BIN=%USERPROFILE%\miniconda3\Scripts\conda.exe"
if not defined CONDA_BIN if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe" set "CONDA_BIN=%USERPROFILE%\anaconda3\Scripts\conda.exe"
if not defined CONDA_BIN if exist "%USERPROFILE%\miniforge3\Scripts\conda.exe" set "CONDA_BIN=%USERPROFILE%\miniforge3\Scripts\conda.exe"
if not defined CONDA_BIN if exist "%ProgramData%\miniconda3\Scripts\conda.exe" set "CONDA_BIN=%ProgramData%\miniconda3\Scripts\conda.exe"
if not defined CONDA_BIN if exist "%ProgramData%\anaconda3\Scripts\conda.exe" set "CONDA_BIN=%ProgramData%\anaconda3\Scripts\conda.exe"

set "DROCAT_VERSION=4.5.0"
for /f "tokens=3" %%v in ('findstr /C:"APP_VERSION = " "%SCRIPT_DIR%ui\config.py"') do set "DROCAT_VERSION=%%v"
set "DROCAT_VERSION=!DROCAT_VERSION:"=!"
set "ENV_BASE=drocat-!DROCAT_VERSION!"
set "REPAIRED=0"

:resolve_reset
set "ENV_NAME="
set /a N=0
:resolve
if !N! EQU 0 (
    set "CAND=!ENV_BASE!"
) else (
    set /a SUFFIX=N+1
    set "CAND=!ENV_BASE!-!SUFFIX!"
)
if defined CONDA_BIN (
    call "!CONDA_BIN!" run -n "!CAND!" python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>nul
    if not errorlevel 1 (
        set "ENV_NAME=!CAND!"
        goto env_found
    )
)
set /a N+=1
if !N! LEQ 20 goto resolve
goto repair

:env_found
call "!CONDA_BIN!" run -n "!ENV_NAME!" python -c "import nicegui,numpy,pandas,neuprint,neuronbridge_client" >nul 2>nul
if errorlevel 1 goto repair
call "!CONDA_BIN!" run -n "!ENV_NAME!" python -m pip check >nul 2>nul
if errorlevel 1 goto repair
goto launch

:repair
if "!REPAIRED!"=="1" goto error
set "REPAIRED=1"
echo Preparing the pinned DROCAT environment...
where pwsh >nul 2>nul
if not errorlevel 1 (
    pwsh -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%install.ps1"
) else (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%install.ps1"
)
if errorlevel 1 goto error
where conda >nul 2>nul && set "CONDA_BIN=conda"
if not defined CONDA_BIN if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" set "CONDA_BIN=%USERPROFILE%\miniconda3\Scripts\conda.exe"
if not defined CONDA_BIN if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe" set "CONDA_BIN=%USERPROFILE%\anaconda3\Scripts\conda.exe"
if not defined CONDA_BIN if exist "%USERPROFILE%\miniforge3\Scripts\conda.exe" set "CONDA_BIN=%USERPROFILE%\miniforge3\Scripts\conda.exe"
if not defined CONDA_BIN if exist "%ProgramData%\miniconda3\Scripts\conda.exe" set "CONDA_BIN=%ProgramData%\miniconda3\Scripts\conda.exe"
if not defined CONDA_BIN if exist "%ProgramData%\anaconda3\Scripts\conda.exe" set "CONDA_BIN=%ProgramData%\anaconda3\Scripts\conda.exe"
goto resolve_reset

:launch
echo Starting DROCAT v!DROCAT_VERSION! in !ENV_NAME!...
cd /d "%SCRIPT_DIR%"
call "!CONDA_BIN!" run -n "!ENV_NAME!" --no-capture-output python ui\app.py
exit /b %ERRORLEVEL%

:error
echo ERROR: DROCAT could not prepare a usable environment.
pause
exit /b 1
