@echo off
REM Double-click / terminal launcher for Windows.
REM Self-healing: prepares the versioned environment on first run (via
REM archive\install\install.ps1), repairs it when inconsistent, resolves
REM port conflicts interactively, and launches the web UI.
setlocal EnableDelayedExpansion
set "PYTHONNOUSERSITE=1"
set "SCRIPT_DIR=%~dp0"
set "CONDA_BIN="

cls
echo DROCAT - Drosophila Connectome Analysis Toolkit
echo Preparing the versioned environment and launching the UI...
echo.

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

REM Token hint: remind users who skipped the installer prompt.
set "NP_TOKEN="
if exist "%SCRIPT_DIR%token_info_local.txt" for /f "usebackq tokens=*" %%t in (`findstr /B "NEUPRINT_TOKEN=" "%SCRIPT_DIR%token_info_local.txt"`) do set "NP_TOKEN=%%t"
set "HINT=0"
if not defined NP_TOKEN set "HINT=1"
if defined NP_TOKEN echo !NP_TOKEN! | findstr /C:"YOUR_NEUPRINT_TOKEN_HERE" >nul && set "HINT=1"
if defined NP_TOKEN echo !NP_TOKEN! | findstr /C:"NEUPRINT_TOKEN=''" >nul && set "HINT=1"
if "!HINT!"=="1" echo Tip: the NeuPrint token is not configured yet - set it in the Settings tab or token_info_local.txt (the CAVE token is optional; only needed for FlyWire FAFB online fetching).
goto launch

:repair
if "!REPAIRED!"=="1" goto error
set "REPAIRED=1"
echo Preparing the pinned DROCAT environment...
where pwsh >nul 2>nul
if not errorlevel 1 (
    pwsh -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%archive\install\install.ps1"
) else (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%archive\install\install.ps1"
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
REM Port-conflict guard: when the UI port is busy, ask the user whether
REM to start on a new port, kill the existing DROCAT process, or cancel.
set "APP_PORT="
for /f "tokens=3" %%p in ('findstr /C:"APP_PORT =" "%SCRIPT_DIR%ui\config.py"') do set "APP_PORT=%%p"
if not defined APP_PORT set "APP_PORT=8080"
if defined DROCAT_UI_PORT set "APP_PORT=!DROCAT_UI_PORT!"

set "OWNER_PID="
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /R ":!APP_PORT! .*LISTENING"') do if not defined OWNER_PID set "OWNER_PID=%%p"
if not defined OWNER_PID goto start_ui

set "OWNER_CMD="
for /f "delims=" %%c in ('powershell -NoProfile -Command "(Get-CimInstance Win32_Process -Filter 'ProcessId=!OWNER_PID!').CommandLine"') do set "OWNER_CMD=%%c"
echo.
echo Port !APP_PORT! is already in use by PID !OWNER_PID!:
echo     !OWNER_CMD!
echo   [1] Start DROCAT on a new port
echo !OWNER_CMD! | findstr /I "ui\app.py drocat" >nul
if errorlevel 1 (
    echo   [2] Not allowed: the process on port !APP_PORT! is not DROCAT - stop it manually, then retry
) else (
    echo   [2] Kill the existing DROCAT process and restart on port !APP_PORT!
)
echo   [3] Cancel
set "CHOICE="
set /p "CHOICE=Your choice [1-3]: "
if "!CHOICE!"=="1" goto pick_new_port
if "!CHOICE!"=="2" (
    echo !OWNER_CMD! | findstr /I "ui\app.py drocat" >nul
    if errorlevel 1 goto conflict_abort
    echo Stopping PID !OWNER_PID!...
    taskkill /PID !OWNER_PID! /F >nul 2>nul
    goto wait_port_free
)

:conflict_abort
echo ERROR: could not resolve the port conflict on !APP_PORT!.
echo Stop the process manually and retry, or set DROCAT_UI_PORT to a free port.
pause
exit /b 1

:pick_new_port
set /a NEW_PORT=!APP_PORT!+1
:pick_new_port_loop
if !NEW_PORT! GTR 65535 goto conflict_abort
netstat -ano | findstr /R ":!NEW_PORT! .*LISTENING" >nul 2>nul
if not errorlevel 1 (
    set /a NEW_PORT+=1
    goto pick_new_port_loop
)
echo Port !APP_PORT! is busy; starting on port !NEW_PORT! instead.
set "APP_PORT=!NEW_PORT!"
goto start_ui

:wait_port_free
set /a WAIT=0
:wait_port_free_loop
set "OWNER_PID="
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /R ":!APP_PORT! .*LISTENING"') do if not defined OWNER_PID set "OWNER_PID=%%p"
if not defined OWNER_PID goto start_ui
set /a WAIT+=1
if !WAIT! GEQ 20 goto conflict_abort
ping -n 2 127.0.0.1 >nul
goto wait_port_free_loop

:start_ui
echo Starting DROCAT v!DROCAT_VERSION! in !ENV_NAME! at http://127.0.0.1:!APP_PORT!...
cd /d "%SCRIPT_DIR%"
set "DROCAT_UI_PORT=!APP_PORT!"
call "!CONDA_BIN!" run -n "!ENV_NAME!" --no-capture-output python ui\app.py
if errorlevel 1 goto launch_failed
exit /b 0

:launch_failed
echo.
echo DROCAT could not start. Review the messages above.
pause
exit /b 1

:error
echo ERROR: DROCAT could not prepare a usable environment.
pause
exit /b 1
