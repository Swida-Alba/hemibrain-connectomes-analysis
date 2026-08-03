@echo off
REM =============================================================================
REM DROCAT UI Launcher for Windows (self-healing)
REM Creates the conda env and installs dependencies on first run, then starts
REM the web UI inside the environment.
REM =============================================================================
setlocal

set SCRIPT_DIR=%~dp0
set CONDA_BIN=

echo Starting DROCAT UI...

REM Locate conda
where conda >nul 2>nul && set CONDA_BIN=conda
if not defined CONDA_BIN if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" set CONDA_BIN=%USERPROFILE%\miniconda3\Scripts\conda.exe
if not defined CONDA_BIN if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe" set CONDA_BIN=%USERPROFILE%\anaconda3\Scripts\conda.exe
if not defined CONDA_BIN (
    echo ERROR: conda not found. Run install.bat first.
    pause
    exit /b 1
)

REM Resolve environment: use 'drocat' if it is free/usable; otherwise leave
REM any existing env untouched and pick the next free name (drocat-2, ...).
set "ENV_NAME="
set "N=0"
:env_resolve
if "%N%"=="0" (set "CAND=drocat") else (set "CAND=drocat-%N%")
call %CONDA_BIN% run -n %CAND% python -c "import sys, nicegui; assert sys.version_info[:2]==(3,11)" >nul 2>nul
if not errorlevel 1 (
    set "ENV_NAME=%CAND%"
    goto :env_found
)
call %CONDA_BIN% env list | findstr /R /C:"^%CAND% " >nul 2>nul
if errorlevel 1 (
    set "ENV_NAME=%CAND%"
    goto :env_create
)
if "%N%"=="0" echo WARNING: existing 'drocat' env is not usable - using a new env.
set /a N+=1
if %N% GTR 20 (
    echo ERROR: could not resolve a usable drocat environment.
    pause
    exit /b 1
)
goto :env_resolve

:env_create
echo Creating environment %ENV_NAME% (first run)...
call %CONDA_BIN% create -n %ENV_NAME% python=3.11 -y || goto :err

:env_found
if not "%ENV_NAME%"=="drocat" echo Using environment: %ENV_NAME%

REM First run: create the environment and install dependencies
call %CONDA_BIN% run -n %ENV_NAME% python -c "import nicegui" >nul 2>nul
if errorlevel 1 (
    echo Installing dependencies (first run)...
    call %CONDA_BIN% run -n %ENV_NAME% --no-capture-output python -m pip install -r "%SCRIPT_DIR%requirements-windows.txt" || goto :err
    call %CONDA_BIN% run -n %ENV_NAME% --no-capture-output python -m pip install neuronbridge-python --no-deps
    call %CONDA_BIN% run -n %ENV_NAME% --no-capture-output python -m pip install -r "%SCRIPT_DIR%ui\requirements.txt" || goto :err
    call %CONDA_BIN% run -n %ENV_NAME% --no-capture-output python -m pip install -e "%SCRIPT_DIR%" --no-deps || goto :err
)

cd /d "%SCRIPT_DIR%"
call %CONDA_BIN% run -n %ENV_NAME% --no-capture-output python ui\app.py
exit /b 0

:err
echo Installation failed. See messages above.
pause
exit /b 1
