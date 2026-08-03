@echo off
REM =============================================================================
REM DROCAT UI Launcher for Windows (self-healing)
REM Creates the conda env and installs dependencies on first run, then starts
REM the web UI inside the environment.
REM =============================================================================
setlocal

set ENV_NAME=drocat
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

REM Guard against an existing env with the wrong Python version
set "ENV_PY="
for /f "delims=" %%v in ('call %CONDA_BIN% run -n %ENV_NAME% python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2^>nul') do set "ENV_PY=%%v"
if defined ENV_PY if not "%ENV_PY%"=="3.11" (
    echo ERROR: existing env uses Python %ENV_PY% (expected 3.11).
    echo Recreate it: conda env remove -n drocat -y, then re-run install.bat
    pause
    exit /b 1
)

REM First run: create the environment and install dependencies
call %CONDA_BIN% run -n %ENV_NAME% python -c "import nicegui" >nul 2>nul
if errorlevel 1 (
    echo Creating environment and installing dependencies (first run)...
    call %CONDA_BIN% create -n %ENV_NAME% python=3.11 -y || goto :err
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
