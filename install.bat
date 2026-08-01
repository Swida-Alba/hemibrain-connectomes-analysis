@echo off
REM =============================================================================
REM DROCAT One-Click Installer for Windows
REM =============================================================================
REM This script installs DROCAT with all dependencies using Miniconda.
REM Usage: Double-click or run: install.bat
REM =============================================================================

setlocal enabledelayedexpansion

echo.
echo ===============================================================
echo      DROCAT - Drosophila Connectome Analysis Toolkit
echo                     One-Click Installer (Windows)
echo ===============================================================
echo.

REM Configuration
set ENV_NAME=drocat
set PYTHON_VERSION=3.11
set SCRIPT_DIR=%~dp0

REM =============================================================================
REM Step 1: Check/Install Miniconda
REM =============================================================================
echo [1/5] Checking for Conda...

where conda >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [OK] Conda found.
    goto :conda_ready
)

REM Check common install locations
if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" (
    set PATH=%USERPROFILE%\miniconda3;%USERPROFILE%\miniconda3\Scripts;%PATH%
    echo [OK] Conda found at %USERPROFILE%\miniconda3
    goto :conda_ready
)

if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe" (
    set PATH=%USERPROFILE%\anaconda3;%USERPROFILE%\anaconda3\Scripts;%PATH%
    echo [OK] Conda found at %USERPROFILE%\anaconda3
    goto :conda_ready
)

echo Conda not found. Installing Miniconda...

REM Download Miniconda
set MINICONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
set INSTALLER_PATH=%TEMP%\miniconda_installer.exe

echo Downloading Miniconda...
powershell -Command "Invoke-WebRequest -Uri '%MINICONDA_URL%' -OutFile '%INSTALLER_PATH%'"

echo Installing Miniconda (silent)...
start /wait "" "%INSTALLER_PATH%" /InstallationType=JustMe /RegisterPython=0 /S /D=%USERPROFILE%\miniconda3

del "%INSTALLER_PATH%" 2>nul

REM Add to PATH
set PATH=%USERPROFILE%\miniconda3;%USERPROFILE%\miniconda3\Scripts;%USERPROFILE%\miniconda3\Library\bin;%PATH%

echo [OK] Miniconda installed successfully.

:conda_ready

REM =============================================================================
REM Step 2: Create Conda Environment
REM =============================================================================
echo.
echo [2/5] Creating conda environment '%ENV_NAME%'...

call conda env list | findstr /C:"%ENV_NAME%" >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo Environment '%ENV_NAME%' already exists. Updating...
) else (
    call conda create -n %ENV_NAME% python=%PYTHON_VERSION% -y
)

call conda activate %ENV_NAME%

echo [OK] Environment ready.

REM =============================================================================
REM Step 3: Install Dependencies
REM =============================================================================
echo.
echo [3/5] Installing dependencies...

cd /d "%SCRIPT_DIR%"

echo Installing core dependencies (this may take a few minutes)...
pip install -r requirements.txt --quiet

REM Handle neuronbridge-python Windows issue
echo Installing neuronbridge-python (with Windows workaround)...
pip install neuronbridge-python --no-deps --quiet 2>nul || echo [WARN] neuronbridge-python may not be available on Windows.

echo Installing UI dependencies...
pip install -r ui/requirements.txt --quiet

echo [OK] Dependencies installed.

REM =============================================================================
REM Step 4: Install DROCAT Package
REM =============================================================================
echo.
echo [4/5] Installing DROCAT package...

pip install -e . --quiet

echo [OK] DROCAT installed in editable mode.

REM =============================================================================
REM Step 5: Create Launcher Scripts
REM =============================================================================
echo.
echo [5/5] Creating launcher scripts...

REM Create run_ui.bat
(
echo @echo off
echo REM DROCAT UI Launcher
echo call conda activate %ENV_NAME%
echo cd /d "%%~dp0"
echo python ui/app.py
) > "%SCRIPT_DIR%run_ui.bat"

echo [OK] Launcher created: run_ui.bat

REM =============================================================================
REM Installation Complete
REM =============================================================================
echo.
echo ===============================================================
echo               Installation Complete!
echo ===============================================================
echo.
echo To launch DROCAT UI:
echo   run_ui.bat
echo.
echo Or manually:
echo   conda activate %ENV_NAME%
echo   python ui/app.py
echo.
echo The UI will open at: http://127.0.0.1:8080
echo.
echo NOTE: You may need to restart your terminal for conda to work.
echo.
echo First time? Configure your NeuPrint token in the Settings tab.
echo.

pause
