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
    echo Environment '%ENV_NAME%' already exists. Checking Python version...
    set "ENV_PY="
    for /f "delims=" %%v in ('call conda run -n %ENV_NAME% python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2^>nul') do set "ENV_PY=%%v"
    if "!ENV_PY!"=="%PYTHON_VERSION%" (
        echo Reusing existing env ^(Python %PYTHON_VERSION%^). Dependencies will be updated in place.
    ) else (
        echo [WARN] Existing env uses Python !ENV_PY! ^(expected %PYTHON_VERSION%^).
        echo Recreating the env will remove any other packages installed in it.
        set "RECREATE="
        set /p RECREATE="Recreate the env with Python %PYTHON_VERSION%? [y/N] "
        if /i "!RECREATE!"=="y" (
            echo Removing existing env...
            call conda env remove -n %ENV_NAME% -y
            call conda create -n %ENV_NAME% python=%PYTHON_VERSION% -y
        ) else (
            echo Aborting. To fix manually: conda env remove -n %ENV_NAME% -y, then re-run install.bat
            pause
            exit /b 1
        )
    )
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
call conda run -n %ENV_NAME% --no-capture-output python -m pip install -r requirements-windows.txt
if errorlevel 1 (
    echo [ERROR] Core dependency install failed.
    pause
    exit /b 1
)

REM Handle neuronbridge-python Windows issue
echo Installing neuronbridge-python (with Windows workaround)...
call conda run -n %ENV_NAME% --no-capture-output python -m pip install neuronbridge-python --no-deps
if errorlevel 1 echo [WARN] neuronbridge-python could not be installed; NeuronBridge panels will be limited.

echo Installing UI dependencies...
call conda run -n %ENV_NAME% --no-capture-output python -m pip install -r ui\requirements.txt
if errorlevel 1 (
    echo [ERROR] UI dependency install failed.
    pause
    exit /b 1
)

echo [OK] Dependencies installed.

REM =============================================================================
REM Step 4: Install DROCAT Package
REM =============================================================================
echo.
echo [4/5] Installing DROCAT package...

call conda run -n %ENV_NAME% --no-capture-output python -m pip install -e . --no-deps
if errorlevel 1 (
    echo [ERROR] Editable install failed.
    pause
    exit /b 1
)

echo [OK] DROCAT installed in editable mode.

REM =============================================================================
REM Step 5: Create Launcher Scripts
REM =============================================================================
echo.
echo [5/5] Creating launcher scripts...

REM Create run_ui.bat (keep an existing launcher)
if exist "%SCRIPT_DIR%run_ui.bat" (
    echo [OK] run_ui.bat already exists - keeping it.
    goto :verify
)
(
echo @echo off
echo REM DROCAT UI Launcher
echo setlocal
echo set ENV_NAME=%ENV_NAME%
echo set SCRIPT_DIR=%%~dp0
echo set CONDA_BIN=
echo where conda ^>nul 2^>nul ^&^& set CONDA_BIN=conda
echo if not defined CONDA_BIN if exist "%%USERPROFILE%%\miniconda3\Scripts\conda.exe" set CONDA_BIN=%%USERPROFILE%%\miniconda3\Scripts\conda.exe
echo if not defined CONDA_BIN if exist "%%USERPROFILE%%\anaconda3\Scripts\conda.exe" set CONDA_BIN=%%USERPROFILE%%\anaconda3\Scripts\conda.exe
echo if not defined CONDA_BIN ^(echo ERROR: conda not found. Run install.bat first. ^& pause ^& exit /b 1^)
echo set "ENV_PY="
echo for /f "delims=" %%%%v in ('call %%CONDA_BIN%% run -n %%ENV_NAME%% python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2^>nul') do set "ENV_PY=%%%%v"
echo if defined ENV_PY if not "%%ENV_PY%%"=="3.11" ^(
echo     echo ERROR: existing env uses Python %%ENV_PY%% ^(expected 3.11^).
echo     echo Recreate it: conda env remove -n drocat -y, then re-run install.bat
echo     pause
echo     exit /b 1
echo ^)
echo call %%CONDA_BIN%% run -n %%ENV_NAME%% python -c "import nicegui" ^>nul 2^>nul
echo if errorlevel 1 ^(
echo     echo Creating environment and installing dependencies ^(first run^)...
echo     call %%CONDA_BIN%% create -n %%ENV_NAME%% python=3.11 -y ^|^| goto :err
echo     call %%CONDA_BIN%% run -n %%ENV_NAME%% --no-capture-output python -m pip install -r "%%SCRIPT_DIR%%requirements-windows.txt" ^|^| goto :err
echo     call %%CONDA_BIN%% run -n %%ENV_NAME%% --no-capture-output python -m pip install neuronbridge-python --no-deps
echo     call %%CONDA_BIN%% run -n %%ENV_NAME%% --no-capture-output python -m pip install -r "%%SCRIPT_DIR%%ui\requirements.txt" ^|^| goto :err
echo     call %%CONDA_BIN%% run -n %%ENV_NAME%% --no-capture-output python -m pip install -e "%%SCRIPT_DIR%%" --no-deps ^|^| goto :err
echo ^)
echo cd /d "%%SCRIPT_DIR%%"
echo call %%CONDA_BIN%% run -n %%ENV_NAME%% --no-capture-output python ui\app.py
echo exit /b 0
echo :err
echo echo Installation failed. See messages above.
echo pause
echo exit /b 1
) > "%SCRIPT_DIR%run_ui.bat"

echo [OK] Launcher created: run_ui.bat

:verify
echo.
echo [6/6] Verifying installation...
call conda run -n %ENV_NAME% --no-capture-output python -m pip check
call conda run -n %ENV_NAME% --no-capture-output python -c "import numpy,pandas,polars,scipy,matplotlib,plotly,networkx,neuprint,nicegui; import neuronbridge; print('Core imports OK')"
if errorlevel 1 echo [WARN] Some imports failed - check the messages above.

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
