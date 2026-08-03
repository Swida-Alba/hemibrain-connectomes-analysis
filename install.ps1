# =============================================================================
# DROCAT One-Click Installer for Windows (PowerShell)
# =============================================================================
# This script installs DROCAT with all dependencies using Miniconda.
# Usage: Right-click -> Run with PowerShell, or: powershell -ExecutionPolicy Bypass -File install.ps1
# =============================================================================

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "===============================================================" -ForegroundColor Cyan
Write-Host "     DROCAT - Drosophila Connectome Analysis Toolkit" -ForegroundColor Cyan
Write-Host "                One-Click Installer (Windows)" -ForegroundColor Cyan
Write-Host "===============================================================" -ForegroundColor Cyan
Write-Host ""

# Configuration
$PythonVersion = "3.11"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VersionLine = Select-String -Path "$ScriptDir\ui\config.py" -Pattern '^APP_VERSION = "([^"]+)"' -ErrorAction SilentlyContinue
$DrocatVersion = "4.5.0"
if ($VersionLine) { $DrocatVersion = $VersionLine.Matches[0].Groups[1].Value }
$EnvBase = "drocat-$DrocatVersion"

# =============================================================================
# Helper Functions
# =============================================================================
function Write-Step {
    param([string]$Step, [string]$Message)
    Write-Host "[$Step] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[OK] $Message" -ForegroundColor Green
}

function Write-Warning-Custom {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

# Run a command inside the drocat environment via `conda run`.
# Falls back to plain `conda run` on older conda versions that do not
# support --no-capture-output.
function Invoke-InEnv {
    param([string[]]$Command)
    & $CondaPath run -n $EnvName --no-capture-output @Command 2>$null
    if ($LASTEXITCODE -eq 2) {
        & $CondaPath run -n $EnvName @Command
    }
}

# =============================================================================
# Step 1: Check/Install Miniconda
# =============================================================================
Write-Step "1/5" "Checking for Conda..."

$CondaPath = $null

# Check if conda is in PATH
$CondaCmd = Get-Command conda -ErrorAction SilentlyContinue
if ($CondaCmd) {
    $CondaPath = $CondaCmd.Source
    Write-Success "Conda found: $CondaPath"
}
else {
    # Check common locations
    $PossiblePaths = @(
        "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
        "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
        "$env:USERPROFILE\miniforge3\Scripts\conda.exe",
        "C:\ProgramData\miniconda3\Scripts\conda.exe",
        "C:\ProgramData\anaconda3\Scripts\conda.exe",
        "C:\ProgramData\miniforge3\Scripts\conda.exe"
    )
    
    foreach ($Path in $PossiblePaths) {
        if (Test-Path $Path) {
            $CondaPath = $Path
            $env:PATH = "$(Split-Path $Path);$env:PATH"
            Write-Success "Conda found: $Path"
            break
        }
    }
}

if (-not $CondaPath) {
    Write-Warning-Custom "Conda not found. Installing Miniconda..."
    
    $MinicondaUrl = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
    $InstallerPath = "$env:TEMP\miniconda_installer.exe"
    $InstallDir = "$env:USERPROFILE\miniconda3"
    
    Write-Host "Downloading Miniconda..."
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri $MinicondaUrl -OutFile $InstallerPath -UseBasicParsing
    
    Write-Host "Installing Miniconda (silent)..."
    Start-Process -FilePath $InstallerPath -ArgumentList "/InstallationType=JustMe", "/RegisterPython=0", "/S", "/D=$InstallDir" -Wait
    
    Remove-Item $InstallerPath -Force -ErrorAction SilentlyContinue
    
    # Verify the silent install actually produced conda (fails otherwise)
    if (-not (Test-Path "$InstallDir\Scripts\conda.exe")) {
        Write-Error "Miniconda installation failed - $InstallDir\Scripts\conda.exe not found."
        exit 1
    }
    
    # Update PATH
    $env:PATH = "$InstallDir;$InstallDir\Scripts;$InstallDir\Library\bin;$env:PATH"
    $CondaPath = "$InstallDir\Scripts\conda.exe"
    
    Write-Success "Miniconda installed successfully"
}

# Initialize conda for PowerShell
& $CondaPath init powershell 2>$null | Out-Null

# =============================================================================
# Step 2: Create Conda Environment
# =============================================================================
Write-Host ""
Write-Step "2/5" "Creating conda environment '$EnvName'..."

# If the versioned base env already exists: reuse it (update deps in place)
# when it has the right Python - this keeps installer re-runs consistent with
# the launchers, which prefer the FIRST usable drocat-<version> env. A wrong
# Python env is never touched; the next free name (-2, -3, ...) is used instead.
# NOTE: escape the name for -match ('.' in the version is a regex wildcard).
$EnvList = & $CondaPath env list 2>$null
$EnvBaseRx = [regex]::Escape($EnvBase)
function Test-EnvUsable([string]$Name) {
    & $CondaPath run -n $Name python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" 2>$null | Out-Null
    return ($LASTEXITCODE -eq 0)
}
if ($EnvList -match "^$EnvBaseRx\s") {
    if (Test-EnvUsable $EnvBase) {
        $EnvName = $EnvBase
        Write-Success "Reusing existing env '$EnvName' (Python $PythonVersion) - dependencies will be updated in place."
    }
    else {
        Write-Warning-Custom "Conda env '$EnvBase' exists but does not use Python $PythonVersion - leaving it untouched."
        $envNum = 2
        while ($EnvList -match "^$EnvBaseRx-$envNum\s") { $envNum++ }
        $EnvName = "$EnvBase-$envNum"
        Write-Success "Creating a new environment instead: $EnvName"
        & $CondaPath create -n $EnvName python=$PythonVersion -y
    }
}
else {
    $EnvName = $EnvBase
    & $CondaPath create -n $EnvName python=$PythonVersion -y
}
# Fail fast if the env was not actually created (e.g. resolver error above)
$CheckList = & $CondaPath env list 2>$null
if (-not ($CheckList -match "^$([regex]::Escape($EnvName))\s")) {
    Write-Error "Failed to create conda environment '$EnvName'. See messages above."
    exit 1
}

Write-Success "Environment ready (all steps run inside the env via 'conda run')"

# =============================================================================
# Step 3: Install Dependencies
# =============================================================================
Write-Host ""
Write-Step "3/5" "Installing dependencies..."

Set-Location $ScriptDir

# Windows uses requirements-windows.txt (two-step neuronbridge install):
# memray (a neuronbridge-python dependency) does not support Windows, so we
# install the compatible deps first, then neuronbridge-python --no-deps.
Write-Host "Installing core dependencies from requirements-windows.txt..."
Invoke-InEnv @("python", "-m", "pip", "install", "-r", "requirements-windows.txt")
if ($LASTEXITCODE -ne 0) {
    Write-Host "Core dependency install failed (exit $LASTEXITCODE)." -ForegroundColor Red
    exit 1
}

Write-Host "Installing neuronbridge-python (--no-deps, Windows memray workaround)..."
Invoke-InEnv @("python", "-m", "pip", "install", "neuronbridge-python", "--no-deps")
if ($LASTEXITCODE -ne 0) {
    Write-Warning-Custom "neuronbridge-python could not be installed; NeuronBridge panels will be limited"
}

Write-Host "Installing UI dependencies..."
Invoke-InEnv @("python", "-m", "pip", "install", "-r", "ui/requirements.txt")
if ($LASTEXITCODE -ne 0) {
    Write-Host "UI dependency install failed (exit $LASTEXITCODE)." -ForegroundColor Red
    exit 1
}

Write-Success "Dependencies installed"

# =============================================================================
# Step 4: Install DROCAT Package
# =============================================================================
Write-Host ""
Write-Step "4/5" "Installing DROCAT package..."

Invoke-InEnv @("python", "-m", "pip", "install", "-e", ".", "--no-deps")
if ($LASTEXITCODE -ne 0) {
    Write-Host "Editable install failed (exit $LASTEXITCODE)." -ForegroundColor Red
    exit 1
}

Write-Success "DROCAT installed in editable mode"

# =============================================================================
# Step 5: Create Launcher Scripts
# =============================================================================
Write-Host ""
Write-Step "5/5" "Creating launcher scripts..."

# Create run_ui.bat (self-healing launcher; only if missing)
$RunBat = "$ScriptDir\run_ui.bat"
if (Test-Path $RunBat) {
    Write-Warning-Custom "run_ui.bat already exists - keeping it"
}
else {
$LauncherContent = @"
@echo off
REM DROCAT UI Launcher
setlocal
set SCRIPT_DIR=%~dp0
set CONDA_BIN=
set "DROCAT_VERSION=4.5.0"
for /f "tokens=3" %%v in ('findstr /C:"APP_VERSION = " "%SCRIPT_DIR%ui\config.py"') do set "DROCAT_VERSION=%%v"
set "DROCAT_VERSION=%DROCAT_VERSION:"=%"
set "ENV_BASE=drocat-%DROCAT_VERSION%"
where conda >nul 2>nul && set CONDA_BIN=conda
if not defined CONDA_BIN if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" set CONDA_BIN=%USERPROFILE%\miniconda3\Scripts\conda.exe
if not defined CONDA_BIN if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe" set CONDA_BIN=%USERPROFILE%\anaconda3\Scripts\conda.exe
if not defined CONDA_BIN (
    echo ERROR: conda not found. Run install.bat first.
    pause
    exit /b 1
)
set "ENV_NAME="
set "N=0"
:env_resolve
if "%N%"=="0" (set "CAND=%ENV_BASE%") else (set "CAND=%ENV_BASE%-%N%")
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
if "%N%"=="0" echo WARNING: existing "%ENV_BASE%" env is not usable - using a new env.
set /a N+=1
if %N% GTR 20 (
    echo ERROR: could not resolve a usable %ENV_BASE% environment.
    pause
    exit /b 1
)
goto :env_resolve
:env_create
echo Creating environment %ENV_NAME% (first run)...
call %CONDA_BIN% create -n %ENV_NAME% python=3.11 -y || goto :err
:env_found
if not "%ENV_NAME%"=="%ENV_BASE%" echo Using environment: %ENV_NAME%
call %CONDA_BIN% run -n %ENV_NAME% python -c "import nicegui" >nul 2>nul
if errorlevel 1 (
    echo Installing dependencies (first run)...
    call %CONDA_BIN% run -n %ENV_NAME% --no-capture-output python -m pip install -r "%SCRIPT_DIR%requirements-windows.txt" || goto :err
    call %CONDA_BIN% run -n %ENV_NAME% --no-capture-output python -m pip install neuronbridge-python --no-deps
    call %CONDA_BIN% run -n %ENV_NAME% --no-capture-output python -m pip install -r "%SCRIPT_DIR%ui\requirements.txt" || goto :err
    call %CONDA_BIN% run -n %ENV_NAME% --no-capture-output python -m pip install -e "%SCRIPT_DIR%" --no-deps || goto :err
)
cd /d "%SCRIPT_DIR%"
call %CONDA_BIN% run -n %ENV_NAME% --no-capture-output python ui/app.py
exit /b 0
:err
echo Installation failed. See messages above.
pause
exit /b 1
"@
$LauncherContent | Out-File -FilePath $RunBat -Encoding ASCII
Write-Success "Launcher created: run_ui.bat"
}

# =============================================================================
# Step 6: Verify the installation
# =============================================================================
Write-Host ""
Write-Step "6/6" "Verifying installation..."

Invoke-InEnv @("python", "-m", "pip", "check")
if ($LASTEXITCODE -ne 0) {
    Write-Warning-Custom "pip check reported dependency conflicts"
}

Invoke-InEnv @("python", "-c", "import numpy,pandas,polars,scipy,matplotlib,plotly,networkx,neuprint,nicegui; import neuronbridge; print('OK')")
if ($LASTEXITCODE -eq 0) {
    Write-Success "Core imports verified"
}
else {
    Write-Warning-Custom "Some imports failed - check the messages above"
}

# =============================================================================
# Installation Complete
# =============================================================================
Write-Host ""
Write-Host "===============================================================" -ForegroundColor Green
Write-Host "              Installation Complete!" -ForegroundColor Green
Write-Host "===============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "To launch DROCAT UI:" -ForegroundColor Cyan
Write-Host "  .\run_ui.bat"
Write-Host ""
Write-Host "Or manually:" -ForegroundColor Cyan
Write-Host "  conda activate $EnvName"
Write-Host "  python ui/app.py"
Write-Host ""
Write-Host "The UI will open at: http://127.0.0.1:8080" -ForegroundColor Cyan
Write-Host ""
Write-Host "NOTE: You may need to restart your terminal for conda to work." -ForegroundColor Yellow
Write-Host ""
Write-Host "First time? Configure your NeuPrint token in the Settings tab." -ForegroundColor Cyan
Write-Host ""

# Keep window open
Read-Host "Press Enter to exit"
