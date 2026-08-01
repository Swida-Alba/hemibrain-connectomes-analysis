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
$EnvName = "drocat"
$PythonVersion = "3.11"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

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
        "C:\ProgramData\miniconda3\Scripts\conda.exe",
        "C:\ProgramData\anaconda3\Scripts\conda.exe"
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

# Check if environment exists
$EnvList = & $CondaPath env list 2>$null
if ($EnvList -match $EnvName) {
    Write-Warning-Custom "Environment '$EnvName' already exists. Updating..."
}
else {
    & $CondaPath create -n $EnvName python=$PythonVersion -y
}

# Activate environment
& $CondaPath activate $EnvName

Write-Success "Environment ready"

# =============================================================================
# Step 3: Install Dependencies
# =============================================================================
Write-Host ""
Write-Step "3/5" "Installing dependencies..."

Set-Location $ScriptDir

Write-Host "Installing core dependencies (this may take a few minutes)..."
pip install -r requirements.txt --quiet

# Handle neuronbridge-python Windows issue
Write-Host "Installing neuronbridge-python (with Windows workaround)..."
try {
    pip install neuronbridge-python --no-deps --quiet 2>$null
}
catch {
    Write-Warning-Custom "neuronbridge-python may not be available on Windows"
}

Write-Host "Installing UI dependencies..."
pip install -r ui/requirements.txt --quiet

Write-Success "Dependencies installed"

# =============================================================================
# Step 4: Install DROCAT Package
# =============================================================================
Write-Host ""
Write-Step "4/5" "Installing DROCAT package..."

pip install -e . --quiet

Write-Success "DROCAT installed in editable mode"

# =============================================================================
# Step 5: Create Launcher Scripts
# =============================================================================
Write-Host ""
Write-Step "5/5" "Creating launcher scripts..."

# Create run_ui.bat
$LauncherContent = @"
@echo off
REM DROCAT UI Launcher
call conda activate $EnvName
cd /d "%~dp0"
python ui/app.py
"@
$LauncherContent | Out-File -FilePath "$ScriptDir\run_ui.bat" -Encoding ASCII

# Create run_ui.ps1
$PsLauncherContent = @"
# DROCAT UI Launcher (PowerShell)
`$ScriptDir = Split-Path -Parent `$MyInvocation.MyCommand.Path
conda activate $EnvName
Set-Location `$ScriptDir
python ui/app.py
"@
$PsLauncherContent | Out-File -FilePath "$ScriptDir\run_ui.ps1" -Encoding UTF8

Write-Success "Launchers created: run_ui.bat, run_ui.ps1"

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
