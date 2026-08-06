# DROCAT one-click installer for Windows.

$ErrorActionPreference = "Stop"
$env:PYTHONNOUSERSITE = "1"

$PythonVersion = "3.11"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
# The installer lives in archive\install; the repository root is two levels up.
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path
$DrocatVersion = "4.5.0"
$VersionLine = Select-String -Path "$ProjectRoot\ui\config.py" -Pattern '^APP_VERSION = "([^"]+)"' -ErrorAction SilentlyContinue
if ($VersionLine) { $DrocatVersion = $VersionLine.Matches[0].Groups[1].Value }
$EnvBase = "drocat-$DrocatVersion"

function Write-Step([string]$Step, [string]$Message) {
    Write-Host "[$Step] $Message" -ForegroundColor Cyan
}

function Find-Conda {
    if ($env:CONDA_EXE -and (Test-Path $env:CONDA_EXE)) {
        return $env:CONDA_EXE
    }
    $Command = Get-Command conda -ErrorAction SilentlyContinue
    if ($Command) {
        if ($Command.CommandType -eq "Application" -and $Command.Source) {
            return $Command.Source
        }
        # An initialized PowerShell session commonly exposes conda as a
        # function rather than an executable. Preserve that command spelling.
        return "conda"
    }
    $Candidates = @(
        "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
        "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
        "$env:USERPROFILE\miniforge3\Scripts\conda.exe",
        "C:\ProgramData\miniconda3\Scripts\conda.exe",
        "C:\ProgramData\anaconda3\Scripts\conda.exe"
    )
    foreach ($Candidate in $Candidates) {
        if (Test-Path $Candidate) { return $Candidate }
    }
    return $null
}

function Install-Miniconda {
    $Url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
    $Installer = Join-Path $env:TEMP "drocat-miniconda-$PID.exe"
    $InstallDir = "$env:USERPROFILE\miniconda3"
    if ((Test-Path $InstallDir) -and -not (Test-Path "$InstallDir\Scripts\conda.exe")) {
        $InstallDir = "$env:USERPROFILE\miniconda3-drocat"
    }
    try {
        Write-Host "Downloading Miniconda..."
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $Url -OutFile $Installer -UseBasicParsing
        Write-Host "Installing Miniconda..."
        $Process = Start-Process -FilePath $Installer -ArgumentList "/InstallationType=JustMe", "/RegisterPython=0", "/S", "/D=$InstallDir" -Wait -PassThru
        if ($Process.ExitCode -ne 0 -or -not (Test-Path "$InstallDir\Scripts\conda.exe")) {
            throw "Miniconda installation failed (exit $($Process.ExitCode))."
        }
        return "$InstallDir\Scripts\conda.exe"
    }
    finally {
        Remove-Item $Installer -Force -ErrorAction SilentlyContinue
    }
}

function Get-EnvironmentNames {
    $Raw = (& $script:CondaPath env list --json | Out-String)
    if ($LASTEXITCODE -ne 0) { throw "Could not list conda environments." }
    $Data = $Raw | ConvertFrom-Json
    return @($Data.envs | ForEach-Object { Split-Path $_ -Leaf })
}

function Test-EnvironmentPython([string]$Name) {
    & $script:CondaPath run -n $Name python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" 2>$null | Out-Null
    return ($LASTEXITCODE -eq 0)
}

function Invoke-InEnvironment([string[]]$Command) {
    & $script:CondaPath run -n $script:EnvName --no-capture-output @Command
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed in $script:EnvName (exit $LASTEXITCODE): $($Command -join ' ')"
    }
}

Write-Host ""
Write-Host "DROCAT - Drosophila Connectome Analysis Toolkit" -ForegroundColor Cyan
Write-Host "One-Click Installer for Windows" -ForegroundColor Cyan
Write-Host ""

Write-Step "1/5" "Checking for Conda"
$script:CondaPath = Find-Conda
if (-not $script:CondaPath) {
    Write-Host "Conda was not found; installing Miniconda." -ForegroundColor Yellow
    $script:CondaPath = Install-Miniconda
}
Write-Host "Using $script:CondaPath" -ForegroundColor Green

Write-Step "2/5" "Selecting a Python $PythonVersion environment"
$Existing = Get-EnvironmentNames
$script:EnvName = $null
for ($Index = 0; $Index -le 20; $Index++) {
    $Candidate = if ($Index -eq 0) { $EnvBase } else { "$EnvBase-$($Index + 1)" }
    if ($Existing -contains $Candidate) {
        if (Test-EnvironmentPython $Candidate) {
            $script:EnvName = $Candidate
            Write-Host "Reusing $Candidate" -ForegroundColor Green
            break
        }
        Write-Host "Skipping $Candidate because it is not Python $PythonVersion." -ForegroundColor Yellow
        continue
    }
    $script:EnvName = $Candidate
    Write-Host "Creating $Candidate..."
    & $script:CondaPath create -n $Candidate "python=$PythonVersion" -y
    if ($LASTEXITCODE -ne 0) { throw "Could not create $Candidate." }
    break
}
if (-not $script:EnvName) { throw "Could not select a usable $EnvBase environment." }

Set-Location $ProjectRoot
Write-Step "3/5" "Installing pinned dependencies"
Invoke-InEnvironment @("python", "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")

& $script:CondaPath run -n $script:EnvName python -m pip show neuronbridge-python *> $null
if ($LASTEXITCODE -eq 0) {
    Write-Host "Removing legacy neuronbridge-python dependency..."
    Invoke-InEnvironment @("python", "-m", "pip", "uninstall", "-y", "neuronbridge-python")
}
Invoke-InEnvironment @("python", "-m", "pip", "install", "--upgrade", "-r", "requirements-windows.txt", "-r", "ui\requirements.txt")

Write-Step "4/5" "Installing DROCAT"
Invoke-InEnvironment @("python", "-m", "pip", "install", "-e", ".", "--no-deps")

Write-Step "5/5" "Verifying the environment"
Invoke-InEnvironment @("python", "-m", "pip", "check")
Invoke-InEnvironment @("python", "skills\drocat-install\scripts\verify_install.py", "--project", $ProjectRoot)

Write-Host ""
Write-Host "Installation complete." -ForegroundColor Green
Write-Host "Environment: $script:EnvName"
Write-Host "Launch with: run_DROCAT.bat"

# --- Token configuration notice ---
# Tokens are NOT collected in the terminal: they are set in the UI Settings
# tab after launch, or by editing token_info_local.txt at the repository
# root. The NeuPrint token is required for NeuPrint datasets; the CAVE token
# is optional and only needed for FlyWire FAFB online fetching.
Write-Host ""
Write-Host "[Token setup]" -ForegroundColor Cyan
$TokenFile = Join-Path $ProjectRoot "token_info_local.txt"
$NeuprintNow = ""
$CaveNow = ""
if (Test-Path $TokenFile) {
    $Content = Get-Content $TokenFile -Raw -ErrorAction SilentlyContinue
    if ($Content -match "NEUPRINT_TOKEN='([^']*)'") { $NeuprintNow = $Matches[1] }
    if ($Content -match "CAVE_TOKEN='([^']*)'") { $CaveNow = $Matches[1] }
}
if ($NeuprintNow -and $NeuprintNow -ne "YOUR_NEUPRINT_TOKEN_HERE") {
    Write-Host "NeuPrint token already configured in token_info_local.txt."
} else {
    Write-Host "NeuPrint token not configured - required for NeuPrint datasets."
}
if ($CaveNow -and $CaveNow -ne "YOUR_CAVE_TOKEN_HERE") {
    Write-Host "CAVE token already configured in token_info_local.txt."
} else {
    Write-Host "CAVE token optional - only needed for FlyWire FAFB online fetching."
}
Write-Host "Set tokens in the UI Settings tab after launching, or edit token_info_local.txt"
Write-Host "(repository root, format: NEUPRINT_TOKEN='...' / CAVE_TOKEN='...')."
