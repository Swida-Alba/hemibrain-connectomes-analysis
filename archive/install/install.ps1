# DROCAT one-click installer for Windows.

$ErrorActionPreference = "Stop"
$env:PYTHONNOUSERSITE = "1"

$PythonVersion = "3.11"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
# The installer lives in archive\install; the repository root is two levels up.
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path

# Guard against an unwritable shared pip cache (restricted ACLs on
# %LOCALAPPDATA%\pip\cache abort the whole install): use a project-local
# cache directory, and fall back to disabling the pip cache entirely if
# even that directory cannot be created.
try {
    $PipCacheDir = Join-Path $ProjectRoot "cache\pip"
    New-Item -ItemType Directory -Force -Path $PipCacheDir -ErrorAction Stop | Out-Null
    $env:PIP_CACHE_DIR = $PipCacheDir
} catch {
    $env:PIP_NO_CACHE_DIR = "1"
}

$DrocatVersion = "4.5.0"
$VersionLine = Select-String -Path "$ProjectRoot\ui\config.py" -Pattern '^APP_VERSION = "([^"]+)"' -ErrorAction SilentlyContinue
if ($VersionLine) { $DrocatVersion = $VersionLine.Matches[0].Groups[1].Value }
$EnvBase = "drocat-$DrocatVersion"
# config.json ships clean on GitHub (committed defaults) and wins per key;
# the gitignored config_local.json is the developer-specific fallback (never
# created automatically - developers add it manually when needed).
$ConfigFile = Join-Path $ProjectRoot "config.json"
$ConfigLocal = Join-Path $ProjectRoot "config_local.json"

# Version-specific custom env (envs.<version>) and tokens: config.json wins
# per key (the file a GitHub-pulled copy edits directly); the gitignored
# config_local.json is the developer-specific fallback for empty entries.
# envs.<version> is only consulted for the CURRENT release, so upgrading
# DROCAT never reuses an older release's custom environment.
$Config = $null
if (Test-Path $ConfigFile) {
    try {
        $Config = Get-Content $ConfigFile -Raw | ConvertFrom-Json
    } catch {
        $Config = $null
    }
}
$ConfigLocalData = $null
if (Test-Path $ConfigLocal) {
    try {
        $ConfigLocalData = Get-Content $ConfigLocal -Raw | ConvertFrom-Json
    } catch {
        $ConfigLocalData = $null
    }
}
$EnvOverride = ""
if ($Config -and $Config.envs) {
    $EnvOverride = [string]$Config.envs."$DrocatVersion"
}
if (-not $EnvOverride -and $ConfigLocalData -and $ConfigLocalData.envs) {
    $EnvOverride = [string]$ConfigLocalData.envs."$DrocatVersion"
}

function Set-ConfigEnvOverride([string]$Version, [string]$EnvName) {
    # Persist the selected environment into the LOCAL config so an empty
    # entry is filled with the auto-created name. The committed config.json
    # is never rewritten, and a configured custom name is never touched.
    if (-not (Test-Path $ConfigLocal)) { return }
    try {
        $Cfg = Get-Content $ConfigLocal -Raw | ConvertFrom-Json
    } catch {
        return
    }
    if (-not $Cfg.envs) {
        $Cfg | Add-Member -NotePropertyName "envs" -NotePropertyValue ([pscustomobject]@{}) -Force
    }
    $Cfg.envs | Add-Member -NotePropertyName $Version -NotePropertyValue $EnvName -Force
    $Cfg | ConvertTo-Json -Depth 5 | Set-Content $ConfigLocal -Encoding UTF8
}

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
    # EAP Continue: Windows PowerShell 5.1 aborts on ANY native stderr line
    # under EAP Stop (same root cause as Invoke-InEnvironment).
    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $Raw = (& $script:CondaPath env list --json | Out-String)
        if ($LASTEXITCODE -ne 0) { throw "Could not list conda environments." }
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
    $Data = $Raw | ConvertFrom-Json
    return @($Data.envs | ForEach-Object { Split-Path $_ -Leaf })
}

function Test-EnvironmentPython([string]$Name) {
    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $script:CondaPath run -n $Name python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" 2>$null | Out-Null
        return ($LASTEXITCODE -eq 0)
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
}

function Invoke-InEnvironment([string[]]$Command) {
    # Windows PowerShell 5.1 converts ANY native stderr line into a
    # terminating NativeCommandError when $ErrorActionPreference is "Stop"
    # (pip warnings such as "Connection interrupted while downloading"
    # would abort an otherwise-successful install). Run native commands
    # with EAP Continue and rely on $LASTEXITCODE instead.
    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $script:CondaPath run -n $script:EnvName --no-capture-output @Command
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed in $script:EnvName (exit $LASTEXITCODE): $($Command -join ' ')"
        }
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
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
if ($EnvOverride) {
    if ($Existing -contains $EnvOverride) {
        if (Test-EnvironmentPython $EnvOverride) {
            $script:EnvName = $EnvOverride
            Write-Host "Reusing $EnvOverride (custom env from config.json)" -ForegroundColor Green
        } else {
            # A configured custom env must be the env used: never silently
            # switch to a default name and clobber the config entry.
            throw "$EnvOverride (custom env from config.json) exists but is not Python $PythonVersion. Fix or remove it, or clear the envs entry."
        }
    } else {
        $script:EnvName = $EnvOverride
        Write-Host "Creating $EnvOverride (custom env from config.json)..."
        $PreviousErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            # Create against conda-forge with --override-channels: a fresh
            # Miniconda 26.x blocks non-interactive `conda create` against the
            # default (repo.anaconda.com) channels until the Anaconda Terms of
            # Service are accepted (CondaToSNonInteractiveError). conda-forge
            # requires no such acceptance; post-creation deps are pip-installed.
            & $script:CondaPath create -n $EnvOverride -c conda-forge --override-channels "python=$PythonVersion" -y
        } finally {
            $ErrorActionPreference = $PreviousErrorActionPreference
        }
        if ($LASTEXITCODE -ne 0) { throw "Could not create $EnvOverride." }
    }
}
if (-not $script:EnvName) {
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
        $PreviousErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            # See the custom-env branch above: conda-forge + --override-channels
            # sidesteps the Anaconda ToS gate on fresh Miniconda 26.x installs.
            & $script:CondaPath create -n $Candidate -c conda-forge --override-channels "python=$PythonVersion" -y
        } finally {
            $ErrorActionPreference = $PreviousErrorActionPreference
        }
        if ($LASTEXITCODE -ne 0) { throw "Could not create $Candidate." }
        break
    }
}
if (-not $script:EnvName) { throw "Could not select a usable $EnvBase environment." }
# Persist the auto-selected environment into the LOCAL config when no
# custom name was configured (empty entry -> auto-fill). The committed
# config.json is never rewritten; a custom name is never touched.
if (-not $EnvOverride) { Set-ConfigEnvOverride $DrocatVersion $script:EnvName }

Set-Location $ProjectRoot
Write-Step "3/5" "Installing pinned dependencies"
Invoke-InEnvironment @("python", "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")

# Probe for the legacy neuronbridge-python distribution. On a fresh install
# pip prints "WARNING: Package(s) not found" to stderr, which Windows
# PowerShell 5.1 turns into a terminating NativeCommandError under EAP Stop;
# missing is the desired state, so swallow it and rely on $LASTEXITCODE.
$LegacyNeuronbridge = $false
try {
    & $script:CondaPath run -n $script:EnvName python -m pip show neuronbridge-python *> $null
    $LegacyNeuronbridge = ($LASTEXITCODE -eq 0)
} catch {
    $LegacyNeuronbridge = $false
}
if ($LegacyNeuronbridge) {
    Write-Host "Removing legacy neuronbridge-python dependency..."
    Invoke-InEnvironment @("python", "-m", "pip", "uninstall", "-y", "neuronbridge-python")
}
Invoke-InEnvironment @("python", "-m", "pip", "install", "--upgrade", "-r", "requirements-windows.txt", "-r", "ui\requirements.txt")

Write-Step "4/5" "Installing DROCAT"
Invoke-InEnvironment @("python", "-m", "pip", "install", "-e", ".", "--no-deps")

Write-Step "5/5" "Verifying the environment"
Invoke-InEnvironment @("python", "-m", "pip", "check")
# pip can silently install a metadata-only wheel when a PEP 517 sdist build
# is corrupted mid-build on a flaky network (observed with img2pdf and
# asciitree: "Successfully installed" but the modules are missing). The
# bundled verifier catches it; before aborting, clear the pip wheel cache
# (which would otherwise re-serve the bad wheels) and rebuild once.
$Verified = $false
for ($Attempt = 1; $Attempt -le 2 -and -not $Verified; $Attempt++) {
    if ($Attempt -eq 2) {
        Write-Host "Verification failed; clearing the pip wheel cache and rebuilding dependencies once." -ForegroundColor Yellow
        Remove-Item (Join-Path $PipCacheDir "wheels") -Recurse -Force -ErrorAction SilentlyContinue
        Invoke-InEnvironment @("python", "-m", "pip", "install", "--upgrade", "-r", "requirements-windows.txt", "-r", "ui\requirements.txt")
    }
    try {
        Invoke-InEnvironment @("python", "skills\drocat-install\scripts\verify_install.py", "--project", $ProjectRoot)
        $Verified = $true
    } catch {
        $Verified = $false
    }
}
if (-not $Verified) { throw "Environment verification failed after one rebuild retry." }

Write-Host ""
Write-Host "Installation complete." -ForegroundColor Green
Write-Host "Environment: $script:EnvName"
Write-Host "Launch with: run_DROCAT.bat"

# --- Token configuration notice ---
# Tokens are NOT collected in the terminal: they are set in the UI Settings
# tab after launch, or by editing config.json at the repository root (the
# gitignored config_local.json is the optional developer fallback). The
# NeuPrint token is required for NeuPrint datasets; the CAVE token is
# optional and only needed for FlyWire FAFB online fetching.
Write-Host ""
Write-Host "[Token setup]" -ForegroundColor Cyan
$NeuprintNow = ""
$CaveNow = ""
foreach ($Cfg in @($Config, $ConfigLocalData)) {
    if ($Cfg -and $Cfg.tokens) {
        if (-not $NeuprintNow -and $Cfg.tokens.neuprint) { $NeuprintNow = [string]$Cfg.tokens.neuprint }
        if (-not $CaveNow -and $Cfg.tokens.cave) { $CaveNow = [string]$Cfg.tokens.cave }
    }
}
if ($NeuprintNow -and $NeuprintNow -ne "YOUR_NEUPRINT_TOKEN_HERE") {
    Write-Host "NeuPrint token already configured in config.json or config_local.json."
} else {
    Write-Host "NeuPrint token not configured - required for NeuPrint datasets."
}
if ($CaveNow -and $CaveNow -ne "YOUR_CAVE_TOKEN_HERE") {
    Write-Host "CAVE token already configured in config.json or config_local.json."
} else {
    Write-Host "CAVE token optional - only needed for FlyWire FAFB online fetching."
}
Write-Host "Set tokens in the UI Settings tab after launching, or edit config.json"
Write-Host "(repository root, format: tokens.neuprint / tokens.cave)."
Write-Host "Get a NeuPrint token from: https://neuprint.janelia.org/account"
Write-Host "Get a CAVE token from: https://codex.flywire.ai/auth_token"
