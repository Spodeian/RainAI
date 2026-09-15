<#
.SYNOPSIS
    RainAI Automated Training, Export & Benchmark Orchestrator Launcher
.DESCRIPTION
    Runs the automated hardware-adaptive training pipeline with version-aware 
    PyTorch backend selection targeting CUDA 13 wheels for Python 3.14+.
.EXAMPLE
    .\train.ps1 -Profile smoke-test
    .\train.ps1 -Profile balanced
    .\train.ps1 -Profile production
#>
param(
    [ValidateSet("smoke-test", "balanced", "production", "export-only", "custom")]
    [string]$Profile = "balanced",

    [string[]]$Phases,
    [int]$EpochsVae,
    [int]$EpochsMamba,
    [int]$BatchSize,
    [int]$MaxBatches,
    [int]$AccumulationSteps,
    [string]$Device,
    [int]$SliceLevel,
    [int]$NumSlices,
    [switch]$NoAmp,
    [switch]$UseDisc,
    [switch]$ChunkCurriculum,
    [switch]$UseActiveLearning,
    [switch]$PinMemory,
    [switch]$Fresh,
    [switch]$PrepareData,
    [switch]$RebuildData,
    [int]$BenchmarkIters,
    [string]$LogDir,
    [string]$LogFile,
    [switch]$Verbose,
    [switch]$Quiet
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Setup Python Virtual Environment and Paths
$VenvPath = Join-Path $ScriptDir ".venv"
$PythonExe = Join-Path $VenvPath "Scripts\python.exe"
$RequirementsPath = Join-Path $ScriptDir "requirements.txt"

# 1. Ensure Virtual Environment Exists
if (-not (Test-Path $PythonExe) -or $Fresh) {
    if (Test-Path $VenvPath) {
        Write-Host "[*] Clearing existing virtual environment for fresh provisioning..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $VenvPath
    }

    Write-Host "[*] Creating Python virtual environment (.venv)..." -ForegroundColor Yellow
    $SysPython = "python"
    if (-not (Get-Command $SysPython -ErrorAction SilentlyContinue)) {
        $SysPython = "python3"
    }

    try {
        & $SysPython -m venv $VenvPath
    } catch {
        Write-Error "Failed to create virtual environment using '$SysPython'. Ensure Python is installed and added to PATH."
        exit 1
    }
}

# 2. Python Version & Hardware Probing for PyTorch Index Selection
Write-Host "[*] Probing host Python version and hardware architecture..." -ForegroundColor Cyan

$PyVersionInfo = & $PythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
Write-Host "[+] Active Python Version: $PyVersionInfo" -ForegroundColor Cyan

$HasCuda = $false
$TorchIndexUrl = "https://download.pytorch.org/whl/cpu"

try {
    if (Get-Command "nvidia-smi" -ErrorAction SilentlyContinue) {
        $NvidiaSmiOutput = & nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1
        if ($LASTEXITCODE -eq 0 -and $NvidiaSmiOutput) {
            $HasCuda = $true
            # Route Python 3.14+ to CUDA 13 wheels, and older python versions to CUDA 12.4
            if ([version]$PyVersionInfo -ge [version]"3.14") {
                $TorchIndexUrl = "https://download.pytorch.org/whl/cu130"
            } else {
                $TorchIndexUrl = "https://download.pytorch.org/whl/cu124"
            }
            Write-Host "[+] NVIDIA CUDA GPU detected. Target Index: $TorchIndexUrl" -ForegroundColor Green
        }
    }
} catch {
    Write-Warning "[!] Could not query nvidia-smi. Defaulting to CPU backend index."
}

# 3. Synchronize Packages & Dependencies
Write-Host "[*] Synchronizing virtual environment packages..." -ForegroundColor Cyan
& $PythonExe -m pip install --upgrade pip setuptools wheel

Write-Host "[*] Installing/updating PyTorch ecosystem backend..." -ForegroundColor Cyan
& $PythonExe -m pip install --upgrade torch torchvision torchaudio --index-url $TorchIndexUrl

if (Test-Path $RequirementsPath) {
    Write-Host "[*] Synchronizing workspace dependencies from requirements.txt..." -ForegroundColor Cyan
    & $PythonExe -m pip install -r $RequirementsPath --extra-index-url $TorchIndexUrl
} else {
    Write-Warning "[!] requirements.txt not found in project root."
}

$ArgsList = @("-X", "utf8", "scripts\auto_train.py", "--profile", $Profile)

# Append dynamic configuration flags matching script parameters
if ($Phases) {
    $ArgsList += "--phases"
    $ArgsList += $Phases
}
if ($PSBoundParameters.ContainsKey('EpochsVae')) { $ArgsList += @("--epochs-vae", $EpochsVae) }
if ($PSBoundParameters.ContainsKey('EpochsMamba')) { $ArgsList += @("--epochs-mamba", $EpochsMamba) }
if ($PSBoundParameters.ContainsKey('BatchSize')) { $ArgsList += @("--batch-size", $BatchSize) }
if ($PSBoundParameters.ContainsKey('MaxBatches')) { $ArgsList += @("--max-batches", $MaxBatches) }
if ($PSBoundParameters.ContainsKey('AccumulationSteps')) { $ArgsList += @("--accumulation-steps", $AccumulationSteps) }
if ($Device) { $ArgsList += @("--device", $Device) }
if ($PSBoundParameters.ContainsKey('SliceLevel')) { $ArgsList += @("--slice-level", $SliceLevel) }
if ($PSBoundParameters.ContainsKey('NumSlices')) { $ArgsList += @("--num-slices", $NumSlices) }
if ($NoAmp) { $ArgsList += "--no-amp" }
if ($UseDisc) { $ArgsList += "--use-disc" }
if ($ChunkCurriculum) { $ArgsList += "--chunk-curriculum" }
if ($UseActiveLearning) { $ArgsList += "--use-active-learning" }
if ($PinMemory) { $ArgsList += "--pin-memory" }
if ($Fresh) { $ArgsList += "--fresh" }
if ($PrepareData) { $ArgsList += "--prepare-data" }
if ($RebuildData) { $ArgsList += "--rebuild-data" }
if ($BenchmarkIters) { $ArgsList += @("--benchmark-iters", $BenchmarkIters) }
if ($LogDir) { $ArgsList += @("--log-dir", $LogDir) }
if ($LogFile) { $ArgsList += @("--log-file", $LogFile) }
if ($Verbose) { $ArgsList += "--verbose" }
if ($Quiet) { $ArgsList += "--quiet" }

# Execute training orchestrator
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "[*] Launching RainAI Training Orchestrator [Profile: $Profile]..." -ForegroundColor Cyan
& $PythonExe $ArgsList
exit $LASTEXITCODE
