<#
.SYNOPSIS
    RainAI Automated Training, Export & Benchmark Orchestrator Launcher
.DESCRIPTION
    Runs the automated hardware-adaptive training pipeline, automatically provisioning 
    the Python virtual environment and installing dependencies from requirements.txt if needed.
.EXAMPLE
    .\train.ps1 -Profile smoke-test
    .\train.ps1 -Profile balanced
    .\train.ps1 -Profile production
    .\train.ps1 -Phases vae export -SliceLevel 2
    .\train.ps1 -Profile balanced -UseActiveLearning -ChunkCurriculum
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

# Setup Python Virtual Environment and Requirements
$VenvPath = Join-Path $ScriptDir ".venv"
$PythonExe = Join-Path $VenvPath "Scripts\python.exe"
$RequirementsPath = Join-Path $ScriptDir "requirements.txt"

if (-not (Test-Path $PythonExe)) {
    Write-Host "[*] Python virtual environment (.venv) not found. Creating one..." -ForegroundColor Yellow
    
    # Locate system python command
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

    Write-Host "[*] Upgrading pip inside virtual environment..." -ForegroundColor Yellow
    & $PythonExe -m pip install --upgrade pip

    if (Test-Path $RequirementsPath) {
        Write-Host "[*] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
        & $PythonExe -m pip install -r $RequirementsPath
    } else {
        Write-Warning "[!] requirements.txt not found in project root. Skipping dependency installation."
    }
} else {
    # Optional safety check: ensure requirements.txt changes are accounted for if needed
    if (Test-Path $RequirementsPath) {
        # Quick silent check or let pip handle caching/fulfillment
        & $PythonExe -m pip install -q -r $RequirementsPath
    }
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

# Execute the deep learning training lifecycle pass with UTF-8 encoding stream
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "[*] Launching RainAI Training Orchestrator [Profile: $Profile]..." -ForegroundColor Cyan
& $PythonExe $ArgsList
exit $LASTEXITCODE
