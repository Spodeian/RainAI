<#
.SYNOPSIS
    RainAI Automated Training, Export & Benchmark Orchestrator Launcher
.DESCRIPTION
    Runs the automated hardware-adaptive training pipeline using the local Python virtual environment.
.EXAMPLE
    .\train.ps1 -Profile smoke-test
    .\train.ps1 -Profile balanced
    .\train.ps1 -Profile production
    .\train.ps1 -Phases vae export -SliceLevel 2
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

# Locate localized Python execution environments safely
$PythonExe = Join-Path $ScriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    Write-Warning "[*] .venv\Scripts\python.exe not found. Using system python..."
    $PythonExe = "python"
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
if ($Fresh) { $ArgsList += "--fresh" }
if ($PrepareData) { $ArgsList += "--prepare-data" }
if ($RebuildData) { $ArgsList += "--rebuild-data" }
if ($BenchmarkIters) { $ArgsList += @("--benchmark-iters", $BenchmarkIters) }
if ($LogDir) { $ArgsList += @("--log-dir", $LogDir) }
if ($LogFile) { $ArgsList += @("--log-file", $LogFile) }
if ($Verbose) { $ArgsList += "--verbose" }
if ($Quiet) { $ArgsList += "--quiet" }

# Execute the deep learning training lifecycle pass
& $PythonExe $ArgsList
exit $LASTEXITCODE
