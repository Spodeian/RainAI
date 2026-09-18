<#
.SYNOPSIS
    RainAI Automated Training, Export & Benchmark Orchestrator Launcher
.DESCRIPTION
    Runs the automated hardware-adaptive training pipeline with version-aware 
    PyTorch backend selection targeting CUDA/ROCm wheels on Windows & Linux.
    Automatically provisions Python environments, compiles Native Rust binaries,
    and continuously synchronizes raw/synthetic datasets prior to execution.
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
    [switch]$NoAmp,
    [switch]$UseDisc,
    [switch]$ChunkCurriculum,
    [switch]$Fresh,
    [switch]$PrepareData,
    [switch]$RebuildData,
    [int]$BenchmarkIters,
    [string]$LogDir,
    [string]$LogFile,
    [switch]$RustEngine,
    [switch]$Verbose,
    [switch]$Quiet
)

$ErrorActionPreference = "Stop"

# 0. Global UTF-8 Encoding, Code Page & Console Initialization
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8

# Force Windows Console Code Page to 65001 (UTF-8) for all child processes
if ($IsWindows -or $env:OS -match "Windows") {
    chcp 65001 | Out-Null
    
    # Enable Virtual Terminal Processing (ANSI color escape support in Windows Console Host)
    try {
        $VTCode = @"
using System;
using System.Runtime.InteropServices;
public class VTConsole {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern IntPtr GetStdHandle(int nStdHandle);
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool GetConsoleMode(IntPtr hConsoleHandle, out uint lpMode);
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool SetConsoleMode(IntPtr hConsoleHandle, uint dwMode);
    public static void EnableVT() {
        IntPtr hOut = GetStdHandle(-11);
        if (GetConsoleMode(hOut, out uint mode)) {
            SetConsoleMode(hOut, mode | 0x0004);
        }
    }
}
"@
        if (-not ([System.Management.Automation.PSTypeName]"VTConsole").Type) {
            Add-Type -TypeDefinition $VTCode
        }
        [VTConsole]::EnableVT()
    } catch {}
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONLEGACYWINDOWSSTDIO = "0"
$env:TERM = "xterm-256color"
$env:COLORTERM = "truecolor"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Setup Python Virtual Environment and Paths
$VenvPath = Join-Path $ScriptDir ".venv"
$PythonExe = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $PythonExe)) { $PythonExe = Join-Path $VenvPath "bin/python" }
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

# 2. Python Version & Hardware Architecture Probing
Write-Host "[*] Probing host Python environment and hardware accelerator..." -ForegroundColor Cyan

$PyVersionInfo = (& $PythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
Write-Host "[+] Active Python Version: $PyVersionInfo" -ForegroundColor Cyan

$GpuType = "None"
$TorchIndexUrl = "https://download.pytorch.org/whl/cu132"

try {
    if (Get-Command "nvidia-smi" -ErrorAction SilentlyContinue) {
        $NvidiaSmiOutput = (& nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1).Trim()
        if ($LASTEXITCODE -eq 0 -and $NvidiaSmiOutput) {
            $GpuType = "NVIDIA"
            Write-Host "[+] NVIDIA CUDA GPU detected (Driver: $NvidiaSmiOutput). Target Index: $TorchIndexUrl" -ForegroundColor Green
        }
    }
    if ($GpuType -eq "None" -and (Get-Command "rocm-smi" -ErrorAction SilentlyContinue)) {
        $GpuType = "AMD"
        $TorchIndexUrl = "https://download.pytorch.org/whl/rocm7.14"
        Write-Host "[+] AMD ROCm GPU detected. Target Index: $TorchIndexUrl" -ForegroundColor Green
    }
} catch {
    Write-Warning "[!] Hardware accelerator probing unhandled. Defaulting to CPU backend index."
    $TorchIndexUrl = "https://download.pytorch.org/whl/cpu"
}

# 3. Streamlined Dependency Synchronization (Python)
Write-Host "[*] Synchronizing Python dependencies..." -ForegroundColor Cyan
& $PythonExe -m pip install --upgrade --disable-pip-version-check --progress-bar off --quiet pip setuptools wheel

if (Test-Path $RequirementsPath) {
    & $PythonExe -m pip install -r $RequirementsPath --extra-index-url $TorchIndexUrl --disable-pip-version-check --progress-bar off --quiet --no-warn-script-location
} else {
    Write-Warning "[!] requirements.txt not found in project root."
}

# 4. Native Rust Toolchain Compilation ("+ others")
Write-Host "[*] Verifying native Rust data pipeline binaries..." -ForegroundColor Cyan
if (Get-Command "cargo" -ErrorAction SilentlyContinue) {
    $CargoToml = Join-Path $ScriptDir "crates\utilities\Cargo.toml"
    if (Test-Path $CargoToml) {
        Write-Host "    -> Compiling utilities crate (Ingest, Synth, Upmix, Features, Studio)..." -ForegroundColor DarkGray
        & cargo build --release --manifest-path $CargoToml --quiet
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "[!] Cargo build failed. Ensure Rust is up to date."
        } else {
            Write-Host "    -> Native binaries successfully compiled." -ForegroundColor Green
        }
    }
} else {
    Write-Warning "[!] Cargo not found in PATH. Native Rust binaries cannot be compiled."
}

# 5. Automated Data Synchronization & Health Watchdog
# By enabling data preparation by default, the pipeline will instantly fetch any new 
# dataset URLs added to ingest.rs, synthesize missing textures, and refresh the manifest.
if (-not $PrepareData -and -not $RebuildData) {
    Write-Host "[*] Engaging automated data synchronization (fetching new datasets)..." -ForegroundColor Yellow
    $PrepareData = $true
}

# 6. Orchestrate Training Run
$AutoTrainScript = Join-Path $ScriptDir "src\training\auto_train.py"
$ArgsList = @("-X", "utf8", $AutoTrainScript, "--profile", $Profile)

if ($Phases) {
    $ArgsList += "--phases"
    foreach ($phase in ($Phases -split ',')) {
        if ($phase.Trim()) {
            $ArgsList += $phase.Trim()
        }
    }
}
if ($PSBoundParameters.ContainsKey('EpochsVae')) { $ArgsList += @("--epochs-vae", $EpochsVae) }
if ($PSBoundParameters.ContainsKey('EpochsMamba')) { $ArgsList += @("--epochs-mamba", $EpochsMamba) }
if ($PSBoundParameters.ContainsKey('BatchSize')) { $ArgsList += @("--batch-size", $BatchSize) }
if ($PSBoundParameters.ContainsKey('MaxBatches')) { $ArgsList += @("--max-batches", $MaxBatches) }
if ($PSBoundParameters.ContainsKey('AccumulationSteps')) { $ArgsList += @("--accumulation-steps", $AccumulationSteps) }
if ($Device) { $ArgsList += @("--device", $Device) }
if ($NoAmp) { $ArgsList += "--no-amp" }
if ($UseDisc) { $ArgsList += "--use-disc" }
if ($ChunkCurriculum) { $ArgsList += "--chunk-curriculum" }
if ($Fresh) { $ArgsList += "--fresh" }
if ($PrepareData) { $ArgsList += "--prepare-data" }
if ($RebuildData) { $ArgsList += "--rebuild-data" }
if ($BenchmarkIters) { $ArgsList += @("--benchmark-iters", $BenchmarkIters) }
if ($LogDir) { $ArgsList += @("--log-dir", $LogDir) }
if ($LogFile) { $ArgsList += @("--log-file", $LogFile) }
if ($RustEngine) { $ArgsList += "--rust-engine" }
if ($Verbose) { $ArgsList += "--verbose" }
if ($Quiet) { $ArgsList += "--quiet" }

Write-Host "[*] Launching RainAI Training Orchestrator [Profile: $Profile]..." -ForegroundColor Cyan
& $PythonExe $ArgsList
exit $LASTEXITCODE