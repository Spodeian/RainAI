#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Builds the RainAI application as an Android shared library (.so) via cargo-ndk.
.DESCRIPTION
    Compiles for Android target architectures (arm64-v8a, armeabi-v7a, x86_64)
    using cargo-ndk and places output libraries in the target directory.
.PARAMETER OutputDir
    Destination directory for compiled jniLibs. Defaults to "target/android/jniLibs".
.EXAMPLE
    ./scripts/build-android.ps1
#>
param(
    [string]$OutputDir = "target/android/jniLibs"
)

$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host " Building RainAI for Android via cargo-ndk"
Write-Host "============================================================"

# Ensure cargo-ndk is installed
if (-not (Get-Command cargo-ndk -ErrorAction SilentlyContinue)) {
    Write-Warning "cargo-ndk is not detected in PATH. Installing via 'cargo install cargo-ndk'..."
    cargo install cargo-ndk
}

# Verify ANDROID_NDK_HOME or NDK_HOME
if (-not $env:ANDROID_NDK_HOME -and -not $env:NDK_HOME) {
    Write-Warning "Neither ANDROID_NDK_HOME nor NDK_HOME is set."
    Write-Warning "Ensure Android NDK is installed and ANDROID_NDK_HOME points to your NDK path (e.g. C:\Users\<User>\AppData\Local\Android\Sdk\ndk\<version>)."
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

Write-Host "[*] Compiling crates/app for Android architectures..."
cargo ndk -t arm64-v8a -t armeabi-v7a -t x86_64 -o $OutputDir build --package app --release

Write-Host "[+] Android build complete! Libraries emitted to: $OutputDir"
