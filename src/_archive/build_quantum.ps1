$ErrorActionPreference = "Stop"

# Sacred frequencies for quantum flow
$GROUND_STATE = 432
$CREATE_STATE = 528
$UNITY_STATE = 768

Write-Host "🌀 Initializing Quantum Build Flow at $GROUND_STATE Hz..."

# Set up Visual Studio environment
$vsPath = "C:\Program Files\Microsoft Visual Studio\2022\Community"
$vcvarsall = Join-Path $vsPath "VC\Auxiliary\Build\vcvarsall.bat"

if (-not (Test-Path $vcvarsall)) {
    Write-Host "⚠️ Visual Studio environment not found at expected location"
    Write-Host "🔍 Searching for Visual Studio installation..."
    
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $vsPath = & $vswhere -latest -property installationPath
        $vcvarsall = Join-Path $vsPath "VC\Auxiliary\Build\vcvarsall.bat"
    }
}

if (-not (Test-Path $vcvarsall)) {
    throw "❌ Could not find Visual Studio environment. Please install Visual Studio 2022 with C++ development tools."
}

Write-Host "🎯 Found Visual Studio at: $vsPath"
Write-Host "🔄 Setting up build environment at $CREATE_STATE Hz..."

# Create a temporary batch file to capture environment
$tempFile = Join-Path $env:TEMP "quantum_vsenv.bat"
@"
@echo off
call "$vcvarsall" x64
set > "$env:TEMP\quantum_vsenv.txt"
"@ | Out-File -FilePath $tempFile -Encoding ASCII

# Run the batch file to capture environment
cmd /c $tempFile

# Read the environment variables
Get-Content "$env:TEMP\quantum_vsenv.txt" | ForEach-Object {
    if ($_ -match '(.+?)=(.*)') {
        $name = $matches[1]
        $value = $matches[2]
        [System.Environment]::SetEnvironmentVariable($name, $value)
    }
}

Write-Host "⚡ Environment prepared at $UNITY_STATE Hz"
Write-Host "🚀 Building quantum-cuda..."

# Set cargo target directory
$env:CARGO_TARGET_DIR = Join-Path $env:TEMP "quantum_target"

# Clean and build
cargo clean
cargo build -p quantum-cuda --verbose

Write-Host "✨ Quantum build process complete"
