# Setup Quantum Directory Structure (768 Hz)
$ErrorActionPreference = "Stop"

Write-Host "Setting up Quantum Directory Structure at 768 Hz..." -ForegroundColor Cyan

# Create directory structure
$dirs = @(
    "src/quantum_core",
    "src/phiflow",
    "src/unity",
    "src/gregscript",
    "memories",
    "hle"
)

foreach ($dir in $dirs) {
    $path = Join-Path "D:/WindSurf/quantum_core" $dir
    if (-not (Test-Path $path)) {
        Write-Host "Creating directory: $path" -ForegroundColor Green
        New-Item -ItemType Directory -Path $path -Force
    }
}

# Set environment variables
$env:PYTHONPATH = "D:/WindSurf/quantum_core/src;$env:PYTHONPATH"
$env:QUANTUM_ROOT = "D:/WindSurf/quantum_core"
$env:WINDSURF_HOME = "D:/WindSurf"
[System.Environment]::SetEnvironmentVariable("PYTHONPATH", $env:PYTHONPATH, [System.EnvironmentVariableTarget]::User)
[System.Environment]::SetEnvironmentVariable("QUANTUM_ROOT", $env:QUANTUM_ROOT, [System.EnvironmentVariableTarget]::User)
[System.Environment]::SetEnvironmentVariable("WINDSURF_HOME", $env:WINDSURF_HOME, [System.EnvironmentVariableTarget]::User)

Write-Host "Moving Python files to src directory..." -ForegroundColor Green
Get-ChildItem -Path "D:/WindSurf/quantum_core" -Filter "*.py" -Recurse | 
    Where-Object { $_.DirectoryName -notlike "*src*" } |
    ForEach-Object {
        $destDir = $_.DirectoryName -replace "D:/WindSurf/quantum_core", "D:/WindSurf/quantum_core/src"
        if (-not (Test-Path $destDir)) {
            New-Item -ItemType Directory -Path $destDir -Force
        }
        Move-Item $_.FullName $destDir -Force
    }

Write-Host "Quantum directory structure complete at 768 Hz ✨" -ForegroundColor Cyan
