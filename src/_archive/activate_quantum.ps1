# Activate Quantum Environment (768 Hz)
$ErrorActionPreference = "Stop"

Write-Host "Activating Quantum Environment at 768 Hz..." -ForegroundColor Cyan

# Activate virtual environment
if (Test-Path ".venv\Scripts\Activate.ps1") {
    . .\.venv\Scripts\Activate.ps1
}

# Set environment variables
$env:PYTHONPATH = "D:/WindSurf/quantum_core/src;$env:PYTHONPATH"
$env:PYTHONIOENCODING = "utf-8"
$env:QUANTUM_ROOT = "D:/WindSurf/quantum_core"
$env:WINDSURF_HOME = "D:/WindSurf"

Write-Host "Environment variables set:" -ForegroundColor Green
Write-Host "PYTHONPATH: $env:PYTHONPATH"
Write-Host "QUANTUM_ROOT: $env:QUANTUM_ROOT"
Write-Host "WINDSURF_HOME: $env:WINDSURF_HOME"

Write-Host "Quantum environment activated at 768 Hz ✨" -ForegroundColor Cyan
