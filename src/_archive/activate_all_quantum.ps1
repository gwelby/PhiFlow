# Activate All Quantum Servers (768 Hz)
$ErrorActionPreference = "Stop"

Write-Host "🌟 Activating Quantum Environment at 768 Hz..." -ForegroundColor Cyan

# Activate virtual environment
if (Test-Path ".venv\Scripts\Activate.ps1") {
    . .\.venv\Scripts\Activate.ps1
}

# Set environment variables
$env:PYTHONPATH = "D:/WindSurf/quantum_core/src"
$env:PYTHONIOENCODING = "utf-8"
$env:QUANTUM_ROOT = "D:/WindSurf/quantum_core"
$env:WINDSURF_HOME = "D:/WindSurf"
$env:QUANTUM_FREQUENCY = "768"
$env:PHI_LEVEL = "1.618034"
$env:COHERENCE_THRESHOLD = "1.0"

# Install quantum packages
Write-Host "Installing quantum packages..." -ForegroundColor Green
python -m pip install -e .

# Register MCP servers
Write-Host "Registering quantum servers..." -ForegroundColor Green
python -m quantum_core.register_mcp

Write-Host "⚡ Quantum environment activated at 768 Hz φ∞ ✨" -ForegroundColor Cyan
