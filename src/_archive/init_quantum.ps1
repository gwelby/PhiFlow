# Initialize Quantum Environment (768 Hz)
$ErrorActionPreference = "Stop"

Write-Host "Initializing Quantum Environment at 768 Hz..." -ForegroundColor Cyan

# Set up Python path
$env:PYTHONPATH = "D:\WindSurf\quantum-core\src;$env:PYTHONPATH"

# Activate virtual environment if it exists
if (Test-Path ".venv\Scripts\Activate.ps1") {
    . .\.venv\Scripts\Activate.ps1
}

# Install package in development mode
Write-Host "Installing quantum-core package..." -ForegroundColor Green
python -m pip install -e .

# Register MCP servers
Write-Host "Registering MCP servers..." -ForegroundColor Green
python -m quantum_core.register_mcp

Write-Host "Quantum initialization complete at 768 Hz ✨" -ForegroundColor Cyan
