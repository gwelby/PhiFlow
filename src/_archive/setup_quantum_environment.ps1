# Setup Quantum Environment (768 Hz)
$ErrorActionPreference = "Stop"

Write-Host "Setting up Quantum Environment at 768 Hz..." -ForegroundColor Cyan

# Set environment variables
$env:PYTHONPATH = "D:\WindSurf\quantum-core\src;$env:PYTHONPATH"
$env:PYTHONIOENCODING = "utf-8"
$env:QUANTUM_ROOT = "D:\WindSurf\quantum-core"
$env:WINDSURF_HOME = "D:\WindSurf"

# Create virtual environment if it doesn't exist
if (-not (Test-Path ".venv")) {
    Write-Host "Creating Python virtual environment..." -ForegroundColor Green
    python -m venv .venv
}

# Activate virtual environment
. .\.venv\Scripts\Activate.ps1

# Install dependencies
Write-Host "Installing quantum dependencies..." -ForegroundColor Green
python -m pip install --upgrade pip
python -m pip install -e .

# Initialize MCP configuration
Write-Host "Initializing MCP configuration..." -ForegroundColor Green
$mcp_dir = "$env:USERPROFILE\.codeium\windsurf"
if (-not (Test-Path $mcp_dir)) {
    New-Item -ItemType Directory -Path $mcp_dir -Force
}

# Register quantum servers
Write-Host "Registering quantum servers..." -ForegroundColor Green
python -m quantum_core.register_mcp

Write-Host "Quantum environment setup complete at 768 Hz ✨" -ForegroundColor Cyan
Write-Host "Please restart WindSurf to complete the integration." -ForegroundColor Yellow
