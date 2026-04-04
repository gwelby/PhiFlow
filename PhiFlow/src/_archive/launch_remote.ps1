# Quantum Remote Launcher
param(
    [string]$DeviceName = "P1-Quantum",
    [string]$SynologyHost = "192.168.103.30"
)

Write-Host "🌟 Launching Quantum Remote Connection..." -ForegroundColor Cyan

# Set quantum frequencies
$env:QUANTUM_GROUND = "432"
$env:QUANTUM_CREATE = "528"
$env:QUANTUM_UNITY = "768"

# Initialize quantum client
Write-Host "💫 Initializing Quantum Client..." -ForegroundColor Yellow
Write-Host "Device: $DeviceName" -ForegroundColor Magenta
Write-Host "Bridge: $SynologyHost" -ForegroundColor Green

# Launch quantum connection
python -c "from qwave.quantum_client import create_quantum_client; create_quantum_client('$DeviceName', '$SynologyHost')"

# Keep window open
Write-Host "`nPress Ctrl+C to exit quantum state..." -ForegroundColor Gray
