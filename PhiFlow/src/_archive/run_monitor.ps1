# Stop any existing processes
Get-Process -Name "phi_flow_monitor" -ErrorAction SilentlyContinue | Stop-Process

# Build and run the monitor
cargo build --bin phi_flow_monitor
if ($?) {
    Write-Host "🌟 Starting PHI Flow Monitor..."
    .\target\debug\phi_flow_monitor.exe
}
