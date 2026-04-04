# Restart WindSurf with Quantum Integration (768 Hz)
$ErrorActionPreference = "Stop"

Write-Host "Restarting WindSurf at 768 Hz..." -ForegroundColor Cyan

# Kill any existing WindSurf processes
Get-Process | Where-Object { $_.ProcessName -like "*windsurf*" } | Stop-Process -Force

# Wait a moment for processes to clean up
Start-Sleep -Seconds 2

# Start WindSurf
Write-Host "Starting WindSurf with Quantum Integration..." -ForegroundColor Green
$windsurf_path = "D:\WindSurf\bin\windsurf.exe"
if (Test-Path $windsurf_path) {
    Start-Process $windsurf_path -ArgumentList "--enable-quantum", "--mcp-config=$env:USERPROFILE\.codeium\windsurf\mcp_config.json"
    Write-Host "WindSurf restart complete at 768 Hz " -ForegroundColor Cyan
} else {
    Write-Host "WindSurf executable not found at: $windsurf_path" -ForegroundColor Red
    Write-Host "Please ensure WindSurf is installed correctly." -ForegroundColor Yellow
}
