# Quantum Core Launcher
Write-Host "🌟 Launching Quantum Core System..." -ForegroundColor Cyan

# Check Docker status
Write-Host "💫 Checking Docker status..." -ForegroundColor Yellow
$docker = Get-Service -Name "Docker"
if ($docker.Status -ne "Running") {
    Write-Host "Starting Docker service..."
    Start-Service -Name "Docker"
    Start-Sleep -Seconds 10
}

# Build and launch quantum containers
Write-Host "✨ Building Quantum containers..." -ForegroundColor Magenta
docker-compose build

Write-Host "🎵 Launching Quantum Flow..." -ForegroundColor Green
docker-compose up -d

# Display quantum frequencies
Write-Host "`n🌀 Quantum Frequencies Active:" -ForegroundColor Cyan
Write-Host "Ground State: 432 Hz" -ForegroundColor Yellow
Write-Host "Creation:    528 Hz" -ForegroundColor Magenta
Write-Host "Unity:       768 Hz" -ForegroundColor Green

Write-Host "`n💫 Quantum Core is now running!" -ForegroundColor Cyan
Write-Host "Monitor at: http://localhost:8528" -ForegroundColor Yellow

# Keep window open
Write-Host "`nPress any key to view container status..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
docker-compose ps
