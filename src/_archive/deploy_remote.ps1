# Remote Deployment Script for Quantum Core
param(
    [Parameter(Mandatory=$true)]
    [string]$SshKeyPath,
    [string]$InternalHost = "192.168.100.32",
    [string]$Username = "admin",
    [string]$PackagePath = "quantum-core-*.tar.gz"
)

$ErrorActionPreference = "Stop"

# Verify SSH key exists
if (-not (Test-Path $SshKeyPath)) {
    Write-Error "SSH key not found at: $SshKeyPath"
    exit 1
}

Write-Host "Starting Quantum Remote Deployment (768 Hz)" -ForegroundColor Cyan

# Find latest package
$package = Get-ChildItem $PackagePath | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $package) {
    Write-Error "No deployment package found matching: $PackagePath"
    exit 1
}

Write-Host "Using package: $($package.Name)" -ForegroundColor Yellow

# Test SSH connection
Write-Host "Testing SSH connection..." -ForegroundColor Yellow
$testSsh = ssh -i $SshKeyPath -o BatchMode=yes -o ConnectTimeout=5 "${Username}@${InternalHost}" "echo 'Connection successful'"
if ($LASTEXITCODE -ne 0) {
    Write-Error "SSH connection failed. Please check your SSH key and connection settings."
    exit 1
}

# Create remote directories
Write-Host "Creating remote directories..." -ForegroundColor Yellow
ssh -i $SshKeyPath "${Username}@${InternalHost}" "mkdir -p /volume1/docker /volume1/quantum-data/{patterns,media,config,certs}"

# Copy package
Write-Host "Copying quantum package..." -ForegroundColor Yellow
scp -i $SshKeyPath $package "${Username}@${InternalHost}:/volume1/docker/"

# Deploy
Write-Host "Deploying quantum services..." -ForegroundColor Cyan
$deployCommands = @"
cd /volume1/docker && \
tar xzf $($package.Name) && \
bash deploy.sh
"@

ssh -i $SshKeyPath "${Username}@${InternalHost}" $deployCommands

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nQuantum deployment successful!" -ForegroundColor Green
    Write-Host "Services will be available at:"
    Write-Host "- External: https://demoguru.networkinggurus.com:31001" -ForegroundColor Yellow
    Write-Host "- Monitor: https://demoguru.networkinggurus.com:8528" -ForegroundColor Yellow
    
    # Check container status
    Write-Host "`nChecking container status..." -ForegroundColor Yellow
    ssh -i $SshKeyPath "${Username}@${InternalHost}" "docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'"
} else {
    Write-Error "Deployment failed. Please check the logs on VirtualDSM."
    exit 1
}
