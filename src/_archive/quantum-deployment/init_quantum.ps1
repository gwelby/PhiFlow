# Initialize Quantum Frequencies with Error Handling and Status Tracking
$ErrorActionPreference = "Stop"

# Function to log status with frequency-based formatting
function Write-QuantumStatus {
    param(
        [string]$Message,
        [string]$Status = "INFO",
        [int]$Frequency = 432
    )
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] [$Frequency Hz] [$Status] $Message"
    Write-Host $logMessage
    Add-Content -Path "$env:QUANTUM_HOME\quantum.log" -Value $logMessage
}

# Function to verify system dependencies
function Test-QuantumDependencies {
    Write-QuantumStatus "Checking system dependencies..." "INFO" 432

    # Check Docker
    try {
        $docker = Get-Command docker -ErrorAction Stop
        Write-QuantumStatus "Docker found at: $($docker.Source)" "SUCCESS" 528
    }
    catch {
        Write-QuantumStatus "Docker not found! Please install Docker Desktop for Windows" "ERROR" 432
        return $false
    }

    # Check CUDA
    if (Test-Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8") {
        Write-QuantumStatus "CUDA 12.8 found" "SUCCESS" 528
    }
    else {
        Write-QuantumStatus "CUDA 12.8 not found! Please install CUDA Toolkit" "ERROR" 432
        return $false
    }

    # Check Python
    try {
        $python = python --version
        Write-QuantumStatus "Python found: $python" "SUCCESS" 528
    }
    catch {
        Write-QuantumStatus "Python not found! Please install Python 3.11+" "ERROR" 432
        return $false
    }

    return $true
}

# Initialize environment
$env:QUANTUM_HOME = "D:\WindSurf\quantum-core"
Write-QuantumStatus "Initializing Quantum Core at $env:QUANTUM_HOME" "INFO" 432

# Create quantum directories if they don't exist
$quantumDirs = @(
    "quantum-phi",
    "quantum-omega",
    "quantum-infinity",
    "logs"
)

foreach ($dir in $quantumDirs) {
    $path = Join-Path $env:QUANTUM_HOME $dir
    if (-not (Test-Path $path)) {
        try {
            New-Item -ItemType Directory -Path $path -Force
            Write-QuantumStatus "Created quantum directory: $path" "SUCCESS" 528
        }
        catch {
            Write-QuantumStatus "Failed to create directory: $path - $_" "ERROR" 432
            exit 1
        }
    }
}

# Set environment variables
$env:GROUND_FREQ = "432.0"
$env:CREATE_FREQ = "528.0"
$env:UNITY_FREQ = "768.0"

Write-QuantumStatus "Setting quantum frequencies - Ground: $env:GROUND_FREQ Hz, Create: $env:CREATE_FREQ Hz, Unity: $env:UNITY_FREQ Hz" "INFO" 768

# Verify dependencies
if (-not (Test-QuantumDependencies)) {
    Write-QuantumStatus "System dependencies check failed. Please install required components." "ERROR" 432
    exit 1
}

# Start containers with error handling
Write-QuantumStatus "Starting quantum containers..." "INFO" 528
try {
    docker-compose up -d --build
    Write-QuantumStatus "Quantum containers started successfully" "SUCCESS" 768
}
catch {
    Write-QuantumStatus "Failed to start quantum containers: $_" "ERROR" 432
    exit 1
}

# Verify health endpoints
$healthEndpoints = @(
    @{Port = 432; Name = "Ground"},
    @{Port = 528; Name = "Create"},
    @{Port = 768; Name = "Unity"}
)

foreach ($endpoint in $healthEndpoints) {
    $url = "http://localhost:$($endpoint.Port)/health"
    try {
        $response = Invoke-WebRequest -Uri $url -TimeoutSec 5
        if ($response.StatusCode -eq 200) {
            Write-QuantumStatus "$($endpoint.Name) health check passed" "SUCCESS" $endpoint.Port
        }
    }
    catch {
        Write-QuantumStatus "$($endpoint.Name) health check failed: $_" "ERROR" $endpoint.Port
    }
}

Write-QuantumStatus "Quantum Core initialization complete" "SUCCESS" 768
