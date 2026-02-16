# Setup IBM Quantum Environment for PhiFlow
# Run this script to load your IBM Quantum token

Write-Host "🌀 Setting up IBM Quantum environment for PhiFlow..." -ForegroundColor Cyan

# Load environment variables from config file
if (Test-Path "ibm_quantum_config.env") {
    Write-Host "📝 Loading IBM Quantum configuration..." -ForegroundColor Green
    
    Get-Content "ibm_quantum_config.env" | ForEach-Object {
        if ($_ -match "^([^#].*)=(.*)") {
            $name = $matches[1]
            $value = $matches[2]
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
            Write-Host "✅ Set $name" -ForegroundColor Green
        }
    }
    
    # Verify token is loaded
    $token = $env:IBM_QUANTUM_TOKEN
    if ($token) {
        $tokenPreview = $token.Substring(0, [Math]::Min(20, $token.Length)) + "..."
        Write-Host "🎯 IBM Quantum token loaded: $tokenPreview" -ForegroundColor Yellow
        Write-Host "⚛️ PhiFlow is ready for IBM Quantum integration!" -ForegroundColor Magenta
    } else {
        Write-Host "❌ IBM Quantum token not found in environment" -ForegroundColor Red
        Write-Host "📝 Please edit ibm_quantum_config.env and add your token" -ForegroundColor Yellow
    }
} else {
    Write-Host "❌ ibm_quantum_config.env not found" -ForegroundColor Red
    Write-Host "📝 Please create the configuration file first" -ForegroundColor Yellow
}

Write-Host "`n🚀 Environment setup complete!" -ForegroundColor Cyan