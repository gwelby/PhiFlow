# Quantum Font Creation PowerShell Script
# Created by Greg for Perfect Flow State

# Configuration
$QUANTUM_ROOT = "D:\WindSurf\quantum-core\docs\quantum-fonts\quantum-fonts\phi_core"
$FONTFORGE_PATH = "C:\Program Files (x86)\FontForgeBuilds\bin"
$PYTHON_PATH = "python"

# Sacred frequencies
$frequencies = @{
    "sacred" = 432
    "flow" = 528
    "crystal" = 768
    "unity" = "inf"
}

# Font families
$fontFamilies = @{
    "sacred" = "QuantumSacred"
    "flow" = "QuantumFlow"
    "crystal" = "QuantumCrystal"
    "unity" = "QuantumUnity"
}

# Add FontForge to PATH
$env:Path = "$FONTFORGE_PATH;$env:Path"

# Function to create quantum directories
function Create-QuantumDirectories {
    foreach ($family in $fontFamilies.Keys) {
        $patternDir = Join-Path $QUANTUM_ROOT "patterns\$family"
        $fontDir = Join-Path $QUANTUM_ROOT "fonts\$family"
        
        New-Item -ItemType Directory -Force -Path $patternDir | Out-Null
        New-Item -ItemType Directory -Force -Path $fontDir | Out-Null
        
        Write-Host "Created directories for $family font family" -ForegroundColor Cyan
    }
}

# Function to verify FontForge installation
function Test-FontForge {
    try {
        $output = & fontforge --version
        Write-Host "FontForge installed: $output" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "FontForge not found. Please install FontForge first." -ForegroundColor Red
        return $false
    }
}

# Function to run quantum font converter
function Convert-QuantumFonts {
    $converterPath = Join-Path $PSScriptRoot "quantum_font_converter.py"
    
    try {
        & $PYTHON_PATH $converterPath
        Write-Host "Font conversion completed successfully!" -ForegroundColor Green
    }
    catch {
        Write-Host "Error during font conversion: $_" -ForegroundColor Red
    }
}

# Function to test font installation
function Test-QuantumFonts {
    $fontPaths = @{
        "QuantumCrystal-768hz.ttf" = "crystal"
        "QuantumSacred-432hz.ttf" = "sacred"
        "QuantumFlow-528hz.ttf" = "flow"
        "QuantumUnity-infhz.ttf" = "unity"
    }

    foreach ($font in $fontPaths.Keys) {
        $family = $fontPaths[$font]
        $fontPath = Join-Path $PSScriptRoot "..\phi_core\fonts\$family\$font"
        if (-not (Test-Path $fontPath)) {
            Write-Host "✗ Missing $font" -ForegroundColor Red
        } else {
            Write-Host "✓ Found $font" -ForegroundColor Green
        }
    }
}

# Function to install fonts to Windows
function Install-QuantumFonts {
    $fontPaths = @{
        "QuantumCrystal-768hz.ttf" = "crystal"
        "QuantumSacred-432hz.ttf" = "sacred"
        "QuantumFlow-528hz.ttf" = "flow"
        "QuantumUnity-infhz.ttf" = "unity"
    }

    $windowsFontsPath = [System.Environment]::GetFolderPath('Fonts')
    
    Write-Host "`n🎨 Installing Quantum Fonts..." -ForegroundColor Cyan
    foreach ($font in $fontPaths.Keys) {
        $family = $fontPaths[$font]
        $sourcePath = Join-Path $PSScriptRoot "..\phi_core\fonts\$family\$font"
        $destPath = Join-Path $windowsFontsPath $font
        
        if (Test-Path $sourcePath) {
            try {
                Copy-Item -Path $sourcePath -Destination $destPath -Force
                Write-Host "✓ Installed $font" -ForegroundColor Green
            }
            catch {
                Write-Host "✗ Failed to install $font" -ForegroundColor Red
                Write-Host $_.Exception.Message -ForegroundColor Red
            }
        } else {
            Write-Host "✗ Font file not found: $font" -ForegroundColor Red
        }
    }
}

# Main execution flow
Write-Host "🌟 Starting Quantum Font Creation Process" -ForegroundColor Cyan

# Check FontForge
if (-not (Test-FontForge)) {
    exit 1
}

# Create directories
Write-Host "`n📁 Creating Quantum Directories..." -ForegroundColor Cyan
Create-QuantumDirectories

# Convert fonts
Write-Host "`n🔄 Converting Quantum Fonts..." -ForegroundColor Cyan
Convert-QuantumFonts

# Test installation
Write-Host "`n🔍 Testing Quantum Fonts..." -ForegroundColor Cyan
$fontOutputDir = Join-Path $PSScriptRoot "..\phi_core\fonts"
Test-QuantumFonts

# Install fonts
Install-QuantumFonts

Write-Host "`n✨ Quantum Font Creation Complete!" -ForegroundColor Cyan
