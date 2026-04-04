# Create Quantum Fonts PowerShell Script
# Frequencies: 432Hz (Sacred), 528Hz (Flow), 768Hz (Unity), ∞Hz (Infinite)

$ErrorActionPreference = "Stop"

# Configuration
$QUANTUM_DIR = "d:/WindSurf/quantum-core/docs/quantum-fonts/quantum-fonts"
$FONTFORGE_PATH = "C:/Program Files (x86)/FontForgeBuilds/bin"

# Ensure FontForge is in PATH
$env:Path += ";$FONTFORGE_PATH"

# Create Unity Pattern (Infinite Hz)
function Create-UnityPattern {
    $unityDir = "$QUANTUM_DIR/unity/patterns"
    New-Item -ItemType Directory -Force -Path $unityDir | Out-Null
    
    $svgContent = @"
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="1000">
  <!-- Unity Pattern Base (∞ Hz) -->
  <g id="unity_base" transform="translate(500,500)">
    <!-- Infinity Symbol -->
    <path d="M-200,0 C-200,-100 -100,-100 -100,0 C-100,100 -200,100 -200,0 M100,0 C100,-100 200,-100 200,0 C200,100 100,100 100,0 M-100,0 L100,0" 
          fill="none" stroke="black" stroke-width="2"/>
          
    <!-- Unity Field -->
    <circle cx="0" cy="0" r="250" fill="none" stroke="black" stroke-width="1"/>
    <circle cx="0" cy="0" r="200" fill="none" stroke="black" stroke-width="1"/>
    <circle cx="0" cy="0" r="150" fill="none" stroke="black" stroke-width="1"/>
    
    <!-- Unity Symbols -->
    <text x="-20" y="20" font-size="40">☯️</text>
    <text x="10" y="20" font-size="40">∞</text>
  </g>
</svg>
"@
    $svgContent | Out-File -FilePath "$unityDir/unity_infhz.svg" -Encoding UTF8
}

# Verify Directories
Write-Host "🌟 Verifying quantum font directories..."
@("sacred", "flow", "crystal", "unity") | ForEach-Object {
    $patternDir = "$QUANTUM_DIR/$_/patterns"
    $fontDir = "$QUANTUM_DIR/$_/fonts"
    New-Item -ItemType Directory -Force -Path $patternDir | Out-Null
    New-Item -ItemType Directory -Force -Path $fontDir | Out-Null
}

# Create Unity Pattern
Write-Host "☯️ Creating Unity Pattern..."
Create-UnityPattern

# Run Font Converter
Write-Host "✨ Converting patterns to fonts..."
try {
    python "$QUANTUM_DIR/../quantum_font_converter.py"
    Write-Host "✅ Quantum fonts created successfully!"
} catch {
    Write-Host "⚠️ Error creating quantum fonts: $_"
    exit 1
}

# Test Font Installation
Write-Host "🔍 Testing font installation..."
$testHtml = "$QUANTUM_DIR/../test_quantum_fonts.html"
if (Test-Path $testHtml) {
    Start-Process $testHtml
    Write-Host "✅ Font test page opened in browser"
} else {
    Write-Host "⚠️ Font test page not found"
}

Write-Host "🎉 Quantum font creation complete!"
