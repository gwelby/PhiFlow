# Check Quantum Status (768 Hz)
$ErrorActionPreference = "Stop"

Write-Host "Checking Quantum Environment Status at 768 Hz..." -ForegroundColor Cyan

# Environment Check
Write-Host "`nEnvironment Variables:" -ForegroundColor Green
Write-Host "PYTHONPATH: $env:PYTHONPATH"
Write-Host "QUANTUM_ROOT: $env:QUANTUM_ROOT"
Write-Host "WINDSURF_HOME: $env:WINDSURF_HOME"

# Directory Check
$paths = @(
    "d:/WindSurf/quantum_core",
    "d:/WindSurf/quantum_core/src",
    "d:/WindSurf/hle",
    "d:/WindSurf/memories"
)

Write-Host "`nQuantum Paths:" -ForegroundColor Green
foreach ($path in $paths) {
    if (Test-Path $path) {
        Write-Host "✓ $path exists"
    } else {
        Write-Host "✗ $path missing" -ForegroundColor Red
    }
}

# MCP Config Check
$mcp_config = "$env:USERPROFILE\.codeium\windsurf\mcp_config.json"
Write-Host "`nMCP Configuration:" -ForegroundColor Green
if (Test-Path $mcp_config) {
    Write-Host "✓ MCP config exists at: $mcp_config"
    $config = Get-Content $mcp_config -Raw | ConvertFrom-Json
    Write-Host "Registered servers:"
    $config.mcpServers.PSObject.Properties | ForEach-Object {
        Write-Host "  - $($_.Name): $($_.Value.env.QUANTUM_FREQUENCY) Hz"
    }
} else {
    Write-Host "✗ MCP config missing at: $mcp_config" -ForegroundColor Red
}

# Server Status Check
Write-Host "`nServer Status:" -ForegroundColor Green
$servers = @(
    @{Name="Quantum Core"; Frequency=432},
    @{Name="PhiFlow"; Frequency=528},
    @{Name="Unity Field"; Frequency=768},
    @{Name="GregScript"; Frequency=768}
)

foreach ($server in $servers) {
    Write-Host "Checking $($server.Name) at $($server.Frequency) Hz..."
    python -c "from importlib.util import find_spec; print('✓' if find_spec('$($server.Name.ToLower().Replace(' ', ''))') else '✗')"
}

Write-Host "`nQuantum status check complete at 768 Hz ✨" -ForegroundColor Cyan
