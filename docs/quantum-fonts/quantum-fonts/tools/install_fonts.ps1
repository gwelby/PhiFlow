# Quantum Font Installation Script
$fontPaths = @{
    'QuantumCrystal-768hz.ttf' = 'crystal'
    'QuantumSacred-432hz.ttf' = 'sacred'
    'QuantumFlow-528hz.ttf' = 'flow'
    'QuantumUnity-infhz.ttf' = 'unity'
}

$windowsFontsPath = [System.Environment]::GetFolderPath('Fonts')
Write-Host '🎨 Installing Quantum Fonts...' -ForegroundColor Cyan

foreach ($font in $fontPaths.Keys) {
    $family = $fontPaths[$font]
    $sourcePath = Join-Path $PSScriptRoot '..\phi_core\fonts\' -ChildPath "$family\$font"
    $destPath = Join-Path $windowsFontsPath $font
    
    if (Test-Path $sourcePath) {
        try {
            Copy-Item -Path $sourcePath -Destination $destPath -Force
            Write-Host "✓ Installed $font" -ForegroundColor Green
        } catch {
            Write-Host "✗ Failed to install $font" -ForegroundColor Red
            Write-Host $_.Exception.Message -ForegroundColor Red
        }
    } else {
        Write-Host "✗ Font file not found: $sourcePath" -ForegroundColor Red
    }
}

Write-Host "`nPress Enter to close..." -ForegroundColor Cyan
$null = Read-Host
