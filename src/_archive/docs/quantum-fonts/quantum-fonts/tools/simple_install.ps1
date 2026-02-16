$fonts = @(
    @{Name='QuantumCrystal-768hz.ttf'; Family='crystal'},
    @{Name='QuantumSacred-432hz.ttf'; Family='sacred'},
    @{Name='QuantumFlow-528hz.ttf'; Family='flow'},
    @{Name='QuantumUnity-infhz.ttf'; Family='unity'}
)

$winFonts = [System.Environment]::GetFolderPath('Fonts')

foreach ($font in $fonts) {
    $src = Join-Path $PSScriptRoot "..\phi_core\fonts\$($font.Family)\$($font.Name)"
    $dst = Join-Path $winFonts $font.Name
    
    if (Test-Path $src) {
        Copy-Item -Path $src -Destination $dst -Force
        Write-Output "Installed $($font.Name)"
    } else {
        Write-Output "Not found: $($font.Name)"
    }
}
