# PhiFlow Direction Field Merge - Master Script
# =============================================
# This script applies all patches to complete the PhiIR structure merge
# Run from: d:\Projects\PhiFlow-compiler\PhiFlow

$ErrorActionPreference = "Stop"
$patchesDir = "d:\Projects\PhiFlow\patches"
$baseDir = "d:\Projects\PhiFlow-compiler\PhiFlow"

Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║   PhiFlow Direction Field Merge - Patch Application   ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

Set-Location $baseDir
Write-Host "Working directory: $baseDir" -ForegroundColor Gray
Write-Host ""

# Try git apply first (cleaner)
Write-Host "[Method 1] Attempting git apply..." -ForegroundColor Yellow
$patchFiles = Get-ChildItem -Path $patchesDir -Filter "*.patch" | Sort-Object Name
$successCount = 0
$failCount = 0

foreach ($patch in $patchFiles) {
    Write-Host "  Applying $($patch.Name)..." -NoNewline
    try {
        git apply --ignore-whitespace $patch.FullName 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host " ✓" -ForegroundColor Green
            $successCount++
        } else {
            Write-Host " ⚠ (already applied or conflict)" -ForegroundColor Yellow
            $failCount++
        }
    } catch {
        Write-Host " ✗" -ForegroundColor Red
        $failCount++
    }
}

Write-Host ""
Write-Host "Git apply results: $successCount succeeded, $failCount failed/skipped" -ForegroundColor Cyan
Write-Host ""

# If git apply failed for some patches, fall back to PowerShell script
if ($failCount -gt 0) {
    Write-Host "[Method 2] Falling back to PowerShell fixes..." -ForegroundColor Yellow
    & "d:\Projects\PhiFlow\apply_direction_fixes.ps1"
}

# Verify compilation
Write-Host ""
Write-Host "[Verification] Building library..." -ForegroundColor Yellow
Write-Host ""
cargo build --lib 2>&1 | Select-String -Pattern "Compiling|Finished|error"

Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║                  Next Steps                            ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Run tests:" -ForegroundColor White
Write-Host "   cargo test --lib" -ForegroundColor Gray
Write-Host ""
Write-Host "2. If tests pass, commit the merge:" -ForegroundColor White
Write-Host "   git add -A" -ForegroundColor Gray
Write-Host "   git commit -m 'Merge: Add direction field to Resonate node'" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Push to origin:" -ForegroundColor White
Write-Host "   git push origin compiler" -ForegroundColor Gray
Write-Host ""
