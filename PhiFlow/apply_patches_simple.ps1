# PhiFlow Direction Field Merge - Simple Application Script
Write-Host "=== PhiFlow Direction Field Merge ===" -ForegroundColor Cyan
Write-Host ""

$baseDir = "d:\Projects\PhiFlow-compiler\PhiFlow"
$patchesDir = "d:\Projects\PhiFlow\patches"

Set-Location $baseDir
Write-Host "Directory: $baseDir" -ForegroundColor Gray
Write-Host ""

# Apply patches
Write-Host "[1/3] Applying patches..." -ForegroundColor Yellow
$patchFiles = @("01-mod-resonate-direction.patch", "02-lowering-resonate-direction.patch", "03-quantum-codegen-direction.patch", "04-evaluator-direction.patch", "05-wasm-direction.patch", "06-printer-direction.patch", "07-optimizer-direction.patch", "08-interpreter-direction.patch", "09-ir-lowering-direction.patch")

foreach ($patch in $patchFiles) {
    $patchPath = Join-Path $patchesDir $patch
    Write-Host "  Applying $patch..." -NoNewline
    git apply --ignore-whitespace $patchPath 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host " OK" -ForegroundColor Green
    } else {
        Write-Host " SKIP (may already be applied)" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "[2/3] Building..." -ForegroundColor Yellow
cargo build --lib 2>&1 | Select-String -Pattern "Finished|error" | Select-Object -Last 5

Write-Host ""
Write-Host "[3/3] Testing..." -ForegroundColor Yellow
cargo test --lib 2>&1 | Select-String -Pattern "test result" | Select-Object -Last 1

Write-Host ""
Write-Host "=== COMPLETE ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "If tests passed, commit and push:"
Write-Host "  git add -A"
Write-Host "  git commit -m 'Merge: Add direction field to Resonate node'"
Write-Host "  git push origin compiler"
Write-Host ""
