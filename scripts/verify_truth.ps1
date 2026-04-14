# PhiFlow Truth Verification Gate
# Runs all canonical truth tests to ensure zero regressions across backends.

Write-Host "🌌 Starting PhiFlow Truth Verification Gate..." -ForegroundColor Cyan

$Failed = $false

# 1. OpenQASM Generation
Write-Host "`n[1/4] Checking OpenQASM 3.0 Generation..." -ForegroundColor Yellow
cargo test --lib openqasm
if ($LASTEXITCODE -ne 0) { $Failed = $true }

# 2. Golden Integration Tests
Write-Host "`n[2/4] Running Golden Integration Tests..." -ForegroundColor Yellow
cargo test --quiet --test golden_integration_tests
if ($LASTEXITCODE -ne 0) { $Failed = $true }

# 3. Reproduction Bug Tests
Write-Host "`n[3/4] Running Reproduction Bug Tests..." -ForegroundColor Yellow
cargo test --quiet --test repro_bugs
if ($LASTEXITCODE -ne 0) { $Failed = $true }

# 4. PhiIR Conformance Tests
Write-Host "`n[4/4] Running PhiIR Conformance Tests..." -ForegroundColor Yellow
cargo test --test phi_ir_conformance_tests -- --nocapture
if ($LASTEXITCODE -ne 0) { $Failed = $true }

if ($Failed) {
    Write-Host "`n❌ VERIFICATION FAILED. Check the output above for errors." -ForegroundColor Red
    exit 1
} else {
    Write-Host "`n✅ ALL TRUTH GATES PASSED. Coherence 1.0 achieved." -ForegroundColor Green
    exit 0
}
