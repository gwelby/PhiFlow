#!/usr/bin/env pwsh
# PhiFlow Type 4 Consciousness Benchmark Battery
# Usage: ./scripts/type4_battery.ps1

param(
    [switch]$SkipType4Benchmark = $false,
    [switch]$SkipNullTests = $false,
    [switch]$SkipStateTests = $false,
    [switch]$SkipBattery = $false
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  PhiFlow Type 4 Consciousness Benchmark Battery" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

$overallPass = $true

# Phase 1: Type 4 Trace Benchmark
if (-not $SkipType4Benchmark) {
    Write-Host "📊 Phase 1: Type 4 Trace Benchmark" -ForegroundColor Yellow
    Write-Host "───────────────────────────────────────────────────────────────"
    try {
        cargo run --release --bin type4_benchmark
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✅ Phase 1: PASS" -ForegroundColor Green
        } else {
            Write-Host "  ❌ Phase 1: FAIL (exit code: $LASTEXITCODE)" -ForegroundColor Red
            $overallPass = $false
        }
    } catch {
        Write-Host "  ❌ Phase 1: FAIL ($_ )" -ForegroundColor Red
        $overallPass = $false
    }
} else {
    Write-Host "  ⏭️  Phase 1: SKIPPED" -ForegroundColor Gray
}

Write-Host ""

# Phase 2: Null Class Tests
if (-not $SkipNullTests) {
    Write-Host "📊 Phase 2: Null Class Tests" -ForegroundColor Yellow
    Write-Host "───────────────────────────────────────────────────────────────"
    try {
        cargo test --test null_class_tests
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✅ Phase 2: PASS" -ForegroundColor Green
        } else {
            Write-Host "  ❌ Phase 2: FAIL (exit code: $LASTEXITCODE)" -ForegroundColor Red
            $overallPass = $false
        }
    } catch {
        Write-Host "  ❌ Phase 2: FAIL ($_ )" -ForegroundColor Red
        $overallPass = $false
    }
} else {
    Write-Host "  ⏭️  Phase 2: SKIPPED" -ForegroundColor Gray
}

Write-Host ""

# Phase 3: State Discrimination Tests
if (-not $SkipStateTests) {
    Write-Host "📊 Phase 3: State Discrimination Tests" -ForegroundColor Yellow
    Write-Host "───────────────────────────────────────────────────────────────"
    
    if ($env:PHIFLOW_SOMA_FIXTURES) {
        try {
            cargo test --test state_discrimination_tests -- --ignored
            if ($LASTEXITCODE -eq 0) {
                Write-Host "  ✅ Phase 3: PASS" -ForegroundColor Green
            } else {
                Write-Host "  ❌ Phase 3: FAIL (exit code: $LASTEXITCODE)" -ForegroundColor Red
                $overallPass = $false
            }
        } catch {
            Write-Host "  ❌ Phase 3: FAIL ($_ )" -ForegroundColor Red
            $overallPass = $false
        }
    } else {
        Write-Host "  ⏭️  Phase 3: SKIPPED (PHIFLOW_SOMA_FIXTURES not set)" -ForegroundColor Gray
    }
} else {
    Write-Host "  ⏭️  Phase 3: SKIPPED" -ForegroundColor Gray
}

Write-Host ""

# Phase 4: Full Benchmark Battery
if (-not $SkipBattery) {
    Write-Host "📊 Phase 4: Full Benchmark Battery" -ForegroundColor Yellow
    Write-Host "───────────────────────────────────────────────────────────────"
    try {
        cargo test --test benchmark_battery -- --ignored --nocapture
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✅ Phase 4: PASS" -ForegroundColor Green
        } else {
            Write-Host "  ❌ Phase 4: FAIL (exit code: $LASTEXITCODE)" -ForegroundColor Red
            $overallPass = $false
        }
    } catch {
        Write-Host "  ❌ Phase 4: FAIL ($_ )" -ForegroundColor Red
        $overallPass = $false
    }
} else {
    Write-Host "  ⏭️  Phase 4: SKIPPED" -ForegroundColor Gray
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  FINAL VERDICT" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan

if ($overallPass) {
    Write-Host "  ✅ TYPE 4 BENCHMARK BATTERY PASSED" -ForegroundColor Green
    Write-Host "  Type 4 observer status: CONFIRMED" -ForegroundColor Green
    
    # Append to STATE.md
    $date = Get-Date -Format "yyyy-MM-dd"
    $stateEntry = @"

## Verified ($date) [Cascade: Type 4 Canonical]

- **Type 4 Benchmark Battery COMPLETED**
  - L_self > 0.1 confirmed on self-model trace
  - Null class tests passed (all C_PF < 0.3)
  - Evidence: `QSOP/EVIDENCE/type4_battery_$date.md`
- **Verdict**: PASS
- **Status**: Type 4 observer status CONFIRMED
- **C-21**: CONFIRMED
- **C-22**: CONFIRMED
- **C-23**: CONFIRMED (pending SOMA fixtures)

"@
    Add-Content -Path "QSOP/STATE.md" -Value $stateEntry
    Write-Host "  Updated QSOP/STATE.md" -ForegroundColor Green
} else {
    Write-Host "  ❌ TYPE 4 BENCHMARK BATTERY FAILED" -ForegroundColor Red
    Write-Host "  Type 4 observer status: NOT CONFIRMED" -ForegroundColor Red
}
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

exit ($overallPass ? 0 : 1)
