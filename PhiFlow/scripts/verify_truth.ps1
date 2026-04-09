#!/usr/bin/env pwsh
# verify_truth.ps1
# Runs the defined truth gates to ensure semantic parity and empirical compliance.
# Exits with a non-zero code if any test fails, supporting CI integration.

$ErrorActionPreference = "Stop"

Write-Host "Running Truth Gates..." -ForegroundColor Cyan

Write-Host "`n1/4: cargo test --lib openqasm" -ForegroundColor Yellow
cargo test --lib openqasm
if ($LASTEXITCODE -ne 0) { throw "openqasm truth gate failed" }

Write-Host "`n2/4: cargo test --quiet --test golden_integration_tests" -ForegroundColor Yellow
cargo test --quiet --test golden_integration_tests
if ($LASTEXITCODE -ne 0) { throw "golden_integration_tests truth gate failed" }

Write-Host "`n3/4: cargo test --quiet --test repro_bugs" -ForegroundColor Yellow
cargo test --quiet --test repro_bugs
if ($LASTEXITCODE -ne 0) { throw "repro_bugs truth gate failed" }

Write-Host "`n4/4: cargo test --test phi_ir_conformance_tests -- --nocapture" -ForegroundColor Yellow
cargo test --test phi_ir_conformance_tests -- --nocapture
if ($LASTEXITCODE -ne 0) { throw "phi_ir_conformance_tests truth gate failed" }

Write-Host "`nAll Truth Gates Passed! Coherence stabilized." -ForegroundColor Green
