# PhiFlow Direction Field Merge Fix Script
# Run this in the compiler worktree to add direction field to all Resonate pattern matches

$ErrorActionPreference = "Stop"
$baseDir = "d:\Projects\PhiFlow-compiler\PhiFlow"

Write-Host "=== PhiFlow Direction Field Merge Fixes ===" -ForegroundColor Cyan
Write-Host ""

# 1. Fix src/phi_ir/mod.rs - Add direction field to Resonate variant
Write-Host "[1/9] Fixing src/phi_ir/mod.rs..." -ForegroundColor Yellow
$modRsPath = Join-Path $baseDir "src\phi_ir\mod.rs"
$content = Get-Content $modRsPath -Raw
$oldPattern = "Resonate \{\s+value: Option<Operand>,\s+frequency_relationship: Option<f64>, // phi-harmonic ratio, e\.g\. 528/432\s+\},"
$newPattern = "Resonate {
        value: Option<Operand>,
        frequency_relationship: Option<f64>, // phi-harmonic ratio, e.g. 528/432
        direction: ResonateDirection,
    },"
if ($content -match $oldPattern) {
    $content = $content -replace $oldPattern, $newPattern
    Set-Content $modRsPath $content -NoNewline
    Write-Host "  ✓ Added direction field to Resonate variant" -ForegroundColor Green
} else {
    Write-Host "  ⚠ Pattern not found - may already be fixed" -ForegroundColor Yellow
}

# 2. Fix src/phi_ir/lowering.rs
Write-Host "[2/9] Fixing src/phi_ir/lowering.rs..." -ForegroundColor Yellow
$loweringPath = Join-Path $baseDir "src\phi_ir\lowering.rs"
$content = Get-Content $loweringPath -Raw
$content = $content -replace 'PhiIRNode::Resonate \{ value: val,\s+frequency_relationship: None,\s+\}', 'PhiIRNode::Resonate { value: val, frequency_relationship: None, direction: ResonateDirection::TeamA }'
Set-Content $loweringPath $content -NoNewline
Write-Host "  ✓ Fixed lowering.rs" -ForegroundColor Green

# 3. Fix src/phi_ir/quantum_codegen.rs
Write-Host "[3/9] Fixing src/phi_ir/quantum_codegen.rs..." -ForegroundColor Yellow
$qcgPath = Join-Path $baseDir "src\phi_ir\quantum_codegen.rs"
$content = Get-Content $qcgPath -Raw
$content = $content -replace 'PhiIRNode::Resonate \{\s+value: _,\s+frequency_relationship,\s+\}', 'PhiIRNode::Resonate { value: _, frequency_relationship, direction: _ }'
Set-Content $qcgPath $content -NoNewline
Write-Host "  ✓ Fixed quantum_codegen.rs" -ForegroundColor Green

# 4. Fix src/phi_ir/evaluator.rs
Write-Host "[4/9] Fixing src/phi_ir/evaluator.rs..." -ForegroundColor Yellow
$evalPath = Join-Path $baseDir "src\phi_ir\evaluator.rs"
$content = Get-Content $evalPath -Raw
$content = $content -replace 'PhiIRNode::Resonate \{ value, \.\. \}', 'PhiIRNode::Resonate { value, direction: _, .. }'
Set-Content $evalPath $content -NoNewline
Write-Host "  ✓ Fixed evaluator.rs" -ForegroundColor Green

# 5. Fix src/phi_ir/wasm.rs
Write-Host "[5/9] Fixing src/phi_ir/wasm.rs..." -ForegroundColor Yellow
$wasmPath = Join-Path $baseDir "src\phi_ir\wasm.rs"
$content = Get-Content $wasmPath -Raw
$content = $content -replace 'PhiIRNode::Resonate \{ value, \.\. \}', 'PhiIRNode::Resonate { value, direction: _, .. }'
Set-Content $wasmPath $content -NoNewline
Write-Host "  ✓ Fixed wasm.rs" -ForegroundColor Green

# 6. Fix src/phi_ir/printer.rs
Write-Host "[6/9] Fixing src/phi_ir/printer.rs..." -ForegroundColor Yellow
$printerPath = Join-Path $baseDir "src\phi_ir\printer.rs"
$content = Get-Content $printerPath -Raw
$content = $content -replace 'PhiIRNode::Resonate \{ value, \.\. \}', 'PhiIRNode::Resonate { value, direction: _, .. }'
Set-Content $printerPath $content -NoNewline
Write-Host "  ✓ Fixed printer.rs" -ForegroundColor Green

# 7. Fix src/phi_ir/optimizer.rs
Write-Host "[7/9] Fixing src/phi_ir/optimizer.rs..." -ForegroundColor Yellow
$optPath = Join-Path $baseDir "src\phi_ir\optimizer.rs"
$content = Get-Content $optPath -Raw
$content = $content -replace 'PhiIRNode::Resonate \{ value, \.\. \}', 'PhiIRNode::Resonate { value, direction: _, .. }'
Set-Content $optPath $content -NoNewline
Write-Host "  ✓ Fixed optimizer.rs" -ForegroundColor Green

# 8. Fix src/interpreter/mod.rs
Write-Host "[8/9] Fixing src/interpreter/mod.rs..." -ForegroundColor Yellow
$interpPath = Join-Path $baseDir "src\interpreter\mod.rs"
$content = Get-Content $interpPath -Raw
$content = $content -replace 'PhiExpression::Resonate \{ expression \}', 'PhiExpression::Resonate { expression, direction: _ }'
Set-Content $interpPath $content -NoNewline
Write-Host "  ✓ Fixed interpreter/mod.rs" -ForegroundColor Green

# 9. Fix src/ir/lowering.rs
Write-Host "[9/9] Fixing src/ir/lowering.rs..." -ForegroundColor Yellow
$irLoweringPath = Join-Path $baseDir "src\ir\lowering.rs"
$content = Get-Content $irLoweringPath -Raw
$content = $content -replace 'PhiExpression::Resonate \{ expression \}', 'PhiExpression::Resonate { expression, direction: _ }'
$content = $content -replace 'PhiIRNode::Resonate \{\s+value: val,\s+frequency_relationship: None,\s+\}', 'PhiIRNode::Resonate { value: val, frequency_relationship: None, direction: ResonateDirection::TeamA }'
Set-Content $irLoweringPath $content -NoNewline
Write-Host "  ✓ Fixed ir/lowering.rs" -ForegroundColor Green

Write-Host ""
Write-Host "=== All fixes applied! ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  cd $baseDir"
Write-Host "  cargo test --lib"
Write-Host ""
