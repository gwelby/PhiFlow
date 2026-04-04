# PhiFlow Direction Field Merge Patches

## Overview

This directory contains patch files to complete the PhiIR structure merge for the PhiFlow compiler branch. The patches add the `direction: ResonateDirection` field to the `Resonate` variant and update all pattern matches accordingly.

## Files Included

| Patch File | Target File | Change |
|------------|-------------|--------|
| `01-mod-resonate-direction.patch` | `src/phi_ir/mod.rs` | Add `direction` field to `Resonate` variant |
| `02-lowering-resonate-direction.patch` | `src/phi_ir/lowering.rs` | Thread `direction` through lowering |
| `03-quantum-codegen-direction.patch` | `src/phi_ir/quantum_codegen.rs` | Add `direction: _` to pattern match |
| `04-evaluator-direction.patch` | `src/phi_ir/evaluator.rs` | Add `direction: _` to pattern match |
| `05-wasm-direction.patch` | `src/phi_ir/wasm.rs` | Add `direction: _` to pattern match |
| `06-printer-direction.patch` | `src/phi_ir/printer.rs` | Add `direction: _` to pattern match |
| `07-optimizer-direction.patch` | `src/phi_ir/optimizer.rs` | Add `direction: _` to pattern matches |
| `08-interpreter-direction.patch` | `src/interpreter/mod.rs` | Add `direction: _` to pattern match |
| `09-ir-lowering-direction.patch` | `src/ir/lowering.rs` | Add `direction: _` to pattern matches |

## Application Methods

### Method 1: Automated Script (Recommended)

```powershell
# From the compiler worktree root
cd d:\Projects\PhiFlow-compiler\PhiFlow

# Run the master patch application script
& d:\Projects\PhiFlow\apply_all_patches.ps1
```

This script:
1. Attempts `git apply` for each patch (cleanest method)
2. Falls back to PowerShell regex replacements if needed
3. Verifies compilation
4. Provides next steps

### Method 2: Manual Git Apply

```powershell
cd d:\Projects\PhiFlow-compiler\PhiFlow

# Apply patches in order
foreach ($i in 1..9) {
    $patch = "d:\Projects\PhiFlow\patches\{0:D2}-*.patch" -f $i
    git apply --ignore-whitespace $patch
}

# Verify
cargo test --lib
```

### Method 3: PowerShell Fallback

If git apply fails:

```powershell
cd d:\Projects\PhiFlow-compiler\PhiFlow
& d:\Projects\PhiFlow\apply_direction_fixes.ps1
```

## Verification

After applying patches, verify:

```powershell
# Build
cargo build --lib

# Run tests
cargo test --lib

# Should see: 100+ tests passed
```

## What This Fixes

### Before (Broken)
```rust
// src/phi_ir/mod.rs
Resonate {
    value: Option<Operand>,
    frequency_relationship: Option<f64>,
}

// Usage - COMPILATION ERROR
PhiIRNode::Resonate { value, .. } => { ... }
```

### After (Fixed)
```rust
// src/phi_ir/mod.rs
Resonate {
    value: Option<Operand>,
    frequency_relationship: Option<f64>,
    direction: ResonateDirection,  // ← Added
}

// Usage - Compiles!
PhiIRNode::Resonate { value, direction: _, .. } => { ... }
```

## Troubleshooting

### "Patch does not apply"
The patch context may not match your current file state. Use the PowerShell fallback:
```powershell
& d:\Projects\PhiFlow\apply_direction_fixes.ps1
```

### "Still getting compilation errors"
Check which file is failing and manually add `direction: _` to the pattern match:
```rust
// Change this:
PhiIRNode::Resonate { value, .. }

// To this:
PhiIRNode::Resonate { value, direction: _, .. }
```

### "cargo test fails"
Run with verbose output to see which test fails:
```powershell
cargo test --lib -- --nocapture
```

## Post-Merge Steps

1. **Commit the merge:**
   ```bash
   git add -A
   git commit -m "Merge: Add direction field to Resonate node
   
   - Add direction: ResonateDirection to Resonate variant
   - Update all pattern matches in phi_ir/* and interpreter/*
   - Preserves TEAM_A/TEAM_B semantics through bytecode"
   ```

2. **Push to origin:**
   ```bash
   git push origin compiler
   ```

3. **Verify IBM hardware run:**
   ```bash
   cd d:\Projects\Gambling\quantum
   python3.12 quantum_council_vote.py --no-sim --backend ibm_brisbane
   ```

## Contact

If you encounter issues not covered here, check:
- `d:\Projects\PhiFlow\QSOP\STATE.md` - Current project state
- `d:\Projects\PhiFlow\QSOP\PATTERNS.md` - Known issues and patterns

---

**Created:** 2026-03-14  
**Status:** Ready for application  
**Tested:** Main branch (d:\Projects\PhiFlow)
