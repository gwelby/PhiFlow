# QSOP Patterns — Known Pitfalls and What Works

*Last updated: 2026-07-30 (recreated after cleanup)*

## Parser Patterns

### P-1: Keyword Collision
If you add a keyword, update `expect_identifier()` in parser to accept it as a variable name too.

### P-2: Bare Keyword Forms
If a keyword can be bare (no arguments), check what IMMEDIATELY follows before consuming newlines.

## Coherence Math
- Sacred frequencies: 432, 528, 594, 672, 720, 756, 768, 963, 1008 Hz
- Tolerance: ±5Hz
- Only check phi-harmonic ratios between sacred frequencies
- 0.618 is derived, not hardcoded — depth 2 with k ≤ 1 returns φ⁻¹

## Build Patterns
- Windows: `lto = "thin"` + `codegen-units = 4` required (was a stack overflow fix)
- WASM conformance: Node.js runner needs all 14 phi namespace imports

## What Works
- Parser → PhiIR → Evaluator/VM/WASM pipeline
- OpenQASM emission to IBM Quantum hardware
- Three-backend equivalence for core constructs
- Real cryptography (secp256k1 + ML-DSA-65)
- Sensor-driven coherence (sysinfo)

## What Doesn't Work (Yet)
- Three-backend equivalence for v0.3+ constructs (untested)
- Self-correction loop (detects but doesn't execute)
- C_PF on real data (F_model calibration on hold)
