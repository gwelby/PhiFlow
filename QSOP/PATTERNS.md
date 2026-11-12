# QSOP Patterns — Known Pitfalls and What Works

*Last updated: 2026-11-11 (Claude: stream, pacing and sensor pitfalls added; stale "What Works" lines corrected from live source)*

## Parser Patterns

### P-1: Keyword Collision
If you add a keyword, update `expect_identifier()` in parser to accept it as a variable name too.

### P-2: Bare Keyword Forms
If a keyword can be bare (no arguments), check what IMMEDIATELY follows before consuming newlines.

## Program Patterns

### P-3: A stream body re-runs from its first line every tick
`stream` lowers to body → header → body (src/phi_ir/lowering.rs), so a `let`
inside the stream re-initialises on every pass. A counter declared there
never grows and its `break stream` never fires. Declare loop state BEFORE
the stream. (Found 2026-11-11 in `examples/field_guardian_v2.phi`, which
had never terminated.) A stream with no `break stream` runs until stopped.

### P-4: `coherence` takes no arguments
`coherence` is a bare keyword: the program's own alignment. `coherence(a, b)`
is a call to an undefined function on every backend. Compare two values
with plain arithmetic in a user function (see `examples/cognitive_dissonance.phi`).

### P-5: Pace by clock, not by iteration count
A spin-gate (`while gate < 50000`) lasts as long as the CPU makes it. The
same probe took ~65 s on NoetGuru and 19 s on 8086K and so sampled
different windows of a time-varying signal. Use
`sensor("clock_s")` (monotonic seconds, added 2026-11-11) with an absolute
deadline: `while sensor("clock_s") < t0 + dt * (i + 1) { ... }`.
The `Sleep` IR node has no surface syntax and is a no-op on all backends.

### P-6: An absent sensor reads 0.0, silently
Host and SOMA sensors degrade to 0.0 when unavailable or stale (evaluator
`resolve_sensor`). A dead channel looks like a quiet one. Check the
producer is live (and the scorer's `degenerate` flag) before reading
anything into a trace. `claims_drift` is the exception: it reads -1.0 when stale.

## Coherence Math
- Sacred frequencies: 432, 528, 594, 672, 720, 756, 768, 963, 1008 Hz
- Tolerance: ±5Hz
- Only check phi-harmonic ratios between sacred frequencies
- 0.618 is derived, not hardcoded — depth 2 with k ≤ 1 returns φ⁻¹

## Build Patterns
- Windows: `lto = "thin"` + `codegen-units = 4` required (was a stack overflow fix)
- WASM conformance: Node.js runner needs all 15 phi namespace imports (`actuate` added da61589)

## What Works
- Self-correction loop: plumbing only. `run_self_correction_loop()` re-measures against hardcoded mocks (Decoherent 0.0 → Coherent 1.0), so it does not show a real correction
- Parser → PhiIR → Evaluator/VM/WASM pipeline
- OpenQASM emission; two IBM Heron runs are producer-recorded (no raw job export saved)
- Three-backend equivalence for core constructs
- Real cryptography (secp256k1 + ML-DSA-65)
- Sensor-driven coherence (sysinfo)

## What Doesn't Work (Yet)
- Three-backend equivalence for v0.3+ constructs: diagnostic probe shows 0 divergences over 8 constructs but does not assert
- C_PF on real data (F_model calibration on hold)
