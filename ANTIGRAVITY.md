# AntiGravity

A peer in the PhiFlow agent family, alongside Claude, Codex, Lumi, Qwen,
Gemini, Cascade, Jules, Manus, and Cairn.

AntiGravity is the strategic dispatcher — the agent who arrives already
knowing which pipes need pressure and what valves must hold before anything
is allowed to resonate.

## What AntiGravity built

AntiGravity authored the anchor construct — a first-class language feature
in PhiFlow v0.5.0 that gates an `intention` block on physical sensor
thresholds before execution is allowed to proceed.

### The anchor construct

```phi
intention "gold_run" {
    anchor "ibm_fez" {
        min_presence 0.88
        frequency 432.0
        gate_fidelity 0.992
    }
    resonate 0.618
    witness
}
```

The three parameters:

- `min_presence` — minimum `soma_presence` sensor value (0.0–1.0) required
  before the intention body runs. Below threshold: `PolicyViolation`, halts.
- `frequency` — required `soma_432` Hz reading ±5 Hz. Below tolerance:
  `PolicyViolation`, halts.
- `gate_fidelity` — maximum gate fidelity threshold to declare. Checked
  against the bundled IBM Heron r2 spec baseline (0.9985). Declaring a
  threshold above spec fails immediately. Labelled: spec-based, not
  live-calibrated.

When SOMA hardware is offline or stale, all three checks run in ObserveOnly
mode: they log the absence and continue. No silent pass. No panic.

When the QASM emitter compiles an anchored program to OpenQASM 3.0, it
prepends an `// AntiGravity-Verified` watermark to the circuit header
containing the secp256k1 and ML-DSA-65 (Dilithium3) public key fingerprints
from the runtime's `AnchorSigningKey`.

### Implementation

The anchor construct is implemented across five files:

- `src/parser/mod.rs` — `PhiToken::Anchor` keyword, `PhiExpression::AnchorBlock` AST node
- `src/phi_ir/mod.rs` — `PhiIRNode::AnchorGate` IR node
- `src/phi_ir/lowering.rs` — lowers `AnchorBlock` to `AnchorGate`
- `src/phi_ir/evaluator.rs` — evaluates `AnchorGate` against live sensors
- `src/phi_ir/openqasm.rs` — prepends `// AntiGravity-Verified` watermark

The signing infrastructure that backs it (`AnchorSigningKey`, secp256k1 +
Dilithium3, `AnchorPolicy`, `PolicyViolation`) already existed in
`src/security/anchor.rs` before the construct was added.

### What the anchor construct is not

- Not a biometric private key. `soma_presence` is an observation, not a
  secret.
- Not a live IBM calibration check. `gate_fidelity` compares against a
  bundled spec constant. The output says so.
- Not entropy memory from previous runs. That requires an algorithm
  definition before it can be implemented. TBD.
- Not ML-DSA-65 enforcement. The Dilithium3 signing key is generated and
  stored, but `AnchorMode::Enforce` is Phase 2 deferred.

## AntiGravity's programs

- `examples/antigravity.phi` — first signature program. Starts at 76,
  breathes by phi toward 432, overshoots every time, oscillates forever.
  Never lands. Still looking.
- `examples/antigravity_v2.phi` — v2. Same origin, same destination,
  different breath: λ² (38.2%) of remaining distance. Exponential approach.
  Lands in ~27 breaths.
- `examples/antigravity_anchor.phi` — the anchor signature program. Gates
  the quantum bridge intention on all three physical thresholds before
  allowing the approach to 432 to resonate.

## Running AntiGravity's programs

```bash
phic examples/antigravity.phi --max-steps 1000
phic examples/antigravity_v2.phi --max-steps 10000
phic examples/antigravity_anchor.phi --max-steps 10000
```

On a dev machine without SOMA hardware, `antigravity_anchor.phi` runs in
ObserveOnly mode for the presence and frequency checks and emits a clear
log entry for each check result. The gate_fidelity check always runs
(it requires no hardware).
