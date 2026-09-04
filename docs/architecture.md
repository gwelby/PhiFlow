# PhiFlow Architecture

**Status:** Implementation-accurate as of v0.4.0

## Compiler Pipeline

```
.phi source
    │
    ▼
┌──────────┐
│  Lexer   │  PhiLexer: char stream → token stream
│ (2858 ln)│  Handles: keywords, operators, literals, comments
└────┬─────┘
     │ Vec<PhiToken>
     ▼
┌──────────┐
│  Parser  │  PhiParser: token stream → AST (Vec<PhiExpression>)
│          │  Recursive descent, operator precedence climbing
└────┬─────┘
     │ Vec<PhiExpression>
     ▼
┌──────────┐
│ Lowering │  AST → PhiIR (SSA-form intermediate representation)
│          │  Variables → SSA registers, blocks → PhiIR blocks
└────┬─────┘
     │ PhiIRProgram
     ├────────────────────────────────┐
     ▼                                ▼
┌──────────┐                   ┌──────────┐
│Evaluator │                   │  Emitter  │
│(tree-walk)│                  │ (bytecode)│
└──────────┘                   └────┬─────┘
                                    │ .phivm bytes
                                    ▼
                              ┌──────────┐
                              │   VM     │  Bytecode interpreter
                              │ (1814 ln)│  30+ opcodes, PHIV format
                              └──────────┘

PhiIRProgram also feeds:
    ├─→ WASM codegen (855 ln)  → .wat → .wasm
    ├─→ OpenQASM codegen (1163 ln) → .qasm
    └─→ Optimizer → optimized PhiIR
```

## Module Structure

```
src/
├── parser/           # Lexer + Parser (2858 lines)
├── phi_ir/           # Intermediate representation (9333 lines)
│   ├── coherence.rs  # Canonical coherence formula (229 lines)
│   ├── emitter.rs    # PhiIR → bytecode
│   ├── evaluator.rs  # Tree-walking interpreter
│   ├── lowering.rs   # AST → PhiIR
│   ├── vm.rs         # Bytecode VM (1814 lines)
│   ├── vm_state.rs   # VM runtime state
│   ├── wasm.rs       # PhiIR → WebAssembly Text (855 lines)
│   ├── openqasm.rs   # PhiIR → OpenQASM 3.0 (1163 lines)
│   ├── optimizer.rs  # PhiIR optimization passes
│   ├── printer.rs    # PhiIR pretty-printer
│   ├── quantum_codegen.rs    # Quantum-specific lowering
│   ├── quantum_interaction.rs # Quantum interaction patterns
│   └── topology_transpiler.rs # Hardware topology mapping
├── quantum/          # Quantum backends (2202 lines)
│   ├── simulator.rs  # State-vector simulator (489 lines)
│   ├── ibm_quantum.rs # IBM Quantum REST API (1041 lines)
│   ├── backends.rs   # Backend abstraction
│   ├── backend_topology.rs # Hardware calibration data
│   └── types.rs      # Quantum types and constants
├── consciousness/    # Consciousness math (2266 lines)
│   ├── consciousness_math.rs # Field strength, states
│   ├── monitor.rs    # Consciousness monitoring
│   ├── muse_integration.rs # MUSE EEG bridge (400 lines)
│   ├── sacred_geometry.rs # Geometry generators
│   └── bridge.rs     # Consciousness bridge
├── metrics/          # Consciousness metrics (2463 lines)
│   ├── coherence_panel.rs # PLV, wPLI
│   ├── consciousness_proxy.rs # C_PF composite metric
│   ├── differentiation.rs # Effective rank (PCA/SVD)
│   ├── fisher_information.rs # Fisher information
│   ├── mutual_information.rs # MI estimation
│   ├── self_correlation.rs # L_self loop detection
│   └── trace.rs      # Execution trace recording
├── security/         # Post-quantum anchoring (1373 lines)
│   ├── anchor.rs     # ECDSA + ML-DSA attestation (845 lines)
│   ├── entropy_buffer.rs # Entropy drift detection
│   └── mod.rs        # Security module interface
├── sacred/           # Phi-harmonic utilities (923 lines)
│   ├── frequency_generator.rs
│   └── phi_memory.rs
├── mcp_server/       # MCP protocol server (1050 lines)
├── sensors.rs        # SOMA sensor integration (512 lines)
├── wasm_host.rs      # WASM host runtime (581 lines)
├── host.rs           # Host provider trait (318 lines)
├── phi_core.rs       # Core types (1060 lines)
└── main_cli.rs       # CLI entry point (1517 lines)
```

## Execution Backends

### 1. Evaluator (Tree-Walking)

Direct AST evaluation. Used for development and testing. The evaluator maintains:
- Intention stack (`Vec<String>`)
- Resonance field (`HashMap<String, Vec<PhiValue>>`)
- Variable scopes
- Witness log

### 2. Bytecode VM

Loads `.phivm` binary format and executes bytecode. The VM has:
- 30+ opcodes (see `src/phi_ir/vm.rs`)
- String table for interning
- Block table for control flow
- Step limit for safety

**Binary format:**
```
PHIV (magic, 4 bytes)
version (1 byte)
string_count (u32)
string_table (length-prefixed UTF-8 strings)
block_count (u32)
block_table (offsets and sizes)
instruction_stream (bytecode)
```

### 3. WebAssembly Host

The WASM host (`src/wasm_host.rs`) provides implementations for the consciousness imports:

| Import | Implementation |
|--------|---------------|
| `phi_witness(operand: i32) -> f64` | Records witness event, returns coherence |
| `phi_resonate(value: f64)` | Appends to resonance field |
| `phi_coherence() -> f64` | Returns current C(d, k) |
| `phi_sensor(sensor_id: i32) -> f64` | Reads SOMA sensor |
| `phi_intention_push()` | Increments intention depth |
| `phi_intention_pop()` | Decrements intention depth |

The host tracks the same state as the evaluator (intention stack, resonance field) and computes coherence using the same canonical formula.

## Quantum Compilation

PhiIR → OpenQASM 3.0 mapping:

| PhiFlow Construct | OpenQASM |
|---|---|
| `intention "name"` | Qubit declaration + RY rotation |
| `resonate value` | `ry(value × π)` on the intention's qubit |
| `witness` | Mid-circuit measurement |
| `resonate ... toward TEAM_B` | Inverted rotation direction |

The OpenQASM emitter supports:
- Topology-aware gate decomposition (Heron RZ/SX, Eagle CX)
- Mid-circuit measurement with collapse policies
- Deferred measurement (Final / NonDestructive)
- Hardware stress noise injection
- Post-quantum watermarks (ECDSA + ML-DSA fingerprints)

## Sensor Integration

### SOMA (System Observation and Measurement Architecture)

Reads system-level sensors via `sysinfo`:
- CPU usage percentage
- CPU temperature
- Memory usage
- Network throughput

Reads environmental sensors from external SOMA state file:
- Schumann resonance (7.83 Hz proxy)
- 432 Hz tone detection
- Presence (ring oscillator jitter)
- Fan speed (Hz)
- 60 Hz AC mains detection

Coherence is blended from system sensors and SOMA environmental sensors when available.

### MUSE EEG

Connects to MUSE headband via Python bridge process:
- EEG channels: TP9, AF7, AF8, TP10
- Computed metrics: coherence, clarity, flow state, phi-resonance
- Brainwave bands: alpha, beta, gamma, theta, delta
- Sacred frequency lock detection

Requires physical MUSE hardware and Python bridge script. Not testable in CI.

## Security Layer

### Anchor Attestation

When PhiFlow performs a significant action, it captures:
1. Current SOMA sensor state (observation)
2. Payload hash (SHA-256)
3. Observation hash (SHA-256)
4. Nonce (UUID v4, replay-protected)

Binds them with hybrid signatures:
- ECDSA secp256k1 (via `k256`, RustCrypto)
- ML-DSA-65 / Dilithium3 (via `pqcrypto-dilithium`, NIST FIPS 204)

Canonical signed message:
```
PhiFlow-Attestation-v1
payload_hash=<hex-sha256>
observation_hash=<hex-sha256>
policy_version=1.0.0
```

22 tests verify sign/verify roundtrips, tamper detection, policy enforcement, and nonce replay protection.

## Test Coverage

| Layer | Tests | Status |
|-------|-------|--------|
| Parser unit tests | 51 | All passing |
| PhiIR (evaluator, VM, WASM, OpenQASM) | 44 | All passing |
| Quantum simulator | 18 | All passing |
| Consciousness metrics | 47 | All passing |
| Security (anchor) | 22 | All passing |
| Integration tests | ~318 | All passing |
| **Total Rust** | **~460** | **All passing** |
| Julia layer | 51 | All passing |
| Python layer | 39 | All passing |
| **Grand total** | **~550** | |
