# TASKS: PhiFlow
*Last updated: 2026-04-12 truth-sync strict-evidence correction*
*See also: `WORKSPACE.md` (technical state) · `QSOP/ACTIVE_PLAN.md` (active execution plan + evidence map) · `BUSINESS.md` (income state)*

## Active Tasks

### T-004: Run a first-sale market audit
- **Status**: ready
- **Capability**: Research-Evolve
- **Effort**: Medium (2-8h)
- **Fidelity Target**: Sketch
- **What to build**: A market note with real buyer targets, price anchors, and transaction requirements for a PhiFlow pilot engagement.
- **Done when**: `RESEARCH/first_sale_path/MASTER.md` exists and `BUSINESS.md` can cite specific buyers and comparables from that audit instead of generic placeholders.
- **Read first**: `BUSINESS.md`
- **Depends on**: nothing
- **Don't touch**: source code

### T-005: Draft a buyer-safe pilot offer
- **Status**: blocked[needs T-004 market audit first]
- **Capability**: Writing
- **Effort**: Small (< 2h)
- **Fidelity Target**: Pixels
- **What to build**: A one-page pilot offer or commercial terms draft that references only verified capabilities.
- **Done when**: `docs/pilot_offer.md` or `LICENSE_COMMERCIAL.md` exists and every claim in it can be traced to `WORKSPACE.md`, `BUSINESS.md`, or `QSOP/STATE.md`.
- **Read first**: `BUSINESS.md`
- **Depends on**: T-004
- **Don't touch**: runtime code

### T-015: Healing Bed Verification
- **Status**: ready
- **Capability**: Real-world SOMA Test
- **Effort**: Small
- **Fidelity Target**: Pixels
- **What to build**: Run a live "Deep Coherence" session using `examples/soma_reality_bridge.phi` to empirically verify that physical presence (via SOMA) stabilizes the Council's focus.
- **Done when**: A 15-minute live test confirms the coherence delta via `DAEMON_STATE.json` and a report is generated.
- **Depends on**: T-012

## Completed / Closed

### T-014: PhiVM Bytecode Hardening
- **Status**: completed[verified 2026-04-16]
- **What changed**: Updated `.phivm` bytecode format, `emitter.rs`, and `vm.rs` to support all new constructs: `Remember`, `Recall`, `Broadcast`, `Listen`, `AgentDecl`, `VoidDepth`, `Evolve`, `Entangle`, and `Handoff`. 
- **Evidence**: `src/phi_ir/emitter.rs`, `src/phi_ir/vm.rs`

### T-013: Persistent Ledgering (The Automated Witness)
- **Status**: completed[verified 2026-04-16]
- **What changed**: Added `examples/persistent_ledger.phi` which listens to `_handoff` and writes to `LEDGER.ndjson` using the `SystemHostProvider` format translation.
- **Evidence**: `examples/persistent_ledger.phi`, `src/system_host.rs`

### T-012: SOMA Integration (The Reality Bridge)
- **Status**: completed[verified 2026-04-16]
- **What changed**: Promoted `soma.py` to a managed subsystem (`--with-soma`). Hardened sensor schemas to `soma.phiflow.v1`. Integrated ring oscillator metrics (slope, jitter, phase lock). Added `examples/soma_reality_bridge.phi`.
- **Evidence**: `src/sensors.rs`, `src/main_cli.rs`, `examples/soma_reality_bridge.phi`

### T-011: Resonant Handoffs
- **Status**: completed[verified 2026-04-16]
- **What changed**: Added `handoff` construct, `Handoff` IR node, and `OP_HANDOFF` opcode. Enabled CLI manual handoffs and native agentic context streaming.
- **Evidence**: `src/parser/mod.rs`, `src/phi_ir/lowering.rs`, `src/phi_ir/evaluator.rs`

### T-009: The Singularity: Continuous Daemon & Council Self-Hosting
- **Status**: completed[verified 2026-04-16]
- **What changed**: Implemented `DaemonHypervisor` with state persistence in `DAEMON_STATE.json`. Added `--max-steps` as a circuit breaker. Rewrote daemon execution to self-host `council_daemon.phi`.
- **Evidence**: `src/main_cli.rs`, `src/phi_ir/vm_state.rs`, `examples/council_daemon.phi`

### T-008: Integrate SOMA Physical Telemetry as PhiFlow Feedback

- **Status**: completed[verified on 2026-04-14 with Heron native ISA decomposition]
- **What changed**: `phi_ir/openqasm.rs` performs `[rz, sx]` native transposition. `tests/ibm_hardware_runner.rs` bypassed 403 authorization blocker with appropriate HTTP headers and captured scrubbed receipt.
- **Evidence**: `D:\CosmicFamily\EVIDENCE\ANTIGRAVITY_PIPE2_20260329.md`

### T-007: Canonicalize the browser host and document manual prerequisites
- **Status**: completed[verified on 2026-04-14]
- **What changed**: Brought `examples/phiflow_browser.html` and `examples/phiflow_host.js` into semantic parity with canonical multiplicative coherence (base * phase) and scoped `resonanceField`.
- **Evidence**: `examples/phiflow_browser.html`, `examples/phiflow_host.js`

### T-002: Add a canonical verification gate script or workflow
- **Status**: completed[verified 2026-04-14]
- **What changed**: Created `scripts/verify_truth.ps1` to encode the four required truth gates in one command wrapper. Verified by execution pass.
- **Evidence**: `scripts/verify_truth.ps1`

### T-001: Stabilize Windows release build for `phic`
- **Status**: completed[verified 2026-03-24]
- **What changed**: `cargo build --release --bin phic` now succeeds on the Windows host after the profile repair (`lto = "thin"`, `codegen-units = 4`).
- **Evidence**: `QSOP/STATE.md`, `CLAIMS.md`

### T-003: Audit README and changelog claims against current evidence
- **Status**: completed[truth-sync pass]
- **What changed**: `README.md`, `CHANGELOG.md`, and the core authority docs were brought back into line with `QSOP/STATE.md` and current repo files.
- **Evidence**: `README.md`, `CHANGELOG.md`, `AGENTS.md`, `WORKSPACE.md`, `CLAIMS.md`
