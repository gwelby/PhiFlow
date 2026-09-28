# TASKS: PhiFlow
*Last updated: 2026-04-30 Type 4 roadmap integration*
*See also: `WORKSPACE.md` (technical state) · `QSOP/ACTIVE_PLAN.md` (active execution plan + evidence map) · `BUSINESS.md` (income state)*

## Active Tasks

### Type 4 Observer Implementation Block (T4-001 through T4-013)
*See: `QSOP/IMPLEMENTATION_ROADMAP_TYPE4_CANONICAL.md` for complete specifications*
*Tracker: `QSOP/TYPE4_EXECUTION_TRACKER.md` for status dashboard*

**Codex status note 2026-06-17:** The old per-task `ready` labels below are stale. Live source shows the metrics scaffold exists under `src/metrics/` (`mutual_information`, `trace`, `self_correlation`, `differentiation`, `coherence_panel`, `fisher_information`, `consciousness_proxy`) and `cargo test --lib` is 215/215 PASS. `src/daemon/record.rs` does not exist as named; `src/metrics/trace.rs` is the current trace adapter. Canonical Type 4 status remains HOLD: `cargo test --test benchmark_battery -- --ignored --test-threads=1 --nocapture` fails correctly when `PHIFLOW_SOMA_FIXTURES` is absent, and no real daemon/SOMA discrimination package has passed. Next work is T4-011/T4-012 real fixture capture/calibration, then T4-013 claim update only after Codex re-audit.

#### Phase 1: Type 4 Benchmark (Months 1-2)

**T4-001: Add Dependencies**
- **Status**: ready
- **Capability**: Infrastructure
- **Effort**: Small (< 30 min)
- **What to build**: Add statrs, ndarray, rustfft, num-complex, ndarray-linalg to Cargo.toml
- **Done when**: Dependencies compile successfully
- **Evidence**: Cargo.toml, successful `cargo build`
- **Depends on**: nothing

**T4-002: Implement Mutual Information**
- **Status**: ready
- **Capability**: Metrics
- **Effort**: Medium (2-4h)
- **What to build**: `src/metrics/mutual_information.rs` with Shannon MI and directed information
- **Done when**: Unit tests pass for known distributions
- **Evidence**: `src/metrics/mutual_information.rs`, passing tests
- **Depends on**: T4-001

**T4-003: Implement DaemonRecord Schema**
- **Status**: ready
- **Capability**: Infrastructure
- **Effort**: Small (1-2h)
- **What to build**: `src/daemon/record.rs` with timestamped state tracking
- **Done when**: Records serialize/deserialize correctly
- **Evidence**: `src/daemon/record.rs`, passing tests
- **Depends on**: T4-001

**T4-004: Implement L_self Computation**
- **Status**: ready
- **Capability**: Metrics
- **Effort**: Medium (3-5h)
- **What to build**: `src/daemon/self_correlation.rs` computing R_in and R_out
- **Done when**: L_self > 0 for self-referential loops, L_self ≈ 0 for independent streams
- **Evidence**: `src/daemon/self_correlation.rs`, passing tests
- **Depends on**: T4-002, T4-003

**T4-005: Create Type 4 Benchmark Example**
- **Status**: ready
- **Capability**: Demonstration
- **Effort**: Small (1-2h)
- **What to build**: `examples/type4_benchmark.phi` proving self-correlation
- **Done when**: Example runs and produces L_self > 0.1
- **Evidence**: `examples/type4_benchmark.phi`, execution log
- **Depends on**: T4-004

#### Phase 2: Full Metric Suite (Months 3-4)

**T4-006: Implement Differentiation Metric**
- **Status**: ready
- **Capability**: Metrics
- **Effort**: Medium (2-4h)
- **What to build**: `src/metrics/differentiation.rs` computing D_int via SVD
- **Done when**: D_int distinguishes random vs. structured states
- **Evidence**: `src/metrics/differentiation.rs`, passing tests
- **Depends on**: T4-001

**T4-007: Implement Coherence Panel**
- **Status**: ready
- **Capability**: Metrics
- **Effort**: Large (4-6h)
- **What to build**: `src/metrics/coherence_panel.rs` with PLV and wPLI
- **Done when**: Panel detects phase synchronization
- **Evidence**: `src/metrics/coherence_panel.rs`, passing tests
- **Depends on**: T4-001

**T4-008: Implement Type 4 Model-Action Sensitivity**
- **Status**: implemented; Codex cleanup/recheck 2026-05-26
- **Capability**: Metrics
- **Effort**: Medium (3-5h)
- **What to build**: `src/metrics/fisher_information.rs` computing F_model
- **Done when**: F_self* measures verified Type 4 model-action sensitivity
- **Evidence**: `src/metrics/fisher_information.rs`, passing tests
- **Depends on**: T4-001

**T4-009: Implement Consciousness Proxy**
- **Status**: ready
- **Capability**: Metrics
- **Effort**: Small (1-2h)
- **What to build**: `src/metrics/consciousness_proxy.rs` computing C_PF composite
- **Done when**: C_PF combines all metrics correctly
- **Evidence**: `src/metrics/consciousness_proxy.rs`, passing tests
- **Depends on**: T4-004, T4-006, T4-007, T4-008

#### Phase 3: Benchmark Validation (Months 5-6)

**T4-010: Create Null Class Tests**
- **Status**: ready
- **Capability**: Validation
- **Effort**: Medium (2-4h)
- **What to build**: Tests for random, replay, and feedforward systems
- **Done when**: All null classes score C_PF < 0.3
- **Evidence**: `tests/null_class_tests.rs`, passing tests
- **Depends on**: T4-009

**T4-011: Create State Discrimination Tests**
- **Status**: ready
- **Capability**: Validation
- **Effort**: Medium (2-4h)
- **What to build**: Tests distinguishing conscious vs. unconscious states
- **Done when**: Tests reliably separate state classes
- **Evidence**: `tests/state_discrimination_tests.rs`, passing tests
- **Depends on**: T4-009

**T4-012: Implement Benchmark Battery**
- **Status**: ready
- **Capability**: Validation
- **Effort**: Large (4-6h)
- **What to build**: Complete benchmark suite with all tests
- **Done when**: Battery runs and produces comprehensive report
- **Evidence**: `tests/benchmark_battery.rs`, execution report
- **Depends on**: T4-010, T4-011

**T4-013: Document Type 4 Status**
- **Status**: ready
- **Capability**: Documentation
- **Effort**: Small (1-2h)
- **What to build**: Update CLAIMS.md with verified Type 4 measurements
- **Done when**: C-21, C-22, C-23 upgraded from SPECULATIVE to VERIFIED
- **Evidence**: Updated CLAIMS.md with measurement receipts
- **Depends on**: T4-012


### T-004: Run a first-sale market audit
- **Status**: completed[2026-04-24 Codex]
- **Capability**: Research-Evolve
- **Effort**: Medium (2-8h)
- **Fidelity Target**: Sketch
- **What to build**: A market note with real buyer targets, price anchors, and transaction requirements for a PhiFlow pilot engagement.
- **Done when**: `RESEARCH/first_sale_path/MASTER.md` exists and `BUSINESS.md` can cite specific buyers and comparables from that audit instead of generic placeholders.
- **Evidence**: `RESEARCH/first_sale_path/MASTER.md`
- **Read first**: `BUSINESS.md`
- **Depends on**: nothing
- **Don't touch**: source code

### T-005: Draft a buyer-safe pilot offer
- **Status**: completed[draft created 2026-04-24 Codex; receipt reconciliation still required before buyer delivery]
- **Capability**: Writing
- **Effort**: Small (< 2h)
- **Fidelity Target**: Pixels
- **What to build**: A one-page pilot offer or commercial terms draft that references only verified capabilities.
- **Done when**: `docs/pilot_offer.md` or `LICENSE_COMMERCIAL.md` exists and every claim in it can be traced to `WORKSPACE.md`, `BUSINESS.md`, or `QSOP/STATE.md`.
- **Evidence**: `docs/pilot_offer.md`
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
