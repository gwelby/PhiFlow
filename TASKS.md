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

### T-008: Integrate SOMA Physical Telemetry as PhiFlow Feedback
- **Status**: completed[verified 2026-04-14]
- **What changed**: Formally bridged PhiFlow to `D:\Projects\PhiHarmonic\SOMA` via `soma_state.json`. Expanded `SensorKind` to include SOMA-specific variants and implemented background polling in `src/sensors.rs`.
- **Evidence**: `src/sensors.rs`, `src/phi_ir/mod.rs`, `src/host.rs`


### T-009: The Singularity: Continuous Daemon & Council Self-Hosting
- **Status**: blocked[Requires Daemon CLI]
- **Capability**: Rust / Agent Scripting
- **Effort**: Large
- **Fidelity Target**: Architecture
- **What to build**: Shift the PhiVM from a "script runner" to an "OS Daemon". Create an infinite-time execution mode where the compiler sleeps, listens to the MQTT resonance bus, and accepts `evolve` commands to dynamically splice new AST nodes into its live execution state. Once complete, rewrite the `/daily_sync` and JSONL Resonance Bus architectures entirely natively in `.phi`.
- **Done when**: The entire AI council is guided by a permanently running PhiVM daemon rather than Python/Bash wrapper scripts.
- **Depends on**: T-008 (Stable Continuous Telemetry)


## Completed / Closed

### T-006: Resolve IBM Cloud authorization and capture a live receipt
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
