# TASKS: PhiFlow
*Last updated: 2026-03-29 truth-sync pass*
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

### T-006: Resolve IBM Cloud authorization and capture a live receipt
- **Status**: ready
- **Capability**: Rust + Cloud Ops
- **Effort**: Medium (2-8h)
- **Fidelity Target**: Photo
- **What to build**: Move Pipe 2 from structurally ready to live-confirmed by fixing IBM Cloud Runtime authorization for the existing ignored hardware runner.
- **Done when**: `cargo test --test ibm_hardware_runner -- --ignored --nocapture` succeeds from this checkout and writes a scrubbed receipt with backend, region, job ID, terminal status, and counts summary.
- **Read first**: `QSOP/STATE.md`, `tests/ibm_hardware_runner.rs`, `src/quantum/ibm_quantum.rs`
- **Depends on**: valid IBM Cloud API key + matching `service_crn`
- **Don't touch**: README marketing language until the receipt exists


## Completed / Closed

### T-002: Add a canonical verification gate script or workflow
- **Status**: completed[verified 2026-04-08]
- **What changed**: Created `scripts/verify_truth.ps1` to run all four required truth gates, failing on any regression.
- **Evidence**: `scripts/verify_truth.ps1`

### T-007: Canonicalize the browser host and document manual prerequisites
- **Status**: completed[verified 2026-04-08]
- **What changed**: Brought `examples/phiflow_browser.html` and `examples/phiflow_host.js` into semantic parity with multiplicative coherence (base * phase) and scoped `resonanceField`. Build/serve steps documented in HTML header.
- **Evidence**: `examples/phiflow_browser.html`, `examples/phiflow_host.js`

### T-001: Stabilize Windows release build for `phic`
- **Status**: completed[verified 2026-03-24]
- **What changed**: `cargo build --release --bin phic` now succeeds on the Windows host after the profile repair (`lto = "thin"`, `codegen-units = 4`).
- **Evidence**: `QSOP/STATE.md`, `CLAIMS.md`

### T-003: Audit README and changelog claims against current evidence
- **Status**: completed[truth-sync pass]
- **What changed**: `README.md`, `CHANGELOG.md`, and the core authority docs were brought back into line with `QSOP/STATE.md` and current repo files.
- **Evidence**: `README.md`, `CHANGELOG.md`, `AGENTS.md`, `WORKSPACE.md`, `CLAIMS.md`
