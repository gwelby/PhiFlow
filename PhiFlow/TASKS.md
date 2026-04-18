# TASKS: PhiFlow
*Last updated: 2026-03-15*
*See also: WORKSPACE.md (technical state) · BUSINESS.md (income state)*

## Active Tasks

### T-001: Stabilize Windows release build for `phic`
- **Status**: ready
- **Capability**: Rust
- **Effort**: Medium (2-8h)
- **Fidelity Target**: Photo
- **What to build**: Reduce or gate the release-build footprint enough that `cargo build --release --bin phic` succeeds on this Windows host.
- **Done when**: `cargo build --release --bin phic` exits successfully and `target\\release\\phic.exe examples\\council_vote.phi --target openqasm` emits QASM on the same machine.
- **Read first**: `Cargo.toml`
- **Depends on**: nothing
- **Don't touch**: parser semantics unless the build fix truly requires it

### T-002: Add a canonical verification gate script or workflow
- **Status**: ready
- **Capability**: DevOps
- **Effort**: Small (< 2h)
- **Fidelity Target**: Photo
- **What to build**: A single repo-level script or workflow that runs the focused truth gates used in this workspace.
- **Done when**: one command runs `cargo test --lib openqasm`, `cargo test --quiet --test golden_integration_tests`, and `cargo test --quiet --test repro_bugs`, and fails if any one of them regresses.
- **Read first**: `QSOP/STATE.md`
- **Depends on**: nothing
- **Don't touch**: release profile settings

### T-003: Audit README and changelog claims against current evidence
- **Status**: ready
- **Capability**: Writing
- **Effort**: Small (< 2h)
- **Fidelity Target**: Sketch
- **What to build**: Bring `README.md` and `CHANGELOG.md` down to verified language or clearly mark historical / experimental claims.
- **Done when**: those files no longer present unverified IBM hardware, production-readiness, or full-backend-equivalence claims as current fact without a dated verification note.
- **Read first**: `WORKSPACE.md`
- **Depends on**: nothing
- **Don't touch**: source code

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
- **What to build**: A one-page pilot offer or commercial terms draft that only references verified capabilities.
- **Done when**: `docs/pilot_offer.md` or `LICENSE_COMMERCIAL.md` exists and every claim in it can be traced to `WORKSPACE.md`, `BUSINESS.md`, or `QSOP/STATE.md`.
- **Read first**: `BUSINESS.md`
- **Depends on**: T-004
- **Don't touch**: runtime code
