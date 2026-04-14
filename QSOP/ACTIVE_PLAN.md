# ACTIVE PLAN - PhiFlow
*Last updated: 2026-04-12*
*Purpose: keep the current execution plan, evidence, and research backing in one local reference so work does not depend on session memory.*

## Read First

1. `QSOP/STATE.md`
2. `WORKSPACE.md`
3. `TASKS.md`
4. `QSOP/PATTERNS.md`
5. `CHANGELOG.md`
6. This file

## Operating Rule

Do not treat memory, summaries, or older narrative docs as authority.

Every active lane below must show:
- local repo evidence
- external research backing if the lane depends on outside systems or claims
- explicit unknowns when the backing is still missing

If a lane does not have backing, the correct state is `UNKNOWN` or `RESEARCH NEEDED`, not a guess.

Status convention for this file:
- `verified` requires same-session repro or a retained in-repo run artifact
- `inspection-verified` is only for claims closed by source inspection
- `implemented` means the file/workflow exists but lacks retained passing execution proof

Research control-plane references:
- `D:\Projects\Research\AgentSettings\AGENTS.md`
- `D:\Projects\Research\AgentSettings\rules\claims_require_evidence.md`
- `D:\Projects\Research\RESEARCH.md`

## Current Execution Lanes

### Lane A - IBM Cloud authorization and live receipt

**Task ID:** `T-006`

**Done when:**
- `cargo test --test ibm_hardware_runner -- --ignored --nocapture` succeeds from this checkout
- a scrubbed receipt exists with backend, region, job ID, terminal status, and counts summary

**Local repo evidence:**
- `QSOP/STATE.md`
- `TASKS.md`
- `tests/ibm_hardware_runner.rs`
- `src/quantum/ibm_quantum.rs`
- `apikey.json`
- `tests/fixtures/ibm_runtime_sampler_result.json`

**What is already backed locally:**
- Canonical OpenQASM compile path exists.
- This checkout's runtime path includes `Accept: application/json`, `Authorization: Bearer`, `Service-CRN`, and `IBM-API-Version`.
- This checkout uses the IAM grant type `urn:ietf:params:oauth:grant-type:apikey`, which is closer to the 2026-04-08 research-backed contract.

**Research backing available now:**
- `D:\Projects\Research\AgentSettings\rules\claims_require_evidence.md`
  - IBM live execution cannot be marked done without a passing run path or receipt.
- `D:\Projects\Research\RESULTS\2026-04-08_ibm_cloud_runtime_authorization_for_back\03_FINDINGS.md`
  - The research-backed contract for `/api/v1/backends` is Bearer token auth plus `Service-CRN`, `Accept: application/json`, `IBM-API-Version`, and IAM grant type `urn:ietf:params:oauth:grant-type:apikey`.

**What is still missing:**
- A valid non-placeholder `service_crn` bound to the API key used here.
- A passing ignored live gate with a scrubbed receipt.
- Confirmation that the credential pair has the required IBM Quantum service permissions for backend discovery.

**Non-guess rule for this lane:**
- Do not claim live IBM verification until the ignored hardware runner succeeds and a scrubbed receipt is retained.

**Immediate next action:**
- Validate the root-checkout implementation against the 2026-04-08 research, then use valid credentials to rerun the smoke compile gate and the ignored live gate.

**Verification commands:**
```powershell
cargo test --test ibm_hardware_runner test_ibm_smoke_compiles_to_openqasm -- --nocapture
cargo test --test ibm_hardware_runner -- --ignored --nocapture
```

### Lane B - Browser host semantic parity

**Task ID:** `T-007`

**Current state:** incomplete in this checkout

**Local repo evidence:**
- `src/phi_ir/coherence.rs`
- `examples/phiflow_browser.html`
- `examples/phiflow_host.js`
- `WORKSPACE.md`
- `TASKS.md`

**What is already backed locally:**
- Canonical coherence truth is internal to this repo and lives in `src/phi_ir/coherence.rs`.
- The root-checkout browser and host still use flattened resonance state (`const resonanceField = []`) and `k` derived from array length.

**What is still missing:**
- Scoped `resonanceField` handling that matches current intention scope.
- Browser/host parity with canonical multiplicative coherence in this checkout.
- Retained validation artifact once the root browser host is corrected.

**Non-guess rule for this lane:**
- Do not inherit compiler-worktree browser claims into the root checkout.
- If root browser behavior differs from `src/phi_ir/coherence.rs`, the root code is still wrong.

**Immediate next action:**
- Refactor the root `examples/phiflow_browser.html` and `examples/phiflow_host.js` to use scoped resonance cardinality, then retain validation evidence in this checkout.

### Lane C - One-command verification gate

**Task ID:** `T-002`

**Current state:** implemented, not verified

**Local repo evidence:**
- `scripts/verify_truth.ps1`
- `TASKS.md`
- `WORKSPACE.md`
- `QSOP/STATE.md`

**What is already backed locally:**
- The script exists and wraps the intended truth commands:
  - `cargo test --lib openqasm`
  - `cargo test --quiet --test golden_integration_tests`
  - `cargo test --quiet --test repro_bugs`
  - `cargo test --test phi_ir_conformance_tests -- --nocapture`

**What is still missing:**
- A retained passing execution artifact or same-session repro in this worktree.

**Non-guess rule for this lane:**
- Do not upgrade T-002 to verified without retained passing evidence in the root checkout.

**Immediate next action:**
- Record a retained passing run before promoting T-002 from implemented to verified.

## Working Agreement For Future Sessions

Before active code work starts:
1. Read `QSOP/STATE.md`.
2. Read this file.
3. If the task depends on an external system, check whether a research-backed source is already named here.
4. If no backing exists, create or queue the research question before guessing in code.

Before marking work complete:
1. Name the file changed.
2. Name the verification command run.
3. Name the exact artifact or receipt produced.
4. If external backing is still missing, mark the lane blocked instead of complete.

## Handoff Note

As of 2026-04-12, the root checkout is closer to the research-backed IBM auth contract than the compiler checkout, but it still lacks valid live credentials and a receipt. Browser canonicalization remains incomplete in the root checkout, and the one-command truth gate is implemented but not yet verified by retained execution proof.
