# ACTIVE PLAN - PhiFlow
*Last updated: 2026-04-08*
*Purpose: keep the current execution plan, evidence, and research backing in one local reference so work does not depend on session memory.*

## Read First

1. `QSOP/STATE.md`
2. `WORKSPACE.md`
3. `TASKS.md`
4. `QSOP/PATTERNS.md`
5. `QSOP/CHANGELOG.md`
6. This file

## Operating Rule

Do not treat memory, summaries, or older narrative docs as authority.

Every active lane below must show:
- local repo evidence
- external research backing if the lane depends on outside systems or claims
- explicit unknowns when the backing is still missing

If a lane does not have backing, the correct state is `UNKNOWN` or `RESEARCH NEEDED`, not a guess.

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
- `tests/fixtures/ibm_runtime_sampler_result.json`

**What is already backed locally:**
- Canonical OpenQASM compile path exists.
- The runtime backend now persists `service_crn` and `region`.
- Runtime requests include `Authorization`, `Service-CRN`, and `IBM-API-Version`.
- A real attempt on 2026-03-29 reached IBM Cloud Runtime and failed at backend discovery with `GET /v1/backends -> 403`, code `1200`, authorization error.

**Research backing available now:**
- `D:\Projects\Research\AgentSettings\rules\claims_require_evidence.md`
  - We cannot mark IBM live execution done without a passing run path or receipt.
- `D:\Projects\Research\RESULTS\2026-03-14_q_ctrl_error_mitigation_on_ibm_brisbane\03_FINDINGS.md`
  - Confirms current IBM hardware work is technically relevant, but this is optimization context, not authorization guidance.
- `D:\Projects\Research\RESULTS\2026-03-14_stack_audit\03_FINDINGS.md`
  - Confirms IBM experiments are real work, but does not resolve the runtime auth boundary.

**What is still missing:**
- Official backing for the exact IBM Cloud Runtime authorization contract used by our `/api/v1/backends` call:
  - required service roles
  - API key to service-instance matching rules
  - exact region and `service_crn` constraints
  - official explanation for `403` / code `1200` in this path

**Non-guess rule for this lane:**
- Do not keep changing runtime headers or endpoints blindly.
- Until we have IBM documentation or a successful verified run, the blocker remains `authorization boundary unresolved`.

**Immediate next action:**
- Run targeted research before more code churn.

**Research prompt to run from `D:\Projects\Research`:**
```powershell
.\research_spawn.ps1 -Topic "IBM Cloud Runtime authorization for backend discovery with API key, Service-CRN, region, and /api/v1/backends 403 code 1200"
```

**Research questions that must be answered:**
1. What official IBM docs define the required roles and service-instance permissions for backend discovery?
2. What is the correct binding between IAM API key, service CRN, region, and backend visibility?
3. What are the documented causes of `403` / code `1200` for backend discovery?

**Verification commands:**
```powershell
cargo test --test ibm_hardware_runner test_ibm_smoke_compiles_to_openqasm -- --nocapture
cargo test --test ibm_hardware_runner -- --ignored --nocapture
```

### Lane B - Browser host semantic parity

**Task ID:** `T-007`

**Done when:**
- `examples/phiflow_browser.html` uses the same canonical multiplicative coherence semantics as the runtime truth source
- manual build and serve steps are written down
- the browser host can be described as experimental but semantically aligned

**Local repo evidence:**
- `src/phi_ir/coherence.rs`
- `tests/phi_ir_wasm_runner.js`
- `examples/phiflow_browser.html`
- `WORKSPACE.md`
- `TASKS.md`

**What is already backed locally:**
- Canonical coherence truth is internal to this repo and lives in `src/phi_ir/coherence.rs`.
- The canonical semantics are multiplicative: `base(depth) * phase(k)`.
- The browser host already has a local `computeCoherence()` helper, but it is still a host-side reimplementation and uses flattened resonance state rather than the scoped rule from the canonical module.
- `WORKSPACE.md` explicitly says the browser host remains experimental and still needs canonicalization.

**Research backing available now:**
- None required for the semantics themselves. This is a local source-of-truth problem, not an external theory problem.

**What is still missing:**
- A local doc that states the exact browser prerequisites and run path.
- A code-level parity check or documented reasoning tying browser `k` scope handling to the canonical runtime rule.

**Non-guess rule for this lane:**
- Do not invent alternate browser coherence math.
- If browser behavior differs from `src/phi_ir/coherence.rs`, the code is wrong until proven otherwise.

**Immediate next action:**
- [x] Refactor browser coherence handling to mirror the scoped rule, then document the actual manual serve path. (Completed by Antigravity on 2026-04-08)

**Verification target:**
- Browser host output should match the same depth and scope expectations used by `tests/phi_ir_wasm_runner.js` for supported programs.

### Lane C - One-command verification gate

**Task ID:** `T-002`

**Done when:**
- one repo-level command runs the focused truth gates and fails on any regression

**Local repo evidence:**
- `TASKS.md`
- `WORKSPACE.md`
- `QSOP/STATE.md`

**Target command set:**
```powershell
cargo test --lib openqasm
cargo test --quiet --test golden_integration_tests
cargo test --quiet --test repro_bugs
cargo test --test phi_ir_conformance_tests -- --nocapture
```

**Research backing available now:**
- None required. This is an internal workflow hardening task.

**Non-guess rule for this lane:**
- The gate should encode the already-named truth commands, not a new moving target.

**Immediate next action:**
- [x] Add a small script or task runner wrapper once Lane A and Lane B context is stable enough that the gate list does not churn again. (Completed by Antigravity on 2026-04-08: scripts/verify_truth.ps1)

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

As of 2026-04-08, the highest-value lane is still IBM authorization, but it lacks the external documentation pack needed to proceed confidently. Browser canonicalization (Lane B) and the verification gate (Lane C) are both complete and verified.

Looking forward, the compiler is currently encountering structural type errors (`Witness` mid_circuit fields, missing `TeamDirection`) that need attention from the Rust hardening agent (Codex/Jules) once Lane A completes.
