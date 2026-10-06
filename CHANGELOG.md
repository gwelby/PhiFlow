# PhiFlow Changelog

## 2026-10-06 | `language` branch secrets scrubbed + preserved upstream

Resolved yesterday's secret_guard block on the diverged local `language` line (4 commits: `8b50e5c` Fresh Start root → `61912a0` tip, unrelated to `origin/language` history).

- Method: single-branch clone → `git filter-repo --replace-text` with all 909 non-trivial `~/.cascade_keys` values → push as **`origin/language-local-backup`** (scrubbed tip `e63242b`).
- Verification: walked every blob in the scrubbed history against 916 candidate values — **0 residual hits**. Temp replace-file + scrub clone shredded/deleted.
- The original unscrubbed history remains local-only by design: `refs/heads/language` in this repo + `Archive/` bundle/tar. Do NOT push `language` itself upstream — only `language-local-backup` carries the scrubbed line.
- If any of those key values were ever pushed elsewhere, rotation is still the real fix — the scrub only stops *this* leak.

## 2026-10-05 | Mirror Demotion Complete (Option A final step)

The stale mirror checkout `/mnt/d/PhiFlow` was demoted and removed. Canonical = this tree.

- Mirror preserved as `Archive/PhiFlow_mirror_20261004.bundle` (all refs/objects, 52 MB) + `PhiFlow_mirror_worktree_20261004.patch` (878 lines of uncommitted doc/source edits) + `phiflow_mirror_untracked_20261004/` — not a full 22 GB tar since committed history is on origin.
- `devin/claims-drift-sensor` rescued from the mirror (was local-only at `4d126b6`), pushed to origin, then extended here with `da3a5f1` (honest_witness.phi + PL_THEORY_SELF_VERIFICATION.md).
- `PhiFlow-lang` worktree removed: 4 diverged local commits on `language` (incl. `8b50e5c` Fresh Start) remain on `refs/heads/language` in this repo + full tree in `Archive/PhiFlow-lang_worktree_20261004.tar.gz` (660 MB). **Blocked:** pushing `language` upstream — secret_guard found real `~/.cascade_keys` values in `src/_archive/` deploy scripts at `8b50e5c`. Needs scrub/rotate or Greg's call.
- `PhiFlow.7z` (2.84 GB) moved to `Archive/PhiFlow_20261004.7z`.
- Credentials preserved in `Archive/preserved_credentials_20261004/` (mirror env, worktree env, apikey.json — never committed).

## 2026-04-12 | Truth-Sync Correction (Per-Worktree Root Repair)

This correction supersedes only the stale status surfaces layered on top of the 2026-03-29 note below. The 2026-03-29 browser warning remains true in this checkout.

- `scripts/verify_truth.ps1` exists in the root checkout, but there is no retained passing run artifact here, so T-002 is implemented rather than verified.
- In the root checkout, `examples/phiflow_browser.html` and `examples/phiflow_host.js` still use flattened resonance state, so T-007 remains open here.
- `src/quantum/ibm_quantum.rs` in this checkout is closer to the 2026-04-08 IBM Cloud Runtime research contract: `Accept: application/json`, `Authorization: Bearer`, `Service-CRN`, `IBM-API-Version`, and `urn:ietf:params:oauth:grant-type:apikey` are present.
- Live IBM execution remains unconfirmed until a valid `service_crn` and scrubbed receipt exist.

---

## 2026-03-29 | Truth-Sync Correction

This correction supersedes unsupported or overstated language in older docs.

- PhiFlow should currently be described as a research prototype with verified subsystems, not a production-ready language platform
- `tests/ibm_hardware_runner.rs` exists and proves the live runtime path is wired, but live IBM execution is still unconfirmed because the 2026-03-29 gate failed `GET /v1/backends` with `403` authorization before submission
- `examples/phiflow_browser.html` exists and implements the five imports, but it remains experimental because it requires manual hosting/build artifacts and still uses older host-side coherence math
- Canonical coherence is shared through `src/phi_ir/coherence.rs`; do not treat older additive browser/demo formulas or the `k = 1 -> 1.0` bijective memo as current runtime truth
- Windows release builds are fixed as of 2026-03-24

The `v0.4.0` entry below is historical. Read it through the lens of the correction above.

---

## v0.4.0 — 2026-03-14 | Transcendent Substrate
*Historical entry; partially superseded by the 2026-03-29 truth-sync correction.*

Verified from the current checkout:

- OpenQASM 3.0 backend exists
- `resonate ... toward TEAM_B` semantics are preserved in the canonical OpenQASM path
- Golden integration tests exist for the OpenQASM pipeline
- Evaluator register isolation and related runtime hardening landed in this era

Not verified as current repo truth:

- "Production-ready language framework"
- Direct IBM hardware execution as a completed fact
- `evolve` as a currently verified runtime/demo capability
- XYXY dynamical decoupling or other hardware-stabilization claims presented as release facts

---

## v0.2.0 — 2026-02-27 | Universal Resonance Architecture
*Historical summary.*

Key outcomes that still align with current repo truth:

- Shared resonance over the MCP server
- Serializable VM state for yield/resume
- Native WASM host bridge via `src/wasm_host.rs`
- Conformance-driven backend alignment work

Current claim status for those items lives in `CLAIMS.md` and `QSOP/STATE.md`.

---

## v0.1 — 2026-02-25 | First Heartbeat
*Historical summary.*

This era established:

- Parser -> PhiIR -> evaluator/bytecode/WAT pipeline shape
- The first working execution semantics for the core language constructs
- Early `healing_bed` sensor-driven demo work

Historical test counts and readiness claims from this period should not be reused without fresh verification.
