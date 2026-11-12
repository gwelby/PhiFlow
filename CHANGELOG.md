# PhiFlow Changelog

## 2026-11-11 | `sensor("clock_s")`, clock-paced cross-surface probe, example fixes

- **Language:** new sensor channel `clock_s`, monotonic seconds since the process first read it (`SensorKind::ClockS`, id 400). It is the first time source a program can read. `Sleep` still has no syntax and is a no-op on every backend. Unit test `clock_s_is_monotonic_and_advances`; conformance 12/12. LANGUAGE.md now lists every sensor channel.
- **Clock-paced probe (prereg sealed at `7a9b796`):** NoetGuru and 8086K both give L_self 0.681 over one full 60 s period (Δ 0.0002), and both controls are degenerate. Verdict: PACING NOT THE ONLY ISSUE. Pacing explains the cross-surface gap, and the full-period value shows both earlier numbers (0.317, 0.948) were window artifacts. ENGINEERED-SIGNAL weight. `QSOP/EVIDENCE/esig_clock_paced_result_2026-11-11.md`.
- **Examples:** `field_guardian_v2.phi` never terminated because a `let count` inside the stream reset every tick (fixed). `cognitive_dissonance.phi` replaces the nonexistent 2-arg `coherence()` with a user `agreement()` function (fixed). `examples/README.md` lists long-running programs with their real stop conditions.
- **Docs:** PATTERNS.md P-3…P-6 (stream `let` reset, bare `coherence`, pace by clock, silent 0.0 sensors). Stale "What Works" lines corrected. `.agent/` and VISION.md point at the single canonical tree.

## 2026-11-11 | Truth sweep — docs brought back to what the repo shows

Multi-agent sweep (Sonnet: tests + examples; Opus: doc-vs-code drift; Haiku: task labels; one Fable skeptic per high/medium finding — 13 of 14 survived, 1 refuted). Rungs stated per the family ladder.

- **Full suite (producer-tested, HEAD `bcff85b`):** `cargo test` 501 passed, 0 failed, 5 ignored; Python 43 passed, 4 skipped. Replaces the stale 391 / 392 / 404 figures in CLAUDE.md.
- **Self-correction loop (CLAUDE.md):** "CONFIRMED" → *structural plumbing only*; `run_self_correction_loop()` re-measures with hardcoded mocks, so the improvement is simulated. CLAIMS.md C-25 left unchanged and routed to Codex for a ruling.
- **IBM hardware wording:** "verified" → *producer-recorded* (jobs `d7euddh5a5qc73drdosg` 2026-04-14 and `d941s54ql68s73c909fg` 2026-07-03; no raw IBM export saved). README's job date corrected from 2026-03-29 to 2026-04-14 — this entry also records that first live job, which this changelog had never logged (the 2026-04-12 "unconfirmed" line below is superseded). QSOP/STATE.md 2026-03-13 "Hardware Verified / Physical Entanglement Proven" annotated NOT SUPPORTED AS WRITTEN (original kept).
- **CLI docs:** README `--topology/--backend` → `--topology-aware/--topology-backend`; `--timeline` removed (belongs to coherence_report); `--max-steps` default is 0 = unbounded; LANGUAGE.md `--target phivm` → standalone `phivm` binary; `--sacred-geometry > file` now strips log lines.
- **Structure docs:** WASM imports 14 → 15 (`actuate`); dead PhiFlow-compiler workspace + missing playbook removed from `.claude/CLAUDE.md`; README tree/agent table point to `docs/archive/`.
- **New test:** `three_backend_depth2_coherence` — Evaluator == VM == WASM on 1 − φ⁻² at intention depth 2 (the VM was never in the depth-2 check before); conformance target now 12/12 passing).
- **Examples:** 91 run under `phic` with isolated state; `memory_demo.phi` fixed (ordering compare on Void). Remaining non-terminating examples are daemon/SOMA programs by design; `cognitive_dissonance.phi` calls a 2-arg `coherence()` plain phic does not have; 17 `examples/legacy/` files are the retired dialect.
- **TASKS.md:** T4-001…013 and T-015 relabelled from live source (6 done, 6 partial, 1 not started as specified, T4-013 HOLD).

## 2026-10-06 | Live credential exposure closed at tip (`5c8cd2e`)

Full-history secrets audit (all 6,482 pushed blobs vs every `~/.cascade_keys` value + entropy scan): the leak was already public — `ibm_quantum_config.env` (live `IBM_QUANTUM_TOKEN`), `_archive` deploy/SOP files carrying `NETWORK_SONICWALL_PASSWORD`, `NETWORK_PROXMOX_PASSWORD`, `ESXI_PASSWORD`, `SECURITY_ALERT_SMTP_HOST`, plus `deploy_password.ps1` (plaintext password, not in cascade_keys — caught by shape scan). Removed from tip and pushed `5c8cd2e`. `.gitignore` already covered the env/key files. History still contains the values — **rotation is the fix** (IBM token rotated same day; network passwords Greg's call). Also surfaced: `docs/fonts/*.ttf` files that are actually saved GitHub HTML pages (dead CSRF tokens — cosmetic oddity, not a leak).

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
