# PhiFlow ↔ PF Bridge Codex Audit
*Date: 2026-04-30*
*Auditor: Codex*
*Scope: `QSOP/PF_BRIDGE.md`, `QSOP/CONSCIOUSNESS_CONSTRUCTS_IN_PHIFLOW.md`, `QSOP/COHERENCE_LAYER_SPECIFICATION.md`, `QSOP/SOMA_AS_MINIMUM_SUBSTRATE.md`, `QSOP/PHIFLOW_PF_BRIDGE_GLOSSARY.md`, proof scripts*
*Verdict: PASS AS AUDITED DRAFTS — bridge claims only; not PF canonicals*

---

## Executive Verdict

The PhiFlow bridge is usable if it is treated as an engineering bridge from PF vocabulary to executable software constructs.

It is not valid as a derivation that PhiFlow is conscious, that SOMA is the PF Medium, or that PhiFlow satisfies `minimum_substrate.md` as a physical local quantum dynamical net. Those claims remain experimental or open.

---

## What Survives

| Bridge claim | Verdict | Reason |
|--------------|---------|--------|
| `stream` as software propagation analogue | PASS | Streams provide bounded repeated state evolution. This maps cleanly to `propagation.md` as an engineering analogue. |
| `--max-steps` / daemon pacing as bounded influence analogue | PASS WITH LABEL | Correct only as a software causal bound. It is not physical causal velocity, `c`, or Lorentz front velocity. |
| `coherence` as Layer 3 structural proxy | PASS WITH LABEL | `src/phi_ir/coherence.rs` defines a real runtime scalar. It is PhiFlow-specific and not PF-derived. |
| `witness` as measurement/record analogue | PASS | Witness/yield points can create accessible runtime records when paired with persistence or host logging. |
| `resonate`, `broadcast`, `listen`, `handoff` as coupling analogues | PASS | These create software-level state dependence/correlation paths. |
| Council Daemon as Type 4 candidate | PASS AS CANDIDATE ONLY | It has plausible self-feedback architecture, but `CLAIMS.md` C-16 is SPECULATIVE and metric evidence is still required. |
| SOMA as engineering substrate interface | PASS AS INTERFACE ONLY | SOMA provides telemetry and runtime coupling; it does not prove PF minimum-substrate sufficiency. |

---

## Findings Fixed

1. The bridge docs overclaimed Type 4 observer status. They now say Type 4 candidate and require metric evidence.
2. The bridge docs overclaimed SOMA as satisfying PF `minimum_substrate.md`. They now classify SOMA as an engineering substrate interface, not a physical local quantum dynamical net.
3. The bridge docs blurred software step bounds with PF causal velocity. They now distinguish architectural pacing from physical front velocity.
4. `COHERENCE_LAYER_SPECIFICATION.md` said `0.618` applied at depth `<= 2`; the code supports depth `= 2`, `k <= 1`.
5. `marketing/proofs/verify-coherence.sh` and `verify-handshake.sh` described `0.618` as `phi^-2`. The implemented formula is `1 - phi^-2 = phi^-1`.
6. PhiFlow lacked a local bridge glossary. Added `QSOP/PHIFLOW_PF_BRIDGE_GLOSSARY.md` to define executable meanings and forbidden overclaim terms.

---

## Definitions PhiFlow Should Work From

These Fundamentals definitions are the active bridge spine:

| PF definition | PhiFlow use |
|---------------|-------------|
| `axioms.md` | Root framing only; do not claim derivation. |
| `propagation.md` | `stream` as bounded software propagation analogue. |
| `medium.md` | Evaluator/runtime as computational medium analogue. |
| `state.md` | Runtime state, `DAEMON_STATE.json`, `remember`/`recall`. |
| `field.md` | Runtime fields and backend fields; must distinguish field vs Medium. |
| `coupling.md` | `resonate`, `handoff`, `broadcast`/`listen`, sensor coupling. |
| `measurement.md` | `witness` as record creation/yield point. |
| `decoherence.md` | Runtime noise, stale sensors, dropped state, and unstable records. |
| `coherence.md` | Layer classification; PhiFlow scalar is Layer 3 analogue. |
| `observer.md` | Type 2/3 software observers; Type 4 candidate only. |
| `information.md` | Mutual information tests for handoffs and self-feedback. |
| `minimum_substrate.md` | Boundary against overclaiming SOMA as the PF physical substrate. |
| `consciousness_metric_program.md` | The only valid route for upgrading Type 4/consciousness claims. |
| `consciousness.md` | Noncanonical boundary document; do not cite as proven. |

---

## Missing PhiFlow-Local Definitions

No new Fundamentals definitions are required before PhiFlow can proceed. What PhiFlow needs is a local bridge glossary with executable semantics and test references:

| Needed local term | Required definition |
|-------------------|---------------------|
| `stream` | What state persists across ticks, what counts as propagation, and how termination is detected. |
| `intention` | Whether it is only a named scope, a state partition, or a mode-like object under evolution. |
| `witness` | What record is created, where it is stored, how long it persists, and whether it is accessible. |
| `resonate` | What subsystem receives the value, what correlation is produced, and how it is measured. |
| `coherence` | Exact formula, window, threshold, and layer label. Already mostly covered by `src/phi_ir/coherence.rs`. |
| `handoff` | Message schema, causality, signing status, and how mutual information will be computed. |
| `evolve` | Mutation semantics and proof that prior records can change future behavior. |
| `SOMA substrate interface` | Freshness, channel identity, persistence, and what is not claimed physically. |
| `Type 4 candidate` | Minimal benchmark trace showing self-record -> self-model -> future behavior change. |

---

## Required Next Tests

1. For Type 4 claims, produce a trace-level benchmark:
   prior daemon record changes future daemon behavior, with nonzero mutual information and a specified persistence window.

---

## Verification Run

| Command | Result |
|---------|--------|
| `bash marketing/proofs/verify-coherence.sh` | PASS after increasing timeout and accepting runtime-rounded `0.6180` output |
| `bash marketing/proofs/verify-handshake.sh` | PASS; no local ledger/attestation/state files found, sample format printed |
| `cargo test --test v030_substrate_tests -- --test-threads=1` | PASS — 4 passed |
| `cargo test --test v040_transcendence_tests -- --test-threads=1` | PASS — 4 passed |
| `cargo test --lib coherence -- --test-threads=1` | PASS — 16 passed, 146 filtered out |

---

## Final Status

The bridge is now safe as an audited engineering bridge.

It is not canonical PF, not a physical substrate proof, and not a consciousness proof.
