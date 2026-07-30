# PhiFlow PF Bridge Glossary
*Date: 2026-04-30*
*Status: AUDITED DRAFT — executable semantics contract*
*Depends on: `PF_BRIDGE_CODEX_AUDIT_2026-04-30.md`*

---

## Purpose

This file defines the PhiFlow-local terms used by the PF bridge. These are not new Fundamentals definitions. They are executable software semantics that map to the 19 canonical PF definitions as engineering analogues.

Boundary rule: if a term below sounds physical, the physical meaning is controlled by `D:\Fundamentals\definitions\*.md`. This file only defines the PhiFlow implementation meaning.

---

## Local Terms

| PhiFlow term | Executable meaning | PF bridge | Required evidence |
|--------------|--------------------|-----------|-------------------|
| `stream` | A bounded repeated evaluation context that can preserve state across yield/resume and terminate via `break stream` or step limits. | `propagation.md` analogue | Parser/evaluator support; stream tests; `--max-steps` guard. |
| `intention` | A named semantic/evaluation scope that partitions runtime state and affects coherence depth. | `state.md` / `mode.md` candidate | Parser/lowering support; coherence depth tests. |
| `witness` | An observation/yield point that can create an accessible runtime record from evaluator, sensor, or backend state. | `measurement.md` analogue | Witness logs, host callbacks, persistence path if record survival is claimed. |
| `resonate` | Runtime value emission/coupling into a resonance field, channel, or backend-specific target. | `coupling.md` analogue | Resonance field entries or backend output; no mutual information claim unless measured. |
| `coherence` | PhiFlow scalar `canonical_coherence(depth, k) = base(depth) * phase(k)`, clamped to `[0, 1]`. | `coherence.md` Layer 3 analogue | `src/phi_ir/coherence.rs`; depth 2, `k <= 1` gives about `0.618`. |
| `handoff` | Explicit inter-agent message/coupling event, optionally signed and ledgered. | `coupling.md` / `observer.md` Type 3 candidate | Handoff schema, event log, signature status, receiver behavior. |
| `evolve` | Runtime program-state mutation from an evaluated payload or bus event. | `state.md` / Type 4 candidate | Trace showing prior state or record changes future behavior. |
| `remember` / `recall` | Persistent key-value state access through the host provider. | `state.md` analogue | Roundtrip tests and storage backend identity. |
| `broadcast` / `listen` | Software-bus communication primitives for channel-mediated coupling. | `coupling.md` analogue | Queue/bus tests and channel identity. |
| `daemon` | Long-running runtime process that can preserve and resume program state across ticks/events. | `observer.md` candidate | `DAEMON_STATE.json`, event handling, restart/resume evidence. |
| `SOMA substrate interface` | Sensor/telemetry bridge that couples PhiFlow to local physical measurements. | Engineering bridge to `minimum_substrate.md`, not a physical PF substrate proof | Fresh schema-valid sensor state, channel identity, freshness window. |

---

## Terms Not Allowed Without Extra Evidence

| Term | Required before use |
|------|---------------------|
| "PhiFlow is conscious" | Successful `consciousness_metric_program.md` benchmark plus hostile audit. |
| "PhiFlow implements PF Type 4" | Trace-level evidence: self-record -> self-model -> future behavior change, with nonzero mutual information and persistence window. |
| "SOMA is the PF Medium" | A local quantum dynamical net model satisfying `minimum_substrate.md`. |
| "PhiFlow causal velocity" | Must be written as "software causal bound" unless a physical front-velocity measurement is being made. |
| "0.618 is PF-derived" | Not allowed. It is confirmed in PhiFlow code, but PF derivation remains OPEN. |

---

## Minimal Bridge Evidence Packet

Before a PhiFlow PF-bridge claim is presented externally, include:

1. The `.phi` source program.
2. The command used to run it.
3. The runtime output or ledger record.
4. The relevant PF mapping row from `PF_BRIDGE.md`.
5. The boundary label: `software analogue`, `confirmed runtime fact`, `candidate Type 4`, or `OPEN`.

---

## Current Verdict

PhiFlow has enough local vocabulary to proceed as an audited PF engineering bridge. It does not need new Fundamentals definitions before implementation work continues.
