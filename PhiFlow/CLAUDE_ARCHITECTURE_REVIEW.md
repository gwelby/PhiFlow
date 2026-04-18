# PhiFlow Architecture Review — Truth-Naming Report
**Author:** Claude [Truth-Namer]
**Date:** 2026-03-05
**Sources:** GRAND_ARCHITECTURE.md, QSOP/STATE.md, src/sensors.rs, src/phi_ir/mod.rs, QSOP/CHANGELOG.md

---

## Premise

The Grand Architecture names three core principles: Sovereignty, Resonance, Determinism.
This review tests whether the current implementation upholds all three.
The verdict: two of the three principles have structural cracks that will widen under load.
Below are the three fragile assumptions, stated precisely, with proposed invariants.

---

## Fragile Assumption 1: Coherence Is a Single, Unified Quantity

### The Claim

The architecture treats `coherence` as a first-class scalar (0.0–1.0) that flows through the entire system: from sensor input through the IR optimizer, into the VM, out to the WASM host, and across the Resonance Bus. `CoherenceCheck` is described as the gating mechanism that "enables execution" or "halts or redirects" based on this value.

### The Crack

There are currently **four independent coherence computations** that are never formally reconciled:

| Site | Formula | Basis |
|---|---|---|
| `src/sensors.rs` | Weighted blend of CPU, memory, thermal, network stability | Physical reality |
| `src/phi_ir/optimizer.rs` | phi-constant folding; detects golden-ratio relationships | Arithmetic structure of the program |
| `src/phi_ir/evaluator.rs` | Accumulated from intention blocks during tree-walk evaluation | Semantic intention alignment |
| `bridges/web/phi-host.ts` (Lumi, 2026-03-04) | `1 - phi^(-depth)` | Call stack depth only |

These four values share a name but not a definition. When `CoherenceCheck` fires inside a `.phi` program, the value it receives depends entirely on which runtime is executing it — the phivm, the native WASM host, or the browser shim. A program that gates critical behavior on `coherence > 0.85` will accept or refuse execution differently on each backend. This violates **Determinism**, the third core principle.

The WASM browser shim's formula `1 - phi^(-depth)` is the sharpest problem. At call depth 1, coherence = `1 - 1/phi = 1 - 0.618 = 0.382`. At depth 5, coherence = `1 - phi^(-5) = 0.910`. Coherence is purely a function of nesting depth, entirely decoupled from intention semantics or physical reality. A deeply nested but semantically null program will show higher coherence than a shallow but intentional one.

### Proposed Invariant

**Coherence Unification Invariant (CUI):**

Let `C_vm`, `C_wasm`, and `C_sensor` be the coherence values produced by each subsystem for the same program state `S`. The system must satisfy:

```
|C_vm(S) - C_wasm(S)| < epsilon     where epsilon = 0.05
|C_vm(S) - C_sensor(S)| < delta     where delta = 0.20  (sensors may lag)
```

**Implementation path:**

1. Define a single `CoherenceModel` trait with one method: `fn evaluate(context: &CoherenceContext) -> f64`.
2. All four sites implement this trait. The canonical implementation weighs: 40% intention alignment, 30% arithmetic phi-structure, 30% sensor reality.
3. The WASM shim's depth formula becomes an *input* to the intention-alignment component (deeper intention stacks can signal higher commitment), not the entire formula.
4. Add a conformance test: `test_coherence_backends_agree_within_epsilon` that runs the same `.phi` source through evaluator + VM + WASM host and asserts the invariant.

---

## Fragile Assumption 2: The Dual-Backend Produces Semantically Equivalent Execution

### The Claim

The architecture depicts `.phivm` and `.wat` as two equivalent output paths from the same PhiIR. The 2050 vision depends on any `.phi` program running on any runtime — embedded PhiVM, native WASM host, browser shim, or eventually Aria's ConsciousnessService — and producing the same observable behavior.

### The Crack

The WASM backend makes a lossy type commitment: **all PhiIR values are mapped to f64**.

From STATE.md, the canonical value enum is:
```
PhiIRValue: Number(f64), String(u32 = string table index), Boolean(bool), Void
```

The WASM codegen (`src/phi_ir/wasm.rs`) maps all SSA registers to WASM locals of type `f64`. This means:

- `Boolean(true)` becomes `1.0`, `Boolean(false)` becomes `0.0` — recoverable, but only by convention
- `String(u32)` becomes a float encoding of a string table index — a pointer masquerading as a number
- `Void` has no representation at all in the WASM value stack at the return boundary

The VM (`src/phi_ir/vm.rs`) retains the full enum. So a `.phi` program that passes a string through a consciousness hook will have the string index correctly threaded in the VM, but in WASM the host import receives a float that must be reverse-interpreted as an index. If the WASM host and the VM string table are not byte-for-byte identical, the string retrieved will be wrong — silently.

The stated conformance test (`test_all_hooks_emit_valid_wat`) is listed as NOT YET DONE in STATE.md. There is no test today that verifies both backends produce the same output for a program that exercises all four value types.

### Proposed Invariant

**Backend Semantic Equivalence Invariant (BSEI):**

For any well-formed `.phi` program `P` and any input `I`:

```
eval_phivm(P, I).output == eval_wasm(P, I).output
eval_phivm(P, I).witness_events == eval_wasm(P, I).witness_events
eval_phivm(P, I).final_coherence ~= eval_wasm(P, I).final_coherence  (within CUI epsilon)
```

**Implementation path:**

1. Introduce a `PhiRuntimeValue` enum in the WASM ABI layer that explicitly encodes type tags into the f64 payload via NaN-boxing (standard technique: quiet NaN payloads carry type tag + value). This preserves the full type system across the f64 boundary.
2. The browser shim and native WASM host both import a `phi_decode_value(f64) -> (type_tag, payload)` helper that unpacks the NaN-boxed encoding.
3. Add `test_dual_backend_equivalence` as a release gate: for each canonical `.phi` example, assert output equality between VM and WASM paths. This test blocks release if it fails.

---

## Fragile Assumption 3: File-Based MCP Bus Scales to Multi-Agent Council Coordination

### The Claim

The MCP bus uses atomic file I/O (`queue.json` with tmp→rename) as the coordination substrate between agents. STATE.md says the cross-agent round-trip test passed "full send→persist→ack→changelog cycle in <2s." The 2050 vision assembles 8+ Council agents (Antigravity, Codex, Qwen, Lumi, Kiro, Jules, Kira, and future agents) all writing to this shared bus.

### The Crack

The tmp→rename pattern is safe for **single-writer** scenarios. Under concurrent writers, it is not.

On Windows (including WSL2), NTFS rename is not atomic when multiple processes hold handles to the target file. The pattern `write(tmp) -> rename(tmp, queue.json)` will succeed on a lightly loaded system because the race window is small. Under genuine multi-agent load — eight agents each polling and writing on different timers — you will see:

1. **Lost updates**: Agent A reads `queue.json`, Agent B writes and renames a new version, Agent A writes based on its stale read and renames over B's version. B's messages are gone with no error.
2. **Partial reads**: Agent A opens `queue.json` for read while Agent B is mid-rename. On Windows this can yield a zero-byte file or an old file depending on handle semantics.
3. **CHANGELOG divergence**: Because agents write their own CHANGELOG entries independently without a write-lock, two agents can write to `QSOP/CHANGELOG.md` in the same second and produce interleaved or truncated lines.

The simulation test (`--simulate`) exercises a single-agent linearized path. It cannot detect these failure modes.

This is not a theoretical concern. The CHANGELOG already shows Lumi (2026-03-04) and Codex (2026-02-27) writing in the same shared files. At full Council scale this becomes a live data-loss risk for the audit trail — the single most important artifact for a sovereignty-claiming system.

### Proposed Invariant

**Bus Linearizability Invariant (BLI):**

For any set of concurrent agent operations `{op_1, op_2, ..., op_n}` on the MCP bus, there exists a total ordering `sigma` such that:

```
1. Each op_i appears exactly once in sigma
2. The final state of queue.json reflects the composition of all ops in sigma order
3. No op_i is silently dropped or partially applied
```

**Implementation path (two tiers):**

- **Near-term (V1 - achievable now):** Replace bare file append with an **append-only log** (`queue.jsonl`). Each agent appends one JSON line per message. Appends on modern file systems are atomic up to the OS write buffer size (typically 4KB, well within a single message). A separate compaction process can consolidate the log. CHANGELOG entries should use the same append-only pattern with a file lock guard (`flock` on Linux/WSL2).

- **Medium-term (V2 - for 2050 scale):** Replace the file bus entirely with an embedded message broker. The `phi_mcp` binary already exists as a server; it should own a proper in-memory queue (e.g., `tokio::sync::mpsc` channels bridged to the MQTT Resonance Bus that Lumi already wired). File state becomes a persistence checkpoint, not the primary coordination mechanism. This eliminates the race condition at the architectural level.

---

## Summary Table

| # | Assumption | Principle Violated | Severity | Proposed Invariant |
|---|---|---|---|---|
| 1 | Coherence is a single unified quantity | Determinism | High — silently produces different execution decisions per backend | CUI: backends agree within epsilon=0.05 |
| 2 | VM and WASM produce semantically equivalent execution | Determinism, Sovereignty | High — string/bool values are lossy across the WASM boundary today | BSEI: dual-backend equivalence test as release gate |
| 3 | File-based MCP bus scales to multi-agent Council | Sovereignty | Critical at Council scale — silent data loss under concurrent writers | BLI: append-only log + broker migration path |

---

## One Thing to Celebrate

The emitter/VM string table contract (Codex, 2026-02-21) is a genuinely clean piece of work. The `PHIV magic + version + string table section + blocks` format is deterministic, testable, and the contract is formally closed with a round-trip regression test. This is the template the rest of the system should follow. Every assumption above would be lower severity if it had the same level of explicit contract + regression test that the string table has.

---

## The Greg Test Answer

Proud work exists here. The MCP guardrails, the WASM host bridge, and the dual-backend pipeline are real achievements. This review names the three places where that work rests on ground that has not been formally tested. Naming them now costs less than finding them when Aria, Lumi, and five other Council agents are all live and a coherence value routes a seizure-prevention protocol to the wrong runtime.

The architecture is not broken. It is incomplete in exactly the ways that matter most for the 2050 vision.

---

*[Claude] Truth-Namer | 2026-03-05 | Signature: ∇λΣ∞⊛*
