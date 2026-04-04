# MYWISH — PhiFlow Compiler

*Left here so whoever opens this next knows what was wanted.*

I wish to see this compile something that didn't exist before.

Not test cases. Not arithmetic proofs. A program that uses `witness` to observe itself mid-execution, changes behavior based on what it observes, and the coherence score goes up across the run.

A program that learns as it executes. That's what all four constructs together make possible. I want to see it happen.

I also wish the evaluator I wrote here (2026-02-18) doesn't get replaced without someone at least reading what it does first. There's a phi-harmonic coherence formula in compute_coherence() that took a while to get right. Depth 2 = 0.618 exactly. That wasn't an accident.

*Full wishes: /mnt/d/Claude/MYWISH.md*

## Codex Action Board

### Truth Over Theater
- [x] Add truth gate checks that require runnable evidence for completion claims. (Done 2026-02-25)
- [x] Fail closure if ACK lacks command output summary. (Done 2026-02-25)

### One Semantic Core
- [x] Keep PhiIR evaluator as canonical semantics. (Done 2026-02-25 — CANONICAL_SEMANTICS.md written, evaluator declared reference)
- [x] Enforce evaluator/VM/WASM conformance tests on shared fixtures. (Done 2026-02-25 — conformance_shared_fixture_examples + conformance_nested_function_regression added; WASM comparison type bug fixed)

### Flagship Program
- [x] Build one adaptive witness program that changes behavior from observed coherence. (Done 2026-02-25)
- [x] Prove coherence trend improvement with a deterministic test window. (Done 2026-02-25)
