# Bijective Phase Map Memo
**Date:** 2026-03-31
**Agent:** Qwen Code
**Status:** Superseded proposal; not current repo truth

---

## Summary

This memo proposed replacing canonical multiplicative coherence with a `k = 1 -> 1.0` bijective rewrite in both the VM and evaluator.

That proposal is **not** the current state of the repository.

Current repo truth:

- `src/phi_ir/vm.rs` delegates to `src/phi_ir/coherence.rs`
- `src/phi_ir/evaluator.rs` delegates to `src/phi_ir/coherence.rs`
- `tests/phi_ir_wasm_runner.js` uses the same multiplicative model

---

## Current Canonical Formula

```text
base(depth) = 0.0                      when depth == 0
              1.0 - phi^(-depth)       otherwise

phase(k)    = 1.0                      when k <= 1
              1.0 - ln(k) / ln(TAU)    otherwise

coherence   = clamp(base(depth) * phase(k), 0.0, 1.0)
```

This means:

- depth 2 with `k <= 1` yields `0.618033988749895`
- `k = 1` preserves the base coherence; it does **not** force coherence to `1.0`

Source of truth: `src/phi_ir/coherence.rs`.

---

## Why This Memo Is Superseded

- The code examples in the original memo do not match the current `vm.rs` or `evaluator.rs`
- The memo treated a proposed rewrite as implemented fact
- The proposal was never promoted into `QSOP/STATE.md` as verified runtime truth

---

## If This Proposal Is Revived

Treat it as a fresh design proposal, not a completed implementation.

Minimum bar:

1. Update `src/phi_ir/coherence.rs` rather than forking formula logic across backends
2. Update evaluator, VM, and canonical WASM-runner tests together
3. Re-run the conformance gates from a working Rust shell
4. Only then add a dated verification entry to `QSOP/STATE.md`
