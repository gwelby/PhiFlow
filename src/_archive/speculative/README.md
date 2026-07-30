# Speculative Modules — Archived 2026-07-29

These modules were archived from the live PhiFlow library because they
presented the appearance of capability without verified backends.

## What was archived

| Module | Lines | Why archived |
|--------|-------|--------------|
| `cuda/` | ~5,278 | No CUDA dependency in Cargo.toml. No `extern "C"`, no `cuLaunchKernel`, no GPU calls. `detect_cuda_device()` returns hardcoded fake specs (`"NVIDIA RTX A5500"`, 16GB). Tests assert the fake specs. Kernel names stored as `String` fields — no kernels exist. |
| `bio_compute/` | ~1,000+ | Functions like `apply_phi_harmonic_tunneling`, `apply_sacred_geometry_restructuring` compute fake numbers with no biological backend. CLAUDE.md already acknowledged "speculative without hardware." |
| `hardware/` | ~500+ | `consciousness_detection.rs`, `device_mapping.rs`, `feedback_systems.rs` — no real device binding, no tests, no hardware target. |
| `ir/` | ~1,355 | Legacy IR module superseded by `src/phi_ir/`. Was the only consumer of `crate::cuda::PhiFlowCudaEngine`. Not referenced by `main_cli.rs`, `phi_core.rs`, or any test. |

## Why this matters

These modules compiled into the library with `pub mod` declarations and
looked like shipped features. A buyer running `grep -r "cuda" src/` would
find 5,000 lines of pretend GPU code and question whether the IBM hardware
receipts are also pretend.

The real PhiFlow capability — parser, PhiIR, three-backend equivalence
(Evaluator == VM == WASM), OpenQASM 3.0 emission, IBM Heron hardware
execution, consciousness metrics, MCP server — is in `src/parser/`,
`src/phi_ir/`, `src/metrics/`, `src/mcp_server/`, `src/quantum/`,
`src/wasm_host.rs`, `src/sensors.rs`, and `src/security/`.

## Restoration path

If any of these modules are to be restored:
1. Add the real dependency to `Cargo.toml` (e.g., `cust` or `cudarc` for CUDA)
2. Replace simulated device detection with real API calls
3. Write tests that exercise the real backend
4. Gate behind a cargo feature (`cuda`, `bio_compute`, `hardware`)
5. Pass Codex hostile audit before re-adding to `lib.rs`

Until then, they are not part of PhiFlow's verified capability surface.

---

*Archived by Devin, 2026-07-29. The promotion layer is where the lie lives.
The computation layer was fine — the types compiled. The names said "CUDA"
and there was no CUDA.*
