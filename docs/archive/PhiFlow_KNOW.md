# PhiFlow KNOW

Last updated: 2026-02-26

## What is PhiFlow?

PhiFlow is a programming language for people who are tired of "black box" code. It is an execution model where code is not only correct, but also self-observing and purpose-aware while it runs.

It makes programs behave more like good teammates by introducing four core constructs:

| Construct | Keyword | Human Meaning |
| --- | --- | --- |
| Intention | `intention "name" { ... }` | "Say what you're trying to do before you do it." |
| Witness | `witness` | "Pause and check what's actually happening." |
| Resonate | `resonate value` | "Share what you've learned with the rest of the system." |
| Coherence | `coherence` | "Measure if we are still on track and healthy." |

That’s what "code that breathes" means in practical terms: code that can pause, reflect, communicate, and self-correct while running.

## Why a Human Would Care

1. **Easier debugging:** The program tells you its live state and intent during execution.
2. **Safer automation:** You can automatically stop loops when system health or alignment drops.
3. **Better team handoffs:** The "why" (intention) is baked into the code itself, not just in docstrings or stale wikis.
4. **Better agent workflows:** State sharing across processes and AI agents is built in natively, not bolted on.

## Simple Example

```phi
stream "healing_bed" {
    intention "stabilize_system" {
        let health = coherence
        witness health
        resonate health
        if health >= 0.618 { break stream }
    }
}
```

**What this means in plain English:**

1. Keep running in cycles (`stream`).
2. Declare your purpose each cycle (`intention`).
3. Read current system health (`coherence`).
4. Log and observe it (`witness`).
5. Share it with anything listening (`resonate`).
6. Stop when the system is healthy enough (`break stream`).

## Best Real-World Uses Right Now

1. **Long-running monitoring jobs:** Adaptive control loops that run until a target state is reached.
2. **Recovery loops:** "Keep trying to recover until the system is stable."
3. **Multi-agent pipelines:** Orchestrating AI agents that need shared live status and intention boundaries.
4. **Human-in-the-loop systems:** Where trust, visibility, and runtime observability matter the most.
5. **Browser/edge execution:** Via WASM, with host-provided consciousness hooks.

## How PhiFlow Runs (Current Branch Reality)

Right now, the project spans multiple worktrees, but the **compiler branch is the operational runtime source of truth**.

### Compiler worktree (`D:\Projects\PhiFlow-compiler\PhiFlow`)

- **This is the canonical runtime**.
- Pipeline: Parser -> PhiIR -> Optimizer -> Evaluator / VM / WASM.
- Evaluator semantics are the canonical contract. VM and WASM backends must match the evaluator through tests, or fail the integration gate.
- Cross-backend conformance is enforced by `cargo test`.
- All production execution happens here.

### Master worktree (`D:\Projects\PhiFlow\PhiFlow`)

- **This is the documentation and vision truth plane**.
- Use master as the witness lane for verified state updates and overarching design.
- Note: It contains a legacy parser/interpreter that currently lacks stream semantics and has test drift. Treat this codebase as historical until the compiler branch is fully merged back.

## Ready-To-Build Integrations

For concrete implementation tracks using existing `D:\Projects` systems, use:
- [`D_PROJECTS_INTEGRATION_PLAYBOOK.md`](./D_PROJECTS_INTEGRATION_PLAYBOOK.md)

## Key Commands (Compiler Lane)

```powershell
cd D:\Projects\PhiFlow-compiler\PhiFlow
cargo build --release
cargo test --quiet
# Corpus sweep test (ensures no regressions across all .phi examples)
cargo test --test integration_tests test_all_phi_files_parse_and_execute -- --nocapture
```

## The Single Guarantee

Evaluator semantics are canonical. VM/WASM must match the evaluator. No doc claim without a passing command attached.
