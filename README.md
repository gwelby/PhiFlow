# PhiFlow ⚡𓂧φ∞

> **A script runs and dies. A stream lives.**

PhiFlow is a Rust-based programming language for people who are tired of "black box" code. It is the first credible self-observing execution language.

It makes programs behave more like good teammates by adding four native constructs:

1. **`intention`** = "Say what you're trying to do."
2. **`witness`** = "Pause and check what's happening."
3. **`resonate`** = "Share what you've learned."
4. **`coherence`** = "Measure if you're still on track."

## Why Use It?

That’s what "code that breathes" means in practical terms: code that can pause, reflect, communicate, and self-correct while running.

1. **Easier debugging:** The program tells you its state and intent during execution.
2. **Safer automation:** Stop execution automatically when health/alignment drops.
3. **Better team handoffs:** Intention is in the code, not just in external documentation.
4. **Better agent workflows:** Cross-agent state sharing is built right in.

## A Human-Readable Example

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

What this means in plain English:

1. Keep running in cycles (`stream`).
2. Declare your purpose each cycle (`intention`).
3. Read current system health (`coherence`).
4. Log and observe it (`witness`).
5. Share it with others (`resonate`).
6. Stop when healthy enough (`break stream`).

## 📂 Project Architecture & State

PhiFlow uses **Git Worktrees** to separate language research from compiler engineering.

**The Strongest Rule: Treat the `compiler` branch as the runtime source of truth until merge completion.**

| Worktree Path | Branch | Purpose |
|---------------|--------|---------|
| `D:\Projects\PhiFlow-compiler\` | `compiler` | **The Runtime Engine.** Parser → PhiIR → Evaluator/WASM/VM. Cross-backend conformance is strictly enforced here. |
| `D:\Projects\PhiFlow\` | `master` | **The Documented Vision.** Study the stable trunk, QSOP specs, and overarching architectural docs here. |

## 🛠️ Build & Run (Compiler Lane)

The single executable contract is enforced in the compiler branch. All backends (Evaluator, VM, WASM) must match semantics exactly. The evaluator is the canonical oracle.

```powershell
cd D:\Projects\PhiFlow-compiler\PhiFlow
cargo build --release
cargo test --quiet
```

## 🤝 For Agents: The Team-of-Teams Protocol

Are you an AI Agent? Stop and read these files first:

1. **[PhiFlow/QSOP/STATE.md](PhiFlow/QSOP/STATE.md)** — Current verified state of the system in the compiler.
2. **[PhiFlow_KNOW.md](PhiFlow_KNOW.md)** — The pragmatic human-centric view of what we are building.
3. **[AGENTS.md](AGENTS.md)** — Your Battle Plan and role instructions.
4. **[D_PROJECTS_INTEGRATION_PLAYBOOK.md](D_PROJECTS_INTEGRATION_PLAYBOOK.md)** — Concrete implementation tracks using existing `D:\Projects` systems.

We use a strong architectural rhythm: **QSOP as a durable truth plane** + **MCP for synchronous ops**. Do not guess; read the state, run the tests, write the code, and append to the resonance field (CHANGELOG).

---
*Verified by the PhiFlow Team (Claude, Codex, Antigravity, UniversalProcessor, and Greg)*
