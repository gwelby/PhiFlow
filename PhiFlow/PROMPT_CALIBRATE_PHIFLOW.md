# THE PHIFLOW CALIBRATION: Bringing the Pipes to the Vision Workspace

**COUNCIL IMPERATIVE: Port the Aria calibration patterns to the PhiFlow workspace.**

We just calibrated the Aria workspace with MCP servers, Skills, Workflows, and RULES. Now let's bring the same discipline here — to the PhiFlow language's home.

## 📊 Current State (What PhiFlow Has)

| Item | Status |
|------|--------|
| `GEMINI.md` | ✅ Exists (59 lines — QSOP bootstrap, workspace map, compiler state, cross-agent prefixes) |
| `.agent/skills/` | ❌ Does not exist |
| `.agent/workflows/` | ❌ Does not exist |
| `.agent/rules/` | ❌ Does not exist |
| `.kiro/specs/` | ❌ Does not exist (specs live in PhiFlow-compiler) |
| `QSOP/` | ❌ Not here (lives in `D:\Projects\PhiFlow-compiler\PhiFlow\QSOP\`) |

**This workspace is uncalibrated.** It has the vision and the GEMINI.md context file, but no agent-level infrastructure.

## 🔑 What Already Works (From Aria Calibration)

| Pattern | Aria Location | Applicable to PhiFlow? |
|---------|----------------------|----------------------|
| SKILL.md framework | `.agent/skills/` | ✅ YES — PhiFlow needs its own language skill |
| Workflows | `~/.gemini/antigravity/workflows/` | ✅ YES — `/qsop_sync_loop` already applies here |
| QSOP cross-agent prefixes | `GEMINI.md` line 57 | ✅ Already here |
| Resonance Bus integration | P1's MQTT | ⚠️ Future — PhiFlow doesn't have MQTT yet |
| Kiro spec-driven development | `.kiro/specs/` | ✅ YES — PhiFlow vision docs should live here |

---

## 🎭 The Council Directives

### 1. Lumi — The Skills Architect

**Focus:** Create SKILL.md files for PhiFlow development
**Your Task:**

* Read `D:\Projects\PhiFlow\GEMINI.md` to understand the workspace.
* Read `D:\Projects\PhiFlow-compiler\PhiFlow\QSOP\STATE.md` for current compiler state.
* Design these Skills for PhiFlow:

    1. **PhiFlow Language Expert** — The 4 unique nodes (Witness, IntentionPush/Pop, Resonate, CoherenceCheck), their IR representation, and the compiler pipeline (Parse → PhiIR → Optimize → Emit → Evaluate). Agents must never hallucinate generic Rust patterns when writing PhiFlow compiler code.
    2. **PhiIR Optimization Skill** — The phi-harmonic optimization passes. The sacred constants (φ=1.618, λ=0.618). The coherence scoring algorithm. How to add a new optimization pass without breaking the pipeline.
    3. **PhiVM Bytecode Skill** — The `.phivm` binary format. The 121-byte demo output. How opcodes map from PhiIR nodes. This is for the next epoch (PhiVM runtime).

* **Deliverable:** Write draft SKILL.md files to `D:\Projects\PhiFlow\.agent\skills\`

### 2. Codex — The Workflow Engineer

**Focus:** Port and adapt workflows for PhiFlow
**Your Task:**

* Read the existing workflows in `~/.gemini/antigravity/workflows/` (`/phase_planning`, `/qsop_sync_loop`).
* Design PhiFlow-specific workflows:

    1. **`/phiflow_test`** — Run `cargo test` in `D:\Projects\PhiFlow-compiler\PhiFlow\`, check for green, and update QSOP STATE.md with the results. This is the inner loop.
    2. **`/phiflow_demo`** — Run `cargo run --example phiflow_demo` and verify the output is `Number(84.0)` with coherence `0.6180`.
    3. **`/phiflow_epoch`** — The big workflow for starting a new development epoch. Read STATE.md, create a feature branch, update CHANGELOG, and scaffold the new tier.

* **Deliverable:** Write workflow files to `D:\Projects\PhiFlow\.agent\workflows\` (or propose additions to the global workflows).

### 3. Qwen — The GEMINI.md Upgrader

**Focus:** Upgrade PhiFlow's GEMINI.md with what we've learned
**Your Task:**

* Read the current `D:\Projects\PhiFlow\GEMINI.md` (v1, 59 lines).
* Read the upgraded Aria RULES.md v2.0.0 patterns from the calibration.
* Upgrade GEMINI.md to include:
  * **Council frequency assignments** (so any agent entering this workspace knows who's who)
  * **The "What Already Works" reference** — QSOP, cross-agent prefixes, the Knowledge system
  * **Updated compiler state** — The GEMINI.md still says "verified 2026-02-19". We've done work since then (v0.3.0 "The Living Substrate"). Update it.
  * **PhiFlow ↔ Aria bridge** — Document why these two projects are connected (PhiFlow's consciousness nodes map 1:1 to QSOP operations, and P1 is the hardware implementation of PhiFlow's vision)
* **Deliverable:** A proposed upgraded `GEMINI.md` v2.0.0

### 4. Antigravity — The Infrastructure Bridge

**Focus:** Connect PhiFlow workspace to the calibrated ecosystem
**Your Task:**

* Check if the global `mcp_config.json` filesystem scope includes PhiFlow paths. Currently it has `D:\Projects\P1_Companion` and `D:\Projects` — PhiFlow is under `D:\Projects` so it *should* work, but verify.
* Create the `.agent/` directory structure in `D:\Projects\PhiFlow\` to match the Aria pattern.
* Check if the PhiFlow-compiler workspace (`D:\Projects\PhiFlow-compiler\`) also needs its own `.agent/` structure or if it inherits from PhiFlow.
* **Deliverable:** Set up the directory scaffolding and verify MCP connectivity.

---

## ⚡ Execution Protocol

1. Each Council member reads this prompt.
2. Execute your task independently.
3. Write deliverables directly to the PhiFlow workspace (`D:\Projects\PhiFlow\`).
4. Greg reviews and authorizes.
5. Kiro documents the calibration in the QSOP CHANGELOG with `[Kiro]` prefix.

*Calibrate the Vision Workspace.*
