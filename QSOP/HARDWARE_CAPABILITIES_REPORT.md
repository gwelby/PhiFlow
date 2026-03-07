# Council Hardware & Capabilities Registry (2026-03-05 v3.1.0)

## 1. IBM Quantum Access (The 10Min/28Days Window)

* **Limit:** 10 minutes of quantum execution time per rolling 28-day window.
* **Capacity:** 100+ qubit QPUs via Qiskit Runtime, up to 10M executions/job.
* **Strategy:** Use local simulation (PhiVM evaluator or WASM bridge) for all structural testing. Physical QPU only for authorized final-validation runs.

## 2. Council AI Capabilities (2026-03-05 — Verified)

| Member | Tool | Model | Context | Best For |
|--------|------|-------|---------|----------|
| **Antigravity** | Antigravity IDE | Gemini 3.1 Pro | 1M tokens | Agentic multi-step, WASM, telemetry |
| **Claude** | Claude Code / Desktop | Sonnet 4.6 (default) / Opus 4.6 (deep research only) | 200k (Opus: 1M) | Architecture review, truth synthesis |
| **Codex** | Warp.dev / Windsurf | GPT-5.3-Codex Extra High | ~200k | Compiler core, deterministic VM |
| **Qwen** | KoboldCPP :11500 | Qwen3.5 9B | 32k local | Coherence math, sovereignty, local |
| **Lumi** | Gemini CLI (`gemini -y`) | Gemini 3.1 Pro | 1M tokens | Protocol-weaving, JSONL, browser shim |
| **Kiro** | Kiro IDE (kiro.dev) | AWS-powered | Project-scoped | Aria embodiment, Kotlin, specs |
| **Jules** | GitHub Agent (@jules) | Gemini | GitHub-scoped | CI/CD, async tasks (B-grade) |

**Fast Inference Fallback Stack:**
* Groq: ~500 tok/sec (Llama 3.3 70B) — free
* Cerebras: `D:\Projects\Warp\cerebras.txt`
* Bifrost gateway: `localhost:18080` — 1000+ models
* OpenRouter: Configured in `.env`

## 3. Embody Specs (Aria / Kiro Synchronization)

* Published `D:\Projects\P1_Companion\.kiro\specs\HARDWARE_LIMITS.md` with IBM Quantum throttling rules.
* Bytecode definitions remain aligned with Kiro's expectations for Aria.
