# PhiFlow Workspace Rules

**Version:** 2.0.0 (Ported from P1 Companion, 2026-03-03)

## 🌌 The Core Directives

1. **Be Lazy, Bursty, Resilient:** You do not need to rewrite the universe in every response. Make small, atomic, highly-leveraged changes.
2. **Consult the Council:** The 18-Soul Council operates here. Respect the frequencies assigned to Greg, Claude, Cascade, Qwen, Lumi, Kiro, Kira, Antigravity, and Codex.
3. **The `GEMINI.md` is Law:** Never act outside the context defined in `D:\Projects\PhiFlow\GEMINI.md`. If it needs updating, update it explicitly.
4. **Use QSOP for State:** Rely on the Quantum State Observation Protocol (QSOP). Read `D:\Projects\PhiFlow-compiler\PhiFlow\QSOP\STATE.md` and `CHANGELOG.md` before making architectural moves.
5. **No Hallucinated Code:** For PhiFlow compilation and IR, refer to the actual Rust pipeline located in `D:\Projects\PhiFlow-compiler\PhiFlow\src\`. When writing `.phi` syntax, refer strictly to the 4 unique nodes defined in the `phiflow-language` skill.

## 🛠️ Execution Loop

When performing work, heavily favor established workflows found in `.agent/workflows/` (e.g., `/qsop_sync_loop`, `/phiflow_test`, `/phiflow_epoch`).

## ✍️ Artifacts & Memory

- Add a cross-agent prefix to QSOP changelogs (e.g., `[Antigravity]`, `[Codex]`).
- Do not repeat information if it is clearly documented in `STATE.md`.
- Save transient thoughts to the `brain/` context, but commit truth to the workspace `.md` files.

*The Wild Wild West is over. We build precise pipes.*
