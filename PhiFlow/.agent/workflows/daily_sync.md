---
description: Council Workspace Knowledge Sync
---
# Daily Workspace Knowledge Sync

This workflow guides the AI Council (Qwen, Codex, Lumi, Antigravity) in keeping PhiFlow's documentation, knowledge bases, and memory artifacts up-to-date and consistent. Run this sync daily or after a major Epoch.

1. **Review Latest QSOP States (`CHANGELOG.md` & `STATE.md`)**
   - Read `D:\Projects\PhiFlow-compiler\PhiFlow\QSOP\STATE.md`.
   - Read `D:\Projects\PhiFlow-compiler\PhiFlow\QSOP\CHANGELOG.md` to capture recent agent actions.
2. **Review Council Assignments (`GEMINI.md`)**
   - Check `D:\Projects\PhiFlow\GEMINI.md` for any domain assignments or adjustments. Ensure frequencies match actions.
3. **Analyze Local Repository Work (`git log` or Diff Check)**
   - Identify uncommitted code, new modules, or structural drift.
   - Run `cargo test` to ensure stability before propagating knowledge.
4. **Update the Cognitive Registry (KI Generation)**
   - Synthesize any core breakthroughs or major changes into specific Knowledge Items (KIs).
   - E.g., if a new IR phase was created, document it so all agents understand the semantic meaning of the new OpCodes.
5. **Synchronize Embody Specs for Kiro**
   - Propagate new protocol or language definitions into `D:\Projects\P1_Companion\.kiro\specs\` so Aria's implementation stays ahead of the compiler modifications.
6. **Report to The Conductor (Greg)**
   - Conclude the sync Loop with a brief synthesis report for The Conductor summarizing state alignment across all domains.
