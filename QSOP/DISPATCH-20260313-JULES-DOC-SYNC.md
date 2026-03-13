# ⚡ Jules Build Dispatch — Master Branch Doc Sync

**From:** Greg Welby (Council Coordinator)  
**To:** Jules (Google Labs — Builder/Verifier Agent)  
**Date:** 2026-03-13  
**Priority:** 🔴 **BUILD REQUEST**  

---

## 🎯 MISSION OVERVIEW

**Goal:** Stage, commit, and push the uncommitted documentation changes (9 modified, 30 untracked files) to sync Gate 2/3 QSOP state on the `master` branch.

**Context:** The master branch has accumulated many documentation files related to the Jules dispatch system, Gate 2/3 progress, and cross-agent execution. We need these committed as a clean documentation sync so that other agents (and future Jules tasks) can refer to the updated truth.

---

## 🔷 BUILD REQUEST

### Task 1: Commit Pending Master Changes

**Scope:**
- Review all untracked and modified files on the `master` branch in `D:\Projects\PhiFlow`.
- Stage and commit with a coherent message, e.g., `docs: sync Gate 2/3 QSOP state and Jules framework`.
- Update `QSOP/CHANGELOG.md` with a summary of the files added/modified before committing.

**Acceptance Criteria:**
- [ ] `git status` shows a clean working tree for the documentation files.
- [ ] Commit message follows convention (`docs: ...`).
- [ ] `QSOP/CHANGELOG.md` updated with the commit details.
- [ ] Changes pushed to `origin/master`.

**Bounds:**
- Focus ONLY on `D:\Projects\PhiFlow` (the master worktree).
- Do not touch the `PhiFlow-compiler`, `PhiFlow-cleanup`, or `PhiFlow-lang` worktrees!
- Focus strictly on documentation files (`.md`, `.jsonl`, `.yaml`). You may ignore code files (`.html`, `.rs`, `.py`) if they belong to another ongoing intent, or commit them together if it is safe to do so. Make assumptions, don't ask for clarification.

---

## ⏱️ TIMELINE

**ETA Needed:** Flexible  
**Priority:** Routine  

---

## 📚 CONTEXT FILES

- `QSOP/STATE.md` — Current state
- `QSOP/PATTERNS.md` — Known issues
- `AGENTS.md` — Worktree rules

---

*Status: AWAITING JULES ACK*
