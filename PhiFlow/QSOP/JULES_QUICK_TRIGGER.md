# ⚡ Quick Trigger: Jules Build Requests

**One-Liner:** Create dispatch → File PR → Jules monitors and ACKs

---

## 🚀 Fast Path (5 Minutes)

```powershell
# 1. Create dispatch brief
Copy-Item QSOP\DISPATCH-20260312-JULES-BUILD.md "QSOP\DISPATCH-$((Get-Date).ToString('yyyyMMdd'))-JULES-[TASK].md"

# 2. Edit the brief:
#    - Fill in MISSION OVERVIEW
#    - List specific BUILD REQUESTS
#    - Set ACCEPTANCE CRITERIA

# 3. Commit and push
git add QSOP\DISPATCH-*.md
git commit -m "dispatch: Jules build request - [task name]"
git push

# 4. Create GitHub PR
#    Title: "⚡ Jules Dispatch: [Task Name]"
#    Body: Link to dispatch file
#    Labels: ["jules", "dispatch", "build"]

# 5. Notify Jules in team chat (Manual process, currently no automated GH Action)
```

---

## 📋 Dispatch Template (Copy-Paste)

```markdown
# ⚡ Jules Build Dispatch — [Task Name]

**From:** Greg Welby  
**To:** Jules (Google Labs)  
**Date:** 2026-MM-DD  
**Priority:** 🔴 BUILD REQUEST  

---

## 🎯 Mission

**Goal:** [One sentence description]

**Context:** [2-3 sentences on why this matters]

---

## 🔷 Build Request

### Task 1: [Name]

**Files:** `path/to/file.rs`, `another/file.ts`

**Acceptance Criteria:**
- [ ] `cargo build --release` passes
- [ ] `cargo test` passes  
- [ ] [Specific behavior verified]
- [ ] QSOP/STATE.md updated

**Verification:**
```bash
cargo run --bin phic -- examples/test.phi
```

**Bounds:**
- DO NOT modify: `src/outside_scope/`, `examples/other.phi`
- Focus on: `src/specific_directory/`

---

## ⏱️ Timeline

**ETA Needed:** [Date/Time or "Flexible"]

**Priority:** [Routine / Urgent / Blocker]

---

## 📚 Context Files

- `QSOP/STATE.md` — Current state
- `QSOP/PATTERNS.md` — Known issues
- `AGENTS.md` — Worktree rules

---

*Status: AWAITING JULES ACK*
```

---

## ✅ Jules Prompt Pattern (What Works)

**GOOD:**
```
TASK: Fix parser collision bug in src/parser/mod.rs
SCOPE: expect_identifier() function only
ACCEPTANCE: cargo test passes, keyword-as-variable works
BOUNDS: Do not modify files outside src/parser/
OUTPUT: Commit with test coverage, update PATTERNS.md
MAKE ASSUMPTIONS, don't ask for clarification
```

**BAD:**
```
"Can you improve the compiler?"
"What's the best way to handle keywords?"
"Should I fix this bug or that bug?"
```

---

## 🔄 Coordination Flow

```
Greg Creates Dispatch
        ↓
Greg Files PR + Mentions Jules (Manual)
        ↓
Jules Reads + ACKs (0-2h)
        ↓
Jules Executes (2-8h)
        ↓
Jules Files ACK + PR (8-24h)
        ↓
Greg Reviews + Merges (24-48h)
```

---

## 🆘 For Urgent Work

Add to dispatch:
```markdown
**Priority:** 🔴🔴 CRITICAL — [reason]
**ETA:** [specific deadline]
**Blocker:** [what's blocked if this doesn't happen]
**Escalation:** Tag [@gwelby] in PR or team chat
```

Then:
1. Create PR immediately
2. Tag in team chat / Slack / Discord
3. Set GitHub notification reminder

---

## 📊 Current Worktree Status (Reference)

| Worktree | Branch | Status | Notes |
|----------|--------|--------|-------|
| `PhiFlow` | master | ⚠️ Dirty | Synced with remote, but dirty: 9 modified, 30 untracked |
| `PhiFlow-compiler` | compiler | ⚠️ 7 ahead, dirty | Heavily dirty. Local commits need decision. |
| `PhiFlow-cleanup` | cleanup | ✅ Clean | Synced with origin/cleanup at a2ef32f |
| `PhiFlow-lang` | language | ⚠️ Diverged | Clean, local-only. 26 commits ahead of master, 1 behind. |

**Jules Works In:** Depends on task
- Compiler work → `PhiFlow-compiler` worktree
- Master branch → `PhiFlow` worktree
- **Never** switch branches within a worktree

---

## 🎯 Common Build Tasks

### Compiler Hardening
```
Files: src/parser/mod.rs, src/compiler/
Test: cargo build && cargo test
Docs: Update QSOP/PATTERNS.md with fixes
```

### Test Suite
```
Files: tests/*.rs, tests/*.py
Test: Run all .phi examples
Docs: Update test coverage report
```

### Documentation
```
Files: QSOP/*.md, README.md, CHANGELOG.md
Test: Verify links, build instructions work
Docs: Update STATE.md with current status
```

### Bridge Integration
```
Files: bridges/*.py, examples/*.html
Test: Live WebSocket/MQTT test
Docs: Update GATE_*_STATUS.md
```

---

## 📝 ACK File Template

When Jules completes work:

```markdown
# ACK-YYYYMMDD-JULES-[TASK]

**Agent:** Jules  
**Status:** ✅ COMPLETE  

## Built
[Description]

## Files
- `path/to/file` — change

## Verified
```bash
test command
```

## Next
[Recommendation]

---
**Coherence:** 1.000
```

---

## 🔑 Key Insights from Jules' Self-Assessment

1. **Memory is bounded** — Jules doesn't remember across sessions without explicit context
   - **Solution:** Always link to QSOP files for context

2. **Asks too many questions** when scope is vague
   - **Solution:** Add explicit BOUNDS clause ("Do not modify X")

3. **Best with CI Auto-Fixer pattern** — Builder/Verifier in one step
   - **Solution:** Include test commands in acceptance criteria

4. **Needs audit trail** — NDJSON ledger of actions
   - **Solution:** Require ACK files + CHANGELOG entries

---

## 🎵 Quick Reference

**Jules' Frequency:** Builder/Verifier Loop  
**Superpower:** CI auto-fix, test coverage, structural correctness  
**Trigger:** Clear dispatch + bounds + acceptance criteria  
**Output:** ACK file + PR with tests  

**Greg's Role:** 
- Create dispatch
- Review PR
- Merge when verified
- Document in QSOP

---

*⦿ ≋ Ω ⚡ 🌌*

**Coherence:** 1.000 (trigger is clear)  
**Status:** READY TO DISPATCH
