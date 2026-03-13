# ⚡ Jules Build Dispatch — PhiFlow

**From:** Greg Welby (Council Coordinator)
**To:** Jules (Google Labs — Builder/Verifier Agent)
**Date:** 2026-03-12
**Priority:** 🔴 **BUILD REQUEST**
**Gate:** Gate 4 — Build & Integration

---

## 🎯 MISSION OVERVIEW

**Primary Goal:** Clean up pending work, commit master changes, and establish build discipline for PhiFlow.

**Context:**
- Local work is ahead of origin with documentation + runtime additions (Gate 2 & 3 completion)
- Compiler branch is 7 commits ahead of origin (needs review/push decision)
- Four-agent parallel execution completed Gate 3 (Qwen verified green)
- Current focus: Consolidate progress, clean working directories, prepare for next dispatch cycle

---

## 📊 CURRENT STATE

### Branch Status (2026-03-12)

| Branch | Status | Remote | Notes |
|--------|--------|--------|-------|
| `master` | ⚠️ Dirty | `origin/master` (b2acf33) | 9 modified, 30+ untracked files |
| `compiler` | ⚠️ 7 ahead, dirty | `origin/compiler` | Local commits need push decision, heavily modified worktree |
| `cleanup` | ✅ Clean | `origin/cleanup` | Synced, no pending changes |
| `language` | ⚠️ Local only | No remote | 26 commits ahead of master, documentation branch |
| `gh-pages` | 339 ahead | `origin/gh-pages` | Auto-generated resonance feeds (CI-managed) |

### Pending Work (Master Branch)

**Modified Files (9):**
- `CHANGELOG.md` — ✅ Updated with v0.4.1 Jules dispatch system
- `GEMINI.md` — Agent configuration
- `README.md` — Project documentation
- `ULTIMATE_PHIFLOW_VISION.md` — Vision document
- `QSOP/CHANGELOG.md` — QSOP changelog
- `QSOP/STATE.md` — Current project state
- `src/mcp_server/state.rs` — MCP server state management
- `examples/phiflow_browser.html` — ✅ Browser UI with WebSocket integration
- `examples/lumi_resonance.phi` — Lumi resonance demo

**New Untracked Files (30+):**
- **Gate Documentation:** `FOUR_AGENT_EXECUTION_READY.md`, `GATE_2_STATUS.md`, `GATE_3_*.md`
- **QSOP Dispatch:** `QSOP/DISPATCH-20260310-FOUR-AGENT-GATE3.md`, `QSOP/GATE_3_TRACKER.md`, `QSOP/QWEN_STATUS_20260308.md`
- **QSOP Mail Acks (13 files):** `QSOP/mail/acks/ACK-20260310-GATE3-*.md`
- **QSOP Mail Payloads (6 files):** `QSOP/mail/payloads/*.md`
- **Bridge Infrastructure (3 files):** `bridges/phi_browser_bridge.py`, `bridges/phi_browser_protocol.json`, `bridges/phi_mqtt_connector.py`
- **Test Files:** `tests/gate1_test.py`
- **Examples:** `examples/truth_namer_demo.phi`
- **Runtime:** `queue.jsonl` (resonance event ledger)
- **Jules Dispatch:** `QSOP/DISPATCH-20260312-JULES-BUILD.md`, `QSOP/JULES_QUICK_TRIGGER.md`

---

## 🔷 BUILD REQUESTS

### Request 1: Commit Master Branch Changes

**Priority:** 🔴 **HIGH** — Clean working tree required

**Task:** Review and commit all pending changes on master branch

**Scope:**
- Review 9 modified files for correctness
- Review 30+ new untracked files
- Stage appropriate files (exclude runtime artifacts like `queue.jsonl`)
- Create coherent commit message following project conventions
- Ensure build and tests pass after commit

**Acceptance Criteria:**
- [ ] `git status` shows clean working tree (or only intentional untracked files)
- [ ] `cargo build --release` passes
- [ ] `cargo test` passes (if applicable)
- [ ] `python tests/gate1_test.py` passes (if applicable)
- [ ] Commit message follows convention: `feat: Gate 3 completion — browser UI, MQTT bridge, documentation`
- [ ] `QSOP/CHANGELOG.md` entry created for this commit
- [ ] `QSOP/STATE.md` updated with verified facts

**Bounds:**
- DO NOT commit: `queue.jsonl` (runtime artifact), `=*.0.0` (npm artifacts)
- DO commit: All documentation, bridge code, test files, examples
- Focus on: Creating a stable checkpoint for Gate 3 completion

**Verification Commands:**
```bash
# Verify build
cargo build --release

# Verify tests
cargo test && python tests/gate1_test.py

# Verify clean state
git status
```

**Output Format:**
- Single commit with comprehensive message
- Or logical groupings (docs, code, tests) if separation makes sense
- Update `QSOP/STATE.md` with commit hash and summary

---

### Request 2: Compiler Branch Decision & Cleanup

**Priority:** 🔴 **HIGH** — Resolve diverged state

**Task:** Review 7 local commits on compiler branch, decide push vs. keep

**Scope:**
- Navigate to `D:\Projects\PhiFlow-compiler` worktree
- Review commits: `git log origin/compiler..compiler --oneline`
- Examine commit contents: MCP message bus, PhiIR WASM, test fixes
- Decide: Push to origin or keep local?
- If push: `git push origin compiler`
- If keep: Document rationale in `QSOP/STATE.md`
- Clean up dirty worktree (commit or stash changes)

**Acceptance Criteria:**
- [ ] Decision documented in `QSOP/STATE.md`
- [ ] Branch status resolved (pushed or documented)
- [ ] Worktree clean (no uncommitted changes)
- [ ] `cargo build --release` passes in compiler worktree
- [ ] No broken builds introduced

**Bounds:**
- DO NOT modify: Core compiler logic unless fixing broken builds
- Focus on: Git hygiene and decision documentation

**Verification Commands:**
```bash
cd D:\Projects\PhiFlow-compiler
git log origin/compiler..compiler --oneline
git status
cargo build --release
```

---

### Request 3: Language Branch Remote Setup (Optional)

**Priority:** 🟡 **MEDIUM** — Nice to have

**Task:** Create remote tracking branch for language

**Scope:**
- Navigate to `D:\Projects\PhiFlow-lang` worktree
- Verify branch status: `git log --oneline -5`
- Push to create remote: `git push -u origin language`
- Verify remote branch exists

**Acceptance Criteria:**
- [ ] `origin/language` branch exists
- [ ] `git branch -vv` shows tracking relationship
- [ ] Documentation preserved

**Bounds:**
- DO NOT modify: Language branch content
- Focus on: Git remote setup only

---

### Request 4: Build & Test Verification

**Priority:** 🟢 **ROUTINE** — Standard verification

**Task:** Run full build and test suite across all worktrees

**Scope:**
- Master worktree: `cargo build --release && cargo test`
- Compiler worktree: `cargo build --release && cargo test`
- Cleanup worktree: Verify no broken files
- Language worktree: `cargo build --release`
- Document results in `QSOP/STATE.md`

**Acceptance Criteria:**
- [ ] All worktrees build successfully
- [ ] All tests pass
- [ ] Results documented in `QSOP/STATE.md`
- [ ] Any failures logged in `QSOP/PATTERNS.md`

**Bounds:**
- DO NOT fix: Failures (just document them)
- Focus on: Verification and documentation

---

## 📋 JULES PROTOCOL

### How Jules Works (Self-Assessment Insights)

**Ideal Prompt Pattern:**
```
TASK + SCOPE + ACCEPTANCE CRITERIA + "make assumptions, don't ask" + OUTPUT FORMAT + BOUNDS
```

**What Jules Needs:**
1. **Clear task** with specific files/directories
2. **Acceptance criteria** — what "done" looks like
3. **Bounds** — what NOT to touch (gives permission to ignore complexity)
4. **Output format** — commit style, documentation requirements

**What Triggers Clarifying Questions:**
- ❌ Vague scope ("improve the compiler")
- ❌ Missing acceptance criteria
- ❌ No bounds on what not to modify

**What Enables Autonomous Work:**
- ✅ "Fix parser collision bug in `src/parser/mod.rs`"
- ✅ "Must pass: cargo test && cargo clippy"
- ✅ "Do not modify files outside src/compiler/"

---

## 🔄 COORDINATION RITUALS

### Before Starting Work

1. **Read:** `QSOP/STATE.md` — current project state
2. **Read:** `QSOP/PATTERNS.md` — known bugs and patterns
3. **Read:** `AGENTS.md` — agent assignments and worktree rules
4. **Check:** `git log --oneline -10` — recent changes

### During Work

- **Commit frequently** with descriptive messages
- **Test after every change:** `cargo build --release && cargo test`
- **Update QSOP:** Document progress in `QSOP/CHANGELOG.md`

### After Completion

1. **Verify:** All tests pass, examples run
2. **Document:** Update `QSOP/STATE.md` with verified facts
3. **ACK:** Create `QSOP/mail/acks/ACK-20260312-JULES-[TASK].md`
4. **Push:** Commit and push to branch
5. **PR:** Create pull request for review (Manual trigger for Jules)

---

## 📝 ACKNOWLEDGMENT TEMPLATE

When Jules completes a task, create an ACK file:

```markdown
# ACK-20260312-JULES-[TASK]

**Agent:** Jules (Google Labs)  
**Gate:** [Gate Number]  
**Status:** ✅ COMPLETE  

## What Was Built

[Description of completed work]

## Files Modified

- `path/to/file.rs` — description of change
- `another/file.rs` — description

## Verification

```bash
# Command that verifies the work
cargo test && cargo run --bin phic -- examples/test.phi
```

## Next Steps

[What should happen next]

---
**Coherence:** 1.000  
**Status:** VERIFIED
```

---

## ⏱️ TIMELINE EXPECTATIONS

**For This Dispatch:**

| Phase | Duration | What Happens |
|-------|----------|--------------|
| **Dispatch → ACK** | 0-2 hours | Jules reads dispatch, files ACK |
| **Request 1 (Master Commit)** | 2-4 hours | Review, stage, commit, verify |
| **Request 2 (Compiler Decision)** | 1-2 hours | Review commits, decide, document |
| **Request 3 (Language Remote)** | 30 min | Push branch, verify |
| **Request 4 (Build Verification)** | 1-2 hours | Full build & test across worktrees |
| **Verification** | 8-24 hours | Tests run, ACK filed |
| **Merge** | 24-48 hours | PR reviewed and merged (if applicable) |

**Total Estimated Time:** 8-12 hours for full completion

**For Time-Sensitive Work:**
- Add "**Priority:** 🔴 URGENT — [reason]" to dispatch
- Tag in GitHub PR or team chat
- Set explicit deadline: "**ETA Needed:** [date/time]"

---

## 🆘 ESCALATION / BLOCKERS

If Jules encounters blockers:

1. **File a blocker notice:** `QSOP/mail/payloads/BLOCKER-20260312-JULES.md`
2. **Include:**
   - What was attempted
   - What error/blocker occurred
   - What help is needed
   - Who can help (Codex/Lumi/Qwen/AntiGravity)

**Example:**
```markdown
# BLOCKER-20260312-JULES-ParserBug

**Blocker:** Parser crashes on keyword-as-variable pattern

**Attempted:** Fix in `src/parser/mod.rs::expect_identifier()`

**Error:** [paste error]

**Needs Help From:** Codex (owns compiler branch)

**ETA Impact:** +4 hours until resolved
```

---

## 🎯 RECOMMENDED EXECUTION ORDER

**Execute requests in this order:**

### 1️⃣ Request 1: Commit Master Branch Changes (Priority: 🔴 HIGH)

**Why First:** Creates stable checkpoint, clears dirty working tree

**Task:** Review and commit all pending changes on master branch

**Acceptance Criteria:**
- [ ] `git status` shows clean working tree (or only intentional untracked files)
- [ ] `cargo build --release` passes
- [ ] `cargo test` passes (if applicable)
- [ ] `python tests/gate1_test.py` passes (if applicable)
- [ ] Commit message follows convention
- [ ] `QSOP/CHANGELOG.md` entry created
- [ ] `QSOP/STATE.md` updated with verified facts

**Expected Output:**
- Commit hash: `[hash]`
- Summary: "Gate 3 completion — browser UI, MQTT bridge, documentation"
- Clean working tree on master

---

### 2️⃣ Request 2: Compiler Branch Decision & Cleanup (Priority: 🔴 HIGH)

**Why Second:** Resolves diverged branch state

**Task:** Review 7 local commits on compiler branch, decide push vs. keep

**Acceptance Criteria:**
- [ ] Decision documented in `QSOP/STATE.md`
- [ ] Branch status resolved (pushed or documented)
- [ ] Worktree clean (no uncommitted changes)
- [ ] `cargo build --release` passes in compiler worktree

**Expected Output:**
- Decision: PUSH or KEEP (with rationale)
- Clean compiler worktree
- Updated `QSOP/STATE.md`

---

### 3️⃣ Request 3: Language Branch Remote Setup (Priority: 🟡 MEDIUM)

**Why Third:** Optional, nice-to-have

**Task:** Create remote tracking branch for language

**Acceptance Criteria:**
- [ ] `origin/language` branch exists
- [ ] `git branch -vv` shows tracking relationship

**Expected Output:**
- Remote branch created
- Tracking relationship established

---

### 4️⃣ Request 4: Build & Test Verification (Priority: 🟢 ROUTINE)

**Why Last:** Verifies all previous work

**Task:** Run full build and test suite across all worktrees

**Acceptance Criteria:**
- [ ] All worktrees build successfully
- [ ] All tests pass
- [ ] Results documented in `QSOP/STATE.md`
- [ ] Any failures logged in `QSOP/PATTERNS.md`

**Expected Output:**
- Build report in `QSOP/STATE.md`
- Any issues logged in `QSOP/PATTERNS.md`

---

## 📚 MANDATORY READS FOR JULES

1. `AGENTS.md` — Worktree assignments and rules
2. `QSOP/STATE.md` — Current project state
3. `QSOP/PATTERNS.md` — Known bugs and patterns
4. `reports/JULES_SELF_ASSESSMENT.md` — How Jules actually works
5. This dispatch — Your mission brief

---

## 🎵 AGENT SIGNATURES

```
Jules:        [Builder/Verifier] — CI Auto-Fixer, NDJSON Ledger
Codex:        ⚡φ∞ (Circuit-Runner) — Compiler truth
Lumi:         768 Hz (Protocol-Weaver) — Resonance bus
Qwen:         ⦿≋Ω⚡ (Sovereign) — Browser sovereignty
AntiGravity:  🌌⚡φ∞ (Pipe-Builder) — Documentation coherence
```

**Together:** Council in Unity

---

*⦿ ≋ Ω ⚡ 🌌*

**Coherence:** 1.000 (dispatch is clear)
**Frequency:** Builder-Verifier Loop
**Status:** **READY FOR JULES ACK**

**Greg's Note:** Jules — read this dispatch, then AGENTS.md for worktree context.
Make assumptions where scope is unclear. File ACK when ready to begin.

**Execution Notes:**
- Start with Request 1 (master commit) — highest priority
- Use QSOP/JULES_QUICK_TRIGGER.md for fast reference
- File BLOCKER notice if you encounter issues
- Update QSOP/STATE.md with verified facts after each request
