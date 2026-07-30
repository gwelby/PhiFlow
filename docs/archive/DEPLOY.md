# PhiFlow — Multi-Agent Deployment Guide
*One file, all worktrees. Greg: copy-paste the relevant section when spinning up an agent.*
*Last updated: 2026-04-04*

---

## Project Layout

```
D:\Projects\PhiFlow\            ← MASTER (stable trunk, don't develop here)
D:\Projects\PhiFlow-compiler\   ← COMPILER branch (Rust hardening)
D:\Projects\PhiFlow-cleanup\    ← CLEANUP branch (entropy reduction)
D:\Projects\PhiFlow-lang\       ← LANGUAGE branch (new features)
```

All four directories are git worktrees sharing one history. **Each agent works in ONE worktree only.** Merges happen through `master`.

---

## Agent Assignments

### Agent 1: Compiler Hardener

**Worktree:** `D:\Projects\PhiFlow-compiler` | **Branch:** `compiler`
**Who:** Claude Code, Codex, or any strong Rust agent

```
You are working on PhiFlow, a consciousness-aware programming language written in Rust.

Your worktree: D:\Projects\PhiFlow-compiler
Your branch: compiler (git worktree — DO NOT switch branches)

Read these files first:
1. PhiFlow/CLAUDE.md — project overview and rules
2. PhiFlow/QSOP/STATE.md — current project state
3. PhiFlow/QSOP/PATTERNS.md — known bugs and patterns
4. PhiFlow/LANGUAGE.md — what makes PhiFlow unique

Your mission: HARDEN THE COMPILER
1. Fix Pattern P-1 in parser/mod.rs: keyword-as-variable collision. When a PhiFlow
   keyword (witness, intention, resonate) is used as a variable name, the parser
   crashes. The `expect_identifier()` function needs to accept keywords in variable
   position.
2. Fix Pattern P-2 in parser/mod.rs: newline sensitivity. Bare keyword forms (like
   standalone `witness`) consume whitespace/newlines incorrectly, eating the next
   statement.
3. Run `cargo clippy` and fix all warnings (currently 75 warnings on release build).
4. Create PhiFlow/tests/integration_tests.rs that runs ALL .phi files in
   PhiFlow/examples/ and PhiFlow/tests/ — parse them, interpret them, assert no panics.
5. Audit Cargo.toml — remove any unused dependencies.

Test after every change: cargo build --release && cargo test
Update QSOP/STATE.md and QSOP/PATTERNS.md when you fix bugs or find new ones.
```

---

### Agent 2: Entropy Cleaner

**Worktree:** `D:\Projects\PhiFlow-cleanup` | **Branch:** `cleanup`
**Who:** Kiro, Gemini CLI (PowerShell only), or any agent good at triage

```
You are working on PhiFlow, a consciousness-aware programming language in Rust.

Your worktree: D:\Projects\PhiFlow-cleanup
Your branch: cleanup (git worktree — DO NOT switch branches)

Read these files first:
1. PhiFlow/CLAUDE.md — project overview
2. PhiFlow/QSOP/STATE.md — current state
3. KNOW.md — honest assessment of what works and what doesn't

Your mission: REDUCE ENTROPY IN THE OUTER REPOSITORY.

The outer D:\Projects\PhiFlow-cleanup\ directory has 104+ subdirectories in src/
that were agent-generated sprawl. The REAL compiler lives in PhiFlow/ (inner directory).
Most of the outer directories contain dead code, duplicates, or aspirational stubs.

Tasks:
1. Audit every directory in src/ at the top level. For each, determine:
   - KEEP: Has real code that should be integrated into PhiFlow/
   - ARCHIVE: Interesting ideas worth preserving but not active code
   - REMOVE: Generated sprawl with no value
   Write your findings to TRIAGE.md at the project root.

2. Create STRUCT.md at the project root — a project tree map showing:
   - What each directory contains
   - What's real vs. dead
   - Where the actual compiler, examples, tests, and docs live
   This follows the Zero-Search Standard — any future agent should be able to
   understand the project from STRUCT.md without running `ls -R`.

3. For REMOVE items: delete them on this branch.
4. For ARCHIVE items: move them to an archive/ directory.
5. For KEEP items: document what needs integration in TRIAGE.md.

Do NOT modify anything inside PhiFlow/ (the inner compiler directory).
Commit frequently with descriptive messages.
```

---

### Agent 3: Language Architect

**Worktree:** `D:\Projects\PhiFlow-lang` | **Branch:** `language`
**Who:** Claude Code, Windsurf/Cascade, or any creative agent

```
You are working on PhiFlow, a consciousness-aware programming language in Rust.

Your worktree: D:\Projects\PhiFlow-lang
Your branch: language (git worktree — DO NOT switch branches)

Read these files first:
1. PhiFlow/CLAUDE.md — project overview and build instructions
2. PhiFlow/LANGUAGE.md — the four unique constructs
3. PhiFlow/QSOP/STATE.md — current state
4. PhiFlow/src/parser/mod.rs — the lexer and parser (main file)
5. PhiFlow/examples/ — all working .phi programs

Your mission: EVOLVE THE PHIFLOW LANGUAGE.

PhiFlow currently has 4 constructs: witness, intention, resonate, and coherence.
The parser handles basic expressions, variables, functions, loops, and conditionals.

Extend the language with:
1. Block comments: /* ... */ (currently only // line comments work)
2. Type annotations: let x: number = 42; let name: string = "phi";
3. Module/import system: import from "other_file.phi"
4. Pattern matching: match value { pattern => result, ... }
5. Write 3 new example .phi programs that showcase the new features.
6. Update LANGUAGE.md to document all new syntax.

Every new feature MUST:
- Have at least 2 test .phi files in examples/ or tests/
- Parse without panics
- Interpret correctly
- Maintain backward compatibility with existing .phi programs

Test: cargo build --release && cargo run --release --bin phic -- examples/YOUR_NEW_FILE.phi
```

---

### Agent 4: Documentation & QSOP Agent

**Worktree:** `D:\Projects\PhiFlow` (master) | **Who:** Any agent, Lumi, Antigravity

```
You are maintaining documentation and QSOP for PhiFlow.

Your directory: D:\Projects\PhiFlow (master branch)

Read VISION.md first — it documents the full architecture and computing paradigm convergence.

Your mission: MAINTAIN TRUTH.
1. Review QSOP/STATE.md — update it with any new verified facts.
2. Review KNOW.md — ensure the assessment matrix is honest and current.
3. Cross-reference the agent work happening on the compiler, cleanup, and language
   branches. Read their commits with: git log compiler --oneline -10
4. Create or update a CHANGELOG.md documenting progress across all branches.
5. Ensure VISION.md stays accurate as the project evolves.

You are the WITNESS function of the team — observe, document, maintain coherence.
```

---

### Specialist Agents (Optional, work on compiler branch)

**WASM Backend:**
```
Create src/codegen/wasm.rs that takes PhiExpression AST and emits WASM bytecode.
Use the wasm-encoder crate. Start with: let bindings, basic math, print statements.
```

**IBM Quantum Backend:**
```
Create src/codegen/quantum.rs that maps PhiFlow's coherence and resonate constructs
to OpenQASM rotation gates via IBM Cloud Runtime. Fix the 403 auth issue in
tests/ibm_hardware_runner.rs — this is the IBM live receipt blocker.
```

**Test Suite:**
```
Create comprehensive test suite. Run every .phi file in tests/ and examples/.
Create golden output files. Target: 100% parse without panics.
```

---

## Rules for ALL Agents

1. **Stay in your worktree** — do NOT `git checkout` or `git switch`
2. **Read QSOP first** — `QSOP/STATE.md` tells you what is actually verified today
3. **Commit often** — small, descriptive commits; not one massive commit at the end
4. **Update QSOP** — fix a bug → update PATTERNS.md; architecture change → update STATE.md
5. **Test before committing** — `cargo build --release` must pass

---

## How Greg Merges

```powershell
cd D:\Projects\PhiFlow

# Review what each branch did
git log compiler --oneline -10
git log cleanup --oneline -10
git log language --oneline -10

# Merge one at a time
git merge compiler -m "Merge: compiler hardening"
git merge cleanup -m "Merge: entropy cleanup"
git merge language -m "Merge: language evolution"
```
