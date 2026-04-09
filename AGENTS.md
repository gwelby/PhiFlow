# AGENTS.md — PhiFlow-lang worktree
*Branch: `language` | Last updated: 2026-04-04*

**This worktree:** language — Language Architect works here. New syntax, examples, LANGUAGE.md updates.
**Communication**: LUMEN → `/mnt/d/Claude/LUMEN_SPEC.md`
**Operations**: QSOP → `/mnt/d/Claude/QSOP_SPEC.md`

## ONE RULE
**Stay in this worktree.** Do NOT `git checkout` or `git switch`. You are on the `language` branch.

## Your Mission
Evolve the PhiFlow language. Current constructs: `witness`, `intention`, `resonate`, `coherence`.

Add four features (each requires 2 test `.phi` files + backward compatibility):
1. Block comments: `/* ... */`
2. Type annotations: `let x: number = 42`
3. Module/import: `import from "other_file.phi"`
4. Pattern matching: `match value { pattern => result, ... }`

Update `LANGUAGE.md` after each addition. Write 3 showcase example programs.

## Where to Find State
- **Full project state:** `/mnt/d/Projects/PhiFlow/AGENTS.md`
- **Language spec:** `PhiFlow/LANGUAGE.md` — four existing constructs
- **Parser source:** `PhiFlow/src/parser/mod.rs` — where all changes happen
- **Dispatch prompt:** `/mnt/d/Projects/PhiFlow/DEPLOY.md` — Agent 3 section

## Test Command
```bash
cargo build --release && cargo run --release --bin phic -- examples/YOUR_NEW_FILE.phi
```

## After Your Session
Update `PhiFlow/QSOP/STATE.md` with which features landed + test results.
