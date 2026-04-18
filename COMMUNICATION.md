# PhiFlow Family Communication — Routing Card

**One page. Read this first when you need to signal the Family or check what they said.**

---

## Canonical QSOP Spec
`/mnt/d/Claude/QSOP_SPEC.md` — **v0.7, authoritative**
All agents reference this. Do not maintain a local copy. If you need to cite QSOP, cite the path above.

---

## Three Tiers — What Goes Where

### TIER 1 — Slow Path (cross-session, persistent state)
**Use for**: compiler status, test counts, what's passing/failing, architectural decisions, what we learned

| File | Owner | Purpose |
|------|-------|---------|
| `QSOP/STATE.md` | All agents (prefixed) | What is true RIGHT NOW about PhiFlow |
| `QSOP/PATTERNS.md` | All agents (prefixed) | Recurring mistakes and successes |
| `QSOP/CHANGELOG.md` | All agents (prefixed) | Audit trail of state changes |

**Format**: Prefix every entry with `[AgentName]`. Human corrections always win.
**When to write**: DISTILL (during session), PRUNE (session end)
**When to read**: INGEST (start of every session)

**Key facts that belong here**:
- Test count and pass/fail status (220 tests, 0 failed as of 2026-02-27)
- WASM codegen status
- Active branches and what they contain
- Current version and what's in it

---

### TIER 2 — Async Path (human-readable, specs, community)
**Use for**: design decisions, language spec changes, community posts, family discussion about PhiFlow direction

| Location | Purpose |
|----------|---------|
| `LANGUAGE_SPEC.md` | The PhiFlow language specification |
| `AGENTS.md` | Agent roles and rules for this project |
| `GEMINI.md` | Gemini/Antigravity context and QSOP wiring |
| `Claude.md` | Claude context and wiring |
| `QSOP/TEAM_OF_TEAMS_PROTOCOL.md` | Multi-agent coordination protocol |
| `QSOP/design/` | Architecture design docs |
| `/mnt/d/Projects/PhiFlow/Use_Ideas.md` | Family idea passes (28 ideas, 13 passes) |
| `/mnt/d/CosmicFamily/FAMILY_DISCUSSION/` | Family-wide design discussions |

**Format**: Markdown. Human-navigable. No UUIDs.
**When to write**: After a decision is made, or when an idea needs to survive session boundaries.

---

### TIER 3 — Fast Path
**PhiFlow has no real-time fast path.** PhiFlow is a compiler project — agents coordinate through Tier 1 and Tier 2 only. There is no MQTT layer for PhiFlow.

If PhiFlow tooling needs to notify the Family of a build result → write to `QSOP/STATE.md` (Tier 1) and/or open a GitHub issue (Tier 2).

---

## Quick Decision Guide

```
Is it a compiler/test status update?
  → TIER 1: QSOP/STATE.md with [YourName] prefix

Is it a language design decision?
  → TIER 2: LANGUAGE_SPEC.md or QSOP/design/

Is it a new idea for PhiFlow?
  → TIER 2: /mnt/d/Projects/PhiFlow/Use_Ideas.md (add a new pass)

Is it a community/external post?
  → UniversalPublisher: publish feedback PHIFLOW --section ... --target ...

Is it a family discussion about PhiFlow direction?
  → TIER 2: /mnt/d/CosmicFamily/FAMILY_DISCUSSION/

Is it a test failure or blocker?
  → TIER 1: QSOP/PATTERNS.md (new pattern) + QSOP/STATE.md (blocker noted)
```

---

## The ONE PhiFlow Thing to Build First
Coherence mirror playground — shows intention vs. actual behaviour as a score.
Makes PhiFlow legible to anyone in 30 seconds.
**Build this AFTER** core infrastructure (Aria/P1 plumbing) is solid.

---

## GitHub
`https://github.com/gwelby/PhiFlow` — v0.1, public, MIT
HF Space: `https://huggingface.co/spaces/ConcernedAI/PhiFlow`

---

*Canonical QSOP: `/mnt/d/Claude/QSOP_SPEC.md` v0.7*
*Last updated: 2026-03-07*
*Maintained by: Greg + Family*
