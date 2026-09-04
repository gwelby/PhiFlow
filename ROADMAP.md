# PhiFlow — Future Work

## Priority: Write the technical paper

**Status:** Not started
**Priority:** High — this is the highest-leverage move for PhiFlow being remembered.

**What:** One clean paper, purely technical language, presenting:
1. The coherence formula: `C(d,k) = (1 - φ^(-d)) × phase(k)`, with the proof that C(2,1) = φ⁻¹
2. The five primitives: `intention`, `witness`, `coherence`, `resonate`, `stream` — as runtime introspection primitives for autonomous systems
3. The WASM target: coherence primitives that run in any WASM runtime (browser, edge, server)
4. Why this matters for autonomous AI: systems that can read their own alignment state at the language level, not through external monitoring

**What to exclude:** sacred geometry, 432 Hz, flower of life, mystical vocabulary. The math is sound without the framing. The framing limits the audience.

**Target venues:** arXiv (cs.PL or cs.AI), PLDI workshop, or AI safety research community (LessWrong, Alignment Forum)

**Why:** Languages get remembered for their ideas, not their install counts. Lisp is remembered for closures. PhiFlow's idea — runtime coherence primitives with a mathematical structure that emerges from recursive depth — can survive even if the language itself doesn't get mass adoption. The paper is how ideas travel.

**Reference:** Discussed with Devin, 2026-09-04. The 499-test implementation, browser demo, and three-language architecture prove the idea is real. The paper makes the idea portable.
