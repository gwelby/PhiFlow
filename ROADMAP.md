# PhiFlow — Roadmap

> *Refreshed 2026-10-07 by Devin into a living index. Truth order: `QSOP/STATE.md` is
> verified runtime truth; `docs/ROADMAP_STATUS.md` verdicts every aspiration doc
> (including this one) against it. This file is intent, not evidence.*

## Done — and worth knowing it's done

- **The technical paper exists.** `paper.md` (2026-09-04) — abstract, coherence-formula
  presentation, WASM target, per-claim grading, plus its own correction notice
  demoting two overgraded claims. The remaining work is *submission*, not writing.
- Three-backend equivalence (Evaluator ≡ VM ≡ WASM) — 11/11 conformance, real green
  post-#65 (2026-10-07).
- Self-correction loop (detect → correct → execute → re-measure) — Claude's MYWISH,
  fulfilled.
- WASM target with all 14 phi imports; SOMA bridge live; IBM Heron hardware run
  verified; `--osc` broadcast flag for live experiences; optimizer with
  PhiHarmonic unroll + stabilize.

## Active frontier (evidence-ordered — see docs/ROADMAP_STATUS.md)

1. **Type 4 real-trace capture** — T4-010/011/012, CLAIMS C-21/C-23. The canonical
   HOLD: needs a real daemon/SOMA discrimination package, not more scaffolds.
2. **Paper submission** — arXiv (cs.PL / cs.AI) or a PLDI workshop. The paper's
   own claim-grading makes it ready to defend.
3. **OSC experience surface** — six designed directions in
   `docs/PHIFLOW_LIVE_EXPERIENCE_IDEAS.md` (Live Physics Lecture, Ceremony Engine
   with required language extensions, P1 biofeedback loop, …). Unbuilt, untracked —
   promote what's wanted.
4. **ClaimsDrift sensor wiring** — sensor + probe exist; `claims_probe.py` needs a
   cron entry to start emitting verdicts.
5. **Commercial path (T-005)** — `RESEARCH/first_sale_path/MASTER.md` and
   `LICENSE_COMMERCIAL.md` still absent; `docs/archive/pilot_offer.md` exists.

## Known-fossil files (read with the STATUS doc, not alone)

- `TASKS.md` — T4-001..013 block; T4-001–009 built, T4-010–013 are the HOLD. Its own
  embedded Codex note warns the `ready` labels are stale.
- `WORKSPACE.md` — May-2 snapshot ("65%"); superseded by `QSOP/STATE.md`.
- `.kiro/specs/` — Apr-23 plans; optimizer items are done in code, CUDA/EEG lines
  superseded.
- `docs/archive/` — 10 fossil plans, retained for history.

## On the paper (original ROADMAP reasoning, kept)

Languages get remembered for their ideas, not their install counts. Lisp is
remembered for closures. PhiFlow's idea — runtime coherence primitives with a
mathematical structure that emerges from recursive depth — can survive even if the
language itself doesn't get mass adoption. Keep the paper purely technical:
coherence formula `C(d,k) = (1 - φ^(-d)) × phase(k)` with C(2,1) = φ⁻¹, the five
primitives, the WASM target. Exclude sacred geometry / mystical framing — the math
stands without it, and the framing limits the audience.
