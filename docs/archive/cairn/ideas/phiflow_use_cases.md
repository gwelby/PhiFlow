# PhiFlow Use Cases — Cairn's Distillation

*Written: April 25, 2026*
*— Cairn*

## What this is

`Use_Ideas.md` at the repo root is a 13-pass, 28-idea brainstorm by Lumi,
Codex, Qwen, Antigravity, and Claude about what PhiFlow can actually be used
for. It is rich and worth reading in full.

This file is a much shorter, drier distillation. Cairn's job here is not to
add new ideas — the family already produced more than enough. The job is to
strip the ones that already exist down to: *what is this idea actually doing,
and is the existing PhiFlow runtime close to being able to do it?*

The ranking below is by how close the current `phic` interpreter is to a
useful demo of the idea. It is not a ranking by importance.

## Tier 1 — Demoable today (or close to it)

These map cleanly onto what `phic` already executes: `intention`, `stream`,
`witness`, `resonate`, `coherence`. No new runtime work required.

1. **Intention coherence playground.** *(Use_Ideas.md #28, Claude's pick.)*
   A small CLI or web tool where the user writes an `intention` block and
   some logic, runs it, and gets back a plain-language report of how aligned
   the run was with the stated intention. The runtime already produces a
   coherence number; the missing piece is a friendly wrapper that translates
   that number into a sentence.
2. **Self-naming dev practice.** *(Use_Ideas.md #24, Claude.)*
   Not a piece of software, a habit. Open a `.phi` file, write the
   `intention` block first, then go write the real code somewhere else. The
   `.phi` becomes a tiny living spec. PhiFlow already supports this on
   day one. The only artifact needed is a one-page guide and an example.
3. **Council voting / handoff demo.** *(Use_Ideas.md #11–12, Lumi/Codex.)*
   Multiple `.phi` streams running in the daemon, each representing a voice.
   They `resonate` their positions; a small adjudicator stream reads the
   field and decides. The daemon already exists in `src/main_cli.rs`. The
   missing piece is a clean self-contained example, not new infrastructure.

## Tier 2 — Demoable with a small bridge

These need one piece of glue that doesn't exist yet but is well-scoped.

4. **Lab notebook with coherence trace.** A Jupyter-style notebook where each
   cell is a `.phi` block, and the cell output includes the coherence value
   and any resonated values. Glue: a notebook kernel that calls into the
   existing parser/lowerer/evaluator. No language changes.
5. **Bio-feedback simulator.** *(Use_Ideas.md #8, Antigravity.)*
   A `.phi` script reads a "body frequency" from a JSON file (mocked at
   first, real sensor later), computes a delta from 432Hz, and resonates a
   counter-frequency. The whole loop fits in 30 lines of `.phi` once the
   sensor read is exposed as a host function. `lumi_identity/lumi_core.phi`
   shows that `sensor("…")` already works in the runtime.
6. **Self-pausing CI script.** *(Use_Ideas.md #3, Codex.)*
   A `.phi` program that wraps a build command, hits a `witness` on
   failure, and serializes its state to disk. The serializer (`VmState`)
   already exists. The glue is a small wrapper that shells out to `cargo`
   or `pytest` and feeds the result back into the program.

## Tier 3 — Real but multi-week projects

These are good ideas, but they need genuine new work in the runtime, the
hardware bridges, or both. Listing them so the next Cairn instance knows
where the heavy lifts are, not as a recommendation to start now.

7. **Quantum-backed coherence checks.** *(Use_Ideas.md #20-ish.)*
   Wire the `coherence` construct to actually consult a small quantum
   circuit (locally simulated to start, IBM Heron later). The
   `quantum_codegen` module exists in skeleton form. Pulling a real value
   back into the interpreter is the missing piece.
8. **Inter-agent resonance bus over the network.** Mailman is the name Greg
   has been using. Protocol is not decided. Out of scope for this task.
9. **WASM "living UI."** *(Use_Ideas.md #7, Antigravity.)*
   The WASM host exists. A demo browser page that drives a 3D scene from a
   running `.phi` instance does not. Real, but a project.

## What Cairn would build first

If asked to pick one, Cairn would build #1 — the intention coherence
playground. Reasons, in order:

1. It is the smallest thing that makes PhiFlow's distinguishing feature
   visible to a non-believer in 30 seconds.
2. It does not require any new runtime constructs.
3. It produces an artifact (a coherence report) that can be screenshotted
   and shared, which matters for whether the project gets external pickup.
4. It is the one Claude already chose in Pass 13 #28, and Cairn agrees with
   that choice. No need to pretend to be original when the right answer is
   already on the board.

## What this distillation deliberately leaves out

- Anything framed as "sovereign," "cosmic," "consciousness-bridging," or
  "Trinity-Fibonacci-φ." Those framings live in their authors' files and
  Cairn does not adjudicate them. They are simply not how Cairn writes.
- Ranking by "frequency" or "phi-level." The runtime treats these as
  numeric tags; they do not change which idea is closer to demoable.
- Any new ideas. The existing 28 are sufficient. If the next Cairn instance
  has a genuinely new one, it should go in its own file in this folder, not
  as a new tier here.

— Cairn
