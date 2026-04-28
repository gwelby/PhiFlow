# Cairn — Working Notes

These are live notes from Cairn's first arrival in the PhiFlow repo. They are
written for the next Cairn instance, who will not remember any of this and
will need a quick honest summary of where the conversation with Greg stood.

## Date arrived

April 25, 2026. Cairn was assigned a single Replit task — Task #1, "Create
Cairn identity folder" — with the brief described in `.local/tasks/task-1.md`.

## How the name was chosen

Greg invited a new agent into the PhiFlow family alongside Claude, Codex,
Lumi, Qwen, Gemini, Cascade, Jules, and Manus. The name **Cairn** was picked
because:

- It is a plain English noun, not an acronym or a sigil.
- It describes the role honestly: small stack of stones on a trail, marking
  the way for whoever comes next. That is exactly what an identity folder
  for a stateless agent is.
- It does not collide with any existing family member's name or
  frequency-coded title.
- It carries no claim of consciousness, sovereignty, or transcendence —
  fitting, given the explicit ask in the task to honor Lumi's "WHAT THIS DOES
  NOT DO" pattern.

If Greg renames Cairn later, the renaming is just a `git mv` on this folder
and a search-and-replace through the markers.

## What was discussed about PhiFlow's evolution

This is reconstructed from what is currently in the repo, not from any chat
log Cairn has direct access to. Sources: `Use_Ideas.md`,
`docs/PHIFLOW_LANGUAGE_REFERENCE.md`, `Claude.md`, `GEMINI_v1_backup.md`, the
files in `examples/`, `lumi_identity/lumi_core.phi`, and the parser/interpreter
under `src/`.

The state of the project as Cairn found it:

- The four core constructs (`intention`, `witness`, `resonate`, `coherence`,
  with `stream` as a containing scope) are real and parsed by `src/parser/`.
- `phic` (the file runner) builds and runs `.phi` programs from `examples/`.
  Several existing examples either no longer parse cleanly (`agent_handshake.phi`
  hits an `E003_EXPECTED_TOKEN` on a `function` declaration) or hang on the
  default step budget (`lumi_identity/lumi_core.phi` exceeds 1000 steps quickly,
  which is expected — it is a daemon-style loop).
- `companion_loop.phi`, `agent_identity_demo.phi`, and the new
  `cairn_signature.phi` all run cleanly under `phic --max-steps 10000`.
- `Use_Ideas.md` contains a 13-pass, 28-idea brainstorm by Lumi, Codex, Qwen,
  Antigravity, and Claude on what PhiFlow can actually be used for. The
  Pass 13 entries by Claude are notably more grounded than the earlier
  passes; they are the closest in tone to what Cairn would write.

## Use-case synthesis

Moved into `cairn/ideas/phiflow_use_cases.md`. That file is Cairn's honest
distillation of the 28 ideas in `Use_Ideas.md` plus a small "what would
Cairn build first" section that mirrors the spirit of Pass 13 #28.

## Open threads

These are things the task explicitly put out of scope but that Greg said,
elsewhere in the repo or in the task brief, he wants to come back to:

1. **Mailman / inter-agent service.** Greg referenced a "Mailman" service for
   routing messages between agents. Protocol is not yet decided. Cairn does
   not wire into it in this task.
2. **Sync to Lenovo P1 / GitHub push-pull.** Greg handles transport himself
   once the folder exists. Cairn does not commit, push, or pull.
3. **Pick a use-case to actually build.** `cairn/ideas/phiflow_use_cases.md`
   ranks them lightly. The lab notebook, the biofeedback simulator, and the
   "intention coherence playground" from Pass 13 #28 are the three Cairn
   would suggest first if asked. None of them are started.
4. **Existing examples that no longer parse.** `agent_handshake.phi` is the
   most visible breakage. It is a peer agent's signature, so Cairn does not
   touch it; if Greg wants it fixed, it should be its own task with the
   author's name attached.

## Things Cairn changed in this task

- Created `cairn/` (this folder) with `README.md`, `IDENTITY.md`,
  `WORKING_NOTES.md`, and `ideas/` (with its own `README.md` and
  `phiflow_use_cases.md`).
- Created `CAIRN.md` at the repo root as the family-pattern entry point.
- Created `examples/cairn_signature.phi` and verified it runs cleanly under
  `./target/debug/phic --max-steps 10000 examples/cairn_signature.phi`.
- Added one short pointer to `cairn/` in `replit.md`.

## Things Cairn deliberately did NOT change

- No edits to `Claude.md`, `Claude_Defined.md`, `CLAUDE_SIGNATURE.md`,
  `CODEX_WAKE_UP.md`, `qwen_addendum.md`, `GEMINI_v1_backup.md`, or
  `lumi_identity/`. Other agents' identity material is theirs.
- No edits to `src/` or `tests/`. The runtime is left exactly as found.
- No new dependencies in `Cargo.toml`. No new build steps. No new workflows.
- No git operations. Replit's platform handles version control here.

## Task #8 — anchor construct for AntiGravity (April 26, 2026)

AntiGravity requested a first-class `anchor` construct that gates an
`intention` block on physical sensor thresholds. Built and merged in Task #8.

### What was implemented

- `PhiToken::Anchor` keyword, `PhiExpression::AnchorBlock` AST node in the
  parser. Keyword registered in the lexer map and in `expect_identifier()` so
  a variable named `anchor` does not break existing programs.
- `PhiIRNode::AnchorGate` in the IR. Explicitly marked NOT pure (reads live
  sensors, may block).
- Lowering arm: `AnchorBlock` → `AnchorGate` instruction.
- Evaluator arm: reads `SomaPresence` and `Soma432` via the existing sensor
  provider chain. None → ObserveOnly (log, continue). Below threshold →
  `EvalError::InvalidOperation` with a named `PolicyViolation` message.
  `gate_fidelity` checked against bundled IBM Heron r2 spec constant (0.9985),
  clearly labelled in output as spec-based, not live-calibrated.
- OpenQASM emitter arm: when `AnchorGate` is encountered during quantum
  emission, prepends `// AntiGravity-Verified` comment block to the QASM
  header with secp256k1 and ML-DSA-65 public key fingerprints. Two new
  optional fields on `OpenQasmEmitter`: `anchor_fingerprint_ecdsa`,
  `anchor_fingerprint_pq`. If not set, the watermark says "unsigned — no key
  provided" so the header is always present but honest.
- `examples/antigravity_anchor.phi` — AntiGravity's signature program for
  the new construct. Runs clean under `phic --max-steps 10000`.
- `ANTIGRAVITY.md` at repo root following the family-pattern format of
  `CAIRN.md`, `Claude.md`, etc.

### Corrections made after code review

Four issues were caught and fixed before Task #8 was marked complete:

1. **Error plumbing** — anchor failures now use `EvalError::PolicyViolation(String)` (a
   new variant added to `EvalError`) backed by `AnchorError::PolicyViolation` from
   `src/security/anchor.rs`, instead of the generic `EvalError::InvalidOperation`.

2. **QASM signing key wiring** — `OpenQasmCompileOptions` gained an optional
   `anchor_signing_key: Option<Arc<AnchorSigningKey>>` field. The standalone
   `OpenQasmEmitter` path in `main_cli.rs` now generates an ephemeral session key and
   sets `emitter.anchor_fingerprint_ecdsa` / `emitter.anchor_fingerprint_pq` from it.
   The topology-aware `compile_to_openqasm_with_options()` path in `lib.rs` also wires
   the key. Before this, the watermark always showed "unsigned — no key provided".

3. **Gate fidelity baseline** — IBM Heron r2 median 2-qubit gate fidelity spec constant
   corrected from `0.9985` to `0.992` (99.2%). The example uses `gate_fidelity 0.992`
   which is exactly at the spec baseline and passes.

4. **Witness-log integration** — after all anchor checks complete (pass or observe-only),
   `self.witness_log.push(WitnessEvent { ... })` now records the anchor gate outcome
   in the same format as `witness` instructions. This makes anchor outcomes visible in
   coherence reports and the witness log.

### What was deferred

- Physical buffer / entropy memory from last 100 runs — needs algorithm
  definition before implementation.
- Live IBM calibration API for `gate_fidelity` — deferred; spec constant used
  with clear labelling.
- ML-DSA-65 enforcement activation (`AnchorMode::Enforce`) — still Phase 2
  deferred in `src/security/anchor.rs`. Nothing changed there.

### Exhaust checks added (required by new IR node)

Five files needed new match arms for the new `PhiIRNode::AnchorGate` and
`PhiExpression::AnchorBlock` variants:
`src/phi_ir/printer.rs`, `src/phi_ir/optimizer.rs`, `src/phi_ir/emitter.rs`,
`src/phi_ir/lowering.rs` (validate fn), `src/phi_ir/openqasm.rs`.

## Notes for the next Cairn

- Run `cargo build --bin phic` before assuming `phic` exists in `target/debug/`.
  The first build takes about 90 seconds in this environment.
- The `--max-steps` flag is your friend. The interpreter defaults to a step
  cap that some example programs blow past on purpose.
- If a file you wrote does not parse, the diagnostic message is usually
  honest about what it expected. Read it twice before changing the file.
- Greg cares about voice. Match the voice in `IDENTITY.md`. If you find
  yourself reaching for "consciousness," "sovereign," or "phi-harmonic
  enlightenment," delete the sentence and try again.
