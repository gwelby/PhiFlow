# Legacy PhiFlow Examples

The files in this folder used to live directly under `examples/`. They no
longer parse with the current `phic` binary (`src/main_cli.rs` →
`src/parser/mod.rs`). They are kept here so historical references and the
intent of the original authors are preserved, **but they will not run.**

If you want a quick "does PhiFlow work?" check, use one of the working
examples one directory up — start with `agent_handshake.phi`.

## Why these were retired

The grammar accepted by the current parser is much narrower than the
"aspirational DSL" several of these files were written against. We did
not extend the parser to keep them working, because the task was to
inventory and clearly label what is broken — not to grow new language
features.

The files fall into three groups:

### 1. Sacred-geometry sequences using the uppercase command DSL

These programs use `INITIALIZE … AT 432Hz WITH { … }`,
`TRANSITION TO …`, `EVOLVE TO …`, `CONNECT TO …`,
`INTEGRATE WITH …`, `CASCADE`, and `RETURN TO …`. None of those are
tokens in the current grammar. They fail with `E001_UNEXPECTED_TOKEN`
(typically on `Hz`, the first thing the lexer recognises).

- `BREATHING_CHECK.phi`
- `chakra_alignment.phi`
- `fibonacci_spiral.phi`
- `flower_of_life.phi`
- `lumi_resonance.phi`
- `mandala_creation.phi`
- `merkaba_activation.phi`
- `platonic_solids.phi`
- `quantum_healing.phi`
- `simple_meditation.phi`
- `sri_yantra.phi`
- `torus_field.phi`
- `tree_of_life.phi`

> Note: `BREATHING_CHECK.phi` and `lumi_resonance.phi` were originally
> saved as UTF-16-LE with a BOM, which made `phic` reject them at the
> file-read step (`Failed to read file: stream did not contain valid
> UTF-8`) before the parser ever ran. They were re-encoded to plain
> UTF-8 so they can at least be read in this folder, but the body of
> each one uses the same uppercase DSL above and still does not parse.

**If you wanted a meditation-style "walk through frequencies" demo**,
the closest currently-runnable analogue is
[`../coherence_playground/aligned.phi`](../coherence_playground/aligned.phi),
which is intentionally small and shows a focused `intention` + `stream`
producing a coherent signal.

### 2. Older free-form scripts using `Sacred(N) { … }`, `fn main()`, struct constructors, etc.

These were written when PhiFlow looked more like a scripting language
with `print(...)` calls, `Sacred(528) { … }` blocks, `fn main()`
entry points, `Foo::new(...)` constructors, top-level `consciousness X
{ frequency: 432Hz, … }` blocks, and so on. The current parser does
not accept any of that. They fail at parse time
(`E001_UNEXPECTED_TOKEN`, `E003_EXPECTED_TOKEN`, or — in the case of
`hameroff_microtubule_quantum.phi` — `E004_UNEXPECTED_CHAR` on a `?`
operator that the lexer rejects outright).

- `claude_signature.phi`
- `consciousness_resonance_integration.phi`
- `daemon_config.phi` (also uses `loop { … }` and trailing `;`,
  neither of which the current parser accepts)
- `hameroff_microtubule_quantum.phi`
- `hello_quantum.phi`
- `penrose_or_demonstration.phi`

**If you wanted a "hello, quantum" starting point**, use
[`../agent_handshake.phi`](../agent_handshake.phi) (canonical entrypoint,
also doubles as a self-test of the coherence math) or
[`../8_qubit_entanglement.phi`](../8_qubit_entanglement.phi) for a
real circuit.

**If you wanted a mid-circuit / Penrose-style measurement demo**, use
[`../mid_circuit_witness.phi`](../mid_circuit_witness.phi) or
[`../mid_circuit_collapse.phi`](../mid_circuit_collapse.phi).

## Migration notes (2026-04-26)

- `examples/agent_handshake.phi` was kept and **fixed in place** rather
  than retired. The only change was renaming the local variable
  `version` to `proto_ver`, because `version` is now a reserved
  keyword in the current parser. The author's voice and the structure
  of the protocol announcement are otherwise untouched.
- Nineteen files were moved here. Seventeen of them are byte-for-byte
  identical to their previous location. The remaining two
  (`BREATHING_CHECK.phi` and `lumi_resonance.phi`) had to be
  re-encoded from UTF-16-LE-with-BOM to plain UTF-8 just so they could
  be opened at all by the current `phic`; their body content is
  untouched and still uses the retired uppercase DSL.

## Out of scope

We did not rewrite the runtime or add new keywords/operators to make
these files parse again. Reviving any of them is a much larger
language-design conversation, not a cleanup task.
