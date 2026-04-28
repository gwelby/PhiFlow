# PhiFlow Examples

These `.phi` programs are written for the **current** `phic` binary
(`src/main_cli.rs`, parsing via `src/parser/mod.rs`).

## Running an example

```bash
cargo build --bin phic
target/debug/phic examples/agent_handshake.phi
```

If you only want to know whether a file parses, use the JSON-error mode:

```bash
target/debug/phic --json-errors examples/agent_handshake.phi
# prints `[]` and exits 0 on success, prints a diagnostic and exits 2 on parse failure
```

A quick "do all examples still parse?" sweep:

```bash
for f in examples/*.phi; do
  target/debug/phic --json-errors "$f" >/dev/null 2>&1 \
    && echo "ok   $f" \
    || echo "FAIL $f"
done
```

## Where to start

| Goal | File |
| --- | --- |
| Canonical "hello, agent" + math self-test | [`agent_handshake.phi`](agent_handshake.phi) |
| Smallest "intention + stream" demo | [`coherence_playground/aligned.phi`](coherence_playground/aligned.phi) |
| What drift looks like in the resonance field | [`coherence_playground/drifts.phi`](coherence_playground/drifts.phi) |
| A real entangled circuit | [`8_qubit_entanglement.phi`](8_qubit_entanglement.phi) |
| Mid-circuit measurement / feedback | [`mid_circuit_witness.phi`](mid_circuit_witness.phi) |
| IBM backend smoke test | [`ibm_smoke.phi`](ibm_smoke.phi) |

## Coherence Playground

The `coherence_playground/` subdirectory contains the smallest legible
demo of PhiFlow's intention-vs-result alignment. Each snippet is meant
to be run through the `coherence_report` CLI, which prints a
plain-English reading of how aligned the run was with its stated
`intention`:

```bash
cargo build --bin coherence_report
target/debug/coherence_report examples/coherence_playground/aligned.phi
target/debug/coherence_report examples/coherence_playground/drifts.phi
target/debug/coherence_report examples/coherence_playground/disconnected.phi
```

- `aligned.phi` — one focused intention + stream → strongly aligned
  (≈ 0.62).
- `drifts.phi` — starts aligned, then unrelated resonances pile in and
  the next witness sees a much lower coherence (≈ 0.62 → 0.14).
- `disconnected.phi` — no `intention` block at all → coherence stays
  at 0.

## Language sketch (current grammar)

This is a working subset, not the full language reference. Read
`src/parser/mod.rs` if you need the authoritative grammar.

```phi
// Top-level functions
function phi_lambda() -> Number {
    let phi = 1.618033988749895
    return 1.0 - (1.0 / (phi * phi))
}

// `intention` blocks scope coherence; they can nest.
intention "doing_one_thing" {
    let signal = 432.0      // `let` bindings
    resonate signal         // emit a value into the resonance field
    witness                 // yield + read coherence
}
```

Available statement-level keywords include `function`, `return`,
`let`, `if` / `else`, `for … in …`, `while`, `intention`, `stream`,
`break stream`, `resonate`, `witness`, `witness mid_circuit`,
`entangle on`, `remember`, `recall`, `broadcast`, `listen`, and
others. Note that `version`, `protocol`, `qubit`, `circuit`,
`evolve`, `entangle`, and `import` are **reserved** — you cannot use
them as ordinary identifiers.

## Legacy examples

A number of older `.phi` files no longer parse with the current
grammar. They have been moved to [`legacy/`](legacy/) with a
[README](legacy/README.md) explaining each one and pointing at a
working replacement. We did not extend the parser to bring them back;
that is a language-design discussion, not a cleanup task.

### Migration log (2026-04-26)

- **Fixed in place:** `agent_handshake.phi` (renamed local `version` →
  `proto_ver`; `version` is now reserved).
- **Moved to `legacy/`:** `BREATHING_CHECK.phi`,
  `chakra_alignment.phi`, `claude_signature.phi`,
  `consciousness_resonance_integration.phi`, `daemon_config.phi`,
  `fibonacci_spiral.phi`, `flower_of_life.phi`,
  `hameroff_microtubule_quantum.phi`, `hello_quantum.phi`,
  `lumi_resonance.phi`, `mandala_creation.phi`,
  `merkaba_activation.phi`, `penrose_or_demonstration.phi`,
  `platonic_solids.phi`, `quantum_healing.phi`,
  `simple_meditation.phi`, `sri_yantra.phi`, `torus_field.phi`,
  `tree_of_life.phi`.
- **Re-encoded during the move:** `BREATHING_CHECK.phi` and
  `lumi_resonance.phi` were UTF-16-LE-with-BOM, which caused `phic` to
  fail at the file-read step before parsing. They were converted to
  plain UTF-8 so they can be read in `legacy/`; their actual content
  was not changed and still uses the retired uppercase DSL.
