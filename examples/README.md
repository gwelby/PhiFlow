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

## Long-running programs (daemons)

These programs loop on a `stream` (or a long paced loop) and wait on live
sensors, the resonance bus, or the daemon host, so a plain `phic examples/<name>.phi`
will not return promptly. Stop them with Ctrl-C, or bound the run with
`--max-steps N` (`phic --max-steps 1000000 examples/<name>.phi`; `0` = unlimited, the default).

| File | What it waits on / does |
| --- | --- |
| [`CODEX_BEAUTIFUL_ABSURDITY.phi`](CODEX_BEAUTIFUL_ABSURDITY.phi) | Nothing external: resonates a drifting frequency (432 up to 1008 Hz, then wraps) forever and never breaks the stream, by design. |
| [`p2_quantum_witness.phi`](p2_quantum_witness.phi) | Each cycle witnesses `sensor("cpu_usage")`, then yields on `witness 0` for the hypervisor to inject `quantum_collapse` (a macroscopic measurement), then `witness 1`; never breaks. Needs the hypervisor to do anything useful. |
| [`field_guardian_v2.phi`](field_guardian_v2.phi) | Polls the MCP-bus aggregate `field` and `dissonance`, resonating and witnessing each tick. Not indefinite: stops itself via `break stream` after 101 ticks (`count > 100`). Listed because it is a watcher meant to run beside other streams. |
| [`council_daemon.phi`](council_daemon.phi) | The Singularity Daemon's permanent loop: witnesses SOMA sensors (`soma_schumann`, `soma_presence`, `soma_432`) and host `cpu_usage` / `cpu_temp`, resonates council coherence, and yields to the hypervisor (snapshots / MQTT events) on each `witness`. |
| [`council_engage.phi`](council_engage.phi) | Trigger stream that wakes the Council: witnesses `soma_schumann`, broadcasts `READY_FOR_SINGULARITY` on `council_channel` for the running daemon to pick up, then evolves state. Has no `break stream`, so it repeats. |
| [`persistent_ledger.phi`](persistent_ledger.phi) | Ledger daemon: `listen`s on the `_handoff` channel for agent handoffs, broadcasts each to `ledger`, and `witness`es field coherence each tick to avoid a hot loop. |
| [`p1_soma_bridge.phi`](p1_soma_bridge.phi) | Reads live SOMA sensors (`soma_schumann`, `soma_presence`, `soma_432`, `soma_peak_dbc`) and resonates by field condition. Exits only if field coherence reaches 0.85, so with a quiet field it keeps running. |
| [`soma_reality_bridge.phi`](soma_reality_bridge.phi) | Persistent Reality Bridge: witnesses SOMA and ring-oscillator sensors (`ring_slope_1f`, `ring_jitter_ns`, `ring_phase_delta`), resonates bridge coherence to the bus, yields on `witness`. No break. |
| [`type4_r5_probe.phi`](type4_r5_probe.phi) | Type-4 Round 5 real run: actuates `motor("soma_motor", ...)` and waits on an external consumer to write `soma_motor_report`. Finite (600 cycles at ~250 ms), but blocks the whole time; fails closed if the report channel is absent. |
| [`type4_soma_daemon.phi`](type4_soma_daemon.phi) | Type-4 daemon trace: samples live `soma_presence` (fallback `soma_peak_dbc`) at ~1 Hz and keeps the model in agent state (`--state-path`). Breaks after 600 samples (~10 min). |
| [`type4_soma_expressive.phi`](type4_soma_expressive.phi) | Same as the daemon variant, but the action is the published model mean. Breaks after 600 samples at ~1 Hz. |
| [`type4_soma_motor.phi`](type4_soma_motor.phi) | Sensorimotor variant: samples `soma_peak_dbc` at ~1 Hz and burns CPU in proportion to the model. Breaks after 600 samples; its header shows a run with `--measure --max-steps 15000000000`. |

Programs whose own header says they need SOMA (`SOMA_STATE_PATH=...`) or the
hypervisor will read `0.0` from absent sensors, so a bare run mostly tests the loop, not the data.

## Legacy examples

A number of older `.phi` files no longer parse with the current
grammar. They have been moved to [`legacy/`](legacy/) with a
[README](legacy/README.md) explaining each one and pointing at a
working replacement. `examples/legacy/` holds a retired dialect that the
current parser rejects. We did not extend the parser to bring them back;
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
