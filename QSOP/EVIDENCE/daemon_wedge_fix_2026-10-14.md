# Daemon Wedge — Diagnosis & Fix

**Date:** 2026-10-14 · **By:** Devin ∇λΣ∞ · **Defect source:** `type4_daemon_trace_2026-10-09.md` §Defects #1-2

## The wedge (recorded symptom)

`phic --daemon` stalled ~2 council cycles in — burned ~1 core, zero I/O,
stream frozen (not ledger contention; solo council wedged identically).

## Root cause — broken `tokio::select!` throttle

```rust
tokio::select! {
    _ = snapshot_timer.tick() => { hypervisor.save_state(); }
    event_opt = async { mqtt_rx.try_recv().ok() } => { ... }   // ALWAYS ready
    _ = tokio::time::sleep(Duration::from_millis(100)) => {}
}
```

`mqtt_rx` is `std::sync::mpsc::Receiver` — `try_recv()` is synchronous and
non-blocking, so the wrapped async block resolved **instantly on every poll**.
An always-ready select branch starves the sleep branch: the outer loop never
throttled. Every pass granted each stream `step_count + 1000` and ran them
back-to-back at machine speed.

Once the council program reached `Complete` (~2 1000-step slices ≈ "2 council
cycles"), every pass returned instantly — the daemon then spun a full core
evaluating finished streams forever. **~1 core, zero I/O, stream frozen** —
exact signature.

## Live reproduction (pre-fix binary, `target/debug/phic` @ 12:11)

Council = `intention "wedge_probe" { resonate 0.618; witness coherence }`;
ledger + council streams spawned, mosquitto up. In ~40s the daemon emitted
**117,685 `Witness [ledger_daemon]` lines** ≈ 2,900 scheduler passes/sec —
ledger's witness loop ran unthrottled at machine speed. Post-fix expectation:
~10 passes/sec (100ms sleep per pass) and council skipped once Complete.

## Adjacent defects fixed in the same region

1. **No reqwest timeout** — the scheduler loop is single-threaded; one hung
   connection to `:18081` would stall every stream. → 5s client timeout.
2. **`AwaitingQuantumCollapse` permanent freeze** — a stream awaiting a job_id
   was skipped (`continue`) on every failed status poll; a bridge dying
   mid-wait froze it forever. → classical fallback (`quantum_collapse = 0.0`,
   Ready) after 60s of sustained outage, matching the job-ERROR path.
3. **Yield→POST retry live-lock** — a threshold-yielded stream below 0.99
   retried the dead bridge every pass. → per-stream 5s cooldown; the yield
   is held (threshold semantics preserved), just not retried at spin rate.

## Also landed

`--max-steps 0` documented "infinite" actually set `Some(1_000_000_000)` —
a finite ceiling. Now `usize::MAX`; `None` keeps the evaluator's 100k
per-invocation brake. Regression: `tests/step_limit_test.rs` 3/3.

## Verification

- `cargo check --bin phic` clean (44m wall — post-reboot drvfs/9P stall).
- Live A/B: pre-fix 117k witness prints/40s; post-fix daemon run pending
  `cargo build --bin phic` (same drvfs bottleneck).
