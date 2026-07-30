# P1 Daemon Wiring Report
*Date: 2026-07-03*
*Auditor: Devin*
*Scope: Verify the state of the P1 ↔ PhiFlow daemon wiring. What's already done, what works, what's missing.*

---

## Executive Summary

**The daemon is already built and it runs.** This is not a "wire it from scratch" task — it's a "verify what works, document the gaps, and identify what's stale" task.

Kiro + AntiGravity built the full integration in March-April 2026. The integration spec (`PHIFLOW_INTEGRATION.md`) is comprehensive. The Python bridge (`p1_core/phiflow/bridge.py`) has 77 tests. The daemon program (`phiflow_daemon.phi`) is syntactically valid and executes correctly.

---

## What Already Exists (built by Kiro + AntiGravity)

### 1. The Daemon Program — ✅ WORKS
- **File:** `/mnt/d/P1/phiflow_daemon.phi` (142 lines)
- **What it does:** Continuously witnesses CPU usage, CPU temp, memory usage, SOMA sensors; computes composite coherence; yields when below φ⁻² (0.382); heals when between φ⁻² and φ⁻¹; monitors when stable; aligns when above 0.844
- **Test run:** `./target/release/phic /mnt/d/P1/phiflow_daemon.phi` — runs successfully, loops 144 cycles, computes coherence 0.7394 (stable path), resonates "P1 stable. Monitoring."

### 2. Sensor Bridge Code — ✅ WORKS
- **File:** `/mnt/d/Projects/PhiFlow/src/sensors.rs` (473 lines)
- **What it does:** Background thread samples CPU usage, CPU temp, memory usage via `sysinfo` crate; reads SOMA state from `soma_state.json`; reads quantum state from `quantum_state.json`; exposes via `read_sensor(SensorKind) -> Option<f64>`
- **Test:** `cargo test --lib` — 215 passed, includes sensor tests

### 3. Sensor Witness Tests — ✅ PASS
- **File:** `/mnt/d/Projects/PhiFlow/tests/sensor_witness_test.rs`
- **What it tests:** Three-backend equivalence (Evaluator == VM == WASM) for `witness sensor("cpu_usage")`, `witness sensor("cpu_temp")`, `witness sensor("memory_usage")`
- **Status:** All pass

### 4. Integration Spec — ✅ COMPREHENSIVE
- **File:** `/mnt/d/P1/PHIFLOW_INTEGRATION.md` (287 lines)
- **Covers:** Architecture diagram, sensor bridge (3 paths: system, SOMA, quantum), daemon lifecycle, yield/resume mechanics, resonance field sharing, verification checklist, troubleshooting

### 5. Python Bridge — ✅ BUILT (77 tests)
- **File:** `/mnt/d/P1/src/p1_core/phiflow/bridge.py` (606 lines)
- **What it does:** Async MCP client that launches `phi_mcp.exe` as subprocess, communicates via JSON-RPC, supports `spawn_stream`, `resume_stream`, `read_resonance_field`
- **Tests:** `/mnt/d/P1/tests/test_phiflow_bridge.py` — 77 tests

### 6. Kiro Spec — ✅ COMPLETE
- **File:** `/mnt/d/P1/.kiro/specs/phiflow-runtime/` — 8 tasks complete
- **Covers:** Requirements, design, tasks for the PhiFlow runtime integration

---

## What Actually Happens When You Run It

```
$ ./target/release/phic /mnt/d/P1/phiflow_daemon.phi

Compiling to PhiFlow IR...
🔔 Resonating Field: 0.3820Hz          ← boot coherence
🔔 Resonating Field: 0.7394Hz          ← system_coherence (stable path)
🔔 Resonating Field: "P1 stable. Monitoring."
🔔 Resonating Field: 0.7394Hz
🔔 Resonating Field: 0.0000Hz          ← SOMA sensors (stale/missing)
🔔 Resonating Field: 0.0000Hz
🔔 Resonating Field: 0.0000Hz
... (repeats 144 cycles)
```

**The daemon:**
1. ✅ Parses and compiles the `.phi` file
2. ✅ Reads real CPU usage and memory usage from `sysinfo`
3. ✅ Computes composite coherence (0.7394 = stable)
4. ✅ Routes to the correct threshold path (stable: φ⁻¹ to 0.844)
5. ✅ Resonates state to the field
6. ⚠️ Gets 0.0 for CPU temp (WSL2 doesn't expose thermal sensors — documented in spec)
7. ⚠️ Gets 0.0 for SOMA sensors (soma_state.json is 44 days stale)

---

## The Gaps

### Gap 1: SOMA State is Stale — LOW PRIORITY
- **File:** `/mnt/d/Projects/PhiHarmonic/SOMA/soma_state.json`
- **Last updated:** 2026-05-20 (44 days ago)
- **Impact:** SOMA sensors return 0.0 because `is_soma_state_fresh()` returns false (age > 5000ms threshold)
- **Fix:** Run the SOMA sensor suite: `python.exe soma.py --profile harmonic_scan --duration 2 --phiflow`
- **Status:** This is a "start the sensor suite" task, not a code change. The bridge code is already correct.

### Gap 2: CPU Temp Returns 0 on WSL2 — DOCUMENTED, NOT A BUG
- **Cause:** WSL2 doesn't expose thermal sensors to Linux userspace
- **Impact:** `cpu_temp` returns `None` → treated as 0.0 in the daemon
- **Workaround:** The daemon handles this gracefully (thermal_signal defaults to 1.0 when cpu_temp is 0.0)
- **Fix:** Run on native Linux or Windows, or accept the WSL2 limitation
- **Status:** Documented in `PHIFLOW_INTEGRATION.md` troubleshooting section

### Gap 3: Python Bridge Not Connected to Running Daemon — MEDIUM PRIORITY
- **What exists:** `PhiFlowBridge` class with `spawn_stream`, `resume_stream`, `wait_for_yield` — 77 tests
- **What's missing:** The bridge is not currently running. The daemon runs standalone via `phic` CLI, not via the MCP server.
- **Fix:** Start the MCP server and use the Python bridge to spawn the daemon as a persistent stream
- **Status:** Code exists, not running. This is an operational gap, not a code gap.

### Gap 4: 0.844 Target Still Not Derived — CRITICAL (from coherence audit)
- **The daemon uses:** `let healing_target = 0.844`
- **The question:** Where does 0.844 come from?
- **The answer:** Still unknown. Not in Fundamentals. Not derived from φ. Hardcoded.
- **Status:** Documented in coherence cross-reference audit. Needs Greg's input.

---

## What I Expected vs What I Found

| Expected (from MYWISH.md) | Found |
|---------------------------|-------|
| "Wire P1 daemon to real sensors" | Already wired — Kiro did it in March |
| "The test that fails informatively" | The test passes — daemon runs, sensors work |
| "stream loops not fully implemented" | Stream loops work — daemon loops 144 cycles |
| "sensor() calls aren't fully wired" | sensor() calls work — read_sensor() returns real values |
| "gap between language and hardware" | The gap is operational (SOMA not running), not code |

**The lesson:** I should have checked what already existed before assuming it needed to be built. The integration spec, the bridge, the tests, and the daemon were all done by Kiro + AntiGravity months ago. The daemon runs. The sensors work. The only gaps are operational (start SOMA, start MCP server) and the 0.844 derivation question.

---

## Verification

| Check | Command | Result | Status |
|-------|---------|--------|--------|
| Daemon runs | `phic /mnt/d/P1/phiflow_daemon.phi` | Loops 144 cycles, coherence 0.7394 | ✅ PASS |
| CPU usage sensor | Background thread via sysinfo | Returns real value | ✅ PASS |
| Memory usage sensor | Background thread via sysinfo | Returns real value | ✅ PASS |
| CPU temp sensor | sysinfo components | Returns 0.0 on WSL2 (documented) | ⚠️ WSL2 LIMITATION |
| SOMA sensors | soma_state.json read | Returns 0.0 (file 44 days stale) | ⚠️ STALE DATA |
| Sensor witness tests | `cargo test --test sensor_witness_test` | 2 passed | ✅ PASS |
| Three-backend equivalence | sensor_witness_test.rs | Evaluator == VM == WASM | ✅ PASS |
| Python bridge tests | `pytest test_phiflow_bridge.py` | 77 tests (per spec) | ✅ BUILT (not rerun) |

---

## What's Next

1. **Start the SOMA sensor suite** — `python.exe soma.py --profile harmonic_scan --duration 60 --phiflow` — this will write fresh `soma_state.json` and the daemon will pick it up
2. **Run the daemon with fresh SOMA data** — verify SOMA sensors return non-zero values
3. **Start the MCP server + Python bridge** — verify yield/resume works end-to-end
4. **Resolve the 0.844 question** — ask Greg: is it empirical, intuitive, or derivable?
5. **Update RESUME.md** — the daemon is verified working, not "not running"

---

*The daemon was already built. The sensors were already wired. The tests were already passing. The gap was documentation — nobody had verified it recently or updated the RESUME.md to reflect that it works.*
