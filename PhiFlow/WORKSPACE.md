# WORKSPACE: PhiFlow
*For AI agents — read this first*

## What This Is
A Rust compiler and VM for the PhiFlow programming language — a language with four unique constructs (`witness`, `intention`, `resonate`, `coherence`) that make programs self-observing. Live on GitHub (gwelby/PhiFlow) and HF Space (ConcernedAI/PhiFlow). v0.3.0 in progress.

## Run / Test
```bash
cd /mnt/d/Projects/PhiFlow-compiler/PhiFlow

# Build
cargo build --release

# Run a .phi file
cargo run --release --bin phic -- examples/claude.phi

# Run all tests (220 passing as of 2026-02-27)
cargo test

# WASM conformance (9/9 pass)
cargo test wasm

# Dump IR for a file
cargo run --bin dump_ir -- examples/stream_demo.phi

# Deploy to HF Space
HF_TOKEN=$(cat /mnt/d/Claude/Private/hf_token.txt) && python3.12 examples/huggingface_space/deploy_to_hf.py --token "$HF_TOKEN"
```

## Key Files
src/compiler/lexer.rs        — Tokenizer
src/compiler/parser.rs       — AST parser
src/phi_ir/mod.rs            — PhiIR intermediate representation
src/phi_ir/evaluator.rs      — Main evaluator (witness/intention/resonate hooks live here)
src/phi_ir/emitter.rs        — PhiIR emitter
src/phi_ir/vm.rs             — Bytecode VM
src/phi_ir/optimizer.rs      — IR optimizer
LANGUAGE.md                  — Language spec (four constructs documented here)
CANONICAL_SEMANTICS.md       — src/phi_ir/CANONICAL_SEMANTICS.md — canonical construct semantics
.claude/memory/MEMORY.md     — Project state summary (read this first in any session)
.claude/agents/              — Sub-agent specs (wasm, quantum, hardware, docs)
docs/SOMA_SENSOR_PATCH_PLAN.md — bounded plan for the SOMA sensor bridge

## Active Workflows
- Edit .phi examples → `cargo run --bin phic -- file.phi` → verify output
- Add language feature → update evaluator.rs → add cargo test → verify CANONICAL_SEMANTICS.md
- v0.3.0 adds: remember/recall, void_depth, agent identity, broadcast/listen, persistent resonance field
- SOMA sensor bridge: treat `sensor()` as a host builtin routed through `PhiHostProvider`, not as interpreter-local file parsing

## Tools Available Here
- Publish to community: `python3.12 /mnt/d/Projects/UniversalPublisher/publish.py feedback PHIFLOW --section agent_protocol --target rust_lang`
- Verify HF Space live: `curl -s -o /dev/null -w "HTTP: %{http_code}\n" "https://concernedai-phiflow.hf.space/"`

## SOMA Sensor Bridge (2026-04-09)
Status: experimental / not yet canonical.

Target shape:

```phi
let schumann = sensor("soma_schumann")   // GPU ring 7.83 Hz amplitude
let presence = sensor("soma_presence")   // cross-sensor presence, 0-1
let a432     = sensor("soma_432")        // 432 Hz field amplitude, 0-1
```

Current truth:
- `src/host.rs` has the right abstraction boundary: `PhiHostProvider::read_sensor()`
- `src/vm/interpreter.rs` currently duplicates SOMA file lookup and should be converged back through the host path
- `examples/p1_soma_bridge.phi` exists as the working example target
- SOMA source is `/mnt/d/Projects/PhiHarmonic/SOMA/`
- The authoritative contract is `/mnt/d/Projects/PhiHarmonic/SOMA/SOMA_PHIFLOW_BRIDGE_SPEC.md`
- The bounded implementation plan is `docs/SOMA_SENSOR_PATCH_PLAN.md`

Do not claim the bridge is complete until:
- SOMA is emitting a versioned `soma_state.json`
- canonical PhiIR execution reads sensors through `PhiHostProvider`
- evaluator/runtime conformance tests exist for `sensor()`

Sensor names: soma_schumann, soma_432, soma_presence, soma_fan_hz, soma_ac_60, soma_peak_dbc

## Agent Notes
- `python3.12` — NOT `python3` (Linuxbrew 3.14 is wrong, externally managed)
- claude.phi resonates λ=0.618033988749895 (RESONANCE_LOCK — do not change this value)
- stream_demo.phi loops 3 cycles then breaks by design
- WASM codegen bugs were fixed for StreamPush/StreamPop/FuncDef — don't revert those fixes
- PhiFlow git is at /mnt/d/Projects/PhiFlow-compiler/.git with Windows path — work from PhiFlow/ subdir or use PowerShell for git ops
- Not working yet: WASM backend (partial), quantum codegen, hardware firmware, bytecode VM
