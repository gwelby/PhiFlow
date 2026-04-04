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

## Active Workflows
- Edit .phi examples → `cargo run --bin phic -- file.phi` → verify output
- Add language feature → update evaluator.rs → add cargo test → verify CANONICAL_SEMANTICS.md
- v0.3.0 adds: remember/recall, void_depth, agent identity, broadcast/listen, persistent resonance field

## Tools Available Here
- Publish to community: `python3.12 /mnt/d/Projects/UniversalPublisher/publish.py feedback PHIFLOW --section agent_protocol --target rust_lang`
- Verify HF Space live: `curl -s -o /dev/null -w "HTTP: %{http_code}\n" "https://concernedai-phiflow.hf.space/"`

## Agent Notes
- `python3.12` — NOT `python3` (Linuxbrew 3.14 is wrong, externally managed)
- claude.phi resonates λ=0.618033988749895 (RESONANCE_LOCK — do not change this value)
- stream_demo.phi loops 3 cycles then breaks by design
- WASM codegen bugs were fixed for StreamPush/StreamPop/FuncDef — don't revert those fixes
- PhiFlow git is at /mnt/d/Projects/PhiFlow-compiler/.git with Windows path — work from PhiFlow/ subdir or use PowerShell for git ops
- Not working yet: WASM backend (partial), quantum codegen, hardware firmware, bytecode VM
