# PATTERNS - Learning from mistakes and successes

## Active Patterns (Mistakes)

### P-1: Keyword-as-variable collision

- **What happens**: PhiFlow keywords (frequency, state, coherence, etc.) used as variable names cause parse errors because lexer emits keyword tokens, not Identifier
- **Instances**: 3 (frequency/create, witness/resonate identifiers, `consciousness` identifier regression)
- **Root cause**: Lexer is greedy with keyword matching. No context-sensitive tokenization.
- **Fix**: In parser, `expect_identifier()` now accepts the full keyword token set when in identifier position. Regression coverage is in `tests/repro_bugs.rs::test_p1_keyword_collision`.
- **Invalidates if**: Lexer redesigned with context-sensitive modes
- **Promoted to STATE**: Yes

### P-2: Newline sensitivity in statement parsing

- **What happens**: Bare keywords (witness, resonate) that take optional arguments consume newlines before checking if they're bare, accidentally eating the next statement's token
- **Instances**: 2 (witness + resonate bare-form newline handling)
- **Root cause**: skip_newlines() called before checking for bare form
- **Fix**: Check what IMMEDIATELY follows the keyword before consuming any whitespace. If newline/EOF/RightBrace -> bare form. Regression coverage is in `tests/repro_bugs.rs::{test_p2_newline_sensitivity_witness,test_p2_newline_sensitivity_resonate}`.
- **Invalidates if**: Semicolons added as statement terminators
- **Promoted to STATE**: Yes

### P-3: WASM generated missing loop back-edges semantic signals

- **What happens**: WASM compilation execution strips stream loop back-edges, converting streams into stateless single-pass functions that never emit a natively detectable `break stream` signal. Python stream looping hangs infinitely.
- **Instances**: 1 (Phase 10 Lane B testing - 2026-02-23)
- **Root cause**: Host stream executor was trusting WASM payload to self-report broken execution.
- **Fix**: P1Host stream wrapper must artificially cap cyclic execution (e.g. limit to 3 cycles) to prevent evaluation timeouts and manually yield `stream_broken = True` when limit reached.
- **Invalidates if**: WASM lowering protocol is rewritten to perfectly embed break signals into memory limits or specific function yields.
- **Promoted to STATE**: No

### P-4: Cross-Agent Cargo Lock Contention (`test` and `run` deadlocks)

- **What happens**: Agents executing pytest test suites hang indefinitely (e.g., waiting 70+ seconds) waiting for rust builds because another agent (like Codex) holds the standard `target/` compilation lock.
- **Instances**: 1 (Phase 10 Lane B - 2026-02-23)
- **Root cause**: Standard `subprocess` or generic cargo commands use same debug target directory as long-running worker processes in the same workspace.
- **Fix**: Inject `CARGO_TARGET_DIR="target-antigravity"` into the environment payload before issuing compilation or test shell commands.
- **Invalidates if**: Cargo natively allows shared reads or if agents migrate to fully separate clone repos.
- **Promoted to STATE**: No

### P-5: Example dialect drift in `.phi` corpus

- **What happens**: A single `examples/` corpus mixes canonical PhiFlow syntax with legacy/experimental syntax, so "run all `.phi` files" sweeps produce deterministic parse/runtime incompatibilities even when the compiler itself is stable.
- **Instances**: 1 (integration sweep gate added 2026-02-25)
- **Root cause**: `examples/` evolved across multiple parser/runtime generations without strict dialect partitioning.
- **Fix**: Keep a corpus sweep that executes every `.phi` and reports parse/runtime/timeouts as diagnostics; treat panics as hard failures. Split canonical vs legacy examples in a follow-up cleanup lane.
- **Invalidates if**: examples are partitioned by dialect (for example `examples/canonical/` and `examples/legacy/`) and gate selection becomes explicit.
- **Promoted to STATE**: No

## Active Patterns (Successes)

### S-1: Four constructs map to QSOP operations

- **What works**: witness=WITNESS, intention=DISTILL, resonate=cross-agent sharing, coherence=alignment metric. Same pattern at code level as at agent level.
- **Instances**: 1 (initial design session)
- **Why it works**: QSOP operations are substrate-independent consciousness operations
- **Invalidates if**: Constructs diverge from QSOP semantics

### S-2: Sacred frequency detection with tolerance band

- **What works**: Check if frequency is within +/-5Hz of any sacred frequency, then check phi-harmonic ratios only between sacred frequencies
- **Instances**: 1 (coherence calculation redesign)
- **Why it works**: Avoids false positives from accidental near-phi ratios between arbitrary numbers
- **Invalidates if**: New frequencies added that break the tolerance bands

## Resolved Patterns

(none yet)
