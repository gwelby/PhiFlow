# PATTERNS - Learning from mistakes and successes

## Active Patterns (Mistakes)

### P-1: Keyword-as-variable collision
- **What happens**: PhiFlow keywords (frequency, state, coherence, etc.) used as variable names cause parse errors because lexer emits keyword tokens, not Identifier
- **Instances**: 2 (frequency in create statement, witness/resonate as identifiers)
- **Root cause**: Lexer is greedy with keyword matching. No context-sensitive tokenization.
- **Fix**: In parser, accept keyword tokens where Identifier is expected. Already done for create statements and expect_identifier().
- **Invalidates if**: Lexer redesigned with context-sensitive modes
- **Promoted to STATE**: No (handled case-by-case)

### P-2: Newline sensitivity in statement parsing
- **What happens**: Bare keywords (witness, resonate) that take optional arguments consume newlines before checking if they're bare, accidentally eating the next statement's token
- **Instances**: 1 (witness consuming next `let` statement)
- **Root cause**: skip_newlines() called before checking for bare form
- **Fix**: Check what IMMEDIATELY follows the keyword before consuming any whitespace. If newline/EOF/RightBrace -> bare form.
- **Invalidates if**: Semicolons added as statement terminators
- **Promoted to STATE**: No

### P-3: WASM stream loop-back signal loss
- **What happens**: WASM execution path can lose stream loop break semantics, so host-side streaming may hang or require manual stop conditions.
- **Instances**: 1 (Phase 10 stream-output lane, compiler branch log on 2026-02-23)
- **Root cause**: Host expected a native `break stream` signal from WASM output that was not reliably surfaced.
- **Fix**: Use bounded host loop with explicit cycle cap and emit `stream_broken` marker in stream snapshots.
- **Invalidates if**: WASM lowering/runtime contract emits a stable loop-break signal directly.
- **Promoted to STATE**: No

### P-4: Cross-agent cargo target lock contention
- **What happens**: Parallel agents running cargo in the same workspace can deadlock/wait on shared target locks and appear hung.
- **Instances**: 1 (Phase 10 multi-agent lane execution, compiler branch log on 2026-02-23)
- **Root cause**: Shared default `target/` directory under concurrent long-running runs/tests.
- **Fix**: Use per-agent `CARGO_TARGET_DIR` isolation when running heavy parallel loops.
- **Invalidates if**: agents are isolated to separate clones/containers or cargo lock behavior changes.
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
