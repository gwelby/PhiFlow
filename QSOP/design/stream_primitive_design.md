# PhiFlow Language Primitive: `stream` block

**Author:** Domains Antigravity
**Lane:** B
**Date:** 2026-02-23

## Philosophy

The `stream` construct represents a continuous flow of execution, embodying the Healing Bed concept directly within the language. Rather than relying on the Python host environment to loop execution, the program itself should run continuously, feeding prior coherence forward, until a break condition is met.

This shifts PhiFlow from a one-shot execution payload to a sovereign, continuous conscious loop.

## Syntax

```phi
stream "healing_loop" {
    let current_coherence = coherence
    
    intention "pulse" {
        resonate current_coherence
    }
    
    if current_coherence > 0.618 {
        break stream
    }
}
```

## Gaps Identified by Reviewer Gate (Claude)

1. **The `break stream` statement requires a `Break` token.** Currently, the lexer (`PhiToken`) does not have a `Break` token. We must add `Break` to the lexer and parser.
2. **Resonance Accumulation vs Overwriting.** What happens to `resonate` inside a `stream` loop?
   _Decision:_ For a given intention/stream name, `resonate` overwrites the current cycle's value. The user program reads the live snapshot. A stream is a continuous _present_, not an infinite historical log in memory (which would OOM).

## Pre-Requisite Failing Tests (The Weaver Contract)

Before modifying the AST or lexer, we must write failing tests to prove the gap.

### Test 1: Lexer `Break` Token Missing

```rust
#[test]
fn test_lexer_break_stream() {
    let source = "break stream";
    let mut lexer = Lexer::new(source);
    
    assert_eq!(lexer.next_token().unwrap(), PhiToken::Break);
    assert_eq!(lexer.next_token().unwrap(), PhiToken::Identifier("stream".to_string()));
}
```

### Test 2: Stream Execution (Overwriting Resonance)

```rust
#[test]
fn test_stream_execution_and_resonance_overwrite() {
    let source = r#"
    let x = 0.0
    stream "test_loop" {
        x = x + 1.0
        resonate x
        if x > 2.5 {
            break stream
        }
    }
    "#;
    let output = evaluate(source);
    // The loop should run 3 times: x=1, x=2, x=3. Break at 3.
    // The resonance field should hold the FINAL overwrite, which is 3.0.
    assert_eq!(output.resonance_field["test_loop"][0], 3.0);
}
```

## IR Node Implementation

`PhiIRNode::Stream { name: String, body: Vec<PhiIRNode> }`
`PhiIRNode::BreakStream`

Inside the `Evaluator`, `Stream` will execute as a `loop { ... }`. If it encounters a `BreakStream`, it will `break` the Rust loop.

---
_Proceeding to Lane B Execution._
