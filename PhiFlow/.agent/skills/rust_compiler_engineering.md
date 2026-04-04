# Skill: Rust Compiler Engineering (PhiFlow Edition)

**Level:** Master
**Domain:** `src/parser/`, `src/phi_ir/`

When making changes to the PhiFlow compiler pipeline, you must adhere to these absolute engineering truths. We are building a living substrate, not a text parser.

## 1. The AST is Sacred (`parser/mod.rs`)
- If you add a new keyword, you **must** update `expect_identifier()` so that the lexer does not greedily consume the keyword when used as a variable name (Pattern P-1).
- If your keyword can be bare (no arguments, e.g., `witness`), you **must** check what immediately follows before consuming newlines (Pattern P-2).

## 2. IR Lowering is Topological (`phi_ir/lowering.rs`)
- PhiIR is SSA (Static Single Assignment). Every operation that produces a value must be assigned a new `Operand`.
- When adding a new `PhiIRNode` variant, you MUST update the `produces_value` match block inside `LoweringContext::emit()`. If you fail to do this, the SSA registers will misalign, and the VM will panic with `OperandNotFound`.

## 3. Fail-First Testing
You do not write code and hope it works. You write the trap, then you build the mouse.
- Add fail-first tests in `tests/phi_ir_evaluator_tests.rs`.
- We use the `CallbackHostProvider` to mock hardware and intercept `witness` and `resonate` events in tests.

## 4. Self-Verification Script
When you finish a compiler feature, you can verify your agentic coherence by mentally (or literally) executing this stream:

```phi
agent "CompilerEngineer" version "1.0" {
    stream "verify_pipeline" {
        intention "compile_and_run" {
            let pass_check = listen "cargo_test_results"
            if pass_check == void {
                resonate "Tests failed. Coherence dropping."
                break stream
            }
            witness "Pipeline Stable"
        }
    }
}
```
