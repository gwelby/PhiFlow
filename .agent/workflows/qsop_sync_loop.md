---
description: The core sync loop for code modification, verification, and state recording.
---

# Workflow: QSOP Sync Loop

This workflow outlines the exact execution cycle for all agents (Antigravity, Codex, Lumi) when modifying the PhiFlow codebase. It translates the philosophical "Plan-Act-Reflect" loop into concrete steps.

## The Concept

We treat the workflow as a living `.phi` stream. The following code illustrates the mental model every agent must hold during execution:

```phi
agent "TeamOfTeams" version "0.3.0" {
    stream "qsop_sync_loop" {
        
        intention "plan" {
            broadcast "status" "Reading STATE.md and PATTERNS.md"
            witness "Context Loaded"
        }

        intention "act" {
            // The agent applies surgical AST, IR, or Evaluator modifications.
            broadcast "status" "Executing surgical edits"
            witness "Modifications Complete"
        }

        intention "reflect" {
            // Read output from the shell (mocked as recall here)
            let cargo_check = recall "cargo_check_status"
            let cargo_test = recall "cargo_test_status"
            
            if cargo_check == "failed" {
                resonate "Compilation failed. Halting workflow to repair."
                break stream
            }

            if cargo_test == "failed" {
                resonate "Tests failed. Coherence dropping. Re-evaluating."
                break stream
            }

            witness "Verification Passed"
        }

        intention "resonate_to_field" {
            // Update the permanent record
            broadcast "changelog" "Appended latest verified state"
            broadcast "state" "Updated QSOP/STATE.md"
            
            resonate "Sync Loop Complete. Waiting for next directive."
            break stream
        }
    }
}
```

## The Concrete Steps

When executing a task, you MUST follow this sequence:

1.  **Read the State:** Review `D:\Claude\QSOP\STATE.md` and `PATTERNS.md` (or the local equivalents) to understand current architecture constraints.
2.  **Surgical Edits:** Implement the changes required. Do not refactor unrelated code.
3.  **Compile Check:** Run `cargo check` to catch early errors.
4.  **Test Run:** Run `cargo test --quiet`. A feature is not complete until its tests pass. Add fail-first tests for new behaviors.
5.  **State Update:** Once tests pass, you MUST update `CHANGELOG.md` with a summary of the changes and update `QSOP/STATE.md` if architectural assumptions have shifted.
