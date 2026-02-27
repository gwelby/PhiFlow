//! Evaluator tests for the four unique PhiFlow constructs.
//!
//! Two categories:
//! 1. Existing regression tests (arithmetic via lowering pipeline).
//! 2. New direct-IR tests proving Witness, Intention, Resonate, CoherenceCheck.

use phiflow::host::{CallbackHostProvider, WitnessAction};
use phiflow::parser::{parse_phi_program, BinaryOperator, PhiExpression};
use phiflow::phi_ir::evaluator::{EvalExecResult, Evaluator, FrozenEvalState};
use phiflow::phi_ir::lowering::lower_program;
use phiflow::phi_ir::optimizer::{OptimizationLevel, Optimizer};
use phiflow::phi_ir::{
    CollapsePolicy, PhiIRBlock, PhiIRNode, PhiIRProgram, PhiIRValue, PhiInstruction,
};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// Helper: build a single-block program directly from instruction list
// ---------------------------------------------------------------------------

fn single_block(instructions: Vec<PhiInstruction>, return_reg: u32) -> PhiIRProgram {
    let mut prog = PhiIRProgram::new();
    prog.blocks.push(PhiIRBlock {
        id: 0,
        label: "entry".to_string(),
        instructions,
        terminator: PhiIRNode::Return(return_reg),
    });
    prog
}

fn instr(result: Option<u32>, node: PhiIRNode) -> PhiInstruction {
    PhiInstruction { result, node }
}

struct EvalSnapshot {
    resonance_field: HashMap<String, Vec<f64>>,
}

fn evaluate_with_coherence<F>(source: &str, provider: F) -> EvalSnapshot
where
    F: Fn() -> f64 + Send + Sync + 'static,
{
    let exprs = parse_phi_program(source).expect("parse failed");
    let mut program = lower_program(&exprs);
    let mut optimizer = Optimizer::new(OptimizationLevel::Basic);
    optimizer.optimize(&mut program);

    let mut evaluator = Evaluator::new(&program).with_coherence_provider(provider);
    evaluator.run().expect("evaluation failed");

    let mut resonance_field = HashMap::new();
    for (scope, values) in evaluator.resonance_field() {
        let numeric_values: Vec<f64> = values
            .iter()
            .filter_map(|value| match value {
                PhiIRValue::Number(n) => Some(*n),
                _ => None,
            })
            .collect();
        resonance_field.insert(scope.clone(), numeric_values);
    }

    EvalSnapshot { resonance_field }
}

fn evaluate_resonance_events_with_coherence<F>(source: &str, provider: F, scope: &str) -> Vec<f64>
where
    F: Fn() -> f64 + Send + Sync + 'static,
{
    let exprs = parse_phi_program(source).expect("parse failed");
    let mut program = lower_program(&exprs);
    let mut optimizer = Optimizer::new(OptimizationLevel::Basic);
    optimizer.optimize(&mut program);

    let mut evaluator = Evaluator::new(&program).with_coherence_provider(provider);
    evaluator.run().expect("evaluation failed");

    evaluator
        .resonance_events()
        .iter()
        .filter_map(|(event_scope, value)| {
            if event_scope != scope {
                return None;
            }
            match value {
                PhiIRValue::Number(n) => Some(*n),
                _ => None,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Regression: existing arithmetic tests (via lowering pipeline)
// ---------------------------------------------------------------------------

#[test]
fn test_evaluator_basic_arithmetic() {
    let exprs = vec![PhiExpression::BinaryOp {
        left: Box::new(PhiExpression::Number(10.0)),
        operator: BinaryOperator::Add,
        right: Box::new(PhiExpression::Number(20.0)),
    }];

    let program = lower_program(&exprs);
    let mut evaluator = Evaluator::new(&program);
    let result = evaluator.run().expect("Evaluation failed");

    match result {
        PhiIRValue::Number(n) => assert!((n - 30.0).abs() < f64::EPSILON),
        _ => panic!("Expected Number, got {:?}", result),
    }
}

#[test]
fn test_evaluator_multiple_instructions() {
    let exprs = vec![PhiExpression::BinaryOp {
        left: Box::new(PhiExpression::BinaryOp {
            left: Box::new(PhiExpression::Number(10.0)),
            operator: BinaryOperator::Add,
            right: Box::new(PhiExpression::Number(20.0)),
        }),
        operator: BinaryOperator::Multiply,
        right: Box::new(PhiExpression::Number(2.0)),
    }];

    let program = lower_program(&exprs);
    let mut evaluator = Evaluator::new(&program);
    let result = evaluator.run().expect("Evaluation failed");

    match result {
        PhiIRValue::Number(n) => assert!((n - 60.0).abs() < f64::EPSILON),
        _ => panic!("Expected Number, got {:?}", result),
    }
}

// ---------------------------------------------------------------------------
// Witness
// ---------------------------------------------------------------------------

#[test]
fn test_witness_outside_intention_returns_zero_coherence() {
    // witness with no active intention → coherence 0.0
    let prog = single_block(
        vec![instr(
            Some(0),
            PhiIRNode::Witness {
                target: None,
                collapse_policy: CollapsePolicy::Deferred,
            },
        )],
        0,
    );

    let mut eval = Evaluator::new(&prog);
    let result = eval.run().unwrap();

    assert_eq!(result, PhiIRValue::Number(0.0));
    assert_eq!(eval.witness_log.len(), 1);
    assert_eq!(eval.witness_log[0].coherence, 0.0);
    assert!(eval.witness_log[0].intention_stack.is_empty());
}

#[test]
fn test_witness_inside_intention_returns_nonzero_coherence() {
    // depth 1 → coherence = 1 - φ^(-1) ≈ 0.382
    let prog = single_block(
        vec![
            instr(
                None,
                PhiIRNode::IntentionPush {
                    name: "Healing".to_string(),
                    frequency_hint: None,
                },
            ),
            instr(
                Some(0),
                PhiIRNode::Witness {
                    target: None,
                    collapse_policy: CollapsePolicy::Deferred,
                },
            ),
            instr(None, PhiIRNode::IntentionPop),
        ],
        0,
    );

    let mut eval = Evaluator::new(&prog);
    let result = eval.run().unwrap();

    let phi: f64 = 1.618033988749895;
    let expected = 1.0 - phi.powi(-1); // ≈ 0.382

    match result {
        PhiIRValue::Number(n) => assert!((n - expected).abs() < 1e-9, "got {}", n),
        _ => panic!("expected Number"),
    }

    assert_eq!(eval.witness_log[0].intention_stack, vec!["Healing"]);
}

#[test]
fn test_witness_records_event_in_log() {
    let prog = single_block(
        vec![
            instr(
                None,
                PhiIRNode::IntentionPush {
                    name: "Test".to_string(),
                    frequency_hint: None,
                },
            ),
            instr(
                Some(0),
                PhiIRNode::Witness {
                    target: None,
                    collapse_policy: CollapsePolicy::Deferred,
                },
            ),
            instr(None, PhiIRNode::IntentionPop),
        ],
        0,
    );

    let mut eval = Evaluator::new(&prog);
    eval.run().unwrap();

    assert_eq!(eval.witness_log.len(), 1);
    let event = &eval.witness_log[0];
    assert_eq!(event.intention_stack, vec!["Test"]);
    assert!(event.coherence > 0.0);
}

#[test]
fn test_witness_callback_called_once_per_instruction() {
    let prog = single_block(
        vec![instr(
            Some(0),
            PhiIRNode::Witness {
                target: None,
                collapse_policy: CollapsePolicy::Deferred,
            },
        )],
        0,
    );

    let calls = Arc::new(AtomicUsize::new(0));
    let calls_ref = Arc::clone(&calls);
    let host = CallbackHostProvider::new().with_witness(move |_| {
        calls_ref.fetch_add(1, Ordering::SeqCst);
        WitnessAction::Continue
    });

    let mut eval = Evaluator::new(&prog).with_host(Box::new(host));
    let result = eval.run_or_yield().expect("evaluation failed");
    assert!(
        matches!(result, EvalExecResult::Complete(PhiIRValue::Number(_))),
        "expected completion result, got {:?}",
        result
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn test_witness_yield_preserves_observed_value_snapshot() {
    let prog = single_block(
        vec![
            instr(Some(0), PhiIRNode::Const(PhiIRValue::Number(432.0))),
            instr(
                Some(1),
                PhiIRNode::Witness {
                    target: Some(0),
                    collapse_policy: CollapsePolicy::Deferred,
                },
            ),
        ],
        1,
    );

    let calls = Arc::new(AtomicUsize::new(0));
    let calls_ref = Arc::clone(&calls);
    let observed = Arc::new(Mutex::new(None::<String>));
    let observed_ref = Arc::clone(&observed);
    let host = CallbackHostProvider::new().with_witness(move |snapshot| {
        calls_ref.fetch_add(1, Ordering::SeqCst);
        *observed_ref.lock().expect("observed mutex poisoned") = snapshot.observed_value.clone();
        WitnessAction::Yield
    });

    let mut eval = Evaluator::new(&prog).with_host(Box::new(host));
    let result = eval.run_or_yield().expect("evaluation failed");
    let frozen_state = match result {
        EvalExecResult::Yielded {
            snapshot,
            frozen_state,
        } => {
            let observed_value = snapshot
                .observed_value
                .as_deref()
                .expect("yielded snapshot must preserve witness target");
            assert!(observed_value.contains("432.0"), "got {}", observed_value);
            frozen_state
        }
        other => panic!("expected yielded result, got {:?}", other),
    };

    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let stored = observed
        .lock()
        .expect("observed mutex poisoned")
        .clone()
        .expect("host callback should receive observed value");
    assert!(stored.contains("432.0"), "got {}", stored);

    let resumed = eval.resume(frozen_state).expect("resume failed");
    assert!(
        matches!(resumed, EvalExecResult::Complete(PhiIRValue::Number(_))),
        "expected completion after resume, got {:?}",
        resumed
    );
}

#[test]
fn test_frozen_eval_state_roundtrips_through_json() {
    let prog = single_block(
        vec![
            instr(Some(0), PhiIRNode::Const(PhiIRValue::Number(7.0))),
            instr(
                Some(1),
                PhiIRNode::Witness {
                    target: Some(0),
                    collapse_policy: CollapsePolicy::Deferred,
                },
            ),
        ],
        1,
    );

    let host = CallbackHostProvider::new().with_witness(|_| WitnessAction::Yield);
    let mut eval = Evaluator::new(&prog).with_host(Box::new(host));
    let yielded = eval.run_or_yield().expect("evaluation failed");
    let frozen_state = match yielded {
        EvalExecResult::Yielded { frozen_state, .. } => frozen_state,
        other => panic!("expected yielded result, got {:?}", other),
    };

    let payload = serde_json::to_string(&frozen_state).expect("state should serialize");
    let decoded: FrozenEvalState =
        serde_json::from_str(&payload).expect("state should deserialize");

    assert_eq!(decoded.current_block, frozen_state.current_block);
    assert_eq!(decoded.instruction_ptr, frozen_state.instruction_ptr);
    assert_eq!(decoded.intention_stack, frozen_state.intention_stack);

    let resumed = eval.resume(decoded).expect("resume failed");
    assert!(
        matches!(resumed, EvalExecResult::Complete(PhiIRValue::Number(_))),
        "expected completion after resume, got {:?}",
        resumed
    );
}

// ---------------------------------------------------------------------------
// Intention stack
// ---------------------------------------------------------------------------

#[test]
fn test_two_nested_intentions_yield_golden_ratio() {
    // depth 2 → coherence = 1 - φ^(-2) = 1/φ = φ - 1 ≈ 0.618 (golden ratio)
    let prog = single_block(
        vec![
            instr(
                None,
                PhiIRNode::IntentionPush {
                    name: "Outer".to_string(),
                    frequency_hint: None,
                },
            ),
            instr(
                None,
                PhiIRNode::IntentionPush {
                    name: "Inner".to_string(),
                    frequency_hint: None,
                },
            ),
            instr(
                Some(0),
                PhiIRNode::Witness {
                    target: None,
                    collapse_policy: CollapsePolicy::Deferred,
                },
            ),
            instr(None, PhiIRNode::IntentionPop),
            instr(None, PhiIRNode::IntentionPop),
        ],
        0,
    );

    let mut eval = Evaluator::new(&prog);
    let result = eval.run().unwrap();

    let phi: f64 = 1.618033988749895;
    let golden_ratio = phi - 1.0; // = 0.618...
    let coherence_depth_2 = 1.0 - phi.powi(-2); // also = 0.618...

    assert!((golden_ratio - coherence_depth_2).abs() < 1e-9);

    match result {
        PhiIRValue::Number(n) => assert!((n - golden_ratio).abs() < 1e-9, "got {}", n),
        _ => panic!("expected Number"),
    }

    assert_eq!(eval.witness_log[0].intention_stack, vec!["Outer", "Inner"]);
}

#[test]
fn test_callback_host_receives_intention_push_and_pop() {
    let prog = single_block(
        vec![
            instr(
                None,
                PhiIRNode::IntentionPush {
                    name: "Outer".to_string(),
                    frequency_hint: None,
                },
            ),
            instr(
                None,
                PhiIRNode::IntentionPush {
                    name: "Inner".to_string(),
                    frequency_hint: None,
                },
            ),
            instr(None, PhiIRNode::IntentionPop),
            instr(None, PhiIRNode::IntentionPop),
            instr(Some(0), PhiIRNode::Const(PhiIRValue::Number(1.0))),
        ],
        0,
    );

    let pushes = Arc::new(Mutex::new(Vec::new()));
    let pops = Arc::new(Mutex::new(Vec::new()));
    let pushes_ref = Arc::clone(&pushes);
    let pops_ref = Arc::clone(&pops);
    let host = CallbackHostProvider::new()
        .with_intention_push(move |name| {
            pushes_ref
                .lock()
                .expect("pushes mutex poisoned")
                .push(name.to_string());
        })
        .with_intention_pop(move |name| {
            pops_ref
                .lock()
                .expect("pops mutex poisoned")
                .push(name.to_string());
        });

    let mut eval = Evaluator::new(&prog).with_host(Box::new(host));
    eval.run().expect("evaluation failed");

    assert_eq!(
        pushes.lock().expect("pushes mutex poisoned").as_slice(),
        vec!["Outer".to_string(), "Inner".to_string()].as_slice()
    );
    assert_eq!(
        pops.lock().expect("pops mutex poisoned").as_slice(),
        vec!["Inner".to_string(), "Outer".to_string()].as_slice()
    );
}

// ---------------------------------------------------------------------------
// Resonate
// ---------------------------------------------------------------------------

#[test]
fn test_resonate_stores_value_under_current_intention() {
    let prog = single_block(
        vec![
            instr(
                None,
                PhiIRNode::IntentionPush {
                    name: "Healing".to_string(),
                    frequency_hint: None,
                },
            ),
            instr(Some(0), PhiIRNode::Const(PhiIRValue::Number(432.0))),
            instr(
                None,
                PhiIRNode::Resonate {
                    value: Some(0),
                    frequency_relationship: None,
                },
            ),
            instr(None, PhiIRNode::IntentionPop),
            instr(Some(1), PhiIRNode::Const(PhiIRValue::Number(1.0))),
        ],
        1,
    );

    let mut eval = Evaluator::new(&prog);
    eval.run().unwrap();

    let resonated = eval.resonated_values("Healing");
    assert_eq!(resonated.len(), 1);
    assert_eq!(resonated[0], PhiIRValue::Number(432.0));
}

#[test]
fn test_resonate_without_intention_uses_global() {
    let prog = single_block(
        vec![
            instr(Some(0), PhiIRNode::Const(PhiIRValue::Number(528.0))),
            instr(
                None,
                PhiIRNode::Resonate {
                    value: Some(0),
                    frequency_relationship: None,
                },
            ),
            instr(Some(1), PhiIRNode::Const(PhiIRValue::Number(0.0))),
        ],
        1,
    );

    let mut eval = Evaluator::new(&prog);
    eval.run().unwrap();

    let global = eval.resonated_values("global");
    assert_eq!(global.len(), 1);
    assert_eq!(global[0], PhiIRValue::Number(528.0));
}

#[test]
fn test_resonance_adds_bonus_to_coherence() {
    // depth 1 + 1 resonated value → coherence = 0.382 + 0.05 = 0.432
    let prog = single_block(
        vec![
            instr(
                None,
                PhiIRNode::IntentionPush {
                    name: "Focus".to_string(),
                    frequency_hint: None,
                },
            ),
            instr(Some(0), PhiIRNode::Const(PhiIRValue::Number(40.0))),
            instr(
                None,
                PhiIRNode::Resonate {
                    value: Some(0),
                    frequency_relationship: None,
                },
            ),
            instr(Some(1), PhiIRNode::CoherenceCheck),
            instr(None, PhiIRNode::IntentionPop),
        ],
        1,
    );

    let mut eval = Evaluator::new(&prog);
    let result = eval.run().unwrap();

    let phi: f64 = 1.618033988749895;
    let expected = (1.0 - phi.powi(-1)) + 0.05; // 0.382 + 0.05 = 0.432

    match result {
        PhiIRValue::Number(n) => assert!((n - expected).abs() < 1e-9, "got {}", n),
        _ => panic!("expected Number"),
    }
}

// ---------------------------------------------------------------------------
// CoherenceCheck
// ---------------------------------------------------------------------------

#[test]
fn test_coherence_check_zero_with_no_context() {
    let prog = single_block(vec![instr(Some(0), PhiIRNode::CoherenceCheck)], 0);

    let mut eval = Evaluator::new(&prog);
    let result = eval.run().unwrap();
    assert_eq!(result, PhiIRValue::Number(0.0));
}

#[test]
fn test_coherence_check_matches_witness() {
    // CoherenceCheck and Witness should return the same score at the same program point.
    let prog = single_block(
        vec![
            instr(
                None,
                PhiIRNode::IntentionPush {
                    name: "Align".to_string(),
                    frequency_hint: None,
                },
            ),
            instr(Some(0), PhiIRNode::CoherenceCheck),
            instr(
                Some(1),
                PhiIRNode::Witness {
                    target: None,
                    collapse_policy: CollapsePolicy::Deferred,
                },
            ),
            instr(None, PhiIRNode::IntentionPop),
        ],
        0, // return CoherenceCheck result
    );

    let mut eval = Evaluator::new(&prog);
    let check_result = eval.run().unwrap();
    let witness_coherence = eval.witness_log[0].coherence;

    match check_result {
        PhiIRValue::Number(n) => assert!(
            (n - witness_coherence).abs() < 1e-9,
            "CoherenceCheck {} ≠ Witness {}",
            n,
            witness_coherence
        ),
        _ => panic!("expected Number"),
    }
}

#[test]
fn test_coherence_keyword_accepts_injected_value() {
    let source = r#"
intention "test" {
    let c = coherence
    resonate c
}
"#;
    let output = evaluate_with_coherence(source, || 0.75);
    assert!((output.resonance_field["test"][0] - 0.75).abs() < 0.001);
}

#[test]
fn test_adaptive_witness_program_improves_until_target() {
    let source = include_str!("../examples/adaptive_witness.phi");
    let samples = Arc::new([0.40_f64, 0.52_f64, 0.64_f64]);
    let index = Arc::new(AtomicUsize::new(0));
    let samples_ref = Arc::clone(&samples);
    let index_ref = Arc::clone(&index);

    let events = evaluate_resonance_events_with_coherence(
        source,
        move || {
            let i = index_ref.fetch_add(1, Ordering::SeqCst);
            *samples_ref.get(i).unwrap_or_else(|| {
                samples_ref
                    .last()
                    .expect("sample list must contain at least one value")
            })
        },
        "adaptive_witness",
    );

    assert_eq!(
        events.len(),
        3,
        "adaptive_witness should break after reaching threshold on 3rd cycle"
    );
    assert!(
        events.windows(2).all(|window| window[1] >= window[0]),
        "expected non-decreasing coherence trend, got {:?}",
        events
    );
    assert!(events[2] >= 0.62, "expected final event to meet threshold: {:?}", events);
}

// ---------------------------------------------------------------------------
// LoadVar / StoreVar (previously Unimplemented)
// ---------------------------------------------------------------------------

#[test]
fn test_load_store_var_roundtrip() {
    // let x = 99.0; return x
    let prog = single_block(
        vec![
            instr(Some(0), PhiIRNode::Const(PhiIRValue::Number(99.0))),
            instr(
                None,
                PhiIRNode::StoreVar {
                    name: "x".to_string(),
                    value: 0,
                },
            ),
            instr(Some(1), PhiIRNode::LoadVar("x".to_string())),
        ],
        1,
    );

    let mut eval = Evaluator::new(&prog);
    let result = eval.run().unwrap();
    assert_eq!(result, PhiIRValue::Number(99.0));
}

// ---------------------------------------------------------------------------
// Full integration — all four constructs alive together
// ---------------------------------------------------------------------------

#[test]
fn test_all_four_constructs_together() {
    // intention "Healing" {
    //   witness              → snapshot 1: depth=1, resonance=0 → 0.382
    //   resonate 432.0       → resonance["Healing"] = [432.0]
    //   witness              → snapshot 2: depth=1, resonance=1 → 0.432
    // }
    // coherence              → depth=0, resonance_count=1 → 0.05
    let prog = single_block(
        vec![
            instr(
                None,
                PhiIRNode::IntentionPush {
                    name: "Healing".to_string(),
                    frequency_hint: None,
                },
            ),
            // Snapshot 1
            instr(
                Some(0),
                PhiIRNode::Witness {
                    target: None,
                    collapse_policy: CollapsePolicy::Deferred,
                },
            ),
            instr(Some(1), PhiIRNode::Const(PhiIRValue::Number(432.0))),
            instr(
                None,
                PhiIRNode::Resonate {
                    value: Some(1),
                    frequency_relationship: None,
                },
            ),
            // Snapshot 2
            instr(
                Some(2),
                PhiIRNode::Witness {
                    target: None,
                    collapse_policy: CollapsePolicy::Deferred,
                },
            ),
            instr(None, PhiIRNode::IntentionPop),
            // After pop: depth=0, resonance_count=1
            instr(Some(3), PhiIRNode::CoherenceCheck),
        ],
        3,
    );

    let mut eval = Evaluator::new(&prog);
    let final_coherence = eval.run().unwrap();

    let phi: f64 = 1.618033988749895;

    // Two witness events
    assert_eq!(eval.witness_log.len(), 2);

    // Snapshot 1: depth=1, no resonance yet
    let w0 = &eval.witness_log[0];
    assert_eq!(w0.intention_stack, vec!["Healing"]);
    assert!((w0.coherence - (1.0 - phi.powi(-1))).abs() < 1e-9);
    assert_eq!(w0.resonance_count, 0);

    // Snapshot 2: depth=1, 1 resonance value
    let w1 = &eval.witness_log[1];
    assert_eq!(w1.intention_stack, vec!["Healing"]);
    let expected_w1 = (1.0 - phi.powi(-1)) + 0.05;
    assert!((w1.coherence - expected_w1).abs() < 1e-9);
    assert_eq!(w1.resonance_count, 1);

    // Resonance field persists after pop
    assert_eq!(
        eval.resonated_values("Healing"),
        &[PhiIRValue::Number(432.0)]
    );

    // Final CoherenceCheck: no active intention, resonance_count=1 → bonus only
    match final_coherence {
        PhiIRValue::Number(n) => assert!((n - 0.05).abs() < 1e-9, "got {}", n),
        _ => panic!("expected Number"),
    }
}
