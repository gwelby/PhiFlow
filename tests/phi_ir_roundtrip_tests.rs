use phiflow::parser::parse_phi_program;
use phiflow::phi_ir::{
    emitter,
    evaluator::Evaluator,
    lowering::lower_program,
    vm::PhiVm,
    PhiIRProgram, PhiIRValue,
};

fn assert_values_close(lhs: &PhiIRValue, rhs: &PhiIRValue, context: &str) {
    match (lhs, rhs) {
        (PhiIRValue::Number(a), PhiIRValue::Number(b)) => {
            assert!(
                (a - b).abs() < 1e-9,
                "{}: numeric mismatch (lhs={}, rhs={})",
                context,
                a,
                b
            );
        }
        _ => assert_eq!(lhs, rhs, "{}: value mismatch", context),
    }
}

fn run_roundtrip_program(program: &PhiIRProgram) -> (PhiIRValue, PhiIRValue) {
    let mut evaluator = Evaluator::new(program);
    let eval_result = evaluator.run().expect("evaluator failed");

    let bytes = emitter::emit(program);
    let vm_result = PhiVm::run_bytes(&bytes).expect("vm failed");

    (eval_result, vm_result)
}

fn run_roundtrip_source(source: &str) -> (PhiIRValue, PhiIRValue) {
    let expressions = parse_phi_program(source).expect("parse failed");
    let program = lower_program(&expressions);
    run_roundtrip_program(&program)
}

#[test]
fn roundtrip_arithmetic_42() {
    let source = r#"
        let x = 6 * 7
        x
    "#;

    let (eval_result, vm_result) = run_roundtrip_source(source);
    assert_values_close(&eval_result, &vm_result, "arithmetic_42");
    assert_values_close(&eval_result, &PhiIRValue::Number(42.0), "arithmetic_42 expected");
}

#[test]
fn roundtrip_chained_84() {
    let source = r#"
        let x = 10 + 32
        let y = x * 2
        y
    "#;

    let (eval_result, vm_result) = run_roundtrip_source(source);
    assert_values_close(&eval_result, &vm_result, "chained_84");
    assert_values_close(&eval_result, &PhiIRValue::Number(84.0), "chained_84 expected");
}

#[test]
fn roundtrip_boolean_branch() {
    let source = r#"
        let x = 0
        if true { let x = 1 } else { let x = 2 }
        x
    "#;

    let (eval_result, vm_result) = run_roundtrip_source(source);
    assert_values_close(&eval_result, &vm_result, "boolean_branch");
    assert_values_close(&eval_result, &PhiIRValue::Number(1.0), "boolean_branch expected");
}

#[test]
fn roundtrip_witness_node() {
    let source = "witness";

    let (eval_result, vm_result) = run_roundtrip_source(source);
    assert_values_close(&eval_result, &vm_result, "witness_node");
    assert_values_close(
        &eval_result,
        &PhiIRValue::Number(0.0),
        "witness_node expected coherence",
    );
}

#[test]
fn roundtrip_intention_push_pop() {
    let source = r#"
        intention "Healing" { 42 }
    "#;

    let (eval_result, vm_result) = run_roundtrip_source(source);
    assert_values_close(&eval_result, &vm_result, "intention_push_pop");
    assert_values_close(
        &eval_result,
        &PhiIRValue::Number(42.0),
        "intention_push_pop expected",
    );
}

#[test]
fn roundtrip_coherence_check_node() {
    let source = "coherence";
    let (eval_result, vm_result) = run_roundtrip_source(source);
    assert_values_close(&eval_result, &vm_result, "coherence_check_node");
    assert_values_close(&eval_result, &PhiIRValue::Number(0.0), "coherence_check_node expected");
}
