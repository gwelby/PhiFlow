//! Tests for the `actuate motor()` construct — the efferent pair of
//! `witness sensor()`. A program writing to a motor channel must leave the
//! process through the bounded MOTOR.jsonl bus.

use phiflow::parser::parse_phi_program;
use phiflow::phi_ir::{
    emitter, evaluator::Evaluator, lowering::lower_program, vm::PhiVm, wasm::emit_wat,
};
use std::sync::Mutex;

// MOTOR_BUS_PATH is process-global; serialize every test in this file so
// parallel runners cannot redirect or remove the bus mid-write.
static BUS_ENV_LOCK: Mutex<()> = Mutex::new(());

fn run_phi(source: &str) {
    let exprs = parse_phi_program(source).expect("parse failed");
    let program = lower_program(&exprs);
    let mut evaluator = Evaluator::new(program);
    evaluator.run().expect("evaluation failed");
}

#[test]
fn test_actuate_writes_motor_bus() {
    let _guard = BUS_ENV_LOCK.lock().unwrap();
    let dir = std::env::temp_dir().join(format!("phiflow-actuate-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let bus = dir.join("MOTOR.jsonl");
    std::env::set_var("MOTOR_BUS_PATH", &bus);

    run_phi(
        r#"
        let action = 0.75
        actuate motor("soma_motor", action)
        actuate motor("soma_motor", action * 2.0)
        "#,
    );

    let content = std::fs::read_to_string(&bus).expect("motor bus should exist");
    let lines: Vec<&str> = content.lines().filter(|l| !l.trim().is_empty()).collect();
    assert_eq!(lines.len(), 2, "two actuate events expected");

    let first: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
    assert_eq!(first["type"], "actuate");
    assert_eq!(first["channel"], "soma_motor");
    assert_eq!(first["value"], 0.75);
    let second: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
    assert_eq!(second["value"], 1.5);

    let _ = std::fs::remove_dir_all(&dir);
    std::env::remove_var("MOTOR_BUS_PATH");
}

#[test]
fn test_actuate_returns_commanded_value() {
    let _guard = BUS_ENV_LOCK.lock().unwrap();
    let dir = std::env::temp_dir().join(format!("phiflow-actuate-ret-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::env::set_var("MOTOR_BUS_PATH", dir.join("MOTOR.jsonl"));

    // `actuate` returns the commanded value so it can feed model state.
    run_phi(
        r#"
        let sent = actuate motor("probe", 0.42)
        resonate sent as "sent_echo"
        "#,
    );

    let _ = std::fs::remove_dir_all(&dir);
    std::env::remove_var("MOTOR_BUS_PATH");
}

#[test]
fn test_actuate_vm_bytecode_writes_bus() {
    let _guard = BUS_ENV_LOCK.lock().unwrap();
    let dir = std::env::temp_dir().join(format!("phiflow-actuate-vm-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let bus = dir.join("MOTOR.jsonl");
    std::env::set_var("MOTOR_BUS_PATH", &bus);

    let exprs = parse_phi_program(
        r#"
        actuate motor("vm_channel", 0.33)
        "#,
    )
    .expect("parse failed");
    let program = lower_program(&exprs);
    let bytes = emitter::emit(&program);
    PhiVm::run_bytes(&bytes).expect("VM execution failed");

    let content = std::fs::read_to_string(&bus).expect("motor bus should exist");
    let event: serde_json::Value = serde_json::from_str(content.lines().next().unwrap()).unwrap();
    assert_eq!(event["channel"], "vm_channel");
    assert_eq!(event["value"], 0.33);
    assert_eq!(event["source"], "phivm");

    let _ = std::fs::remove_dir_all(&dir);
    std::env::remove_var("MOTOR_BUS_PATH");
}

#[test]
fn test_actuate_wasm_wat_uses_phi_actuate() {
    let exprs = parse_phi_program(
        r#"
        actuate motor("wasm_channel", 0.5)
        "#,
    )
    .expect("parse failed");
    let program = lower_program(&exprs);
    let wat = emit_wat(&program);
    assert!(wat.contains("phi_actuate"), "WAT must use the phi_actuate import");
    assert!(wat.contains("\"actuate\""), "WAT must import \"actuate\"");
}

#[test]
fn test_actuate_rejects_unknown_class() {
    let result = parse_phi_program(
        r#"
        actuate servo("channel", 1.0)
        "#,
    );
    assert!(result.is_err(), "actuate with unknown class should fail");
    let msg = format!("{:?}", result.unwrap_err());
    assert!(msg.contains("motor"), "error should name the motor class: {}", msg);
}
