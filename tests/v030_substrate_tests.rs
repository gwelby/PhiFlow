// v0.3.0 Substrate Tests
// Verifies persistence, dialogue, and agent identity.

use phiflow::phi_ir::lowering::lower_program;
use phiflow::phi_ir::evaluator::{Evaluator, VmExecResult};
use phiflow::parser::parse_phi_program;
use phiflow::host::{CallbackHostProvider, WitnessAction};
use std::sync::{Arc, Mutex};

#[test]
fn test_remember_recall_roundtrip() {
    let source = r#"
        remember "key" 42.0
        let val = recall "key"
        val
    "#;
    let exprs = parse_phi_program(source).unwrap();
    let program = lower_program(&exprs);
    
    let storage = Arc::new(Mutex::new(std::collections::HashMap::new()));
    let storage_clone = storage.clone();
    
    let host = CallbackHostProvider::new()
        .with_persist(move |k, v| {
            storage_clone.lock().unwrap().insert(k.to_string(), v.to_string());
        })
        .with_recall(move |k| {
            storage.lock().unwrap().get(k).cloned()
        });
        
    let mut eval = Evaluator::new(&program).with_host(Box::new(host));
    let result = eval.run().unwrap();
    
    assert_eq!(result.as_number(), Some(42.0));
}

#[test]
fn test_agent_identity_flow() {
    let source = r#"
        agent "TestAgent" version "1.0.0" {
            witness
        }
    "#;
    let exprs = parse_phi_program(source).unwrap();
    let program = lower_program(&exprs);
    
    let mut eval = Evaluator::new(&program);
    eval.run().unwrap();
    
    let log = &eval.witness_log;
    assert_eq!(log.len(), 1);
    assert_eq!(log[0].agent_name, Some("TestAgent".to_string()));
}

#[test]
fn test_broadcast_listen_dialogue() {
    let source = r#"
        broadcast "chan" 123.0
        listen "chan"
    "#;
    let exprs = parse_phi_program(source).unwrap();
    let program = lower_program(&exprs);
    
    let bus = Arc::new(Mutex::new(std::collections::HashMap::new()));
    let bus_clone = bus.clone();
    
    let host = CallbackHostProvider::new()
        .with_broadcast(move |c, v| {
            bus_clone.lock().unwrap().insert(c.to_string(), v.to_string());
        })
        .with_listen(move |c| {
            bus.lock().unwrap().get(c).cloned()
        });
        
    let mut eval = Evaluator::new(&program).with_host(Box::new(host));
    let result = eval.run().unwrap();
    
    assert_eq!(result.as_number(), Some(123.0));
}

#[test]
fn test_yield_resume_machinery() {
    let source = r#"
        witness
        42.0
    "#;
    let exprs = parse_phi_program(source).unwrap();
    let program = lower_program(&exprs);
    
    let mut eval = Evaluator::new(&program).with_host(Box::new(
        CallbackHostProvider::new().with_witness(|_| WitnessAction::Yield)
    ));
    
    let res = eval.run_or_yield().unwrap();
    if let VmExecResult::Yielded { frozen_state, .. } = res {
        let res2 = eval.resume(frozen_state).unwrap();
        if let VmExecResult::Complete(val) = res2 {
            assert_eq!(val.as_number(), Some(42.0));
        } else {
            panic!("Expected completion");
        }
    } else {
        panic!("Expected yield");
    }
}
