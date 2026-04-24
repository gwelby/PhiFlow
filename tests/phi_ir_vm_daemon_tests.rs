use phiflow::parser::parse_phi_program;
use phiflow::phi_ir::lowering::lower_program_checked;
use phiflow::phi_ir::emitter::emit;
use phiflow::phi_ir::vm::{PhiVm, VmExecResult};
use phiflow::host::{CallbackHostProvider, WitnessAction};
use std::sync::{Arc, Mutex};

#[test]
fn test_vm_daemon_yield_resume_cycle() {
    // A simple witness-heavy program
    let source = r#"
        intention "test_daemon" {
            let x = 10
            witness x
            let y = x + 5
            witness y
            y
        }
    "#;

    // 1. Compile to PhiIR
    let ast = parse_phi_program(source).expect("Failed to parse");
    let ir = lower_program_checked(&ast).expect("Failed to lower");

    // 2. Emit to Bytecode
    let bytecode = emit(&ir);

    // 3. Load into VM
    let mut vm = PhiVm::from_bytes(&bytecode).expect("Failed to load bytecode");
    
    // Set up a host that yields on witness
    let yield_count = Arc::new(Mutex::new(0));
    let yield_count_clone = Arc::clone(&yield_count);
    
    let host = Arc::new(CallbackHostProvider::new()
        .with_witness(move |_snapshot| {
            let mut count = yield_count_clone.lock().unwrap();
            *count += 1;
            WitnessAction::Yield
        }));
        
    vm = vm.with_host(host);

    // 4. Run to first yield
    let result1 = vm.run_or_yield().expect("Failed to run");
    let frozen_state1 = match result1 {
        VmExecResult::Yielded { snapshot, frozen_state } => {
            assert_eq!(snapshot.observed_value, Some("Number(10.0)".to_string()));
            frozen_state
        }
        other => panic!("Expected Yielded, got {:?}", other),
    };
    
    assert_eq!(*yield_count.lock().unwrap(), 1);

    // 5. Resume to second yield
    let result2 = vm.resume(frozen_state1).expect("Failed to resume");
    let frozen_state2 = match result2 {
        VmExecResult::Yielded { snapshot, frozen_state } => {
            assert_eq!(snapshot.observed_value, Some("Number(15.0)".to_string()));
            frozen_state
        }
        other => panic!("Expected Yielded, got {:?}", other),
    };
    
    assert_eq!(*yield_count.lock().unwrap(), 2);

    // 6. Resume to completion
    let result3 = vm.resume(frozen_state2).expect("Failed to resume");
    match result3 {
        VmExecResult::Complete(val) => {
            assert_eq!(val.as_number(), Some(15.0));
        }
        other => panic!("Expected Complete, got {:?}", other),
    }
}
