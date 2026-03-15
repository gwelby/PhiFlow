use phiflow::{compile_and_run_phi_ir, phi_ir::PhiIRValue};

#[test]
fn test_phiflow_demo() {
    let result = compile_and_run_phi_ir("6 * 7").expect("PhiFlow pipeline should evaluate");
    assert_eq!(result, PhiIRValue::Number(42.0));
}
