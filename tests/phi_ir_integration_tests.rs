use phiflow::compile_and_run_phi_ir;
use phiflow::phi_ir::PhiIRValue;

#[test]
fn test_compile_run_basic_arithmetic() {
    let source = "10 + 20";
    let result = compile_and_run_phi_ir(source);

    match result {
        Ok(PhiIRValue::Number(n)) => {
            assert!(
                (n - 30.0).abs() < f64::EPSILON,
                "Expected 30.0, got {:?}",
                n
            );
        }
        Ok(val) => panic!("Expected Number, got {:?}", val),
        Err(e) => panic!("Compilation failed: {}", e),
    }
}

#[test]
fn test_compile_run_constant_folding() {
    // This should be optimized by constant folding, but runtime should give same result.
    // The optimizer test already verified that the IR structure is optimized.
    // This test verifies that the optimized IR can be executed.
    let source = "2 * (3 + 4)";
    let result = compile_and_run_phi_ir(source);

    match result {
        Ok(PhiIRValue::Number(n)) => {
            assert!(
                (n - 14.0).abs() < f64::EPSILON,
                "Expected 14.0, got {:?}",
                n
            );
        }
        Ok(val) => panic!("Expected Number, got {:?}", val),
        Err(e) => panic!("Compilation failed: {}", e),
    }
}
