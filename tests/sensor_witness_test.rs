use phiflow::parser::parse_phi_program;
use phiflow::phi_ir::emitter;
use phiflow::phi_ir::evaluator::Evaluator;
use phiflow::phi_ir::lowering::{lower_program_checked, LoweringError};
use phiflow::phi_ir::optimizer::{OptimizationLevel, Optimizer};
use phiflow::phi_ir::vm::PhiVm;
use phiflow::phi_ir::{PhiIRValue, SensorKind};
use phiflow::wasm_host::{run_source_with_host, WasmHostHooks};

fn sensor_provider(sensor: SensorKind) -> Option<f64> {
    match sensor {
        SensorKind::CpuUsage => Some(12.5),
        SensorKind::CpuTemp => Some(55.0),
        SensorKind::MemoryUsage => Some(62.0),
        _ => Some(0.5), // Dummy value for the SOMA sensors added recently
    }
}

fn compile_program(source: &str) -> phiflow::phi_ir::PhiIRProgram {
    let exprs = parse_phi_program(source).expect("parse failed");
    let mut program = lower_program_checked(&exprs).expect("lowering failed");
    let mut optimizer = Optimizer::new(OptimizationLevel::Basic);
    optimizer.optimize(&mut program);
    program
}

#[test]
fn test_sensor_witness_conforms_across_backends() {
    let cases = [
        (r#"witness sensor("cpu_usage")"#, PhiIRValue::Number(12.5)),
        (r#"witness sensor("cpu_temp")"#, PhiIRValue::Number(55.0)),
        (
            r#"witness sensor("memory_usage")"#,
            PhiIRValue::Number(62.0),
        ),
    ];

    for (source, expected) in cases {
        let program = compile_program(source);

        let mut evaluator = Evaluator::new(program.clone()).with_sensor_provider(sensor_provider);
        let eval_result = evaluator.run().expect("evaluator failed");

        let bytes = emitter::emit(&program);
        let vm_result =
            PhiVm::run_bytes_with_sensor_provider(&bytes, sensor_provider).expect("vm failed");

        let wasm_result = run_source_with_host(
            source,
            WasmHostHooks::new().with_sensor_provider(sensor_provider),
        )
        .expect("wasm host failed")
        .result;

        assert_eq!(eval_result, expected);
        assert_eq!(vm_result, expected);
        assert_eq!(wasm_result, expected);
    }
}

#[test]
fn test_sensor_witness_rejects_unknown_sensor() {
    let source = r#"witness sensor("fan_speed")"#;
    let exprs = parse_phi_program(source).expect("parse failed");
    let error = lower_program_checked(&exprs).expect_err("lowering should fail");
    assert_eq!(error, LoweringError::UnknownSensor("fan_speed".to_string()));
}
