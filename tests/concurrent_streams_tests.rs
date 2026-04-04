use phiflow::host::DefaultHostProvider;
use phiflow::parser::{PhiLexer, PhiParser};
use phiflow::phi_ir::evaluator::Evaluator;
use phiflow::phi_ir::lowering::lower_program;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[tokio::test]
async fn test_concurrent_streams_shared_resonance() {
    let source1 = r#"
        intention global_peace {
            param group = 1
            resonate group
        }
    "#;

    let source2 = r#"
        intention global_peace {
            param group = 2
            resonate group
        }
    "#;

    // Parse and lower both programs
    let program1 = lower_program(
        &PhiParser::new(PhiLexer::new(source1).tokenize().unwrap())
            .parse()
            .unwrap(),
    );
    let program2 = lower_program(
        &PhiParser::new(PhiLexer::new(source2).tokenize().unwrap())
            .parse()
            .unwrap(),
    );

    // Create a shared resonance field
    let shared_resonance = Arc::new(Mutex::new(HashMap::new()));

    // Clone the arc for each evaluator
    let shared1 = Arc::clone(&shared_resonance);
    let shared2 = Arc::clone(&shared_resonance);

    // Spawn task 1
    let handle1 = tokio::task::spawn_blocking(move || {
        let mut eval = Evaluator::new(&program1)
            .with_host(Box::new(DefaultHostProvider))
            .with_shared_resonance(shared1);
        eval.run()
    });

    // Spawn task 2
    let handle2 = tokio::task::spawn_blocking(move || {
        let mut eval = Evaluator::new(&program2)
            .with_host(Box::new(DefaultHostProvider))
            .with_shared_resonance(shared2);
        eval.run()
    });

    // Wait for both to complete
    let _ = tokio::try_join!(handle1, handle2).unwrap();

    // Verify shared resonance field
    let guard = shared_resonance.lock().unwrap();
    let values = guard
        .get("global_peace")
        .expect("Should have global_peace intention");

    assert_eq!(values.len(), 2, "Should have 2 values resonated");

    // Convert to float to check simply
    let mut num_values: Vec<f64> = values
        .iter()
        .map(|v| match v {
            phiflow::phi_ir::PhiIRValue::Number(n) => *n,
            _ => panic!("Expected number"),
        })
        .collect();

    num_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(num_values, vec![1.0, 2.0]);
}
