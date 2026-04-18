use phiflow::parser::{BinaryOperator, PhiExpression};
use phiflow::phi_ir::lowering::lower_program;
use phiflow::phi_ir::optimizer::{OptimizationLevel, Optimizer};
use phiflow::phi_ir::printer::PhiIRPrinter;

#[test]
fn test_phi_harmonic_loop_unrolling() {
    // Defines a simple loop:
    // while x > 0 {
    //    x = x - 1;
    // }
    let exprs = vec![
        PhiExpression::LetBinding {
            name: "x".to_string(),
            value: Box::new(PhiExpression::Number(10.0)),
            phi_type: None,
        },
        PhiExpression::WhileLoop {
            condition: Box::new(PhiExpression::BinaryOp {
                left: Box::new(PhiExpression::Variable("x".to_string())),
                operator: BinaryOperator::Greater,
                right: Box::new(PhiExpression::Number(0.0)),
            }),
            body: Box::new(PhiExpression::LetBinding {
                name: "x".to_string(),
                value: Box::new(PhiExpression::BinaryOp {
                    left: Box::new(PhiExpression::Variable("x".to_string())),
                    operator: BinaryOperator::Subtract,
                    right: Box::new(PhiExpression::Number(1.0)),
                }),
                phi_type: None,
            }),
        },
    ];

    let mut program = lower_program(&exprs);

    println!("--- Before Optimization ---");
    println!("{}", PhiIRPrinter::print(&program));

    // Initial state:
    // Block 0: Entry (x = 10), Jump(Header)
    // Block 1: Header (x > 0 ?), Branch(Body, Exit)
    // Block 2: Body (x = x - 1), Jump(Header)
    // Block 3: Exit
    // Total blocks: 4

    let initial_blocks = program.blocks.len();
    assert_eq!(initial_blocks, 4, "Expected 4 blocks initially");

    // Enable Phi-Harmonic Optimization
    let mut optimizer = Optimizer::new(OptimizationLevel::PhiHarmonic);
    optimizer.optimize(&mut program);

    println!("--- After Phi-Harmonic Optimization ---");
    println!("{}", PhiIRPrinter::print(&program));

    // If unrolled by factor 3 (Fibonacci):
    // We expect copies of Body and Header/Check.
    // Structure:
    // Header -> Body1 -> Check1 -> Body2 -> Check2 -> Body3 -> Header
    // New blocks: Check1, Body2, Check2, Body3 (approx +4 blocks)
    // Total blocks should be > 4.

    // Check Coherence Score
    let score = optimizer.monitor.coherence_score;
    println!("Coherence Score: {}", score);
    assert!(score > 0.0, "Coherence score should be positive");
    assert!(
        score < 1.0,
        "Coherence score likely less than perfect 1.0 for this simple program"
    );

    let final_blocks = program.blocks.len();
    assert!(
        final_blocks > initial_blocks,
        "Phi-Harmonic unrolling should increase block count (creating resonance chains)"
    );

    // Ideally, we check for the specific Fibonacci number (Sequence length 3 or 5)
}

#[test]
fn test_phi_harmonic_stabilization() {
    // Defines a loop with low coherence (high control flow, low arithmetic).
    // while 1 > 0 {
    //    x = x; // No-op essentially, but keeps loop valid
    // }
    // Ratio ~ 0.5 (1 Arith / 2 Control).
    let exprs = vec![
        PhiExpression::LetBinding {
            name: "x".to_string(),
            value: Box::new(PhiExpression::Number(1.0)),
            phi_type: None,
        },
        PhiExpression::WhileLoop {
            condition: Box::new(PhiExpression::BinaryOp {
                left: Box::new(PhiExpression::Number(1.0)),
                operator: BinaryOperator::Greater,
                right: Box::new(PhiExpression::Number(0.0)),
            }),
            body: Box::new(PhiExpression::LetBinding {
                name: "x".to_string(),
                value: Box::new(PhiExpression::Variable("x".to_string())),
                phi_type: None,
            }),
        },
    ];

    let mut program = lower_program(&exprs);
    let mut optimizer = Optimizer::new(OptimizationLevel::PhiHarmonic);

    optimizer.optimize(&mut program);

    // Check Coherence Score
    let score = optimizer.monitor.coherence_score;
    println!("Chaotic Coherence Score: {}", score);
    assert!(
        score < 0.618,
        "Score should be low enough to trigger stabilization"
    );

    // Verify Sleep injection
    let has_sleep = program.blocks.iter().any(|b| {
        b.instructions
            .iter()
            .any(|i| matches!(i.node, phiflow::phi_ir::PhiIRNode::Sleep { .. }))
    });

    assert!(
        has_sleep,
        "Stabilizer should have injected Sleep instruction"
    );
}
