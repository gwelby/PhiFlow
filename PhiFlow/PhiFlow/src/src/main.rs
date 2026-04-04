mod phi_core;
mod visualization;
mod parser;
mod interpreter;
use visualization::{Visualizer, save_svg};
use parser::{parse_phi_program};
use interpreter::PhiInterpreter;
use phi_core::{golden_spiral_points, flower_of_life_points, fibonacci_spiral_pattern, dna_helix_points, validate_pattern_consciousness, validate_with_recommendations, CREATION_FREQUENCY};

fn main() {
    println!(" PhiFlow - Consciousness Mathematics Language");
    println!("=========================================");

    // Test enhanced .phi language parsing
    test_enhanced_phi_parsing();

    // Test pattern validation
    test_pattern_validation();

    // Test visualization with validation feedback
    test_validated_visualization();

    // Test .phi interpreter
    test_phi_interpreter();

    // Test List and Function features
    test_list_and_function_features();

    // Test new binary and unary operators
    test_new_operators();

    println!("\nPhiFlow Ground State initialized successfully.");
}

fn test_list_and_function_features() {
    println!("\n Testing List and Function Features:");

    // Test 1: List literal creation
    let program_list_literal = r#"
        let my_numbers = [1.0, 2.0, 3.0]
        let my_strings = ["hello", "world"]
        let mixed_list = [1.0, "test", true]
    "#;
    println!("\nExecuting Program (List Literals):\n```phi\n{}\n```", program_list_literal);
    match parse_phi_program(program_list_literal) {
        Ok(expressions) => {
            let mut interpreter = PhiInterpreter::new();
            match interpreter.execute(expressions) {
                Ok(_) => println!("✅ Program (List Literals) executed."),
                Err(e) => println!("❌ Interpreter error (List Literals): {}", e),
            }
        }
        Err(e) => println!("❌ Parsing error (List Literals): {}", e),
    }

    // Test 2: Function definition and call with parameters and return value
    let program_function_add = r#"
        function add(a: Number, b: Number) -> Number {
            return a + b
        }
        let result = add(10.0, 20.0)
    "#;
    println!("\nExecuting Program (Function Add):\n```phi\n{}\n```", program_function_add);
    match parse_phi_program(program_function_add) {
        Ok(expressions) => {
            let mut interpreter = PhiInterpreter::new();
            match interpreter.execute(expressions) {
                Ok(result) => println!("✅ Program (Function Add) executed. Result: {:?}", result),
                Err(e) => println!("❌ Interpreter error (Function Add): {}", e),
            }
        }
        Err(e) => println!("❌ Parsing error (Function Add): {}", e),
    }

    // Test 3: Function returning a list
    let program_function_return_list = r#"
        function create_list() -> List<Number> {
            return [10.0, 20.0, 30.0]
        }
        let my_new_list = create_list()
    "#;
    println!("\nExecuting Program (Function Return List):\n```phi\n{}\n```", program_function_return_list);
    match parse_phi_program(program_function_return_list) {
        Ok(expressions) => {
            let mut interpreter = PhiInterpreter::new();
            match interpreter.execute(expressions) {
                Ok(result) => println!("✅ Program (Function Return List) executed. Result: {:?}", result),
                Err(e) => println!("❌ Interpreter error (Function Return List): {}", e),
            }
        }
        Err(e) => println!("❌ Parsing error (Function Return List): {}", e),
    }

    // Test 4: Function taking a list as argument and returning a modified list
    let program_function_process_list = r#"
        function process_list(input_list: List<Number>) -> List<Number> {
            let first_element = input_list[0]
            return [first_element + 1.0, input_list[1], input_list[2]]
        }
        let original_list = [1.0, 2.0, 3.0]
        let processed_list = process_list(original_list)
    "#;
    println!("\nExecuting Program (Function Process List):\n```phi\n{}\n```", program_function_process_list);
    match parse_phi_program(program_function_process_list) {
        Ok(expressions) => {
            let mut interpreter = PhiInterpreter::new();
            match interpreter.execute(expressions) {
                Ok(result) => println!("✅ Program (Function Process List) executed. Result: {:?}", result),
                Err(e) => println!("❌ Interpreter error (Function Process List): {}", e),
            }
        }
        Err(e) => println!("❌ Parsing error (Function Process List): {}", e),
    }
}

fn test_new_operators() {
    println!("\n Testing New Binary and Unary Operators:");

    // Test 1: Modulo and Power operators
    let program_modulo_power = r#"
        let x = 10.0 % 3.0
        let y = 2.0 ** 3.0
        let z = 3.0 ** 2.0
    "#;
    println!("\nExecuting Program (Modulo and Power):\n```phi\n{}```", program_modulo_power);
    match parse_phi_program(program_modulo_power) {
        Ok(expressions) => {
            let mut interpreter = PhiInterpreter::new();
            match interpreter.execute(expressions) {
                Ok(result) => println!("✅ Program (Modulo and Power) executed. Result: {:?}", result),
                Err(e) => println!("❌ Interpreter error (Modulo and Power): {}", e),
            }
        }
        Err(e) => println!("❌ Parsing error (Modulo and Power): {}", e),
    }

    // Test 2: Comparison operators
    let program_comparisons = r#"
        let a = 5.0 > 3.0
        let b = 2.0 <= 2.0
        let c = 10.0 < 5.0
        let d = 7.0 >= 7.0
    "#;
    println!("\nExecuting Program (Comparisons):\n```phi\n{}```", program_comparisons);
    match parse_phi_program(program_comparisons) {
        Ok(expressions) => {
            let mut interpreter = PhiInterpreter::new();
            match interpreter.execute(expressions) {
                Ok(result) => println!("✅ Program (Comparisons) executed. Result: {:?}", result),
                Err(e) => println!("❌ Interpreter error (Comparisons): {}", e),
            }
        }
        Err(e) => println!("❌ Parsing error (Comparisons): {}", e),
    }

    // Test 3: Equality and inequality
    let program_equality = r#"
        let x = 5.0 == 5.0
        let y = "hello" != "world"
        let z = true == false
    "#;
    println!("\nExecuting Program (Equality):\n```phi\n{}```", program_equality);
    match parse_phi_program(program_equality) {
        Ok(expressions) => {
            let mut interpreter = PhiInterpreter::new();
            match interpreter.execute(expressions) {
                Ok(result) => println!("✅ Program (Equality) executed. Result: {:?}", result),
                Err(e) => println!("❌ Interpreter error (Equality): {}", e),
            }
        }
        Err(e) => println!("❌ Parsing error (Equality): {}", e),
    }

    // Test 4: Logical operators
    let program_logical = r#"
        let a = true && false
        let b = true || false
        let c = false || false
    "#;
    println!("\nExecuting Program (Logical Operators):\n```phi\n{}```", program_logical);
    match parse_phi_program(program_logical) {
        Ok(expressions) => {
            let mut interpreter = PhiInterpreter::new();
            match interpreter.execute(expressions) {
                Ok(result) => println!("✅ Program (Logical Operators) executed. Result: {:?}", result),
                Err(e) => println!("❌ Interpreter error (Logical Operators): {}", e),
            }
        }
        Err(e) => println!("❌ Parsing error (Logical Operators): {}", e),
    }

    // Test 5: Unary operators
    let program_unary = r#"
        let x = -5.0
        let y = !true
        let z = !false
    "#;
    println!("\nExecuting Program (Unary Operators):\n```phi\n{}```", program_unary);
    match parse_phi_program(program_unary) {
        Ok(expressions) => {
            let mut interpreter = PhiInterpreter::new();
            match interpreter.execute(expressions) {
                Ok(result) => println!("✅ Program (Unary Operators) executed. Result: {:?}", result),
                Err(e) => println!("❌ Interpreter error (Unary Operators): {}", e),
            }
        }
        Err(e) => println!("❌ Parsing error (Unary Operators): {}", e),
    }

    // Test 6: Complex expression with multiple operators
    let program_complex = r#"
        let result = (5.0 + 3.0) * 2.0 ** 2.0 > 30.0 && !false
    "#;
    println!("\nExecuting Program (Complex Expression):\n```phi\n{}```", program_complex);
    match parse_phi_program(program_complex) {
        Ok(expressions) => {
            let mut interpreter = PhiInterpreter::new();
            match interpreter.execute(expressions) {
                Ok(result) => println!("✅ Program (Complex Expression) executed. Result: {:?}", result),
                Err(e) => println!("❌ Interpreter error (Complex Expression): {}", e),
            }
        }
        Err(e) => println!("❌ Parsing error (Complex Expression): {}", e),
    }
}

fn test_enhanced_phi_parsing() {
    println!("\n Testing Enhanced .phi Language Parser:");

    let phi_programs = vec![
        // Basic pattern creation
        r#"create spiral at 528Hz with { rotations: 5.0, scale: 100.0 }"#,

        // Function definition
        r#"function golden_ratio() -> Number { 1.618 }"#,

        // Pattern validation
        r#"validate spiral with [coherence, phi_resonance, universal_alignment]"#,

        // Variable assignment
        r#"let my_number = 10.0"#,
        r#"let my_pattern = create flower at 432Hz with { rings: 3 }"#,

        // Complex program
        r#"
        let frequency = 528.0
        let pattern = create dna at frequency with { turns: 10.0, radius: 25.0 }
        validate pattern with [coherence, consciousness_zone]
        "#,

        // Control flow (will fail until implemented)
        r#"
        if true {
            let x = 1
        } else {
            let y = 0
        }
        "#,
    ];

    for (i, program) in phi_programs.iter().enumerate() {
        println!("\n Program {}:", i + 1);
        println!("```phi\n{}\n```", program);

        match parse_phi_program(program) {
            Ok(expressions) => {
                println!("✅ Parsed {} expressions:", expressions.len());
                for (j, expr) in expressions.iter().enumerate() {
                    println!("   {}: {:?}", j + 1, expr);
                }
            }
            Err(e) => {
                println!("❌ Parse error: {}", e);
            }
        }
    }
}

fn test_pattern_validation() {
    println!("\n Testing Pattern Validation:");

    let viz_temp = Visualizer::new(1.0, 1.0); // Temporary visualizer for 3D projection

    // Test different patterns
    let patterns = vec![
        ("Golden Spiral", golden_spiral_points(5.0, 100, 50.0)),
        ("Flower of Life", flower_of_life_points(3)),
        ("Fibonacci Spiral", fibonacci_spiral_pattern(5.0, 89, 30.0)),
        ("DNA Helix", viz_temp.project_3d_to_2d(&dna_helix_points(2.0, 20.0, 25.0).0)), // Project 3D to 2D
    ];

    for (name, pattern) in patterns {
        println!("\n Validating {}", name);
        let (result, recommendations) = validate_with_recommendations(&pattern);

        println!("    Validation Results:");
        println!("      Coherence: {:.3}", result.coherence);
        println!("      Consciousness Zone: {}", result.consciousness_zone);
        println!("      Phi Resonance: {:.3}", result.phi_resonance);
        println!("      Universal Alignment: {}", result.universal_constant_alignment);
        println!("      Frequency Match: {:.3}", result.frequency_match);
        println!("      Overall Score: {:.3}", result.validation_score);
        println!("      Classification: {}", result.pattern_classification);

        println!("    Recommendations:");
        for rec in recommendations {
            println!("      • {}", rec);
        }
    }
}

fn test_validated_visualization() {
    println!("\n Testing Validated Visualization:");

    let spiral = golden_spiral_points(5.0, 200, 80.0);
    let validation = validate_pattern_consciousness(&spiral);

    let viz = Visualizer::new(800.0, 800.0);
    let svg = viz.pattern_to_svg(&spiral, CREATION_FREQUENCY);

    match save_svg(&svg, "validated_golden_spiral.svg") {
        Ok(_) => {
            println!("✅ Golden spiral visualization saved");
            println!("    Validation Score: {:.1}", validation.validation_score * 100.0); // Format as percentage
            println!("    Consciousness Zone: {}", validation.consciousness_zone);
        }
        Err(e) => println!("❌ Failed to save: {}", e),
    }
}

fn test_phi_interpreter() {
    println!("\n Testing .phi Interpreter:");

    // Test 1: Variable in parameters (working)
    let program_str1 = r#"
        let my_number = 10.0
        create spiral at 528Hz with { rotations: my_number, scale: 100.0 }
    "#;

    println!("Executing Program 1 (variable in parameters):\n```phi\n{}\n```", program_str1);

    match parse_phi_program(program_str1) {
        Ok(expressions) => {
            let mut interpreter = PhiInterpreter::new();
            match interpreter.execute(expressions) {
                Ok(result) => println!("✅ Program 1 executed. Variable in parameters working."),
                Err(e) => println!("❌ Interpreter error: {}", e),
            }
        }
        Err(e) => println!("❌ Parsing error for interpreter test: {}", e),
    }

    // Test 2: Variable frequency resolution (new feature)
    let program_str2 = r#"
        let frequency = 528.0
        create dna at frequency with { turns: 10.0, radius: 25.0 }
    "#;

    println!("\nExecuting Program 2 (frequency variable resolution):\n```phi\n{}\n```", program_str2);

    match parse_phi_program(program_str2) {
        Ok(expressions) => {
            let mut interpreter = PhiInterpreter::new();
            match interpreter.execute(expressions) {
                Ok(result) => println!("✅ Program 2 executed. Frequency variable resolution working!"),
                Err(e) => println!("❌ Interpreter error: {}", e),
            }
        }
        Err(e) => println!("❌ Parsing error for interpreter test: {}", e),
    }
}