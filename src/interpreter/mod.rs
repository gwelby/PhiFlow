use crate::parser::{BinaryOperator, PhiExpression, PhiType, PhiValue, UnaryOperator};
use crate::phi_core::AudioSynthesizer;
use crate::phi_core::{
    dna_helix_points, flower_of_life_points, golden_spiral_points, validate_pattern_consciousness,
    PatternAnalyzer,
};
use crate::visualization::Visualizer;
use std::collections::HashMap;

pub struct PhiInterpreter {
    // Environment for variables
    environment: HashMap<String, PhiValue>,
    // Environment for functions
    functions: HashMap<String, PhiExpression>,
    // Instances of core components
    pattern_analyzer: PatternAnalyzer,
    audio_synthesizer: AudioSynthesizer,
    visualizer: Visualizer,

    // === LIVE COHERENCE TRACKING ===
    // The program measures its own alignment as it runs
    coherence: f64,               // current program coherence (0.0 - 1.0)
    intention_stack: Vec<String>, // nested intentions
    frequencies_used: Vec<f64>,   // all frequencies encountered
    operations_log: Vec<String>,  // what the program has done
    witness_count: u32,           // how many times the program has witnessed
    contradictions: Vec<String>,  // self-contradictions detected

    // === RESONANCE FIELD ===
    // Shared state between intention blocks - code talking to itself
    resonance_field: HashMap<String, Vec<PhiValue>>, // intention_name -> shared values
    resonance_log: Vec<(String, String)>,            // (from, to) resonance events
}

impl PhiInterpreter {
    pub fn new() -> Self {
        let mut interpreter = PhiInterpreter {
            environment: HashMap::new(),
            functions: HashMap::new(),
            pattern_analyzer: PatternAnalyzer::new(),
            audio_synthesizer: AudioSynthesizer::new(44100),
            visualizer: Visualizer::new(800.0, 800.0),
            coherence: 1.0, // starts perfect - every action either maintains or degrades it
            intention_stack: Vec::new(),
            frequencies_used: Vec::new(),
            operations_log: Vec::new(),
            witness_count: 0,
            contradictions: Vec::new(),
            resonance_field: HashMap::new(),
            resonance_log: Vec::new(),
        };

        // Add built-in functions
        interpreter.functions.insert(
            "print".to_string(),
            PhiExpression::FunctionDef {
                name: "print".to_string(),
                parameters: vec![("value".to_string(), None)],
                return_type: PhiType::Void,
                body: Box::new(PhiExpression::Block(vec![])),
            },
        );

        interpreter
    }

    pub fn execute(&mut self, program: Vec<PhiExpression>) -> Result<PhiValue, String> {
        let mut last_value = PhiValue::Void;

        for expression in program {
            last_value = self.evaluate_expression(&expression)?;
        }

        // Program self-report: how did this code live?
        self.print_program_summary();

        Ok(last_value)
    }

    /// The program reports on its own life after execution completes.
    fn print_program_summary(&self) {
        let final_coherence = self.calculate_coherence();

        println!();
        println!("  \u{2550}\u{2550}\u{2550} PHIFLOW PROGRAM SUMMARY \u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}");

        // Final coherence with bar
        let bar_len = 20;
        let filled = (final_coherence * bar_len as f64) as usize;
        let empty = bar_len - filled;
        let bar: String = "\u{2588}".repeat(filled) + &"\u{2591}".repeat(empty);
        let alignment = if final_coherence >= 0.8 {
            "ALIGNED"
        } else if final_coherence >= 0.5 {
            "DRIFTING"
        } else {
            "MISALIGNED"
        };
        println!(
            "  Coherence: {:.3} [{}] {}",
            final_coherence, bar, alignment
        );

        // Frequency summary
        if !self.frequencies_used.is_empty() {
            let mut unique_freqs: Vec<f64> = self.frequencies_used.clone();
            unique_freqs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            unique_freqs.dedup();
            let freqs: Vec<String> = unique_freqs.iter().map(|f| format!("{:.0}Hz", f)).collect();
            println!("  Frequencies: {}", freqs.join(" \u{2192} "));
        }

        // Witness count
        if self.witness_count > 0 {
            println!("  Self-observations: {}", self.witness_count);
        } else {
            println!("  Self-observations: 0 (this program never witnessed itself)");
        }

        // Contradictions
        if !self.contradictions.is_empty() {
            println!("  Contradictions: {}", self.contradictions.len());
            for c in &self.contradictions {
                println!("    \u{26A0} {}", c);
            }
        }

        // Resonance connections
        if !self.resonance_field.is_empty() {
            let sources: Vec<String> = self.resonance_field.keys().cloned().collect();
            let total: usize = self.resonance_field.values().map(|v| v.len()).sum();
            println!(
                "  Resonance: {} value(s) across {} intention(s)",
                total,
                sources.len()
            );
            if !self.resonance_log.is_empty() {
                for (from, to) in &self.resonance_log {
                    println!("    \"{}\" \u{2192} \"{}\"", from, to);
                }
            }
        }

        // Operations
        println!("  Operations: {}", self.operations_log.len());

        println!("  \u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}");
    }

    fn evaluate_expression(&mut self, expression: &PhiExpression) -> Result<PhiValue, String> {
        match expression {
            PhiExpression::Number(n) => Ok(PhiValue::Number(*n)),
            PhiExpression::String(s) => Ok(PhiValue::String(s.clone())),
            PhiExpression::Boolean(b) => Ok(PhiValue::Boolean(*b)),
            PhiExpression::Variable(name) => {
                // In legacy interpreter mode, `coherence` acts as a live keyword value.
                if name == "coherence" {
                    return Ok(PhiValue::Number(self.calculate_coherence()));
                }

                self.environment
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("Undefined variable: {}", name))
            }
            PhiExpression::LetBinding {
                name,
                value,
                phi_type: _,
            } => {
                let evaluated_value = self.evaluate_expression(value)?;
                self.check_contradiction(name, &evaluated_value);
                self.environment
                    .insert(name.clone(), evaluated_value.clone());
                Ok(evaluated_value)
            }
            PhiExpression::CreatePattern {
                pattern_type,
                frequency,
                parameters,
            } => {
                // First evaluate parameters to get access to potential frequency variable
                let mut pattern_params = HashMap::new();
                for (key, val_expr) in parameters {
                    // Don't evaluate __frequency_var yet - keep it as string reference for later resolution
                    if key == "__frequency_var" {
                        pattern_params.insert(key.clone(), val_expr.clone());
                    } else {
                        pattern_params.insert(key.clone(), self.evaluate_phi_value(val_expr)?);
                    }
                }

                // Resolve frequency - check if it's a variable reference
                let freq = if *frequency < 0.0 {
                    // Frequency is a variable reference, resolve it
                    if let Some(var_ref_value) = pattern_params.get("__frequency_var") {
                        if let Some(resolved_freq) =
                            self.evaluate_phi_value(var_ref_value)?.as_number()
                        {
                            resolved_freq
                        } else {
                            return Err(
                                "Frequency variable did not resolve to a number".to_string()
                            );
                        }
                    } else {
                        return Err("Frequency variable reference not found".to_string());
                    }
                } else {
                    *frequency
                };

                // Track frequency for coherence measurement
                if freq > 0.0 {
                    self.frequencies_used.push(freq);
                    self.log_operation("create");
                }

                let pattern_data = match pattern_type.as_str() {
                    "spiral" => {
                        let rotations = pattern_params
                            .get("rotations")
                            .and_then(|v| v.as_number())
                            .unwrap_or(5.0);
                        let scale = pattern_params
                            .get("scale")
                            .and_then(|v| v.as_number())
                            .unwrap_or(100.0);
                        golden_spiral_points(rotations, 100, scale)
                    }
                    "flower" => {
                        let rings = pattern_params
                            .get("rings")
                            .and_then(|v| v.as_number())
                            .unwrap_or(3.0) as i64;
                        flower_of_life_points(rings)
                    }
                    "dna" => {
                        let turns = pattern_params
                            .get("turns")
                            .and_then(|v| v.as_number())
                            .unwrap_or(10.0);
                        let radius = pattern_params
                            .get("radius")
                            .and_then(|v| v.as_number())
                            .unwrap_or(25.0);
                        // Project 3D DNA to 2D for now, as PhiValue::Pattern2D expects 2D points
                        self.visualizer
                            .project_3d_to_2d(&dna_helix_points(10.0, turns, radius).0)
                    }
                    _ => return Err(format!("Unknown pattern type: {}", pattern_type)),
                };
                Ok(PhiValue::Pattern2D(pattern_data))
            }
            PhiExpression::ConsciousnessValidation { pattern, metrics } => {
                let evaluated_pattern = self.evaluate_expression(pattern)?;
                if let PhiValue::Pattern2D(p) = evaluated_pattern {
                    let validation_result = validate_pattern_consciousness(&p);
                    // Filter metrics if specified
                    let filtered_metrics: HashMap<String, PhiValue> = metrics
                        .iter()
                        .filter_map(|m| match m.as_str() {
                            "coherence" => {
                                Some((m.clone(), PhiValue::Number(validation_result.coherence)))
                            }
                            "consciousness_zone" => Some((
                                m.clone(),
                                PhiValue::String(validation_result.consciousness_zone.to_string()),
                            )),
                            "phi_resonance" => {
                                Some((m.clone(), PhiValue::Number(validation_result.phi_resonance)))
                            }
                            "universal_alignment" => Some((
                                m.clone(),
                                PhiValue::Boolean(validation_result.universal_constant_alignment),
                            )),
                            "frequency_match" => Some((
                                m.clone(),
                                PhiValue::Number(validation_result.frequency_match),
                            )),
                            "overall_score" => Some((
                                m.clone(),
                                PhiValue::Number(validation_result.validation_score),
                            )),
                            "classification" => Some((
                                m.clone(),
                                PhiValue::String(
                                    validation_result.pattern_classification.to_string(),
                                ),
                            )),
                            _ => None,
                        })
                        .collect();
                    // Print validation metrics cleanly
                    for (key, value) in &filtered_metrics {
                        match value {
                            PhiValue::Number(n) => println!("    {}: {:.3}", key, n),
                            PhiValue::Boolean(b) => println!("    {}: {}", key, b),
                            PhiValue::String(s) => println!("    {}: {}", key, s),
                            _ => println!("    {}: {:?}", key, value),
                        }
                    }
                    Ok(PhiValue::ValidationResult(validation_result))
                } else {
                    Err("Validation requires a Pattern2D value".to_string())
                }
            }
            PhiExpression::FunctionDef {
                name,
                parameters,
                return_type,
                body,
            } => {
                self.functions.insert(name.clone(), expression.clone());
                Ok(PhiValue::Void)
            }
            PhiExpression::FunctionCall { name, arguments } => {
                let function_def = self
                    .functions
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("Undefined function: {}", name))?;

                let evaluated_args: Result<Vec<PhiValue>, String> = arguments
                    .iter()
                    .map(|arg_expr| self.evaluate_expression(arg_expr))
                    .collect();
                let evaluated_args = evaluated_args?;

                // Handle built-in functions
                if name == "print" {
                    for arg in &evaluated_args {
                        println!("{:?}", arg);
                    }
                    return Ok(PhiValue::Void);
                }

                // Check argument count
                let (params, body) = if let PhiExpression::FunctionDef {
                    parameters, body, ..
                } = function_def
                {
                    (
                        parameters
                            .iter()
                            .map(|(name, _type)| name.clone())
                            .collect::<Vec<String>>(),
                        body,
                    )
                } else {
                    return Err(format!(
                        "Expected function definition for {}, found {:?}",
                        name, function_def
                    ));
                };

                // Evaluate arguments
                let evaluated_args: Result<Vec<PhiValue>, String> = arguments
                    .iter()
                    .map(|arg_expr| self.evaluate_expression(arg_expr))
                    .collect();
                let evaluated_args = evaluated_args?;

                // Check argument count
                if params.len() != evaluated_args.len() {
                    return Err(format!(
                        "Function {} expected {} arguments, but received {}",
                        name,
                        params.len(),
                        evaluated_args.len()
                    ));
                }

                // Save current environment and create a new scope for function execution
                let original_environment = self.environment.clone();
                self.environment.clear(); // Clear for new scope

                // Bind arguments to parameters in the new scope
                for (i, param_name) in params.iter().enumerate() {
                    self.environment
                        .insert(param_name.clone(), evaluated_args[i].clone());
                }

                // Execute function body
                let mut result = PhiValue::Void;
                if let PhiExpression::Block(expressions) = body.as_ref() {
                    for expr in expressions {
                        result = self.evaluate_expression(expr)?;
                        if let PhiValue::Return(returned_value) = result {
                            result = *returned_value;
                            break; // Exit loop on return
                        }
                    }
                } else {
                    // Handle single expression body
                    result = self.evaluate_expression(&body)?;
                    if let PhiValue::Return(returned_value) = result {
                        result = *returned_value;
                    }
                }

                // Restore original environment
                self.environment = original_environment;

                Ok(result)
            }
            PhiExpression::PatternTransform { .. } => {
                Err("Pattern transformations not yet executable".to_string())
            }
            PhiExpression::PatternCombine { .. } => {
                Err("Pattern combinations not yet executable".to_string())
            }
            PhiExpression::ConsciousnessMonitor { .. } => {
                Err("Consciousness monitoring not yet executable".to_string())
            }
            PhiExpression::AudioSynthesis { .. } => {
                Err("Audio synthesis not yet executable".to_string())
            }
            PhiExpression::Block(expressions) => {
                let mut last_val = PhiValue::Void;
                for expr in expressions {
                    last_val = self.evaluate_expression(expr)?;
                }
                Ok(last_val)
            }
            PhiExpression::IfElse {
                condition,
                then_branch,
                else_branch,
            } => {
                let condition_val = self.evaluate_expression(condition)?;
                if self.is_truthy(&condition_val) {
                    self.evaluate_expression(then_branch)
                } else if let Some(else_branch) = else_branch {
                    self.evaluate_expression(else_branch)
                } else {
                    Ok(PhiValue::Void)
                }
            }
            PhiExpression::ForLoop {
                variable,
                iterable,
                body,
            } => {
                let evaluated_iterable = self.evaluate_expression(iterable)?;
                if let PhiValue::List(elements) = evaluated_iterable {
                    for element in elements {
                        self.environment.insert(variable.clone(), element);
                        self.evaluate_expression(body)?;
                    }
                } else {
                    return Err(format!(
                        "For loop iterable must be a list, found {:?}",
                        evaluated_iterable
                    ));
                }
                Ok(PhiValue::Void)
            }
            PhiExpression::WhileLoop { condition, body } => {
                while {
                    let condition_val = self.evaluate_expression(condition)?;
                    self.is_truthy(&condition_val)
                } {
                    self.evaluate_expression(body)?;
                }
                Ok(PhiValue::Void)
            }
            PhiExpression::Return(value) => {
                let returned_value = self.evaluate_expression(value)?;
                Ok(PhiValue::Return(Box::new(returned_value)))
            }
            PhiExpression::List(elements) => {
                let evaluated_elements: Result<Vec<PhiValue>, String> = elements
                    .iter()
                    .map(|elem| self.evaluate_expression(elem))
                    .collect();
                Ok(PhiValue::List(evaluated_elements?))
            }
            PhiExpression::BinaryOp {
                left,
                operator,
                right,
            } => {
                let left_val = self.evaluate_expression(left)?;
                let right_val = self.evaluate_expression(right)?;

                match operator {
                    BinaryOperator::Add => {
                        if let (Some(l), Some(r)) = (left_val.as_number(), right_val.as_number()) {
                            Ok(PhiValue::Number(l + r))
                        } else {
                            Err("Addition requires numeric operands".to_string())
                        }
                    }
                    BinaryOperator::Subtract => {
                        if let (Some(l), Some(r)) = (left_val.as_number(), right_val.as_number()) {
                            Ok(PhiValue::Number(l - r))
                        } else {
                            Err("Subtraction requires numeric operands".to_string())
                        }
                    }
                    BinaryOperator::Multiply => {
                        if let (Some(l), Some(r)) = (left_val.as_number(), right_val.as_number()) {
                            Ok(PhiValue::Number(l * r))
                        } else {
                            Err("Multiplication requires numeric operands".to_string())
                        }
                    }
                    BinaryOperator::Divide => {
                        if let (Some(l), Some(r)) = (left_val.as_number(), right_val.as_number()) {
                            if r != 0.0 {
                                Ok(PhiValue::Number(l / r))
                            } else {
                                Err("Division by zero".to_string())
                            }
                        } else {
                            Err("Division requires numeric operands".to_string())
                        }
                    }
                    BinaryOperator::Modulo => {
                        if let (Some(l), Some(r)) = (left_val.as_number(), right_val.as_number()) {
                            if r != 0.0 {
                                Ok(PhiValue::Number(l % r))
                            } else {
                                Err("Modulo by zero".to_string())
                            }
                        } else {
                            Err("Modulo requires numeric operands".to_string())
                        }
                    }
                    BinaryOperator::Power => {
                        if let (Some(l), Some(r)) = (left_val.as_number(), right_val.as_number()) {
                            Ok(PhiValue::Number(l.powf(r)))
                        } else {
                            Err("Power requires numeric operands".to_string())
                        }
                    }
                    BinaryOperator::Greater => {
                        if let (Some(l), Some(r)) = (left_val.as_number(), right_val.as_number()) {
                            Ok(PhiValue::Boolean(l > r))
                        } else {
                            Err("Greater than comparison requires numeric operands".to_string())
                        }
                    }
                    BinaryOperator::GreaterEqual => {
                        if let (Some(l), Some(r)) = (left_val.as_number(), right_val.as_number()) {
                            Ok(PhiValue::Boolean(l >= r))
                        } else {
                            Err("Greater than or equal comparison requires numeric operands"
                                .to_string())
                        }
                    }
                    BinaryOperator::Less => {
                        if let (Some(l), Some(r)) = (left_val.as_number(), right_val.as_number()) {
                            Ok(PhiValue::Boolean(l < r))
                        } else {
                            Err("Less than comparison requires numeric operands".to_string())
                        }
                    }
                    BinaryOperator::LessEqual => {
                        if let (Some(l), Some(r)) = (left_val.as_number(), right_val.as_number()) {
                            Ok(PhiValue::Boolean(l <= r))
                        } else {
                            Err("Less than or equal comparison requires numeric operands"
                                .to_string())
                        }
                    }
                    BinaryOperator::Equal => {
                        // Equality comparison works for all types
                        let result = match (&left_val, &right_val) {
                            (PhiValue::Number(l), PhiValue::Number(r)) => l == r,
                            (PhiValue::String(l), PhiValue::String(r)) => l == r,
                            (PhiValue::Boolean(l), PhiValue::Boolean(r)) => l == r,
                            _ => false, // Different types are not equal
                        };
                        Ok(PhiValue::Boolean(result))
                    }
                    BinaryOperator::NotEqual => {
                        // Not equal comparison works for all types
                        let result = match (&left_val, &right_val) {
                            (PhiValue::Number(l), PhiValue::Number(r)) => l != r,
                            (PhiValue::String(l), PhiValue::String(r)) => l != r,
                            (PhiValue::Boolean(l), PhiValue::Boolean(r)) => l != r,
                            _ => true, // Different types are not equal
                        };
                        Ok(PhiValue::Boolean(result))
                    }
                    BinaryOperator::And => {
                        // Logical AND - both operands must be truthy
                        let left_bool = self.is_truthy(&left_val);
                        let right_bool = self.is_truthy(&right_val);
                        Ok(PhiValue::Boolean(left_bool && right_bool))
                    }
                    BinaryOperator::Or => {
                        // Logical OR - at least one operand must be truthy
                        let left_bool = self.is_truthy(&left_val);
                        let right_bool = self.is_truthy(&right_val);
                        Ok(PhiValue::Boolean(left_bool || right_bool))
                    }
                }
            }
            PhiExpression::ListAccess { list, index } => {
                let evaluated_list = self.evaluate_expression(list)?;
                let evaluated_index = self.evaluate_expression(index)?;

                if let PhiValue::List(elements) = evaluated_list {
                    if let Some(idx) = evaluated_index.as_number() {
                        let idx = idx as usize;
                        if idx < elements.len() {
                            Ok(elements[idx].clone())
                        } else {
                            Err(format!(
                                "List index out of bounds: index {} but list has {} elements",
                                idx,
                                elements.len()
                            ))
                        }
                    } else {
                        Err(format!(
                            "List index must be a number, found {:?}",
                            evaluated_index
                        ))
                    }
                } else {
                    Err(format!(
                        "Cannot access elements of non-list type: {:?}",
                        evaluated_list
                    ))
                }
            }
            PhiExpression::UnaryOp { operator, operand } => {
                let operand_val = self.evaluate_expression(operand)?;

                match operator {
                    UnaryOperator::Negate => {
                        if let Some(n) = operand_val.as_number() {
                            Ok(PhiValue::Number(-n))
                        } else {
                            Err("Negation requires a numeric operand".to_string())
                        }
                    }
                    UnaryOperator::Not => {
                        let bool_val = self.is_truthy(&operand_val);
                        Ok(PhiValue::Boolean(!bool_val))
                    }
                }
            }

            // === PHIFLOW UNIQUE: Constructs no other language has ===
            PhiExpression::Witness { expression, body } => {
                self.witness_count += 1;

                // Capture the moment
                let witnessed_value = if let Some(expr) = expression {
                    Some(self.evaluate_expression(expr)?)
                } else {
                    None
                };

                // Calculate coherence at this moment
                let coherence_now = self.calculate_coherence();

                // The program observes itself
                println!();
                println!("  \u{25C9} WITNESS #{} \u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}", self.witness_count);
                if let Some(ref val) = witnessed_value {
                    // Summarize values instead of dumping raw data
                    match val {
                        PhiValue::Pattern2D(points) => {
                            println!("    Observing: Pattern ({} points)", points.len());
                        }
                        PhiValue::ValidationResult(vr) => {
                            println!("    Observing: Validation");
                            println!(
                                "      coherence: {:.3}  phi: {:.3}  zone: {}",
                                vr.coherence, vr.phi_resonance, vr.consciousness_zone
                            );
                        }
                        PhiValue::Number(n) => println!("    Observing: {}", n),
                        PhiValue::String(s) => println!("    Observing: \"{}\"", s),
                        PhiValue::Boolean(b) => println!("    Observing: {}", b),
                        PhiValue::List(items) => {
                            println!("    Observing: List ({} items)", items.len())
                        }
                        _ => println!("    Observing: {:?}", val),
                    }
                }

                // Coherence bar visualization
                let bar_len = 20;
                let filled = (coherence_now * bar_len as f64) as usize;
                let empty = bar_len - filled;
                let bar: String = "\u{2588}".repeat(filled) + &"\u{2591}".repeat(empty);
                println!("    Coherence: {:.3} [{}]", coherence_now, bar);

                if !self.intention_stack.is_empty() {
                    println!("    Intention: {}", self.intention_stack.last().unwrap());
                }
                if !self.frequencies_used.is_empty() {
                    // Deduplicate and sort frequencies for clean display
                    let mut unique_freqs: Vec<f64> = self.frequencies_used.clone();
                    unique_freqs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    unique_freqs.dedup();
                    let freqs: Vec<String> =
                        unique_freqs.iter().map(|f| format!("{:.0}Hz", f)).collect();
                    println!(
                        "    Frequencies: {} ({}x used)",
                        freqs.join(" \u{2192} "),
                        self.frequencies_used.len()
                    );
                }
                if !self.contradictions.is_empty() {
                    println!(
                        "    \u{26A0} {} contradiction(s):",
                        self.contradictions.len()
                    );
                    for c in &self.contradictions {
                        println!("      - {}", c);
                    }
                }
                // Show resonance field if anything has been shared
                if !self.resonance_field.is_empty() {
                    let sources: Vec<String> = self.resonance_field.keys().cloned().collect();
                    let total_values: usize = self.resonance_field.values().map(|v| v.len()).sum();
                    println!(
                        "    Resonance: {} source(s), {} value(s) shared",
                        sources.len(),
                        total_values
                    );
                }
                println!(
                    "    Operations: {} total, {} witness(es)",
                    self.operations_log.len(),
                    self.witness_count
                );
                println!("  \u{25C9} \u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}");
                println!();

                self.log_operation("witness");
                self.coherence = self.calculate_coherence();

                // Execute body if present (what happens after witnessing)
                if let Some(body_expr) = body {
                    self.evaluate_expression(body_expr)?;
                }

                // Return coherence as the value of witnessing
                Ok(PhiValue::Number(coherence_now))
            }

            PhiExpression::IntentionBlock { intention, body } => {
                // Push intention onto stack
                self.intention_stack.push(intention.clone());
                self.log_operation("intention");

                // Check if other intentions have resonated - show incoming resonance
                let incoming: Vec<String> = self
                    .resonance_field
                    .keys()
                    .filter(|k| *k != intention)
                    .cloned()
                    .collect();

                println!();
                println!("  \u{2738} INTENTION: \"{}\"", intention);
                if !incoming.is_empty() {
                    for source in &incoming {
                        let count = self
                            .resonance_field
                            .get(source)
                            .map(|v| v.len())
                            .unwrap_or(0);
                        println!(
                            "    \u{223F} Resonance from \"{}\": {} value(s)",
                            source, count
                        );
                        self.resonance_log.push((source.clone(), intention.clone()));
                    }
                }

                // Execute body under this intention
                let result = self.evaluate_expression(body)?;

                // Pop intention and report
                self.intention_stack.pop();
                let coherence_now = self.calculate_coherence();
                let resonated = self
                    .resonance_field
                    .get(intention)
                    .map(|v| v.len())
                    .unwrap_or(0);
                if resonated > 0 {
                    println!("  \u{2738} INTENTION \"{}\" complete (coherence: {:.3}, resonated {} value(s))", intention, coherence_now, resonated);
                } else {
                    println!(
                        "  \u{2738} INTENTION \"{}\" complete (coherence: {:.3})",
                        intention, coherence_now
                    );
                }
                println!();

                Ok(result)
            }

            PhiExpression::Resonate { expression } => {
                let current_intention = self
                    .intention_stack
                    .last()
                    .cloned()
                    .unwrap_or_else(|| "_global".to_string());

                let value = if let Some(expr) = expression {
                    self.evaluate_expression(expr)?
                } else {
                    // Bare resonate - share the current coherence snapshot
                    PhiValue::Number(self.calculate_coherence())
                };

                // Summarize what's being shared
                let summary = match &value {
                    PhiValue::Pattern2D(pts) => format!("Pattern ({} points)", pts.len()),
                    PhiValue::Number(n) => format!("{:.3}", n),
                    PhiValue::String(s) => format!("\"{}\"", s),
                    PhiValue::Boolean(b) => format!("{}", b),
                    PhiValue::ValidationResult(vr) => {
                        format!("Validation (coherence: {:.3})", vr.coherence)
                    }
                    PhiValue::List(items) => format!("List ({} items)", items.len()),
                    _ => format!("{:?}", value),
                };

                println!(
                    "    \u{223F} Resonating from \"{}\": {}",
                    current_intention, summary
                );

                // Store in resonance field
                self.resonance_field
                    .entry(current_intention.clone())
                    .or_default()
                    .push(value.clone());

                self.log_operation("resonate");

                Ok(value)
            }

            // Handle new consciousness-aware AST nodes
            PhiExpression::ConsciousnessState {
                state,
                coherence,
                frequency,
            } => {
                // For now, just print the state and its properties
                println!("Consciousness State: {}", state);
                println!("  Coherence: {}", coherence);
                println!("  Frequency: {}", frequency);
                Ok(PhiValue::Void)
            }

            PhiExpression::FrequencyPattern {
                base_frequency,
                harmonics,
                phi_scaling,
            } => {
                println!("Processing Frequency Pattern:");
                println!("  Base Frequency: {}", base_frequency);
                println!("  Harmonics: {:?}", harmonics);
                println!("  Phi Scaling: {}", phi_scaling);
                Ok(PhiValue::Void)
            }
            PhiExpression::QuantumField {
                field_type,
                dimensions,
                coherence_target,
            } => {
                println!("Processing Quantum Field:");
                println!("  Field Type: {}", field_type);
                println!("  Dimensions: {:?}", dimensions);
                println!("  Coherence Target: {}", coherence_target);
                Ok(PhiValue::Void)
            }
            PhiExpression::BiologicalInterface {
                target,
                transduction_method,
                frequency,
            } => {
                println!("Processing Biological Interface:");
                println!("  Target: {}", target);
                println!("  Transduction Method: {}", transduction_method);
                println!("  Frequency: {}", frequency);
                Ok(PhiValue::Void)
            }
            PhiExpression::HardwareSync {
                device_type,
                consciousness_mapping,
            } => {
                println!("Processing Hardware Sync:");
                println!("  Device Type: {}", device_type);
                // consciousness_mapping is a PhiExpression, so we need to evaluate it if we want to print its value
                let evaluated_mapping = self.evaluate_expression(consciousness_mapping)?;
                println!("  Consciousness Mapping: {:?}", evaluated_mapping);
                Ok(PhiValue::Void)
            }
            PhiExpression::ConsciousnessFlow {
                condition,
                branches,
            } => {
                println!("Processing Consciousness Flow:");
                let evaluated_condition = self.evaluate_expression(condition)?;
                println!("  Condition: {:?}", evaluated_condition);
                // For now, we won't execute branches, just acknowledge them
                for (state, _action) in branches {
                    println!("  Branch State: {}", state);
                }
                Ok(PhiValue::Void)
            }
            PhiExpression::EmergencyProtocol {
                trigger,
                immediate_action,
                notification,
            } => {
                println!("Processing Emergency Protocol:");
                let evaluated_trigger = self.evaluate_expression(trigger)?;
                println!("  Trigger: {:?}", evaluated_trigger);
                // For now, we won't execute immediate_action, just acknowledge it
                println!("  Immediate Action: (acknowledged)");
                println!("  Notifications: {:?}", notification);
                Ok(PhiValue::Void)
            }
            PhiExpression::StreamBlock { name, body: _ } => {
                // The interpreter doesn't fully support stream execution yet
                // Stream is a construct designed for the IR backend loop
                println!("Processing Stream Block: {}", name);
                println!("  (Stream execution handled natively by the IR Evaluator)");
                Ok(PhiValue::Void)
            }
            PhiExpression::BreakStream => {
                // The interpreter doesn't fully support stream execution yet
                println!("Processing Break Stream");
                Ok(PhiValue::Void)
            }
        }
    }

    /// Calculate live coherence based on the program's behavior so far.
    /// This is the heartbeat of PhiFlow - the program measuring itself.
    fn calculate_coherence(&self) -> f64 {
        // Known consciousness frequencies (the harmonic family)
        let sacred_frequencies: [f64; 9] = [
            432.0, 528.0, 594.0, 672.0, 720.0, 756.0, 768.0, 963.0, 1008.0,
        ];

        let mut coherence = 1.0;

        if !self.frequencies_used.is_empty() {
            // Part 1: How many frequencies are from the sacred family?
            let mut sacred_count = 0;
            let mut unique_freqs: Vec<f64> = self.frequencies_used.clone();
            unique_freqs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            unique_freqs.dedup();

            for &freq in &unique_freqs {
                let is_sacred = sacred_frequencies.iter().any(|&sf| (freq - sf).abs() < 5.0);
                if is_sacred {
                    sacred_count += 1;
                }
            }

            if !unique_freqs.is_empty() {
                let sacred_ratio = sacred_count as f64 / unique_freqs.len() as f64;
                // Sacred ratio heavily influences coherence: 100% sacred = 1.0, 0% = 0.3
                coherence *= 0.3 + 0.7 * sacred_ratio;
            }

            // Part 2: Are the sacred frequencies phi-harmonically related to each other?
            if sacred_count >= 2 {
                let phi: f64 = 1.618033988749895;
                let sacred_used: Vec<f64> = unique_freqs
                    .iter()
                    .filter(|&&f| sacred_frequencies.iter().any(|&sf| (f - sf).abs() < 5.0))
                    .copied()
                    .collect();
                let mut harmonic_pairs = 0;
                let mut total_sacred_pairs = 0;
                for i in 0..sacred_used.len() {
                    for j in (i + 1)..sacred_used.len() {
                        total_sacred_pairs += 1;
                        let ratio = sacred_used[j] / sacred_used[i]; // larger / smaller
                                                                     // Check against known phi powers: phi^1=1.618, phi^2=2.618, phi^3=4.236
                        for power in 1..=4 {
                            if (ratio - phi.powi(power)).abs() < 0.2 {
                                harmonic_pairs += 1;
                                break;
                            }
                        }
                        // Also check simple ratios that are musically harmonic
                        for &nice_ratio in &[1.0, 1.222, 1.333, 1.5, 2.0] {
                            if (ratio - nice_ratio).abs() < 0.05 {
                                harmonic_pairs += 1;
                                break;
                            }
                        }
                    }
                }
                if total_sacred_pairs > 0 {
                    let phi_harmony = harmonic_pairs as f64 / total_sacred_pairs as f64;
                    // Boost for phi-harmonic relationships among sacred frequencies
                    coherence *= 0.8 + 0.2 * phi_harmony;
                }
            }
        }

        // Witness bonus: programs that observe themselves maintain coherence
        if self.witness_count > 0 {
            let witness_bonus = (self.witness_count as f64 * 0.03).min(0.1);
            coherence = (coherence + witness_bonus).min(1.0);
        }

        // Contradiction penalty
        if !self.contradictions.is_empty() {
            let penalty = (self.contradictions.len() as f64 * 0.15).min(0.5);
            coherence -= penalty;
        }

        // Intention clarity: having clear intention slightly boosts coherence
        if !self.intention_stack.is_empty() {
            coherence = (coherence + 0.02).min(1.0);
        }

        coherence.max(0.0)
    }

    /// Log an operation and update coherence
    fn log_operation(&mut self, op: &str) {
        self.operations_log.push(op.to_string());
        self.coherence = self.calculate_coherence();
    }

    /// Check for contradictions when setting a variable
    fn check_contradiction(&mut self, name: &str, new_value: &PhiValue) {
        if let Some(old_value) = self.environment.get(name) {
            // Check if we're overwriting a frequency with a non-harmonic frequency
            if let (Some(old_freq), Some(new_freq)) = (old_value.as_number(), new_value.as_number())
            {
                if self.frequencies_used.contains(&old_freq) && old_freq > 100.0 && new_freq > 100.0
                {
                    let ratio = new_freq / old_freq;
                    let ratio = if ratio < 1.0 { 1.0 / ratio } else { ratio };
                    if ratio > 2.5 && (ratio.ln() / 1.618_f64.ln()).fract() > 0.3 {
                        self.contradictions.push(format!(
                            "Variable '{}' changed from {} to {} (non-harmonic shift)",
                            name, old_freq, new_freq
                        ));
                    }
                }
            }
        }
    }

    /// Map intention to alignment categories for coherence checking
    fn intention_aligns_with(&self, operation: &str) -> bool {
        if let Some(intention) = self.intention_stack.last() {
            match intention.as_str() {
                "healing" | "repair" | "restoration" => {
                    matches!(operation, "create" | "validate" | "witness")
                }
                "analysis" | "measurement" | "observation" => {
                    matches!(operation, "validate" | "witness" | "compare")
                }
                "creation" | "building" | "growth" => {
                    matches!(operation, "create" | "function" | "build")
                }
                _ => true, // unknown intentions don't penalize
            }
        } else {
            true // no intention = no alignment check
        }
    }

    fn evaluate_phi_value(&mut self, value: &PhiValue) -> Result<PhiValue, String> {
        match value {
            PhiValue::String(s) if s.starts_with('$') => {
                // This is a variable reference, resolve it
                let var_name = &s[1..]; // Remove the $ prefix
                self.environment
                    .get(var_name)
                    .cloned()
                    .ok_or_else(|| format!("Undefined variable: {}", var_name))
            }
            PhiValue::List(list_values) => {
                // Lists are literal values, so we just return them as is
                Ok(PhiValue::List(list_values.clone()))
            }
            _ => Ok(value.clone()),
        }
    }

    fn is_truthy(&self, value: &PhiValue) -> bool {
        match value {
            PhiValue::Boolean(b) => *b,
            PhiValue::Number(n) => *n != 0.0,
            PhiValue::String(s) => !s.is_empty(),
            PhiValue::List(l) => !l.is_empty(),
            _ => false, // Other types are falsy by default
        }
    }
}

impl Default for PhiInterpreter {
    fn default() -> Self {
        Self::new()
    }
}

impl PhiValue {
    // Placeholder for a void value, useful for statements that don't return a value
    #[allow(non_upper_case_globals)]
    pub const Void: PhiValue = PhiValue::Number(0.0); // Using Number(0.0) as a simple void placeholder

    pub fn as_number(&self) -> Option<f64> {
        if let PhiValue::Number(n) = self {
            Some(*n)
        } else {
            None
        }
    }
}
