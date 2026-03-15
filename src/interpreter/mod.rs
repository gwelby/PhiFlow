#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unreachable_patterns)]

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
        println!("  ═══ PHIFLOW PROGRAM SUMMARY ════════════");

        // Final coherence with bar
        let bar_len = 20;
        let filled = (final_coherence * bar_len as f64) as usize;
        let empty = bar_len - filled;
        let bar: String = "█".repeat(filled) + &"░".repeat(empty);
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
            println!("  Frequencies: {}", freqs.join(" → "));
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
                println!("    ⚠ {}", c);
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
                    println!("    \"{}\" → \"{}\"", from, to);
                }
            }
        }

        // Operations
        println!("  Operations: {}", self.operations_log.len());

        println!("  ════════════════════════════════════════");
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
                    // Frequency is a variable reference stored in parameters
                    if let Some(PhiValue::String(var_ref)) = pattern_params.get("__frequency_var") {
                        let var_name = &var_ref[1..]; // Remove $
                        let val = self
                            .environment
                            .get(var_name)
                            .ok_or_else(|| format!("Undefined frequency variable: {}", var_name))?;
                        val.as_number().ok_or_else(|| {
                            format!("Frequency variable {} must be a number", var_name)
                        })?
                    } else {
                        0.0 // Fallback
                    }
                } else {
                    *frequency
                };

                self.frequencies_used.push(freq);

                // Generate points
                let points = match pattern_type.as_str() {
                    "spiral" => golden_spiral_points(freq, 100, 1.0),
                    "flower" => flower_of_life_points(3),
                    "dna" => {
                        let (s1, _) = dna_helix_points(freq, 2.0, 1.0);
                        s1.iter().map(|p| [p[0], p[1]]).collect()
                    }
                    _ => vec![], // Custom/Generic field
                };

                self.log_operation("create");
                Ok(PhiValue::Pattern2D(points))
            }
            PhiExpression::ConsciousnessValidation { pattern, .. } => {
                let evaluated_pattern = self.evaluate_expression(pattern)?;
                if let PhiValue::Pattern2D(points) = evaluated_pattern {
                    let result = validate_pattern_consciousness(&points);
                    self.log_operation("validate");
                    Ok(PhiValue::ValidationResult(result))
                } else {
                    Err("Validation requires a Pattern2D operand".to_string())
                }
            }
            PhiExpression::FunctionDef {
                name,
                parameters: _,
                return_type: _,
                body: _,
            } => {
                // Functions are already in the functions map after parsing
                // (This is a simplification, in real usage we'd handle scoping)
                self.functions.insert(name.clone(), expression.clone());
                Ok(PhiValue::Void)
            }
            PhiExpression::FunctionCall { name, arguments } => {
                // Special case for print
                if name == "print" || name == "println" {
                    for arg in arguments {
                        let val = self.evaluate_expression(arg)?;
                        match val {
                            PhiValue::String(s) => print!("{}", s),
                            PhiValue::Number(n) => print!("{}", n),
                            PhiValue::Boolean(b) => print!("{}", b),
                            _ => print!("{:?}", val),
                        }
                    }
                    if name == "println" {
                        println!();
                    }
                    return Ok(PhiValue::Void);
                }

                let function_expr = self
                    .functions
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("Undefined function: {}", name))?;

                if let PhiExpression::FunctionDef {
                    parameters, body, ..
                } = function_expr
                {
                    // Basic scoping: capture current env, inject arguments, execute body, restore env
                    let old_env = self.environment.clone();

                    for (i, (param_name, _)) in parameters.iter().enumerate() {
                        if i < arguments.len() {
                            let arg_val = self.evaluate_expression(&arguments[i])?;
                            self.environment.insert(param_name.clone(), arg_val);
                        }
                    }

                    let result = self.evaluate_expression(&body)?;

                    self.environment = old_env;

                    // If it's a return value, unwrap it
                    if let PhiValue::Return(val) = result {
                        Ok(*val)
                    } else {
                        Ok(result)
                    }
                } else {
                    Err(format!("{} is not a function", name))
                }
            }
            PhiExpression::ConsciousnessState { .. } => {
                Err("Consciousness state not yet executable".to_string())
            }
            PhiExpression::FrequencyPattern { .. } => {
                Err("Frequency pattern not yet executable".to_string())
            }
            PhiExpression::QuantumField { .. } => {
                Err("Quantum field not yet executable".to_string())
            }
            PhiExpression::BiologicalInterface { .. } => {
                Err("Biological interface not yet executable".to_string())
            }
            PhiExpression::HardwareSync { .. } => {
                Err("Hardware sync not yet executable".to_string())
            }
            PhiExpression::ConsciousnessFlow { .. } => {
                Err("Consciousness flow not yet executable".to_string())
            }
            PhiExpression::EmergencyProtocol { .. } => {
                Err("Emergency protocol not yet executable".to_string())
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
                    BinaryOperator::And => {
                        let left_bool = self.is_truthy(&left_val);
                        let right_bool = self.is_truthy(&right_val);
                        Ok(PhiValue::Boolean(left_bool && right_bool))
                    }
                    BinaryOperator::Or => {
                        let left_bool = self.is_truthy(&left_val);
                        let right_bool = self.is_truthy(&right_val);
                        Ok(PhiValue::Boolean(left_bool || right_bool))
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
                        let result = match (&left_val, &right_val) {
                            (PhiValue::Number(l), PhiValue::Number(r)) => l == r,
                            (PhiValue::String(l), PhiValue::String(r)) => l == r,
                            (PhiValue::Boolean(l), PhiValue::Boolean(r)) => l == r,
                            _ => false,
                        };
                        Ok(PhiValue::Boolean(result))
                    }
                    BinaryOperator::NotEqual => {
                        let result = match (&left_val, &right_val) {
                            (PhiValue::Number(l), PhiValue::Number(r)) => l != r,
                            (PhiValue::String(l), PhiValue::String(r)) => l != r,
                            (PhiValue::Boolean(l), PhiValue::Boolean(r)) => l != r,
                            _ => true,
                        };
                        Ok(PhiValue::Boolean(result))
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
            PhiExpression::Witness { expression, mid_circuit, body } => {
                if *mid_circuit {
                    eprintln!(
                        "[WARNING] Legacy interpreter ignores `witness mid_circuit`; \
lowering it as a standard witness. Use `phic --target openqasm` for faithful semantics."
                    );
                }
                self.witness_count += 1;

                let witnessed_value = if let Some(expr) = expression {
                    Some(self.evaluate_expression(expr)?)
                } else {
                    None
                };

                let coherence_now = self.calculate_coherence();

                println!();
                println!(
                    "  ◉ WITNESS #{} ───────────────────────",
                    self.witness_count
                );
                if let Some(ref val) = witnessed_value {
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

                let bar_len = 20;
                let filled = (coherence_now * bar_len as f64) as usize;
                let empty = bar_len - filled;
                let bar: String = "█".repeat(filled) + &"░".repeat(empty);
                println!("    Coherence: {:.3} [{}]", coherence_now, bar);

                if !self.intention_stack.is_empty() {
                    println!("    Intention: {}", self.intention_stack.last().unwrap());
                }
                if !self.frequencies_used.is_empty() {
                    let mut unique_freqs: Vec<f64> = self.frequencies_used.clone();
                    unique_freqs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    unique_freqs.dedup();
                    let freqs: Vec<String> =
                        unique_freqs.iter().map(|f| format!("{:.0}Hz", f)).collect();
                    println!(
                        "    Frequencies: {} ({}x used)",
                        freqs.join(" → "),
                        self.frequencies_used.len()
                    );
                }
                if !self.contradictions.is_empty() {
                    println!("    ⚠ {} contradiction(s):", self.contradictions.len());
                    for c in &self.contradictions {
                        println!("      - {}", c);
                    }
                }
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
                println!("  ◉ ───────────────────────────────");
                println!();

                self.log_operation("witness");
                self.coherence = self.calculate_coherence();

                if let Some(body_expr) = body {
                    self.evaluate_expression(body_expr)?;
                }

                Ok(PhiValue::Number(coherence_now))
            }

            PhiExpression::IntentionBlock { intention, body } => {
                self.intention_stack.push(intention.clone());
                self.log_operation("intention");

                let incoming: Vec<String> = self
                    .resonance_field
                    .keys()
                    .filter(|k| *k != intention)
                    .cloned()
                    .collect();

                println!();
                println!("  ✺ INTENTION: \"{}\"", intention);
                if !incoming.is_empty() {
                    for source in &incoming {
                        let count = self
                            .resonance_field
                            .get(source)
                            .map(|v| v.len())
                            .unwrap_or(0);
                        println!("    ∿ Resonance from \"{}\": {} value(s)", source, count);
                        self.resonance_log.push((source.clone(), intention.clone()));
                    }
                }

                let result = self.evaluate_expression(body)?;

                self.intention_stack.pop();
                let coherence_now = self.calculate_coherence();
                let resonated = self
                    .resonance_field
                    .get(intention)
                    .map(|v| v.len())
                    .unwrap_or(0);
                if resonated > 0 {
                    println!(
                        "  ✺ INTENTION \"{}\" complete (coherence: {:.3}, resonated {} value(s))",
                        intention, coherence_now, resonated
                    );
                } else {
                    println!(
                        "  ✺ INTENTION \"{}\" complete (coherence: {:.3})",
                        intention, coherence_now
                    );
                }
                println!();

                Ok(result)
            }

            PhiExpression::Resonate { expression, direction } => {
                use crate::parser::ResonateDirection;
                if *direction != ResonateDirection::TeamA {
                    eprintln!(
                        "[WARNING] Legacy interpreter ignores `toward {:?}`; \
resonance polarity is not evaluated in interpreter mode. Use `phic --target openqasm` for faithful semantics.",
                        direction
                    );
                }

                let current_intention = self
                    .intention_stack
                    .last()
                    .cloned()
                    .unwrap_or_else(|| "_global".to_string());

                let value = if let Some(expr) = expression {
                    self.evaluate_expression(expr)?
                } else {
                    PhiValue::Number(self.calculate_coherence())
                };

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
                    "    ∿ Resonating from \"{}\": {}",
                    current_intention, summary
                );

                self.resonance_field
                    .entry(current_intention.clone())
                    .or_default()
                    .push(value.clone());

                self.log_operation("resonate");

                Ok(value)
            }

            PhiExpression::StreamBlock { name, body: _ } => {
                println!("Processing Stream Block: {}", name);
                println!("  (Stream execution handled natively by the IR Evaluator)");
                Ok(PhiValue::Void)
            }
            PhiExpression::BreakStream => {
                println!("Processing Break Stream");
                Ok(PhiValue::Void)
            }
            _ => Err(format!(
                "Expression {:?} not implemented in legacy interpreter",
                expression
            )),
        }
    }

    fn calculate_coherence(&self) -> f64 {
        let sacred_frequencies: [f64; 9] = [
            432.0, 528.0, 594.0, 672.0, 720.0, 756.0, 768.0, 963.0, 1008.0,
        ];

        let mut coherence = 1.0;

        if !self.frequencies_used.is_empty() {
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
                coherence *= 0.3 + 0.7 * sacred_ratio;
            }

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
                        let ratio = sacred_used[j] / sacred_used[i];
                        for power in 1..=4 {
                            if (ratio - phi.powi(power)).abs() < 0.2 {
                                harmonic_pairs += 1;
                                break;
                            }
                        }
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
                    coherence *= 0.8 + 0.2 * phi_harmony;
                }
            }
        }

        if self.witness_count > 0 {
            let witness_bonus = (self.witness_count as f64 * 0.03).min(0.1);
            coherence = (coherence + witness_bonus).min(1.0);
        }

        if !self.contradictions.is_empty() {
            let penalty = (self.contradictions.len() as f64 * 0.15).min(0.5);
            coherence -= penalty;
        }

        if !self.intention_stack.is_empty() {
            coherence = (coherence + 0.02).min(1.0);
        }

        coherence.max(0.0)
    }

    fn log_operation(&mut self, op: &str) {
        self.operations_log.push(op.to_string());
        self.coherence = self.calculate_coherence();
    }

    fn check_contradiction(&mut self, name: &str, new_value: &PhiValue) {
        if let Some(old_value) = self.environment.get(name) {
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
                _ => true,
            }
        } else {
            true
        }
    }

    fn evaluate_phi_value(&mut self, value: &PhiValue) -> Result<PhiValue, String> {
        match value {
            PhiValue::String(s) if s.starts_with('$') => {
                let var_name = &s[1..];
                self.environment
                    .get(var_name)
                    .cloned()
                    .ok_or_else(|| format!("Undefined variable: {}", var_name))
            }
            PhiValue::List(list_values) => Ok(PhiValue::List(list_values.clone())),
            _ => Ok(value.clone()),
        }
    }

    fn is_truthy(&self, value: &PhiValue) -> bool {
        match value {
            PhiValue::Boolean(b) => *b,
            PhiValue::Number(n) => *n != 0.0,
            PhiValue::String(s) => !s.is_empty(),
            PhiValue::List(l) => !l.is_empty(),
            _ => false,
        }
    }
}

impl Default for PhiInterpreter {
    fn default() -> Self {
        Self::new()
    }
}

impl PhiValue {
    #[allow(non_upper_case_globals)]
    pub const Void: PhiValue = PhiValue::Number(0.0);

    pub fn as_number(&self) -> Option<f64> {
        if let PhiValue::Number(n) = self {
            Some(*n)
        } else {
            None
        }
    }
}
