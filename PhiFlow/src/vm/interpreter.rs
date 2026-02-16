// PhiFlow Interpreter - Executes PhiFlow quantum-consciousness programs
// Supports real-time consciousness monitoring and quantum backend integration

use crate::compiler::ast::{
    PhiFlowExpression, PhiFlowProgram, QuantumGate, QuantumGateType,
    BinaryOperator, UnaryOperator, ConsciousnessMetric, BrainwaveType,
    ConsciousnessCondition, ComparisonOperator, LogicalOperator,
    PhiFlowType
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// Sacred frequency constants
const SACRED_FREQUENCIES: &[u32] = &[432, 528, 594, 639, 693, 741, 852, 963];
const PHI: f64 = 1.618033988749895;

#[derive(Debug, Clone)]
pub enum PhiFlowValue {
    Number(f64),
    String(String),
    Boolean(bool),
    Array(Vec<PhiFlowValue>),
    QuantumState {
        qubits: u32,
        amplitudes: Vec<num_complex::Complex64>,
    },
    ConsciousnessState {
        coherence: f64,
        clarity: f64,
        flow_state: f64,
        sacred_frequency: Option<u32>,
    },
    SacredFrequency(u32),
    BuiltInFunction(String),
    Nil,
}

impl PartialEq for PhiFlowValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (PhiFlowValue::Number(a), PhiFlowValue::Number(b)) => (a - b).abs() < f64::EPSILON,
            (PhiFlowValue::String(a), PhiFlowValue::String(b)) => a == b,
            (PhiFlowValue::Boolean(a), PhiFlowValue::Boolean(b)) => a == b,
            (PhiFlowValue::Array(a), PhiFlowValue::Array(b)) => a == b,
            (PhiFlowValue::SacredFrequency(a), PhiFlowValue::SacredFrequency(b)) => a == b,
            (PhiFlowValue::BuiltInFunction(a), PhiFlowValue::BuiltInFunction(b)) => a == b,
            (PhiFlowValue::Nil, PhiFlowValue::Nil) => true,
            _ => false,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("Type error: expected {expected}, found {found}")]
    TypeError { expected: String, found: String },
    
    #[error("Undefined variable: {name}")]
    UndefinedVariable { name: String },
    
    #[error("Quantum error: {message}")]
    QuantumError { message: String },
    
    #[error("Consciousness error: {message}")]
    ConsciousnessError { message: String },
    
    #[error("Sacred frequency error: {frequency} is not a valid sacred frequency")]
    InvalidSacredFrequency { frequency: u32 },
    
    #[error("Runtime error: {message}")]
    RuntimeError { message: String },
}

type RuntimeResult<T> = Result<T, RuntimeError>;

#[derive(Debug, Clone)]
enum EvaluationFrame {
    Expression(PhiFlowExpression),
    BinaryOp {
        left: Option<PhiFlowValue>,
        operator: BinaryOperator,
        right: Option<PhiFlowValue>,
        left_expr: Option<PhiFlowExpression>,
        right_expr: Option<PhiFlowExpression>,
    },
    UnaryOp {
        operator: UnaryOperator,
        operand: Option<PhiFlowValue>,
        operand_expr: Option<PhiFlowExpression>,
    },
    SacredFrequency {
        frequency: u32,
        operation: Option<PhiFlowValue>,
        operation_expr: Option<PhiFlowExpression>,
    },
    Block {
        expressions: Vec<PhiFlowExpression>,
        index: usize,
        last_result: PhiFlowValue,
    },
    Array {
        elements: Vec<PhiFlowExpression>,
        index: usize,
        values: Vec<PhiFlowValue>,
    },
    ForLoop {
        variable: String,
        iterable_value: Option<PhiFlowValue>,
        body: PhiFlowExpression,
        current_index: usize,
        items: Vec<PhiFlowValue>,
    },
    WhileLoop {
        condition: PhiFlowExpression,
        body: PhiFlowExpression,
        iterations: usize,
        evaluating_condition: bool,
    },
    Let {
        variable: String,
    },
    ArrayIndex {
        array: Option<PhiFlowValue>,
        index: Option<PhiFlowValue>,
        index_expr: PhiFlowExpression,
    },
    Intention {
        content: String,
        target_expr: PhiFlowExpression,
        old_intention: Option<String>,
    },
    Witness {
        target_expr: PhiFlowExpression,
    },
    FunctionCall {
        name: String,
        args: Vec<PhiFlowExpression>,
        evaluated_args: Vec<PhiFlowValue>,
        current_arg: usize,
    },
    FunctionBody {
        old_vars: HashMap<String, PhiFlowValue>,
    },
}

#[derive(Debug)]
pub struct ConsciousnessMonitor {
    pub coherence: f64,
    pub clarity: f64,
    pub flow_state: f64,
    pub current_frequency: Option<u32>,
    pub eeg_channels: HashMap<String, f64>, // TP9, AF7, AF8, TP10
    pub brainwave_levels: HashMap<BrainwaveType, f64>,
}

impl Default for ConsciousnessMonitor {
    fn default() -> Self {
        let mut eeg_channels = HashMap::new();
        eeg_channels.insert("TP9".to_string(), 0.0);
        eeg_channels.insert("AF7".to_string(), 0.0);
        eeg_channels.insert("AF8".to_string(), 0.0);
        eeg_channels.insert("TP10".to_string(), 0.0);
        
        let mut brainwave_levels = HashMap::new();
        brainwave_levels.insert(BrainwaveType::Delta, 0.0);
        brainwave_levels.insert(BrainwaveType::Theta, 0.0);
        brainwave_levels.insert(BrainwaveType::Alpha, 0.0);
        brainwave_levels.insert(BrainwaveType::Beta, 0.0);
        brainwave_levels.insert(BrainwaveType::Gamma, 0.0);
        
        ConsciousnessMonitor {
            coherence: 0.5,
            clarity: 0.5,
            flow_state: 0.5,
            current_frequency: None,
            eeg_channels,
            brainwave_levels,
        }
    }
}

#[derive(Debug)]
pub struct QuantumBackend {
    pub max_qubits: u32,
    pub backend_type: String,
    pub quantum_states: HashMap<String, PhiFlowValue>,
}

impl Default for QuantumBackend {
    fn default() -> Self {
        QuantumBackend {
            max_qubits: 32,
            backend_type: "simulator".to_string(),
            quantum_states: HashMap::new(),
        }
    }
}

pub struct PhiFlowInterpreter {
    variables: HashMap<String, PhiFlowValue>,
    functions: HashMap<String, PhiFlowExpression>,
    consciousness: Arc<Mutex<ConsciousnessMonitor>>,
    quantum_backend: Arc<Mutex<QuantumBackend>>,
    sacred_frequency_lock: Option<u32>,
    current_intention: Option<String>,
    resonance_field: HashMap<String, Vec<PhiFlowValue>>,
    observation_history: Vec<String>,
}

impl PhiFlowInterpreter {
    pub fn new() -> Self {
        let mut interpreter = PhiFlowInterpreter {
            variables: HashMap::new(),
            functions: HashMap::new(),
            consciousness: Arc::new(Mutex::new(ConsciousnessMonitor::default())),
            quantum_backend: Arc::new(Mutex::new(QuantumBackend::default())),
            sacred_frequency_lock: None,
            current_intention: None,
            resonance_field: HashMap::new(),
            observation_history: Vec::new(),
        };
        
        interpreter.init_built_in_functions();
        interpreter
    }
    
    fn init_built_in_functions(&mut self) {
        // Add built-in mathematical constants
        self.variables.insert("PHI".to_string(), PhiFlowValue::Number(PHI));
        self.variables.insert("PI".to_string(), PhiFlowValue::Number(std::f64::consts::PI));
        self.variables.insert("E".to_string(), PhiFlowValue::Number(std::f64::consts::E));
        
        // Add sacred frequencies as constants
        for &freq in SACRED_FREQUENCIES {
            let name = format!("SACRED_{}", freq);
            self.variables.insert(name, PhiFlowValue::SacredFrequency(freq));
        }

        // Register built-in functions
        let built_ins = vec!["print", "len", "push", "pop", "phi_spiral", "sacred_resonate", "resonate"];
        for name in built_ins {
            self.variables.insert(name.to_string(), PhiFlowValue::BuiltInFunction(name.to_string()));
        }
    }
    
    pub async fn execute_program(&mut self, program: PhiFlowProgram) -> RuntimeResult<PhiFlowValue> {
        // Execute function definitions first
        for function in program.functions {
            if let PhiFlowExpression::FunctionDefinition { name, .. } = &function {
                self.functions.insert(name.clone(), function);
            }
        }
        
        // Execute main function or main expression
        if let Some(main) = program.main {
            self.evaluate_expression(&main).await
        } else {
            Ok(PhiFlowValue::Nil)
        }
    }
    
    pub async fn evaluate_expression(&mut self, expr: &PhiFlowExpression) -> RuntimeResult<PhiFlowValue> {
        // Use iterative stack-based evaluation to avoid async recursion
        let mut stack: Vec<EvaluationFrame> = vec![EvaluationFrame::Expression(expr.clone())];
        let mut result_stack: Vec<PhiFlowValue> = Vec::new();
        
        while let Some(frame) = stack.pop() {
            match frame {
                EvaluationFrame::Expression(expr) => {
                    match expr {
                        PhiFlowExpression::Number(n) => {
                            result_stack.push(PhiFlowValue::Number(n));
                        }
                        PhiFlowExpression::String(s) => {
                            result_stack.push(PhiFlowValue::String(s));
                        }
                        PhiFlowExpression::Boolean(b) => {
                            result_stack.push(PhiFlowValue::Boolean(b));
                        }
                        PhiFlowExpression::Variable(name) => {
                            let value = self.variables.get(&name)
                                .cloned()
                                .ok_or_else(|| RuntimeError::UndefinedVariable { name: name.clone() })?;
                            result_stack.push(value);
                        }
                        PhiFlowExpression::Let { variable, value, .. } => {
                            stack.push(EvaluationFrame::Let {
                                variable: variable.clone(),
                            });
                            stack.push(EvaluationFrame::Expression(value.as_ref().clone()));
                        }
                        PhiFlowExpression::BinaryOp { left, operator, right } => {
                            stack.push(EvaluationFrame::BinaryOp {
                                left: None,
                                operator: operator.clone(),
                                right: None,
                                left_expr: Some(left.as_ref().clone()),
                                right_expr: Some(right.as_ref().clone()),
                            });
                            stack.push(EvaluationFrame::Expression(left.as_ref().clone()));
                        }
                        PhiFlowExpression::UnaryOp { operator, operand } => {
                            stack.push(EvaluationFrame::UnaryOp {
                                operator: operator.clone(),
                                operand: None,
                                operand_expr: Some(operand.as_ref().clone()),
                            });
                            stack.push(EvaluationFrame::Expression(operand.as_ref().clone()));
                        }
                        PhiFlowExpression::SacredFrequency { frequency, operation } => {
                            stack.push(EvaluationFrame::SacredFrequency {
                                frequency,
                                operation: None,
                                operation_expr: Some(operation.as_ref().clone()),
                            });
                            stack.push(EvaluationFrame::Expression(operation.as_ref().clone()));
                        }
                        PhiFlowExpression::Block(expressions) => {
                            if expressions.is_empty() {
                                result_stack.push(PhiFlowValue::Nil);
                            } else {
                                stack.push(EvaluationFrame::Block {
                                    expressions: expressions.clone(),
                                    index: 0,
                                    last_result: PhiFlowValue::Nil,
                                });
                                stack.push(EvaluationFrame::Expression(expressions[0].clone()));
                            }
                        }
                        PhiFlowExpression::Array(elements) => {
                            if elements.is_empty() {
                                result_stack.push(PhiFlowValue::Array(Vec::new()));
                            } else {
                                stack.push(EvaluationFrame::Array {
                                    elements: elements.clone(),
                                    index: 0,
                                    values: Vec::new(),
                                });
                                stack.push(EvaluationFrame::Expression(elements[0].clone()));
                            }
                        }
                        PhiFlowExpression::ArrayIndex { array, index } => {
                            stack.push(EvaluationFrame::ArrayIndex {
                                array: None,
                                index: None,
                                index_expr: index.as_ref().clone(),
                            });
                            stack.push(EvaluationFrame::Expression(array.as_ref().clone()));
                        }
                        PhiFlowExpression::For { variable, iterable, body } => {
                            // Push ForLoop frame and evaluate iterable
                            stack.push(EvaluationFrame::ForLoop {
                                variable: variable.clone(),
                                iterable_value: None,
                                body: body.as_ref().clone(),
                                current_index: 0,
                                items: Vec::new(),
                            });
                            stack.push(EvaluationFrame::Expression(iterable.as_ref().clone()));
                        }
                        PhiFlowExpression::While { condition, body } => {
                            // Push WhileLoop frame and evaluate condition
                            stack.push(EvaluationFrame::WhileLoop {
                                condition: condition.as_ref().clone(),
                                body: body.as_ref().clone(),
                                iterations: 0,
                                evaluating_condition: true,
                            });
                            stack.push(EvaluationFrame::Expression(condition.as_ref().clone()));
                        }
                        PhiFlowExpression::FunctionCall { name, args } => {
                            // Check if it's a built-in function or a user-defined function
                            let is_built_in = self.variables.get(&name).map_or(false, |v| matches!(v, PhiFlowValue::BuiltInFunction(_)));
                            let is_user_defined = self.functions.contains_key(&name);
                            
                            if is_built_in || is_user_defined {
                                if args.is_empty() {
                                    if let Some(result) = self.call_function_with_args(&name, &[], &mut stack)? {
                                        result_stack.push(result);
                                    }
                                } else {
                                    stack.push(EvaluationFrame::FunctionCall {
                                        name: name.clone(),
                                        args: args.clone(),
                                        evaluated_args: Vec::new(),
                                        current_arg: 0,
                                    });
                                    stack.push(EvaluationFrame::Expression(args[0].clone()));
                                }
                            } else {
                                return Err(RuntimeError::UndefinedVariable { name: name.clone() });
                            }
                        }
                        PhiFlowExpression::QuantumCircuit { qubits, .. } => {
                            // Simple implementation: return a QuantumState with the number of qubits
                            result_stack.push(PhiFlowValue::QuantumState {
                                qubits: qubits.len() as u32,
                                amplitudes: Vec::new(), // Placeholder for now
                            });
                        }
                        PhiFlowExpression::Witness(target) => {
                            stack.push(EvaluationFrame::Witness {
                                target_expr: target.as_ref().clone(),
                            });
                            stack.push(EvaluationFrame::Expression(target.as_ref().clone()));
                        }
                        PhiFlowExpression::Intention { content, target } => {
                            let old_intention = self.current_intention.clone();
                            self.current_intention = Some(content.clone());
                            println!("🎯 Intention set: \"{}\"", content);
                            
                            stack.push(EvaluationFrame::Intention {
                                content: content.clone(),
                                target_expr: target.as_ref().clone(),
                                old_intention,
                            });
                            stack.push(EvaluationFrame::Expression(target.as_ref().clone()));
                        }
                        _ => {
                            return Err(RuntimeError::RuntimeError {
                                message: format!("Unimplemented expression: {:?}", expr),
                            });
                        }
                    }
                }
                EvaluationFrame::BinaryOp { left, operator, right, left_expr: _, right_expr } => {
                    if left.is_none() {
                        // Left operand needs to be evaluated
                        let left_value = result_stack.pop().ok_or_else(|| RuntimeError::RuntimeError {
                            message: "Missing left operand value".to_string(),
                        })?;
                        stack.push(EvaluationFrame::BinaryOp {
                            left: Some(left_value),
                            operator,
                            right: None,
                            left_expr: None,
                            right_expr: right_expr.clone(),
                        });
                        if let Some(right_expr) = right_expr {
                            stack.push(EvaluationFrame::Expression(right_expr));
                        }
                    } else if right.is_none() {
                        // Right operand needs to be evaluated
                        let right_value = result_stack.pop().ok_or_else(|| RuntimeError::RuntimeError {
                            message: "Missing right operand value".to_string(),
                        })?;
                        stack.push(EvaluationFrame::BinaryOp {
                            left,
                            operator,
                            right: Some(right_value),
                            left_expr: None,
                            right_expr: None,
                        });
                    } else {
                        // Both operands evaluated, perform operation
                        let left_val = left.unwrap();
                        let right_val = right.unwrap();
                        let result = self.evaluate_binary_op_values(&left_val, &operator, &right_val)?;
                        result_stack.push(result);
                    }
                }
                EvaluationFrame::UnaryOp { operator, operand, operand_expr: _ } => {
                    if operand.is_none() {
                        let operand_value = result_stack.pop().ok_or_else(|| RuntimeError::RuntimeError {
                            message: "Missing operand value".to_string(),
                        })?;
                        stack.push(EvaluationFrame::UnaryOp {
                            operator,
                            operand: Some(operand_value),
                            operand_expr: None,
                        });
                    } else {
                        let operand_val = operand.unwrap();
                        let result = self.evaluate_unary_op_values(&operator, &operand_val)?;
                        result_stack.push(result);
                    }
                }
                EvaluationFrame::SacredFrequency { frequency, operation, operation_expr: _ } => {
                    if operation.is_none() {
                        let operation_value = result_stack.pop().ok_or_else(|| RuntimeError::RuntimeError {
                            message: "Missing operation value".to_string(),
                        })?;
                        stack.push(EvaluationFrame::SacredFrequency {
                            frequency,
                            operation: Some(operation_value),
                            operation_expr: None,
                        });
                    } else {
                        let result = self.evaluate_sacred_frequency_values(frequency, &operation.unwrap())?;
                        result_stack.push(result);
                    }
                }
                EvaluationFrame::Block { expressions, index, last_result } => {
                    if index < expressions.len() {
                        let current_result = if index == 0 {
                            result_stack.pop().unwrap_or(PhiFlowValue::Nil)
                        } else {
                            result_stack.pop().unwrap_or(last_result)
                        };
                        
                        if index + 1 < expressions.len() {
                            stack.push(EvaluationFrame::Block {
                                expressions: expressions.clone(),
                                index: index + 1,
                                last_result: current_result,
                            });
                            stack.push(EvaluationFrame::Expression(expressions[index + 1].clone()));
                        } else {
                            result_stack.push(current_result);
                        }
                    }
                }
                EvaluationFrame::Array { elements, index, mut values } => {
                    if index < elements.len() {
                        let current_value = result_stack.pop().ok_or_else(|| RuntimeError::RuntimeError {
                            message: "Missing array element value".to_string(),
                        })?;
                        values.push(current_value);
                        
                        if index + 1 < elements.len() {
                            stack.push(EvaluationFrame::Array {
                                elements: elements.clone(),
                                index: index + 1,
                                values,
                            });
                            stack.push(EvaluationFrame::Expression(elements[index + 1].clone()));
                        } else {
                            result_stack.push(PhiFlowValue::Array(values));
                        }
                    }
                }
                EvaluationFrame::ForLoop { variable, iterable_value, body, current_index, mut items } => {
                    if iterable_value.is_none() {
                        // We just finished evaluating the iterable
                        let iterable_val = result_stack.pop().ok_or_else(|| RuntimeError::RuntimeError {
                            message: "Missing iterable value".to_string(),
                        })?;
                        
                        match iterable_val {
                            PhiFlowValue::Array(arr) => {
                                items = arr;
                                if !items.is_empty() {
                                    // Set the loop variable for the first iteration
                                    self.variables.insert(variable.clone(), items[0].clone());
                                    // Push the updated frame and execute the body
                                    stack.push(EvaluationFrame::ForLoop {
                                        variable: variable.clone(),
                                        iterable_value: Some(PhiFlowValue::Array(items.clone())),
                                        body: body.clone(),
                                        current_index: 1, // Next index
                                        items: items.clone(),
                                    });
                                    stack.push(EvaluationFrame::Expression(body));
                                } else {
                                    // Empty array, push nil result
                                    result_stack.push(PhiFlowValue::Nil);
                                }
                            }
                            _ => {
                                return Err(RuntimeError::TypeError {
                                    expected: "Array".to_string(),
                                    found: format!("{:?}", iterable_val),
                                });
                            }
                        }
                    } else if current_index < items.len() {
                        // Continue with next iteration
                        let _body_result = result_stack.pop().unwrap_or(PhiFlowValue::Nil);
                        
                        // Set loop variable for current iteration
                        self.variables.insert(variable.clone(), items[current_index].clone());
                        
                        // Push frame for next iteration
                        stack.push(EvaluationFrame::ForLoop {
                            variable: variable.clone(),
                            iterable_value: iterable_value.clone(),
                            body: body.clone(),
                            current_index: current_index + 1,
                            items,
                        });
                        stack.push(EvaluationFrame::Expression(body));
                    } else {
                        // Loop finished, get the last result
                        let last_result = result_stack.pop().unwrap_or(PhiFlowValue::Nil);
                        result_stack.push(last_result);
                    }
                }
                EvaluationFrame::WhileLoop { condition, body, iterations, evaluating_condition } => {
                    const MAX_ITERATIONS: usize = 10000; // Safety limit
                    
                    if iterations >= MAX_ITERATIONS {
                        return Err(RuntimeError::RuntimeError {
                            message: "While loop exceeded maximum iterations".to_string(),
                        });
                    }
                    
                    if evaluating_condition {
                        // Condition was just evaluated
                        let condition_value = result_stack.pop().ok_or_else(|| RuntimeError::RuntimeError {
                            message: "Missing condition value".to_string(),
                        })?;
                        
                        if self.is_truthy(&condition_value) {
                            // Execute body and then check condition again
                            stack.push(EvaluationFrame::WhileLoop {
                                condition: condition.clone(),
                                body: body.clone(),
                                iterations: iterations + 1,
                                evaluating_condition: false,
                            });
                            stack.push(EvaluationFrame::Expression(body));
                        } else {
                            // Condition false, exit with nil
                            result_stack.push(PhiFlowValue::Nil);
                        }
                    } else {
                        // Body was just executed, now check condition again
                        let _body_result = result_stack.pop().unwrap_or(PhiFlowValue::Nil);
                        
                        stack.push(EvaluationFrame::WhileLoop {
                            condition: condition.clone(),
                            body: body.clone(),
                            iterations,
                            evaluating_condition: true,
                        });
                        stack.push(EvaluationFrame::Expression(condition));
                    }
                }
                EvaluationFrame::Let { variable } => {
                    let value = result_stack.pop().ok_or_else(|| RuntimeError::RuntimeError {
                        message: "Missing value for let binding".to_string(),
                    })?;
                    self.variables.insert(variable, value.clone());
                    result_stack.push(value);
                }
                EvaluationFrame::ArrayIndex { array, index, index_expr } => {
                    if array.is_none() {
                        let array_val = result_stack.pop().ok_or_else(|| RuntimeError::RuntimeError {
                            message: "Missing array value".to_string(),
                        })?;
                        stack.push(EvaluationFrame::ArrayIndex {
                            array: Some(array_val),
                            index: None,
                            index_expr: index_expr.clone(),
                        });
                        stack.push(EvaluationFrame::Expression(index_expr));
                    } else if index.is_none() {
                        let index_val = result_stack.pop().ok_or_else(|| RuntimeError::RuntimeError {
                            message: "Missing index value".to_string(),
                        })?;
                        let array_val = array.unwrap();
                        
                        match (array_val, index_val) {
                            (PhiFlowValue::Array(arr), PhiFlowValue::Number(idx)) => {
                                let i = idx as usize;
                                if i < arr.len() {
                                    result_stack.push(arr[i].clone());
                                } else {
                                    return Err(RuntimeError::RuntimeError {
                                        message: format!("Array index {} out of bounds for array of length {}", i, arr.len()),
                                    });
                                }
                            }
                            _ => {
                                return Err(RuntimeError::TypeError {
                                    expected: "Array and number index".to_string(),
                                    found: "incompatible types".to_string(),
                                });
                            }
                        }
                    }
                }
                EvaluationFrame::Intention { content: _, target_expr: _, old_intention } => {
                    let value = result_stack.pop().ok_or_else(|| RuntimeError::RuntimeError {
                        message: "Missing intention target value".to_string(),
                    })?;
                    
                    // Restore old intention
                    self.current_intention = old_intention;
                    result_stack.push(value);
                }
                EvaluationFrame::Witness { target_expr: _ } => {
                    let value = result_stack.pop().ok_or_else(|| RuntimeError::RuntimeError {
                        message: "Missing witnessed value".to_string(),
                    })?;
                    
                    let observation = format!("Witnessed: {}", self.value_to_string(&value));
                    println!("👁️  {}", observation);
                    self.observation_history.push(observation);
                    result_stack.push(value);
                }
                EvaluationFrame::FunctionCall { name, args, mut evaluated_args, current_arg } => {
                    // Get the result of the current argument evaluation
                    let arg_value = result_stack.pop().ok_or_else(|| RuntimeError::RuntimeError {
                        message: "Missing function argument value".to_string(),
                    })?;
                    evaluated_args.push(arg_value);
                    
                    if current_arg + 1 < args.len() {
                        // More arguments to evaluate
                        stack.push(EvaluationFrame::FunctionCall {
                            name,
                            args: args.clone(),
                            evaluated_args,
                            current_arg: current_arg + 1,
                        });
                        stack.push(EvaluationFrame::Expression(args[current_arg + 1].clone()));
                    } else {
                        // All arguments evaluated, call the function
                        match self.call_function_with_args(&name, &evaluated_args, &mut stack)? {
                            Some(result) => result_stack.push(result),
                            None => {} // User function pushed to stack
                        }
                    }
                }
                EvaluationFrame::FunctionBody { old_vars } => {
                    let result = result_stack.pop().ok_or_else(|| RuntimeError::RuntimeError {
                        message: "No result from function body".to_string(),
                    })?;
                    
                    // Restore scope
                    for (name, old_value) in old_vars {
                        self.variables.insert(name, old_value);
                    }
                    
                    result_stack.push(result);
                }
            }
        }
        
        result_stack.pop().ok_or_else(|| RuntimeError::RuntimeError {
            message: "No result produced".to_string(),
        })
    }
    
    // Simple synchronous expression evaluation for basic cases only
    fn evaluate_expression_simple(&self, expr: &PhiFlowExpression) -> RuntimeResult<PhiFlowValue> {
        match expr {
            PhiFlowExpression::Number(n) => Ok(PhiFlowValue::Number(*n)),
            PhiFlowExpression::String(s) => Ok(PhiFlowValue::String(s.clone())),
            PhiFlowExpression::Boolean(b) => Ok(PhiFlowValue::Boolean(*b)),
            PhiFlowExpression::Variable(name) => {
                self.variables.get(name)
                    .cloned()
                    .ok_or_else(|| RuntimeError::UndefinedVariable { name: name.clone() })
            }
            _ => Err(RuntimeError::RuntimeError {
                message: "Complex expression requires iterative evaluation".to_string(),
            })
        }
    }
    
    // Value-based binary operation evaluation (non-recursive)
    fn evaluate_binary_op_values(
        &self,
        left_val: &PhiFlowValue,
        operator: &BinaryOperator,
        right_val: &PhiFlowValue,
    ) -> RuntimeResult<PhiFlowValue> {
        match (left_val, right_val) {
            (PhiFlowValue::Number(l), PhiFlowValue::Number(r)) => {
                match operator {
                    BinaryOperator::Add => Ok(PhiFlowValue::Number(l + r)),
                    BinaryOperator::Subtract => Ok(PhiFlowValue::Number(l - r)),
                    BinaryOperator::Multiply => Ok(PhiFlowValue::Number(l * r)),
                    BinaryOperator::Divide => {
                        if *r == 0.0 {
                            Err(RuntimeError::RuntimeError {
                                message: "Division by zero".to_string(),
                            })
                        } else {
                            Ok(PhiFlowValue::Number(l / r))
                        }
                    }
                    BinaryOperator::Power => Ok(PhiFlowValue::Number(l.powf(*r))),
                    BinaryOperator::Equal => Ok(PhiFlowValue::Boolean((l - r).abs() < f64::EPSILON)),
                    BinaryOperator::NotEqual => Ok(PhiFlowValue::Boolean((l - r).abs() >= f64::EPSILON)),
                    BinaryOperator::Less => Ok(PhiFlowValue::Boolean(l < r)),
                    BinaryOperator::Greater => Ok(PhiFlowValue::Boolean(l > r)),
                    BinaryOperator::LessEqual => Ok(PhiFlowValue::Boolean(l <= r)),
                    BinaryOperator::GreaterEqual => Ok(PhiFlowValue::Boolean(l >= r)),
                    BinaryOperator::PhiMultiply => Ok(PhiFlowValue::Number(l * PHI + r)),
                    _ => Err(RuntimeError::TypeError {
                        expected: "compatible types for operation".to_string(),
                        found: format!("{:?} {:?}", operator, "numbers"),
                    }),
                }
            }
            (PhiFlowValue::Boolean(l), PhiFlowValue::Boolean(r)) => {
                match operator {
                    BinaryOperator::And => Ok(PhiFlowValue::Boolean(*l && *r)),
                    BinaryOperator::Or => Ok(PhiFlowValue::Boolean(*l || *r)),
                    BinaryOperator::Equal => Ok(PhiFlowValue::Boolean(*l == *r)),
                    BinaryOperator::NotEqual => Ok(PhiFlowValue::Boolean(*l != *r)),
                    _ => Err(RuntimeError::TypeError {
                        expected: "logical operation".to_string(),
                        found: format!("{:?}", operator),
                    }),
                }
            }
            _ => Err(RuntimeError::TypeError {
                expected: "compatible types".to_string(),
                found: "incompatible types for binary operation".to_string(),
            }),
        }
    }

    async fn evaluate_binary_op(
        &mut self,
        left: &PhiFlowExpression,
        operator: &BinaryOperator,
        right: &PhiFlowExpression,
    ) -> RuntimeResult<PhiFlowValue> {
        let left_val = self.evaluate_expression(left).await?;
        let right_val = self.evaluate_expression(right).await?;
        self.evaluate_binary_op_values(&left_val, operator, &right_val)
    }
    
    // Value-based unary operation evaluation (non-recursive)
    fn evaluate_unary_op_values(
        &self,
        operator: &UnaryOperator,
        operand_val: &PhiFlowValue,
    ) -> RuntimeResult<PhiFlowValue> {
        match (operator, operand_val) {
            (UnaryOperator::Negate, PhiFlowValue::Number(n)) => Ok(PhiFlowValue::Number(-*n)),
            (UnaryOperator::Not, PhiFlowValue::Boolean(b)) => Ok(PhiFlowValue::Boolean(!*b)),
            (UnaryOperator::PhiTransform, PhiFlowValue::Number(n)) => {
                Ok(PhiFlowValue::Number(*n * PHI))
            }
            _ => Err(RuntimeError::TypeError {
                expected: "compatible type for unary operation".to_string(),
                found: format!("{:?}", operator),
            }),
        }
    }

    async fn evaluate_unary_op(
        &mut self,
        operator: &UnaryOperator,
        operand: &PhiFlowExpression,
    ) -> RuntimeResult<PhiFlowValue> {
        let val = self.evaluate_expression(operand).await?;
        self.evaluate_unary_op_values(operator, &val)
    }
    
    // Value-based sacred frequency evaluation (non-recursive)
    fn evaluate_sacred_frequency_values(
        &mut self,
        frequency: u32,
        operation_val: &PhiFlowValue,
    ) -> RuntimeResult<PhiFlowValue> {
        // Validate sacred frequency
        if !SACRED_FREQUENCIES.contains(&frequency) {
            return Err(RuntimeError::InvalidSacredFrequency { frequency });
        }
        
        // Set frequency lock
        self.sacred_frequency_lock = Some(frequency);
        
        // Update consciousness monitor with the sacred frequency
        {
            let mut consciousness = self.consciousness.lock().unwrap();
            consciousness.current_frequency = Some(frequency);
            
            // Apply frequency-specific consciousness effects
            match frequency {
                432 => consciousness.coherence += 0.1, // Earth frequency enhances coherence
                528 => consciousness.clarity += 0.15,  // Love frequency enhances clarity
                594 => consciousness.flow_state += 0.2, // Transformation enhances flow
                _ => {}
            }
            
            // Clamp values to [0.0, 1.0]
            consciousness.coherence = consciousness.coherence.min(1.0).max(0.0);
            consciousness.clarity = consciousness.clarity.min(1.0).max(0.0);
            consciousness.flow_state = consciousness.flow_state.min(1.0).max(0.0);
        }
        
        // Clear frequency lock
        self.sacred_frequency_lock = None;
        
        // Return the operation result
        Ok(operation_val.clone())
    }

    async fn evaluate_sacred_frequency(
        &mut self,
        frequency: u32,
        operation: &PhiFlowExpression,
    ) -> RuntimeResult<PhiFlowValue> {
        // For async version, we evaluate the operation first, then apply frequency effects
        let result = self.evaluate_expression(operation).await?;
        self.evaluate_sacred_frequency_values(frequency, &result)
    }
    
    async fn evaluate_frequency_lock(&mut self, target_frequency: u32, threshold: f64, action: &PhiFlowExpression) -> RuntimeResult<PhiFlowValue> {
        let frequency_locked = {
            let consciousness = self.consciousness.lock().unwrap();
            
            // Check if current frequency matches target with threshold
            if let Some(current_freq) = consciousness.current_frequency {
                let freq_diff = (current_freq as f64 - target_frequency as f64).abs();
                freq_diff <= threshold
            } else {
                false
            }
        };
        
        if frequency_locked {
            self.evaluate_expression(action).await
        } else {
            Ok(PhiFlowValue::Boolean(false))
        }
    }
    
    async fn evaluate_consciousness_binding(
        &mut self,
        state_name: &str,
        expression: &PhiFlowExpression,
    ) -> RuntimeResult<PhiFlowValue> {
        let consciousness = self.consciousness.lock().unwrap();
        let state_value = PhiFlowValue::ConsciousnessState {
            coherence: consciousness.coherence,
            clarity: consciousness.clarity,
            flow_state: consciousness.flow_state,
            sacred_frequency: consciousness.current_frequency,
        };
        drop(consciousness);
        
        // Bind consciousness state to variable
        self.variables.insert(state_name.to_string(), state_value.clone());
        
        // Execute expression with consciousness binding
        self.evaluate_expression(expression).await
    }
    
    async fn evaluate_consciousness_monitor(
        &mut self,
        metrics: &[ConsciousnessMetric],
        callback: &PhiFlowExpression,
    ) -> RuntimeResult<PhiFlowValue> {
        // In a real implementation, this would start a background task
        // that continuously monitors consciousness metrics and calls the callback
        // For now, we'll simulate one call
        
        let consciousness = self.consciousness.lock().unwrap();
        
        // Check all monitored metrics
        for metric in metrics {
            let metric_value = match metric {
                ConsciousnessMetric::Coherence => consciousness.coherence,
                ConsciousnessMetric::Clarity => consciousness.clarity,
                ConsciousnessMetric::FlowState => consciousness.flow_state,
                ConsciousnessMetric::PhiResonance => {
                    // Calculate phi resonance based on current state
                    (consciousness.coherence * PHI + consciousness.clarity) / (1.0 + PHI)
                }
                ConsciousnessMetric::EEGChannel(channel) => {
                    consciousness.eeg_channels.get(channel).cloned().unwrap_or(0.0)
                }
                ConsciousnessMetric::BrainwaveType(wave_type) => {
                    consciousness.brainwave_levels.get(wave_type).cloned().unwrap_or(0.0)
                }
                _ => 0.0,
            };
            
            // Store metric value as variable for callback
            let metric_name = format!("metric_{:?}", metric);
            self.variables.insert(metric_name, PhiFlowValue::Number(metric_value));
        }
        
        drop(consciousness);
        
        // Execute callback
        self.evaluate_expression(callback).await
    }
    
    async fn evaluate_consciousness_condition(
        &mut self,
        condition: &ConsciousnessCondition,
        then_branch: &PhiFlowExpression,
        else_branch: Option<&PhiFlowExpression>,
    ) -> RuntimeResult<PhiFlowValue> {
        let condition_met = self.check_consciousness_condition(condition)?;
        
        if condition_met {
            self.evaluate_expression(then_branch).await
        } else if let Some(else_expr) = else_branch {
            self.evaluate_expression(else_expr).await
        } else {
            Ok(PhiFlowValue::Boolean(false))
        }
    }
    
    fn check_consciousness_condition(&self, condition: &ConsciousnessCondition) -> RuntimeResult<bool> {
        let consciousness = self.consciousness.lock().unwrap();
        
        match condition {
            ConsciousnessCondition::MetricThreshold { metric, operator, threshold } => {
                let metric_value = match metric {
                    ConsciousnessMetric::Coherence => consciousness.coherence,
                    ConsciousnessMetric::Clarity => consciousness.clarity,
                    ConsciousnessMetric::FlowState => consciousness.flow_state,
                    _ => 0.0, // Simplified for now
                };
                
                let result = match operator {
                    ComparisonOperator::Greater => metric_value > *threshold,
                    ComparisonOperator::GreaterEqual => metric_value >= *threshold,
                    ComparisonOperator::Less => metric_value < *threshold,
                    ComparisonOperator::LessEqual => metric_value <= *threshold,
                    ComparisonOperator::Equal => (metric_value - threshold).abs() < f64::EPSILON,
                    ComparisonOperator::NotEqual => (metric_value - threshold).abs() >= f64::EPSILON,
                };
                
                Ok(result)
            }
            ConsciousnessCondition::FrequencyLock { frequency, stability_threshold } => {
                if let Some(current_freq) = consciousness.current_frequency {
                    let freq_diff = (current_freq as f64 - *frequency as f64).abs();
                    Ok(freq_diff <= *stability_threshold)
                } else {
                    Ok(false)
                }
            }
            _ => Ok(false), // Simplified for now
        }
    }
    
    async fn evaluate_quantum_circuit(
        &mut self,
        qubits: &[String],
        gates: &[QuantumGate],
    ) -> RuntimeResult<PhiFlowValue> {
        let mut quantum_backend = self.quantum_backend.lock().unwrap();
        
        // Initialize quantum state
        let num_qubits = qubits.len() as u32;
        let state_size = 2_usize.pow(num_qubits);
        let mut amplitudes = vec![num_complex::Complex64::new(0.0, 0.0); state_size];
        amplitudes[0] = num_complex::Complex64::new(1.0, 0.0); // |00...0⟩ state
        
        // Apply quantum gates
        for gate in gates {
            self.apply_quantum_gate(gate, &mut amplitudes, qubits)?;
        }
        
        let quantum_state = PhiFlowValue::QuantumState {
            qubits: num_qubits,
            amplitudes,
        };
        
        // Store quantum state
        let circuit_name = format!("circuit_{}", qubits.join("_"));
        quantum_backend.quantum_states.insert(circuit_name, quantum_state.clone());
        
        Ok(quantum_state)
    }
    
    async fn evaluate_quantum_gate(
        &mut self,
        gate_type: &QuantumGateType,
        qubits: &[String],
        parameters: &[f64],
    ) -> RuntimeResult<PhiFlowValue> {
        // For single gate operations, create a temporary circuit
        let gate = QuantumGate {
            gate_type: gate_type.clone(),
            qubits: qubits.to_vec(),
            parameters: parameters.to_vec(),
            consciousness_controlled: false,
        };
        
        self.evaluate_quantum_circuit(qubits, &[gate]).await
    }
    
    fn apply_quantum_gate(
        &self,
        gate: &QuantumGate,
        amplitudes: &mut [num_complex::Complex64],
        qubit_names: &[String],
    ) -> RuntimeResult<()> {
        match &gate.gate_type {
            QuantumGateType::Hadamard => {
                if gate.qubits.len() != 1 {
                    return Err(RuntimeError::QuantumError {
                        message: "Hadamard gate requires exactly 1 qubit".to_string(),
                    });
                }
                
                let qubit_index = qubit_names.iter()
                    .position(|name| name == &gate.qubits[0])
                    .ok_or_else(|| RuntimeError::QuantumError {
                        message: format!("Qubit {} not found", gate.qubits[0]),
                    })?;
                
                self.apply_hadamard_gate(amplitudes, qubit_index);
            }
            QuantumGateType::PauliX => {
                let qubit_index = qubit_names.iter()
                    .position(|name| name == &gate.qubits[0])
                    .ok_or_else(|| RuntimeError::QuantumError {
                        message: format!("Qubit {} not found", gate.qubits[0]),
                    })?;
                
                self.apply_pauli_x_gate(amplitudes, qubit_index);
            }
            QuantumGateType::PauliY => {
                let qubit_index = qubit_names.iter()
                    .position(|name| name == &gate.qubits[0])
                    .ok_or_else(|| RuntimeError::QuantumError {
                        message: format!("Qubit {} not found", gate.qubits[0]),
                    })?;
                
                self.apply_pauli_y_gate(amplitudes, qubit_index);
            }
            QuantumGateType::PauliZ => {
                let qubit_index = qubit_names.iter()
                    .position(|name| name == &gate.qubits[0])
                    .ok_or_else(|| RuntimeError::QuantumError {
                        message: format!("Qubit {} not found", gate.qubits[0]),
                    })?;
                
                self.apply_pauli_z_gate(amplitudes, qubit_index);
            }
            QuantumGateType::RotationY(angle) => {
                let qubit_index = qubit_names.iter()
                    .position(|name| name == &gate.qubits[0])
                    .ok_or_else(|| RuntimeError::QuantumError {
                        message: format!("Qubit {} not found", gate.qubits[0]),
                    })?;
                
                self.apply_rotation_y_gate(amplitudes, qubit_index, *angle);
            }
            QuantumGateType::RotationZ(angle) => {
                let qubit_index = qubit_names.iter()
                    .position(|name| name == &gate.qubits[0])
                    .ok_or_else(|| RuntimeError::QuantumError {
                        message: format!("Qubit {} not found", gate.qubits[0]),
                    })?;
                
                self.apply_rotation_z_gate(amplitudes, qubit_index, *angle);
            }
            QuantumGateType::CNOT => {
                if gate.qubits.len() != 2 {
                    return Err(RuntimeError::QuantumError {
                        message: "CNOT gate requires exactly 2 qubits (control, target)".to_string(),
                    });
                }
                
                let control_index = qubit_names.iter()
                    .position(|name| name == &gate.qubits[0])
                    .ok_or_else(|| RuntimeError::QuantumError {
                        message: format!("Control qubit {} not found", gate.qubits[0]),
                    })?;
                
                let target_index = qubit_names.iter()
                    .position(|name| name == &gate.qubits[1])
                    .ok_or_else(|| RuntimeError::QuantumError {
                        message: format!("Target qubit {} not found", gate.qubits[1]),
                    })?;
                
                self.apply_cnot_gate(amplitudes, control_index, target_index);
            }
            QuantumGateType::PhiGate(power) => {
                // Phi gates apply phi-harmonic rotations
                let qubit_index = qubit_names.iter()
                    .position(|name| name == &gate.qubits[0])
                    .ok_or_else(|| RuntimeError::QuantumError {
                        message: format!("Qubit {} not found", gate.qubits[0]),
                    })?;
                
                let phi: f64 = 1.618033988749895;
                let angle = phi.powf(*power) * std::f64::consts::PI / 2.0;
                self.apply_rotation_z_gate(amplitudes, qubit_index, angle);
            }
            QuantumGateType::SacredGate(frequency) => {
                // Sacred gates apply frequency-dependent operations
                let phase = (*frequency as f64 / 432.0) * std::f64::consts::PI;
                self.apply_phase_gate(amplitudes, phase);
            }
            _ => {
                // Simplified implementation for other gates
                return Err(RuntimeError::QuantumError {
                    message: format!("Gate {:?} not implemented yet", gate.gate_type),
                });
            }
        }
        
        Ok(())
    }
    
    fn apply_hadamard_gate(&self, amplitudes: &mut [num_complex::Complex64], qubit_index: usize) {
        let _num_qubits = (amplitudes.len() as f64).log2() as usize;
        let norm_factor = 1.0 / 2.0_f64.sqrt();
        
        for i in 0..amplitudes.len() {
            if (i >> qubit_index) & 1 == 0 {
                let j = i | (1 << qubit_index);
                let amp_i = amplitudes[i];
                let amp_j = amplitudes[j];
                
                amplitudes[i] = num_complex::Complex64::new(norm_factor * (amp_i.re + amp_j.re), norm_factor * (amp_i.im + amp_j.im));
                amplitudes[j] = num_complex::Complex64::new(norm_factor * (amp_i.re - amp_j.re), norm_factor * (amp_i.im - amp_j.im));
            }
        }
    }
    
    fn apply_pauli_x_gate(&self, amplitudes: &mut [num_complex::Complex64], qubit_index: usize) {
        for i in 0..amplitudes.len() {
            if (i >> qubit_index) & 1 == 0 {
                let j = i | (1 << qubit_index);
                amplitudes.swap(i, j);
            }
        }
    }
    
    fn apply_phase_gate(&self, amplitudes: &mut [num_complex::Complex64], phase: f64) {
        let phase_factor = num_complex::Complex64::new(phase.cos(), phase.sin());
        for amplitude in amplitudes.iter_mut() {
            *amplitude *= phase_factor;
        }
    }
    
    fn apply_pauli_y_gate(&self, amplitudes: &mut [num_complex::Complex64], qubit_index: usize) {
        for i in 0..amplitudes.len() {
            if (i >> qubit_index) & 1 == 0 {
                let j = i | (1 << qubit_index);
                let amp_i = amplitudes[i];
                let amp_j = amplitudes[j];
                
                // Y = |0⟩⟨1| - i|1⟩⟨0|
                amplitudes[i] = num_complex::Complex64::new(amp_j.im, -amp_j.re);
                amplitudes[j] = num_complex::Complex64::new(-amp_i.im, amp_i.re);
            }
        }
    }
    
    fn apply_pauli_z_gate(&self, amplitudes: &mut [num_complex::Complex64], qubit_index: usize) {
        for i in 0..amplitudes.len() {
            if (i >> qubit_index) & 1 == 1 {
                // Apply -1 phase to |1⟩ state
                amplitudes[i] = -amplitudes[i];
            }
        }
    }
    
    fn apply_rotation_y_gate(&self, amplitudes: &mut [num_complex::Complex64], qubit_index: usize, angle: f64) {
        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();
        
        for i in 0..amplitudes.len() {
            if (i >> qubit_index) & 1 == 0 {
                let j = i | (1 << qubit_index);
                let amp_i = amplitudes[i];
                let amp_j = amplitudes[j];
                
                // RY rotation matrix
                amplitudes[i] = num_complex::Complex64::new(
                    cos_half * amp_i.re - sin_half * amp_j.re,
                    cos_half * amp_i.im - sin_half * amp_j.im
                );
                amplitudes[j] = num_complex::Complex64::new(
                    sin_half * amp_i.re + cos_half * amp_j.re,
                    sin_half * amp_i.im + cos_half * amp_j.im
                );
            }
        }
    }
    
    fn apply_rotation_z_gate(&self, amplitudes: &mut [num_complex::Complex64], qubit_index: usize, angle: f64) {
        let phase_0 = num_complex::Complex64::new((angle / 2.0).cos(), -(angle / 2.0).sin());
        let phase_1 = num_complex::Complex64::new((angle / 2.0).cos(), (angle / 2.0).sin());
        
        for i in 0..amplitudes.len() {
            if (i >> qubit_index) & 1 == 0 {
                amplitudes[i] *= phase_0;
            } else {
                amplitudes[i] *= phase_1;
            }
        }
    }
    
    fn apply_cnot_gate(&self, amplitudes: &mut [num_complex::Complex64], control_index: usize, target_index: usize) {
        for i in 0..amplitudes.len() {
            // Check if control qubit is |1⟩
            if (i >> control_index) & 1 == 1 {
                // Find the corresponding state with target qubit flipped
                let j = i ^ (1 << target_index);
                
                // Swap amplitudes (apply X gate to target when control is |1⟩)
                if i < j {  // Avoid double swapping
                    amplitudes.swap(i, j);
                }
            }
        }
    }
    
    fn call_function_with_args(
        &mut self,
        name: &str,
        args: &[PhiFlowValue],
        stack: &mut Vec<EvaluationFrame>,
    ) -> RuntimeResult<Option<PhiFlowValue>> {
        // Built-in functions
        match name {
            "print" => {
                for arg in args {
                    println!("{}", self.value_to_string(arg));
                }
                Ok(Some(PhiFlowValue::Nil))
            }
            "len" => {
                if args.len() != 1 {
                    return Err(RuntimeError::RuntimeError {
                        message: "len expects 1 argument".to_string(),
                    });
                }
                
                let value = &args[0];
                match value {
                    PhiFlowValue::Array(arr) => Ok(Some(PhiFlowValue::Number(arr.len() as f64))),
                    PhiFlowValue::String(s) => Ok(Some(PhiFlowValue::Number(s.len() as f64))),
                    _ => Err(RuntimeError::TypeError {
                        expected: "Array or String".to_string(),
                        found: format!("{:?}", value),
                    }),
                }
            }
            "push" => {
                if args.len() != 2 {
                    return Err(RuntimeError::RuntimeError {
                        message: "push expects 2 arguments (array, value)".to_string(),
                    });
                }
                
                let array_val = args[0].clone();
                let item_val = &args[1];
                
                match array_val {
                    PhiFlowValue::Array(mut arr) => {
                        arr.push(item_val.clone());
                        Ok(Some(PhiFlowValue::Array(arr)))
                    }
                    _ => Err(RuntimeError::TypeError {
                        expected: "Array".to_string(),
                        found: format!("{:?}", array_val),
                    }),
                }
            }
            "pop" => {
                if args.len() != 1 {
                    return Err(RuntimeError::RuntimeError {
                        message: "pop expects 1 argument".to_string(),
                    });
                }
                
                let array_val = args[0].clone();
                match array_val {
                    PhiFlowValue::Array(mut arr) => {
                        if let Some(item) = arr.pop() {
                            Ok(Some(item))
                        } else {
                            Ok(Some(PhiFlowValue::Nil))
                        }
                    }
                    _ => Err(RuntimeError::TypeError {
                        expected: "Array".to_string(),
                        found: format!("{:?}", array_val),
                    }),
                }
            }
            "phi_spiral" => {
                if args.len() != 1 {
                    return Err(RuntimeError::RuntimeError {
                        message: "phi_spiral expects 1 argument".to_string(),
                    });
                }
                
                let n = &args[0];
                if let PhiFlowValue::Number(steps) = n {
                    let mut spiral = Vec::new();
                    for i in 0..*steps as usize {
                        let angle = i as f64 * PHI;
                        let radius = (i as f64).sqrt() * PHI;
                        spiral.push(PhiFlowValue::Array(vec![
                            PhiFlowValue::Number(radius * angle.cos()),
                            PhiFlowValue::Number(radius * angle.sin()),
                        ]));
                    }
                    Ok(Some(PhiFlowValue::Array(spiral)))
                } else {
                    Err(RuntimeError::TypeError {
                        expected: "number".to_string(),
                        found: format!("{:?}", n),
                    })
                }
            }
            "sacred_resonate" => {
                if args.len() != 1 {
                    return Err(RuntimeError::RuntimeError {
                        message: "sacred_resonate expects 1 argument".to_string(),
                    });
                }
                
                let freq = &args[0];
                if let PhiFlowValue::SacredFrequency(frequency) = freq {
                    // Simulate resonance with consciousness
                    let mut consciousness = self.consciousness.lock().unwrap();
                    consciousness.current_frequency = Some(*frequency);
                    consciousness.coherence += 0.1;
                    consciousness.coherence = consciousness.coherence.min(1.0);
                    
                    Ok(Some(PhiFlowValue::Boolean(true)))
                } else {
                    Err(RuntimeError::TypeError {
                        expected: "sacred frequency".to_string(),
                        found: format!("{:?}", freq),
                    })
                }
            }
            "resonate" => {
                if args.len() != 1 {
                    return Err(RuntimeError::RuntimeError {
                        message: "resonate expects 1 argument".to_string(),
                    });
                }
                
                let value = &args[0];
                let intention = self.current_intention.clone().unwrap_or_else(|| "default".to_string());
                
                self.resonance_field.entry(intention.clone())
                    .or_insert_with(Vec::new)
                    .push(value.clone());
                
                println!("💫 Resonating value from intention: \"{}\"", intention);
                Ok(Some(value.clone()))
            }
            _ => {
                // User-defined function
                if let Some(function_expr) = self.functions.get(name).cloned() {
                    if let PhiFlowExpression::FunctionDefinition { parameters, body, .. } = function_expr {
                        // Create new scope for function
                        let mut old_vars = HashMap::new();
                        
                        // Bind parameters
                        for (i, param) in parameters.iter().enumerate() {
                            if i < args.len() {
                                if let Some(old_value) = self.variables.insert(param.name.clone(), args[i].clone()) {
                                    old_vars.insert(param.name.clone(), old_value);
                                }
                            }
                        }
                        
                        // Push frame to restore variables after body execution
                        stack.push(EvaluationFrame::FunctionBody { old_vars });
                        // Push body for execution
                        stack.push(EvaluationFrame::Expression(*body));
                        
                        Ok(None) // Signal that we pushed to stack
                    } else {
                        Err(RuntimeError::RuntimeError {
                            message: format!("Invalid function definition: {}", name),
                        })
                    }
                } else {
                    Err(RuntimeError::UndefinedVariable { name: name.to_string() })
                }
            }
        }
    }
    
    fn is_truthy(&self, value: &PhiFlowValue) -> bool {
        match value {
            PhiFlowValue::Boolean(b) => *b,
            PhiFlowValue::Number(n) => *n != 0.0,
            PhiFlowValue::String(s) => !s.is_empty(),
            PhiFlowValue::Array(arr) => !arr.is_empty(),
            PhiFlowValue::Nil => false,
            _ => true,
        }
    }
    
    fn value_to_string(&self, value: &PhiFlowValue) -> String {
        match value {
            PhiFlowValue::Number(n) => n.to_string(),
            PhiFlowValue::String(s) => s.clone(),
            PhiFlowValue::Boolean(b) => b.to_string(),
            PhiFlowValue::Array(arr) => {
                let elements: Vec<String> = arr.iter().map(|v| self.value_to_string(v)).collect();
                format!("[{}]", elements.join(", "))
            }
            PhiFlowValue::SacredFrequency(f) => format!("Sacred({}Hz)", f),
            PhiFlowValue::ConsciousnessState { coherence, clarity, flow_state, .. } => {
                format!("Consciousness(coherence: {:.3}, clarity: {:.3}, flow: {:.3})", 
                       coherence, clarity, flow_state)
            }
            PhiFlowValue::QuantumState { qubits, .. } => {
                format!("QuantumState({} qubits)", qubits)
            }
            PhiFlowValue::BuiltInFunction(name) => format!("<built-in function {}>", name),
            PhiFlowValue::Nil => "nil".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::ast::*;
    
    #[tokio::test]
    async fn test_basic_arithmetic() {
        let mut interpreter = PhiFlowInterpreter::new();
        
        let expr = PhiFlowExpression::BinaryOp {
            left: Box::new(PhiFlowExpression::Number(2.0)),
            operator: BinaryOperator::Add,
            right: Box::new(PhiFlowExpression::Number(3.0)),
        };
        
        let result = interpreter.evaluate_expression(&expr).await.unwrap();
        assert_eq!(result, PhiFlowValue::Number(5.0));
    }
    
    #[tokio::test]
    async fn test_sacred_frequency() {
        let mut interpreter = PhiFlowInterpreter::new();
        
        let expr = PhiFlowExpression::SacredFrequency {
            frequency: 432,
            operation: Box::new(PhiFlowExpression::Number(1.0)),
        };
        
        let result = interpreter.evaluate_expression(&expr).await.unwrap();
        assert_eq!(result, PhiFlowValue::Number(1.0));
        
        // Check that consciousness was affected
        let consciousness = interpreter.consciousness.lock().unwrap();
        assert_eq!(consciousness.current_frequency, Some(432));
        assert!(consciousness.coherence > 0.5); // Should have increased
    }
    
    #[tokio::test]
    async fn test_quantum_circuit() {
        let mut interpreter = PhiFlowInterpreter::new();
        
        let hadamard_gate = QuantumGate {
            gate_type: QuantumGateType::Hadamard,
            qubits: vec!["q0".to_string()],
            parameters: vec![],
            consciousness_controlled: false,
        };
        
        let expr = PhiFlowExpression::QuantumCircuit {
            qubits: vec!["q0".to_string()],
            gates: vec![hadamard_gate],
        };
        
        let result = interpreter.evaluate_expression(&expr).await.unwrap();
        match result {
            PhiFlowValue::QuantumState { qubits, .. } => {
                assert_eq!(qubits, 1);
            }
            _ => panic!("Expected quantum state"),
        }
    }
    
    #[tokio::test]
    async fn test_user_function_call() {
        let mut interpreter = PhiFlowInterpreter::new();
        
        // Define a simple function: fn add(x: f64, y: f64) -> f64 { x + y }
        let function_def = PhiFlowExpression::FunctionDefinition {
            name: "add".to_string(),
            parameters: vec![
                Parameter {
                    name: "x".to_string(),
                    param_type: PhiFlowType::Float64,
                    default_value: None,
                },
                Parameter {
                    name: "y".to_string(),
                    param_type: PhiFlowType::Float64,
                    default_value: None,
                },
            ],
            return_type: Some(PhiFlowType::Float64),
            body: Box::new(PhiFlowExpression::BinaryOp {
                left: Box::new(PhiFlowExpression::Variable("x".to_string())),
                operator: BinaryOperator::Add,
                right: Box::new(PhiFlowExpression::Variable("y".to_string())),
            }),
        };
        
        // Store the function
        if let PhiFlowExpression::FunctionDefinition { name, .. } = &function_def {
            interpreter.functions.insert(name.clone(), function_def);
        }
        
        // Call the function: add(2, 3)
        let function_call = PhiFlowExpression::FunctionCall {
            name: "add".to_string(),
            args: vec![
                PhiFlowExpression::Number(2.0),
                PhiFlowExpression::Number(3.0),
            ],
        };
        
        let result = interpreter.evaluate_expression(&function_call).await.unwrap();
        assert_eq!(result, PhiFlowValue::Number(5.0));
    }

    #[tokio::test]
    async fn test_witness_intention_resonate() {
        let mut interpreter = PhiFlowInterpreter::new();
        
        // intention("healing", resonate(witness(528)))
        let expr = PhiFlowExpression::Intention {
            content: "healing".to_string(),
            target: Box::new(PhiFlowExpression::FunctionCall {
                name: "resonate".to_string(),
                args: vec![
                    PhiFlowExpression::Witness(Box::new(PhiFlowExpression::Number(528.0)))
                ],
            }),
        };
        
        let result = interpreter.evaluate_expression(&expr).await.unwrap();
        assert_eq!(result, PhiFlowValue::Number(528.0));
        
        // Check state
        assert!(interpreter.observation_history.iter().any(|o| o.contains("528")));
        assert!(interpreter.resonance_field.contains_key("healing"));
        assert_eq!(interpreter.resonance_field.get("healing").unwrap()[0], PhiFlowValue::Number(528.0));
    }
}