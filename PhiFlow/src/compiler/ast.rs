// PhiFlow Abstract Syntax Tree - Represents parsed PhiFlow quantum-consciousness programs
// Supports quantum circuits, sacred frequencies, consciousness bindings, and phi operations

use std::collections::HashMap;

// Main expression types in PhiFlow
#[derive(Debug, Clone, PartialEq)]
pub enum PhiFlowExpression {
    // Quantum Expressions
    QuantumCircuit {
        qubits: Vec<String>,
        gates: Vec<QuantumGate>,
    },
    QuantumGate {
        gate_type: QuantumGateType,
        qubits: Vec<String>,
        parameters: Vec<f64>,
    },
    QuantumMeasurement {
        qubits: Vec<String>,
        classical_bits: Vec<String>,
    },
    
    // Sacred Frequency Expressions
    SacredFrequency {
        frequency: u32,
        operation: Box<PhiFlowExpression>,
    },
    FrequencyLock {
        target_frequency: u32,
        threshold: f64,
        action: Box<PhiFlowExpression>,
    },
    PhiResonance {
        phi_power: f64,
        target: Box<PhiFlowExpression>,
    },
    
    // Consciousness Integration
    ConsciousnessBinding {
        state_name: String,
        expression: Box<PhiFlowExpression>,
    },
    ConsciousnessMonitor {
        metrics: Vec<ConsciousnessMetric>,
        callback: Box<PhiFlowExpression>,
    },
    ConsciousnessCondition {
        condition: ConsciousnessCondition,
        then_branch: Box<PhiFlowExpression>,
        else_branch: Option<Box<PhiFlowExpression>>,
    },
    
    // Consciousness State Constructs
    Witness(Box<PhiFlowExpression>),
    Intention {
        content: String,
        target: Box<PhiFlowExpression>,
    },
    
    // Standard Language Constructs
    Variable(String),
    Number(f64),
    String(String),
    Boolean(bool),
    Array(Vec<PhiFlowExpression>),
    ArrayIndex {
        array: Box<PhiFlowExpression>,
        index: Box<PhiFlowExpression>,
    },
    
    // Function Expressions
    FunctionCall {
        name: String,
        args: Vec<PhiFlowExpression>,
    },
    FunctionDefinition {
        name: String,
        parameters: Vec<Parameter>,
        return_type: Option<PhiFlowType>,
        body: Box<PhiFlowExpression>,
    },
    
    // Control Flow
    If {
        condition: Box<PhiFlowExpression>,
        then_branch: Box<PhiFlowExpression>,
        else_branch: Option<Box<PhiFlowExpression>>,
    },
    Match {
        expression: Box<PhiFlowExpression>,
        arms: Vec<MatchArm>,
    },
    For {
        variable: String,
        iterable: Box<PhiFlowExpression>,
        body: Box<PhiFlowExpression>,
    },
    While {
        condition: Box<PhiFlowExpression>,
        body: Box<PhiFlowExpression>,
    },
    
    // Binary and Unary Operations
    BinaryOp {
        left: Box<PhiFlowExpression>,
        operator: BinaryOperator,
        right: Box<PhiFlowExpression>,
    },
    UnaryOp {
        operator: UnaryOperator,
        operand: Box<PhiFlowExpression>,
    },
    
    // Variable Assignment
    Let {
        variable: String,
        type_annotation: Option<PhiFlowType>,
        value: Box<PhiFlowExpression>,
    },
    
    // Block Expression
    Block(Vec<PhiFlowExpression>),
    
    // Return Statement
    Return(Option<Box<PhiFlowExpression>>),
}

// Quantum gate types supported in PhiFlow
#[derive(Debug, Clone, PartialEq)]
pub enum QuantumGateType {
    // Single-qubit gates
    Hadamard,           // H
    PauliX,            // X
    PauliY,            // Y
    PauliZ,            // Z
    
    // Rotation gates
    RotationX(f64),    // RX(angle)
    RotationY(f64),    // RY(angle)
    RotationZ(f64),    // RZ(angle)
    
    // Two-qubit gates
    CNOT,              // CNOT(control, target)
    CZ,                // CZ(control, target)
    SWAP,              // SWAP(qubit1, qubit2)
    
    // PhiFlow-specific gates
    PhiGate(f64),      // Φ^n gate with phi power
    SacredGate(u32),   // Gate tuned to sacred frequency
    ConsciousnessGate { // Gate controlled by consciousness state
        state_binding: String,
        gate_type: Box<QuantumGateType>,
    },
}

// Quantum gate representation
#[derive(Debug, Clone, PartialEq)]
pub struct QuantumGate {
    pub gate_type: QuantumGateType,
    pub qubits: Vec<String>,
    pub parameters: Vec<f64>,
    pub consciousness_controlled: bool,
}

// Consciousness metrics that can be monitored
#[derive(Debug, Clone, PartialEq)]
pub enum ConsciousnessMetric {
    Coherence,
    Clarity,
    FlowState,
    SacredFrequency(u32),
    PhiResonance,
    EEGChannel(String), // TP9, AF7, AF8, TP10
    BrainwaveType(BrainwaveType),
}

// Types of brainwaves
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BrainwaveType {
    Delta,   // 0.5-4 Hz
    Theta,   // 4-8 Hz
    Alpha,   // 8-13 Hz
    Beta,    // 13-30 Hz
    Gamma,   // 30-100 Hz
}

// Consciousness conditions for control flow
#[derive(Debug, Clone, PartialEq)]
pub enum ConsciousnessCondition {
    MetricThreshold {
        metric: ConsciousnessMetric,
        operator: ComparisonOperator,
        threshold: f64,
    },
    FrequencyLock {
        frequency: u32,
        stability_threshold: f64,
    },
    StateMatch {
        target_state: String,
    },
    Composite {
        conditions: Vec<ConsciousnessCondition>,
        operator: LogicalOperator,
    },
}

// PhiFlow type system
#[derive(Debug, Clone, PartialEq)]
pub enum PhiFlowType {
    // Quantum types
    Qubit,
    QuantumCircuit,
    QuantumState,
    QuantumGate,
    
    // Sacred frequency types
    SacredFrequency(u32),
    PhiResonance,
    FrequencyRange { min: u32, max: u32 },
    
    // Consciousness types
    ConsciousnessState,
    EEGData,
    BrainwaveData(BrainwaveType),
    
    // Standard types
    Float64,
    Integer,
    String,
    Boolean,
    Array(Box<PhiFlowType>),
    
    // Function types
    Function {
        parameters: Vec<PhiFlowType>,
        return_type: Box<PhiFlowType>,
    },
    
    // Custom types
    Custom(String),
}

// Function parameters
#[derive(Debug, Clone, PartialEq)]
pub struct Parameter {
    pub name: String,
    pub param_type: PhiFlowType,
    pub default_value: Option<PhiFlowExpression>,
}

// Match arms for pattern matching
#[derive(Debug, Clone, PartialEq)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub guard: Option<PhiFlowExpression>,
    pub body: PhiFlowExpression,
}

// Patterns for matching
#[derive(Debug, Clone, PartialEq)]
pub enum Pattern {
    Variable(String),
    Number(f64),
    String(String),
    Boolean(bool),
    SacredFrequency(u32),
    ConsciousnessState(String),
    Array(Vec<Pattern>),
    Wildcard,
}

// Binary operators
#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOperator {
    // Arithmetic
    Add, Subtract, Multiply, Divide, Modulo, Power,
    
    // Comparison
    Equal, NotEqual, Less, Greater, LessEqual, GreaterEqual,
    
    // Logical
    And, Or,
    
    // PhiFlow-specific
    PhiMultiply,      // Multiply by phi ratio
    SacredResonance,  // Check sacred frequency resonance
    QuantumEntangle,  // Entangle quantum states
}

// Unary operators
#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOperator {
    Negate,
    Not,
    PhiTransform,     // Apply phi transformation
    QuantumMeasure,   // Measure quantum state
}

// Comparison operators for consciousness conditions
#[derive(Debug, Clone, PartialEq)]
pub enum ComparisonOperator {
    Equal,
    NotEqual,
    Less,
    Greater,
    LessEqual,
    GreaterEqual,
}

// Logical operators for combining conditions
#[derive(Debug, Clone, PartialEq)]
pub enum LogicalOperator {
    And,
    Or,
    Not,
}

// Program structure
#[derive(Debug, Clone, PartialEq)]
pub struct PhiFlowProgram {
    pub functions: Vec<PhiFlowExpression>, // Function definitions
    pub main: Option<PhiFlowExpression>,   // Main function
    pub imports: Vec<String>,              // Imported modules
    pub consciousness_config: Option<ConsciousnessConfig>,
    pub quantum_config: Option<QuantumConfig>,
}

// Configuration for consciousness monitoring
#[derive(Debug, Clone, PartialEq)]
pub struct ConsciousnessConfig {
    pub device: String,                    // MUSE device identifier
    pub sampling_rate: u32,                // EEG sampling rate
    pub monitored_metrics: Vec<ConsciousnessMetric>,
    pub sacred_frequencies: Vec<u32>,      // Frequencies to monitor
    pub thresholds: HashMap<String, f64>,  // Metric thresholds
}

// Configuration for quantum backend
#[derive(Debug, Clone, PartialEq)]
pub struct QuantumConfig {
    pub backend_type: String,              // "simulator", "ibm", etc.
    pub max_qubits: u32,                   // Maximum qubits available
    pub api_token: Option<String>,         // API token for cloud providers
    pub optimization_level: u32,           // Circuit optimization level
}

// Visitor trait for AST traversal
pub trait PhiFlowVisitor<T> {
    fn visit_expression(&mut self, expr: &PhiFlowExpression) -> T;
    fn visit_quantum_gate(&mut self, gate: &QuantumGate) -> T;
    fn visit_consciousness_condition(&mut self, condition: &ConsciousnessCondition) -> T;
    fn visit_type(&mut self, phi_type: &PhiFlowType) -> T;
}

// Implementation helpers
impl PhiFlowExpression {
    pub fn get_type(&self) -> PhiFlowType {
        match self {
            PhiFlowExpression::Number(_) => PhiFlowType::Float64,
            PhiFlowExpression::String(_) => PhiFlowType::String,
            PhiFlowExpression::Boolean(_) => PhiFlowType::Boolean,
            PhiFlowExpression::QuantumCircuit { .. } => PhiFlowType::QuantumCircuit,
            PhiFlowExpression::SacredFrequency { frequency, .. } => PhiFlowType::SacredFrequency(*frequency),
            PhiFlowExpression::ConsciousnessBinding { .. } => PhiFlowType::ConsciousnessState,
            PhiFlowExpression::Array(elements) => {
                if let Some(first) = elements.first() {
                    PhiFlowType::Array(Box::new(first.get_type()))
                } else {
                    PhiFlowType::Array(Box::new(PhiFlowType::Custom("empty".to_string())))
                }
            }
            _ => PhiFlowType::Custom("unknown".to_string()),
        }
    }
    
    pub fn is_quantum_expression(&self) -> bool {
        matches!(self, 
            PhiFlowExpression::QuantumCircuit { .. } |
            PhiFlowExpression::QuantumGate { .. } |
            PhiFlowExpression::QuantumMeasurement { .. }
        )
    }
    
    pub fn is_consciousness_expression(&self) -> bool {
        matches!(self,
            PhiFlowExpression::ConsciousnessBinding { .. } |
            PhiFlowExpression::ConsciousnessMonitor { .. } |
            PhiFlowExpression::ConsciousnessCondition { .. } |
            PhiFlowExpression::Witness(_) |
            PhiFlowExpression::Intention { .. }
        )
    }
    
    pub fn is_sacred_frequency_expression(&self) -> bool {
        matches!(self,
            PhiFlowExpression::SacredFrequency { .. } |
            PhiFlowExpression::FrequencyLock { .. } |
            PhiFlowExpression::PhiResonance { .. }
        )
    }
}

impl QuantumGateType {
    pub fn parameter_count(&self) -> usize {
        match self {
            QuantumGateType::Hadamard | QuantumGateType::PauliX | 
            QuantumGateType::PauliY | QuantumGateType::PauliZ => 0,
            QuantumGateType::RotationX(_) | QuantumGateType::RotationY(_) | 
            QuantumGateType::RotationZ(_) | QuantumGateType::PhiGate(_) => 1,
            QuantumGateType::CNOT | QuantumGateType::CZ | QuantumGateType::SWAP => 0,
            QuantumGateType::SacredGate(_) => 0,
            QuantumGateType::ConsciousnessGate { .. } => 0,
        }
    }
    
    pub fn qubit_count(&self) -> usize {
        match self {
            QuantumGateType::Hadamard | QuantumGateType::PauliX | 
            QuantumGateType::PauliY | QuantumGateType::PauliZ |
            QuantumGateType::RotationX(_) | QuantumGateType::RotationY(_) | 
            QuantumGateType::RotationZ(_) | QuantumGateType::PhiGate(_) |
            QuantumGateType::SacredGate(_) => 1,
            QuantumGateType::CNOT | QuantumGateType::CZ | QuantumGateType::SWAP => 2,
            QuantumGateType::ConsciousnessGate { gate_type, .. } => gate_type.qubit_count(),
        }
    }
}

// Sacred frequency constants
pub const SACRED_FREQUENCIES: &[u32] = &[
    432,  // Earth frequency
    528,  // Love frequency
    594,  // Transformation
    639,  // Connection
    693,  // Expression
    741,  // Intuition
    852,  // Spiritual order
    963,  // Unity
];

// Phi constant for mathematical operations
pub const PHI: f64 = 1.618033988749895;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_quantum_gate_creation() {
        let hadamard = QuantumGate {
            gate_type: QuantumGateType::Hadamard,
            qubits: vec!["q0".to_string()],
            parameters: vec![],
            consciousness_controlled: false,
        };
        
        assert_eq!(hadamard.gate_type.qubit_count(), 1);
        assert_eq!(hadamard.gate_type.parameter_count(), 0);
    }
    
    #[test]
    fn test_sacred_frequency_expression() {
        let freq_expr = PhiFlowExpression::SacredFrequency {
            frequency: 528,
            operation: Box::new(PhiFlowExpression::Number(1.0)),
        };
        
        assert!(freq_expr.is_sacred_frequency_expression());
        assert_eq!(freq_expr.get_type(), PhiFlowType::SacredFrequency(528));
    }
    
    #[test]
    fn test_consciousness_condition() {
        let condition = ConsciousnessCondition::MetricThreshold {
            metric: ConsciousnessMetric::Coherence,
            operator: ComparisonOperator::Greater,
            threshold: 0.9,
        };
        
        match condition {
            ConsciousnessCondition::MetricThreshold { threshold, .. } => {
                assert_eq!(threshold, 0.9);
            }
            _ => panic!("Wrong condition type"),
        }
    }
} 