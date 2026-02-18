//! PhiFlow Intermediate Representation (PhiIR)
//!
//! This module defines the "True IR" for the PhiFlow compiler path.
//! Unlike `src/ir` (which is a stack-based VM for the interpreter),
//! `PhiIR` is an SSA-like, backend-agnostic representation designed for
//! compilation to WASM, Quantum circuits, and Hardware bytecode.
//!
//! Key Design Principles:
//! 1. **Two-Layer Architecture**: Layer 1 (this module) is shared. Layer 2 is backend-lowering.
//! 2. **First-Class Consciousness**: Witness, Intention, Resonate, and Coherence are native nodes.
//! 3. **Backend Agnostic**: No WASM types, no Qubit registers, no HAL traits here.

pub mod evaluator;
pub mod lowering;
pub mod optimizer;
pub mod printer;

use crate::compiler::lexer::Token; // Re-using Token if needed, or defining own types

/// A reference to a computed value (SSA-style)
/// In a BasicBlock, this is the index into the instruction list.
pub type Operand = u32;

/// Basic block identifier
pub type BlockId = u32;

/// Sacred frequency annotation (optional, backends can ignore)
#[derive(Debug, Clone, PartialEq)]
pub enum SacredFrequency {
    Ground,         // 432 Hz
    Creation,       // 528 Hz
    Heart,          // 594 Hz
    Voice,          // 672 Hz
    Vision,         // 720 Hz
    Unity,          // 768 Hz
    Source,         // 963 Hz
    Arbitrary(f64), // non-sacred frequency
}

/// Pattern types that CreatePattern can produce
#[derive(Debug, Clone, PartialEq)]
pub enum PatternKind {
    Spiral,
    Flower,
    DNA,
    Mandelbrot,
    Pentagram,
    SriYantra,
    Golden,
    Fibonacci,
    Heart,
    Toroid,
    Field,
}

/// Measurement strategy hint (quantum backend uses this, others ignore)
#[derive(Debug, Clone, PartialEq)]
pub enum CollapsePolicy {
    MidCircuit,     // measure and continue
    Deferred,       // record but don't collapse until end
    NonDestructive, // measure ancilla, preserve main state
}

/// Domain operations (specialized features beyond the four core constructs)
#[derive(Debug, Clone, PartialEq)]
pub enum DomainOp {
    AudioSynthesize,
    ConsciousnessMonitor,
    ConsciousnessState,
    FrequencyPattern,
    QuantumField,
    BiologicalInterface,
    HardwareSync,
    EmergencyProtocol { interrupt_priority: bool },
    ConsciousnessFlow,
    Validate,
}

/// Binary operators
#[derive(Debug, Clone, PartialEq)]
pub enum PhiIRBinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    Eq,
    Neq,
    Lt,
    Lte,
    Gt,
    Gte,
    And,
    Or,
}

/// Unary operators
#[derive(Debug, Clone, PartialEq)]
pub enum PhiIRUnOp {
    Neg,
    Not,
}

/// Function parameter
#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: String,
    // Type annotation is optional in PhiFlow
}

/// Values that can appear as constants in the IR.
/// Backends lower these to their own representations.
#[derive(Debug, Clone, PartialEq)]
pub enum PhiIRValue {
    Number(f64), // WASM: f64, Hardware: f32, Quantum: f64
    String(u32), // index into string_table
    Boolean(bool),
    Void,
}

/// A wrapper around a PhiIRNode that explicitly tracks the result operand.
#[derive(Debug, Clone, PartialEq)]
pub struct PhiInstruction {
    /// The SSA value defined by this instruction, if any.
    pub result: Option<Operand>,
    /// The operation itself.
    pub node: PhiIRNode,
}

/// One instruction in the PhiIR.
/// Each instruction produces at most one value, referenced by its index (Operand).
#[derive(Debug, Clone, PartialEq)]
pub enum PhiIRNode {
    // --- Primitives ---
    /// No operation (used for padding or deleted instructions)
    Nop,

    // --- Standard Computation ---
    /// Load a constant value
    Const(PhiIRValue),

    /// Read a variable from the environment
    LoadVar(String),

    /// Write a value to a variable
    StoreVar { name: String, value: Operand },

    /// Binary operation
    BinOp {
        op: PhiIRBinOp,
        left: Operand,
        right: Operand,
    },

    /// Unary operation
    UnaryOp { op: PhiIRUnOp, operand: Operand },

    /// Call a user-defined or built-in function
    Call { name: String, args: Vec<Operand> },

    /// Return a value from a function
    Return(Operand),

    /// Create a list
    ListNew(Vec<Operand>),

    /// Index into a list
    ListGet { list: Operand, index: Operand },

    /// Define a function (name, parameters, entry block)
    // Note: In a flat IR, functions are usually top-level, but for simplicity
    // we keep the definition node structure here or move it to Program level.
    // IR_DESIGN suggests this node, but `IrProgram` also has `functions` map.
    // We'll treat this as a directive if encountered in control flow,
    // or rely on `IrProgram` structure.
    FuncDef {
        name: String,
        params: Vec<Param>,
        body: BlockId,
    },

    // --- PhiFlow-Unique Nodes (first-class, never lowered in shared IR) ---
    /// Self-observation. The program pauses to observe its own state.
    /// target: None = observe everything. Some(op) = observe a specific value.
    /// collapse_policy: hint for quantum backend (others ignore).
    Witness {
        target: Option<Operand>,
        collapse_policy: CollapsePolicy,
    },

    /// Enter an intention scope. Pushes intention name onto the stack.
    /// WASM: runtime stack. Quantum: register allocation. Hardware: sensor reconfig.
    IntentionPush {
        name: String,
        frequency_hint: Option<SacredFrequency>,
    },

    /// Exit an intention scope. Pops the current intention.
    IntentionPop,

    /// Share state between intention blocks through resonance.
    /// value: None = share all current scope. Some(op) = share specific value.
    Resonate {
        value: Option<Operand>,
        frequency_relationship: Option<f64>, // phi-harmonic ratio, e.g. 528/432
    },

    /// Evaluate program coherence NOW using backend-appropriate method.
    CoherenceCheck,

    /// Create a sacred geometry pattern at a given frequency.
    CreatePattern {
        kind: PatternKind,
        frequency: Operand,
        annotation: SacredFrequency,
        params: Vec<(String, Operand)>,
    },

    // --- Domain Operations (backend-specific interpretation) ---
    /// Specialized operation. Each backend maps these to its own implementation.
    /// Backends that don't support a domain op emit a warning or no-op.
    DomainCall {
        op: DomainOp,
        args: Vec<Operand>,
        string_args: Vec<String>, // for intention names, device types, etc.
    },

    // --- Control Flow (block terminators) ---
    /// Conditional branch
    Branch {
        condition: Operand,
        then_block: BlockId,
        else_block: BlockId,
    },

    /// Unconditional jump
    Jump(BlockId),

    /// Block terminator: fall through (last instruction in block, no explicit jump)
    Fallthrough,
}

/// A basic block: a named sequence of instructions with a terminator.
#[derive(Debug, Clone)]
pub struct PhiIRBlock {
    pub id: BlockId,
    pub label: String, // human-readable name (e.g., "intention_healing_entry")
    pub instructions: Vec<PhiInstruction>,
    pub terminator: PhiIRNode, // must be Branch, Jump, Return, or Fallthrough
}

/// A complete PhiIR program.
#[derive(Debug, Clone)]
pub struct PhiIRProgram {
    /// All basic blocks in the program, in order.
    pub blocks: Vec<PhiIRBlock>,

    /// Entry block ID.
    pub entry: BlockId,

    /// Interned string table (hardware backend needs this for no-heap operation).
    pub string_table: Vec<String>,

    /// Sacred frequency declarations found in the program.
    pub frequencies_declared: Vec<(SacredFrequency, f64)>,

    /// Intention names declared in the program (for register pre-allocation by quantum).
    pub intentions_declared: Vec<String>,
}

impl PhiIRProgram {
    pub fn new() -> Self {
        PhiIRProgram {
            blocks: Vec::new(),
            entry: 0,
            string_table: Vec::new(),
            frequencies_declared: Vec::new(),
            intentions_declared: Vec::new(),
        }
    }
}
