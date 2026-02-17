//! PhiFlow Intermediate Representation (IR)
//!
//! A flat, linear representation suitable for:
//! - Stack-based VM execution
//! - WASM code generation
//! - Future LLVM backend
//!
//! The IR preserves PhiFlow's consciousness-aware constructs as first-class
//! opcodes rather than lowering them away. This is intentional: witness,
//! intention, resonate, and coherence are not sugar — they are the language.

pub mod lowering;
pub mod printer;
pub mod vm;

use std::collections::HashMap;

/// A unique label for jump targets and basic block identification.
pub type Label = usize;

/// A unique register/slot identifier for the virtual register file.
pub type Register = usize;

/// The PhiFlow IR instruction set.
///
/// Design principles:
/// - Stack-based for values (push/pop), registers for named bindings
/// - Consciousness opcodes are first-class, not lowered away
/// - Each instruction is independently interpretable (no implicit state)
#[derive(Debug, Clone, PartialEq)]
pub enum Opcode {
    // ═══════════════════════════════════════════════
    // LITERALS — Push values onto the operand stack
    // ═══════════════════════════════════════════════
    /// Push a floating-point number onto the stack.
    PushNumber(f64),

    /// Push a string literal onto the stack.
    PushString(String),

    /// Push a boolean onto the stack.
    PushBool(bool),

    /// Push Void (unit value) onto the stack.
    PushVoid,

    // ═══════════════════════════════════════════════
    // VARIABLES — Named storage via register file
    // ═══════════════════════════════════════════════
    /// Store top-of-stack into a named variable.
    /// The value remains on the stack (non-consuming).
    Store(String),

    /// Load a named variable and push its value onto the stack.
    Load(String),

    // ═══════════════════════════════════════════════
    // ARITHMETIC & LOGIC — Pop operands, push result
    // ═══════════════════════════════════════════════
    /// Pop two values, push their sum.
    Add,
    /// Pop two values, push their difference (left - right).
    Sub,
    /// Pop two values, push their product.
    Mul,
    /// Pop two values, push their quotient (left / right).
    Div,
    /// Pop two values, push left % right.
    Mod,
    /// Pop two values, push left ^ right.
    Pow,

    // ═══════════════════════════════════════════════
    // COMPARISON — Pop two values, push boolean
    // ═══════════════════════════════════════════════
    /// Pop two values, push (left == right).
    Eq,
    /// Pop two values, push (left != right).
    Ne,
    /// Pop two values, push (left < right).
    Lt,
    /// Pop two values, push (left <= right).
    Le,
    /// Pop two values, push (left > right).
    Gt,
    /// Pop two values, push (left >= right).
    Ge,

    // ═══════════════════════════════════════════════
    // LOGICAL — Pop operand(s), push boolean
    // ═══════════════════════════════════════════════
    /// Pop two values, push (left && right).
    And,
    /// Pop two values, push (left || right).
    Or,
    /// Pop one value, push (!value).
    Not,

    /// Pop one value, push (-value).
    Neg,

    // ═══════════════════════════════════════════════
    // CONTROL FLOW — Jumps and branches
    // ═══════════════════════════════════════════════
    /// Unconditional jump to a label.
    Jump(Label),

    /// Pop top-of-stack; jump to label if truthy.
    JumpIfTrue(Label),

    /// Pop top-of-stack; jump to label if falsy.
    JumpIfFalse(Label),

    /// A label marker (no-op, but marks a jump target).
    LabelMark(Label),

    // ═══════════════════════════════════════════════
    // FUNCTIONS — Call stack management
    // ═══════════════════════════════════════════════
    /// Define a function. The body is a separate block of instructions
    /// referenced by name. Parameters are bound from the stack.
    DefineFunction {
        name: String,
        params: Vec<String>,
        body_label: Label,
    },

    /// Call a function by name. Arguments are already on the stack
    /// (pushed left-to-right, so rightmost arg is TOS).
    Call { name: String, arg_count: usize },

    /// Return from the current function. Pops the return value from TOS.
    Return,

    // ═══════════════════════════════════════════════
    // LISTS — Aggregate operations
    // ═══════════════════════════════════════════════
    /// Pop `count` values from stack, push a List containing them.
    MakeList(usize),

    /// Pop an index (TOS) and a list (TOS-1), push list[index].
    ListAccess,

    // ═══════════════════════════════════════════════
    // I/O — Output
    // ═══════════════════════════════════════════════
    /// Pop TOS and print it.
    Print,

    /// Pop TOS and discard it.
    Pop,

    // ═══════════════════════════════════════════════
    // CONSCIOUSNESS — PhiFlow's unique opcodes
    // These exist in no other IR.
    // ═══════════════════════════════════════════════
    /// WITNESS: The program observes itself.
    ///
    /// If `has_expression` is true, pop TOS as the witnessed value.
    /// Calculates current coherence, emits an observation event,
    /// and pushes the coherence score onto the stack.
    ///
    /// If `has_body` is true, the body instructions follow inline
    /// between Witness and WitnessEnd.
    Witness {
        has_expression: bool,
        has_body: bool,
    },

    /// Marks the end of a witness body block.
    WitnessEnd,

    /// INTENTION: Push a named intention onto the intention stack.
    /// The intention context affects coherence scoring.
    /// Body instructions follow inline between IntentionPush and IntentionPop.
    IntentionPush(String),

    /// Pop the current intention from the intention stack.
    /// Reports coherence for the completed intention.
    IntentionPop,

    /// RESONATE: Share a value to the resonance field.
    ///
    /// If `has_expression` is true, pop TOS and share it.
    /// If false, share the current coherence score.
    /// The value is registered under the current intention context.
    Resonate { has_expression: bool },

    /// COHERENCE: Calculate and push the current coherence score.
    ///
    /// This is the live measurement of how well the program is
    /// aligned with its declared intentions and sacred frequencies.
    Coherence,

    /// FREQUENCY CHECK: Pop TOS (a frequency value) and validate it
    /// against the sacred frequency series. Push a boolean result.
    FrequencyCheck,

    // ═══════════════════════════════════════════════
    // PATTERN CREATION — Sacred geometry
    // ═══════════════════════════════════════════════
    /// Create a sacred geometry pattern.
    /// Parameters are read from the stack based on pattern_type.
    CreatePattern {
        pattern_type: String,
        frequency: f64,
    },

    /// Validate a pattern's consciousness alignment.
    /// Pop TOS (pattern), push ValidationResult.
    ValidatePattern { metrics: Vec<String> },

    // ═══════════════════════════════════════════════
    // LOOP SUPPORT
    // ═══════════════════════════════════════════════
    /// Begin a for-loop iteration.
    /// Pop TOS (iterable list), set up iteration state.
    ForLoopInit { variable: String, end_label: Label },

    /// Check if there are more items in the for-loop.
    /// If exhausted, jump to end_label.
    ForLoopNext {
        variable: String,
        body_label: Label,
        end_label: Label,
    },

    // ═══════════════════════════════════════════════
    // STUB MARKERS — For future hardware backends
    // ═══════════════════════════════════════════════
    /// Placeholder for hardware/quantum/bio nodes not yet lowered.
    /// Preserves the original AST node type for future backend work.
    Stub {
        node_type: String,
        description: String,
    },

    /// Halt execution.
    Halt,
}

/// A basic block in the IR — a linear sequence of instructions
/// with a single entry point and optional label.
#[derive(Debug, Clone)]
pub struct BasicBlock {
    pub label: Option<Label>,
    pub instructions: Vec<Opcode>,
}

impl BasicBlock {
    pub fn new() -> Self {
        BasicBlock {
            label: None,
            instructions: Vec::new(),
        }
    }

    pub fn with_label(label: Label) -> Self {
        BasicBlock {
            label: Some(label),
            instructions: Vec::new(),
        }
    }

    pub fn emit(&mut self, op: Opcode) {
        self.instructions.push(op);
    }
}

/// A complete IR program — the output of lowering.
///
/// Contains the main instruction stream (a flat Vec<Opcode>)
/// and a table of function bodies indexed by name.
#[derive(Debug, Clone)]
pub struct IrProgram {
    /// The top-level instruction stream (main program body).
    pub instructions: Vec<Opcode>,

    /// Function bodies, keyed by function name.
    /// Each function is a Vec<Opcode> starting after parameter binding.
    pub functions: HashMap<String, FunctionDef>,

    /// Label counter for generating unique labels.
    pub next_label: Label,
}

/// A function definition in the IR.
#[derive(Debug, Clone)]
pub struct FunctionDef {
    pub name: String,
    pub params: Vec<String>,
    pub body: Vec<Opcode>,
}

impl IrProgram {
    pub fn new() -> Self {
        IrProgram {
            instructions: Vec::new(),
            functions: HashMap::new(),
            next_label: 0,
        }
    }

    /// Allocate a fresh unique label.
    pub fn fresh_label(&mut self) -> Label {
        let label = self.next_label;
        self.next_label += 1;
        label
    }

    /// Emit an instruction to the main program body.
    pub fn emit(&mut self, op: Opcode) {
        self.instructions.push(op);
    }

    /// Total instruction count (main + all functions).
    pub fn total_instructions(&self) -> usize {
        let func_total: usize = self.functions.values().map(|f| f.body.len()).sum();
        self.instructions.len() + func_total
    }
}
