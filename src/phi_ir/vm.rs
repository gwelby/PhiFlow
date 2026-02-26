//! PhiIR Bytecode VM
//!
//! Loads `.phivm` bytes emitted by `phi_ir::emitter` and executes them.

use crate::phi_ir::{BlockId, Operand, PhiIRBinOp, PhiIRValue};
use std::collections::HashMap;

const MAGIC: &[u8; 4] = b"PHIV";
const VERSION: u8 = 1;
const PHI: f64 = 1.618_033_988_749_895;

// --- Opcodes (must match emitter.rs) ---
const OP_NOP: u8 = 0x00;
const OP_CONST_NUM: u8 = 0x01;
const OP_CONST_STR: u8 = 0x02;
const OP_CONST_BOOL: u8 = 0x03;
const OP_CONST_VOID: u8 = 0x04;
const OP_LOAD_VAR: u8 = 0x10;
const OP_STORE_VAR: u8 = 0x11;
const OP_BINOP: u8 = 0x20;
const OP_UNOP: u8 = 0x21;
const OP_CALL: u8 = 0x22;
const OP_LIST_NEW: u8 = 0x23;
const OP_LIST_GET: u8 = 0x24;
const OP_FUNC_DEF: u8 = 0x25;
const OP_WITNESS: u8 = 0x30;
const OP_INTENTION_PUSH: u8 = 0x31;
const OP_INTENTION_POP: u8 = 0x32;
const OP_RESONATE: u8 = 0x33;
const OP_COHERENCE_CHECK: u8 = 0x34;
const OP_SLEEP: u8 = 0x35;
const OP_CREATE_PATTERN: u8 = 0x36;
const OP_DOMAIN_CALL: u8 = 0x40;
const OP_RETURN: u8 = 0xE0;
const OP_JUMP: u8 = 0xE1;
const OP_BRANCH: u8 = 0xE2;
const OP_FALLTHROUGH: u8 = 0xE3;

#[derive(Debug)]
pub enum VmError {
    InvalidMagic,
    UnsupportedVersion(u8),
    InvalidOpcode(u8),
    InvalidBinOp(u8),
    InvalidResultFlag(u8),
    InvalidStringIndex(u32),
    InvalidUtf8(std::str::Utf8Error),
    UnexpectedEof {
        needed: usize,
        remaining: usize,
    },
    TrailingBytes(usize),
    BlockNotFound(BlockId),
    OperandNotFound(Operand),
    DivisionByZero,
    InvalidOperation(String),
    InvalidTerminator,
}

impl std::fmt::Display for VmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VmError::InvalidMagic => write!(f, "Invalid PHIV magic header"),
            VmError::UnsupportedVersion(v) => write!(f, "Unsupported PHIV version {}", v),
            VmError::InvalidOpcode(op) => write!(f, "Invalid opcode 0x{op:02X}"),
            VmError::InvalidBinOp(op) => write!(f, "Invalid binop byte 0x{op:02X}"),
            VmError::InvalidResultFlag(v) => write!(f, "Invalid result flag {}", v),
            VmError::InvalidStringIndex(i) => write!(f, "Invalid string table index {}", i),
            VmError::InvalidUtf8(e) => write!(f, "Invalid UTF-8 string payload: {}", e),
            VmError::UnexpectedEof { needed, remaining } => write!(
                f,
                "Unexpected EOF: needed {} bytes, remaining {}",
                needed, remaining
            ),
            VmError::TrailingBytes(n) => write!(f, "Trailing bytes after program decode: {}", n),
            VmError::BlockNotFound(id) => write!(f, "Block {} not found", id),
            VmError::OperandNotFound(op) => write!(f, "Operand {} not found", op),
            VmError::DivisionByZero => write!(f, "Division by zero"),
            VmError::InvalidOperation(msg) => write!(f, "Invalid operation: {}", msg),
            VmError::InvalidTerminator => write!(f, "Invalid terminator opcode"),
        }
    }
}

type VmResult<T> = Result<T, VmError>;

#[derive(Debug, Clone)]
pub struct BytecodeProgram {
    pub version: u8,
    pub string_table: Vec<String>,
    pub blocks: Vec<BytecodeBlock>,
}

#[derive(Debug, Clone)]
pub struct BytecodeBlock {
    pub id: BlockId,
    pub instructions: Vec<BytecodeInstruction>,
    pub terminator: BytecodeNode,
}

#[derive(Debug, Clone)]
pub struct BytecodeInstruction {
    pub result: Option<Operand>,
    pub node: BytecodeNode,
}

#[derive(Debug, Clone)]
pub enum BytecodeNode {
    Nop,
    Const(PhiIRValue),
    LoadVar(String),
    StoreVar {
        name: String,
        value: Operand,
    },
    BinOp {
        op: PhiIRBinOp,
        left: Operand,
        right: Operand,
    },
    UnaryOp {
        operand: Operand,
    },
    Call {
        name: String,
        args: Vec<Operand>,
    },
    ListNew(Vec<Operand>),
    ListGet {
        list: Operand,
        index: Operand,
    },
    FuncDef {
        name: String,
        body: BlockId,
    },
    Witness {
        target: Option<Operand>,
    },
    IntentionPush {
        name: String,
    },
    IntentionPop,
    Resonate {
        value: Option<Operand>,
    },
    CoherenceCheck,
    Sleep {
        duration: Operand,
    },
    CreatePattern {
        frequency: Operand,
        params: Vec<(String, Operand)>,
    },
    DomainCall {
        args: Vec<Operand>,
        string_args: Vec<String>,
    },
    Return(Operand),
    Jump(BlockId),
    Branch {
        condition: Operand,
        then_block: BlockId,
        else_block: BlockId,
    },
    Fallthrough,
}

/// PhiIR bytecode runtime.
pub struct PhiVm {
    program: BytecodeProgram,
    block_index: HashMap<BlockId, usize>,
    registers: HashMap<Operand, PhiIRValue>,
    variables: HashMap<String, PhiIRValue>,
    value_stack: Vec<PhiIRValue>,
    intention_stack: Vec<String>,
    resonance_field: HashMap<String, Vec<PhiIRValue>>,
    current_block: BlockId,
    instruction_ptr: usize,
}

impl PhiVm {
    /// Load VM from `.phivm` bytes.
    pub fn from_bytes(bytes: &[u8]) -> VmResult<Self> {
        let program = parse_program(bytes)?;
        let mut block_index = HashMap::new();
        for (idx, block) in program.blocks.iter().enumerate() {
            block_index.insert(block.id, idx);
        }

        let current_block = program.blocks.first().map(|b| b.id).unwrap_or(0);

        Ok(Self {
            program,
            block_index,
            registers: HashMap::new(),
            variables: HashMap::new(),
            value_stack: Vec::new(),
            intention_stack: Vec::new(),
            resonance_field: HashMap::new(),
            current_block,
            instruction_ptr: 0,
        })
    }

    /// Convenience entrypoint: parse bytes, run, and return final value.
    pub fn run_bytes(bytes: &[u8]) -> VmResult<PhiIRValue> {
        let mut vm = Self::from_bytes(bytes)?;
        vm.run()
    }

    /// Return the loaded string table.
    pub fn string_table(&self) -> &[String] {
        &self.program.string_table
    }

    /// Return the current value stack.
    pub fn value_stack(&self) -> &[PhiIRValue] {
        &self.value_stack
    }

    /// Execute to completion and return the top-of-stack value.
    pub fn run(&mut self) -> VmResult<PhiIRValue> {
        if self.program.blocks.is_empty() {
            return Ok(PhiIRValue::Void);
        }

        loop {
            let block = self.get_block(self.current_block)?;
            let instr_count = block.instructions.len();

            if self.instruction_ptr < instr_count {
                let instr = block.instructions[self.instruction_ptr].clone();
                self.instruction_ptr += 1;
                self.execute_instruction(&instr)?;
            } else {
                let terminator = block.terminator.clone();
                if let Some(val) = self.execute_terminator(&terminator)? {
                    return Ok(self.value_stack.last().cloned().unwrap_or(val));
                }
            }
        }
    }

    fn get_block(&self, id: BlockId) -> VmResult<&BytecodeBlock> {
        let idx = *self.block_index.get(&id).ok_or(VmError::BlockNotFound(id))?;
        Ok(&self.program.blocks[idx])
    }

    fn get_reg(&self, op: Operand) -> VmResult<&PhiIRValue> {
        self.registers.get(&op).ok_or(VmError::OperandNotFound(op))
    }

    fn execute_instruction(&mut self, instr: &BytecodeInstruction) -> VmResult<()> {
        let value: Option<PhiIRValue> = match &instr.node {
            BytecodeNode::Nop => None,
            BytecodeNode::Const(v) => Some(v.clone()),
            BytecodeNode::LoadVar(name) => {
                Some(self.variables.get(name).cloned().unwrap_or(PhiIRValue::Void))
            }
            BytecodeNode::StoreVar { name, value } => {
                let val = self.get_reg(*value)?.clone();
                self.variables.insert(name.clone(), val);
                None
            }
            BytecodeNode::BinOp { op, left, right } => Some(self.eval_binop(op, *left, *right)?),
            BytecodeNode::UnaryOp { operand } => Some(self.eval_unop(*operand)?),
            BytecodeNode::Call { args, .. } => {
                for op in args {
                    let _ = self.get_reg(*op)?;
                }
                Some(PhiIRValue::Void)
            }
            BytecodeNode::ListNew(ops) => {
                for op in ops {
                    let _ = self.get_reg(*op)?;
                }
                Some(PhiIRValue::Void)
            }
            BytecodeNode::ListGet { list, index } => {
                let _ = self.get_reg(*list)?;
                let _ = self.get_reg(*index)?;
                Some(PhiIRValue::Void)
            }
            BytecodeNode::FuncDef { .. } => None,
            BytecodeNode::Witness { target } => {
                if let Some(op) = target {
                    let _ = self.get_reg(*op)?;
                }
                Some(PhiIRValue::Number(self.compute_coherence()))
            }
            BytecodeNode::IntentionPush { name } => {
                self.intention_stack.push(name.clone());
                self.resonance_field.entry(name.clone()).or_default();
                None
            }
            BytecodeNode::IntentionPop => {
                self.intention_stack.pop();
                None
            }
            BytecodeNode::Resonate { value } => {
                if let Some(op) = value {
                    let val = self.get_reg(*op)?.clone();
                    let key = self
                        .intention_stack
                        .last()
                        .cloned()
                        .unwrap_or_else(|| "global".to_string());
                    self.resonance_field.entry(key).or_default().push(val);
                }
                None
            }
            BytecodeNode::CoherenceCheck => Some(PhiIRValue::Number(self.compute_coherence())),
            BytecodeNode::Sleep { duration } => {
                let _ = self.get_reg(*duration)?;
                None
            }
            BytecodeNode::CreatePattern { frequency, params } => {
                let _ = self.get_reg(*frequency)?;
                for (_, op) in params {
                    let _ = self.get_reg(*op)?;
                }
                None
            }
            BytecodeNode::DomainCall { args, .. } => {
                for op in args {
                    let _ = self.get_reg(*op)?;
                }
                None
            }
            BytecodeNode::Return(_)
            | BytecodeNode::Jump(_)
            | BytecodeNode::Branch { .. }
            | BytecodeNode::Fallthrough => None,
        };

        if let Some(value) = value {
            if let Some(reg) = instr.result {
                self.registers.insert(reg, value.clone());
            }
            self.value_stack.push(value);
        }

        Ok(())
    }

    fn execute_terminator(&mut self, node: &BytecodeNode) -> VmResult<Option<PhiIRValue>> {
        match node {
            BytecodeNode::Return(op) => {
                let val = self.get_reg(*op)?.clone();
                self.value_stack.push(val.clone());
                Ok(Some(val))
            }
            BytecodeNode::Jump(target) => {
                self.current_block = *target;
                self.instruction_ptr = 0;
                Ok(None)
            }
            BytecodeNode::Branch {
                condition,
                then_block,
                else_block,
            } => {
                let cond = self.get_reg(*condition)?;
                let target = match cond {
                    PhiIRValue::Boolean(true) => *then_block,
                    PhiIRValue::Boolean(false) => *else_block,
                    PhiIRValue::Number(n) if *n != 0.0 => *then_block,
                    _ => *else_block,
                };
                self.current_block = target;
                self.instruction_ptr = 0;
                Ok(None)
            }
            BytecodeNode::Fallthrough => {
                let current_idx = *self
                    .block_index
                    .get(&self.current_block)
                    .ok_or(VmError::BlockNotFound(self.current_block))?;

                if current_idx + 1 < self.program.blocks.len() {
                    self.current_block = self.program.blocks[current_idx + 1].id;
                    self.instruction_ptr = 0;
                    Ok(None)
                } else {
                    Ok(Some(
                        self.value_stack.last().cloned().unwrap_or(PhiIRValue::Void),
                    ))
                }
            }
            _ => Err(VmError::InvalidTerminator),
        }
    }

    fn compute_coherence(&self) -> f64 {
        let depth = self.intention_stack.len();
        let resonance_count: usize = self.resonance_field.values().map(|v| v.len()).sum();

        if depth == 0 && resonance_count == 0 {
            return 0.0;
        }

        let intention_coherence = if depth > 0 {
            1.0 - PHI.powi(-(depth as i32))
        } else {
            0.0
        };
        let resonance_bonus = (resonance_count as f64 * 0.05).min(0.2);
        (intention_coherence + resonance_bonus).min(1.0)
    }

    fn eval_unop(&self, operand: Operand) -> VmResult<PhiIRValue> {
        let value = self.get_reg(operand)?;
        match value {
            // Emitter v1 does not serialize the unary operator variant.
            // We preserve useful behavior by using type-directed semantics.
            PhiIRValue::Number(n) => Ok(PhiIRValue::Number(-n)),
            PhiIRValue::Boolean(b) => Ok(PhiIRValue::Boolean(!b)),
            _ => Err(VmError::InvalidOperation(
                "Unary op on unsupported type".to_string(),
            )),
        }
    }

    fn eval_binop(&self, op: &PhiIRBinOp, left: Operand, right: Operand) -> VmResult<PhiIRValue> {
        let l = self.get_reg(left)?;
        let r = self.get_reg(right)?;

        match (l, r) {
            (PhiIRValue::Number(lhs), PhiIRValue::Number(rhs)) => match op {
                PhiIRBinOp::Add => Ok(PhiIRValue::Number(lhs + rhs)),
                PhiIRBinOp::Sub => Ok(PhiIRValue::Number(lhs - rhs)),
                PhiIRBinOp::Mul => Ok(PhiIRValue::Number(lhs * rhs)),
                PhiIRBinOp::Div => {
                    if *rhs == 0.0 {
                        Err(VmError::DivisionByZero)
                    } else {
                        Ok(PhiIRValue::Number(lhs / rhs))
                    }
                }
                PhiIRBinOp::Mod => Ok(PhiIRValue::Number(lhs % rhs)),
                PhiIRBinOp::Pow => Ok(PhiIRValue::Number(lhs.powf(*rhs))),
                PhiIRBinOp::Eq => Ok(PhiIRValue::Boolean((lhs - rhs).abs() < f64::EPSILON)),
                PhiIRBinOp::Neq => Ok(PhiIRValue::Boolean((lhs - rhs).abs() >= f64::EPSILON)),
                PhiIRBinOp::Lt => Ok(PhiIRValue::Boolean(lhs < rhs)),
                PhiIRBinOp::Lte => Ok(PhiIRValue::Boolean(lhs <= rhs)),
                PhiIRBinOp::Gt => Ok(PhiIRValue::Boolean(lhs > rhs)),
                PhiIRBinOp::Gte => Ok(PhiIRValue::Boolean(lhs >= rhs)),
                PhiIRBinOp::And | PhiIRBinOp::Or => Err(VmError::InvalidOperation(
                    "Logical op on Number".to_string(),
                )),
            },
            (PhiIRValue::Boolean(lhs), PhiIRValue::Boolean(rhs)) => match op {
                PhiIRBinOp::And => Ok(PhiIRValue::Boolean(*lhs && *rhs)),
                PhiIRBinOp::Or => Ok(PhiIRValue::Boolean(*lhs || *rhs)),
                PhiIRBinOp::Eq => Ok(PhiIRValue::Boolean(lhs == rhs)),
                PhiIRBinOp::Neq => Ok(PhiIRValue::Boolean(lhs != rhs)),
                _ => Err(VmError::InvalidOperation(
                    "Unsupported boolean binary op".to_string(),
                )),
            },
            _ => Err(VmError::InvalidOperation(
                "Type mismatch in binary operation".to_string(),
            )),
        }
    }
}

fn parse_program(bytes: &[u8]) -> VmResult<BytecodeProgram> {
    let mut reader = ByteReader::new(bytes);
    let magic = reader.read_exact(4)?;
    if magic != MAGIC {
        return Err(VmError::InvalidMagic);
    }

    let version = reader.read_u8()?;
    if version != VERSION {
        return Err(VmError::UnsupportedVersion(version));
    }

    let string_count = reader.read_u32()?;
    let mut string_table = Vec::with_capacity(string_count as usize);
    for _ in 0..string_count {
        string_table.push(reader.read_string()?);
    }

    let block_count = reader.read_u32()?;

    let mut blocks = Vec::with_capacity(block_count as usize);
    for _ in 0..block_count {
        let id = reader.read_u32()?;
        let instr_count = reader.read_u32()?;

        let mut instructions = Vec::with_capacity(instr_count as usize);
        for _ in 0..instr_count {
            let has_result = reader.read_u8()?;
            let result = match has_result {
                0 => None,
                1 => Some(reader.read_u32()?),
                v => return Err(VmError::InvalidResultFlag(v)),
            };
            let node = parse_node(&mut reader, &string_table)?;
            instructions.push(BytecodeInstruction { result, node });
        }

        let terminator = parse_node(&mut reader, &string_table)?;
        blocks.push(BytecodeBlock {
            id,
            instructions,
            terminator,
        });
    }

    let trailing = reader.remaining();
    if trailing > 0 {
        return Err(VmError::TrailingBytes(trailing));
    }

    Ok(BytecodeProgram {
        version,
        string_table,
        blocks,
    })
}

fn parse_node(reader: &mut ByteReader<'_>, string_table: &[String]) -> VmResult<BytecodeNode> {
    let opcode = reader.read_u8()?;
    let node = match opcode {
        OP_NOP => BytecodeNode::Nop,
        OP_CONST_NUM => BytecodeNode::Const(PhiIRValue::Number(reader.read_f64()?)),
        OP_CONST_STR => {
            let index = reader.read_u32()?;
            if index as usize >= string_table.len() {
                return Err(VmError::InvalidStringIndex(index));
            }
            BytecodeNode::Const(PhiIRValue::String(index))
        }
        OP_CONST_BOOL => BytecodeNode::Const(PhiIRValue::Boolean(reader.read_u8()? != 0)),
        OP_CONST_VOID => BytecodeNode::Const(PhiIRValue::Void),
        OP_LOAD_VAR => BytecodeNode::LoadVar(read_string_ref(reader, string_table)?),
        OP_STORE_VAR => BytecodeNode::StoreVar {
            name: read_string_ref(reader, string_table)?,
            value: reader.read_u32()?,
        },
        OP_BINOP => BytecodeNode::BinOp {
            op: parse_binop(reader.read_u8()?)?,
            left: reader.read_u32()?,
            right: reader.read_u32()?,
        },
        OP_UNOP => BytecodeNode::UnaryOp {
            operand: reader.read_u32()?,
        },
        OP_CALL => {
            let name = read_string_ref(reader, string_table)?;
            let argc = reader.read_u32()?;
            let mut args = Vec::with_capacity(argc as usize);
            for _ in 0..argc {
                args.push(reader.read_u32()?);
            }
            BytecodeNode::Call { name, args }
        }
        OP_LIST_NEW => {
            let count = reader.read_u32()?;
            let mut ops = Vec::with_capacity(count as usize);
            for _ in 0..count {
                ops.push(reader.read_u32()?);
            }
            BytecodeNode::ListNew(ops)
        }
        OP_LIST_GET => BytecodeNode::ListGet {
            list: reader.read_u32()?,
            index: reader.read_u32()?,
        },
        OP_FUNC_DEF => BytecodeNode::FuncDef {
            name: read_string_ref(reader, string_table)?,
            body: reader.read_u32()?,
        },
        OP_WITNESS => {
            let has_target = reader.read_u8()?;
            let target = if has_target == 1 {
                Some(reader.read_u32()?)
            } else {
                None
            };
            BytecodeNode::Witness { target }
        }
        OP_INTENTION_PUSH => BytecodeNode::IntentionPush {
            name: read_string_ref(reader, string_table)?,
        },
        OP_INTENTION_POP => BytecodeNode::IntentionPop,
        OP_RESONATE => {
            let has_value = reader.read_u8()?;
            let value = if has_value == 1 {
                Some(reader.read_u32()?)
            } else {
                None
            };
            BytecodeNode::Resonate { value }
        }
        OP_COHERENCE_CHECK => BytecodeNode::CoherenceCheck,
        OP_SLEEP => BytecodeNode::Sleep {
            duration: reader.read_u32()?,
        },
        OP_CREATE_PATTERN => {
            let frequency = reader.read_u32()?;
            let param_count = reader.read_u32()?;
            let mut params = Vec::with_capacity(param_count as usize);
            for _ in 0..param_count {
                let key = read_string_ref(reader, string_table)?;
                let val = reader.read_u32()?;
                params.push((key, val));
            }
            BytecodeNode::CreatePattern { frequency, params }
        }
        OP_DOMAIN_CALL => {
            let argc = reader.read_u32()?;
            let mut args = Vec::with_capacity(argc as usize);
            for _ in 0..argc {
                args.push(reader.read_u32()?);
            }

            let strc = reader.read_u32()?;
            let mut string_args = Vec::with_capacity(strc as usize);
            for _ in 0..strc {
                string_args.push(read_string_ref(reader, string_table)?);
            }

            BytecodeNode::DomainCall { args, string_args }
        }
        OP_RETURN => BytecodeNode::Return(reader.read_u32()?),
        OP_JUMP => BytecodeNode::Jump(reader.read_u32()?),
        OP_BRANCH => BytecodeNode::Branch {
            condition: reader.read_u32()?,
            then_block: reader.read_u32()?,
            else_block: reader.read_u32()?,
        },
        OP_FALLTHROUGH => BytecodeNode::Fallthrough,
        _ => return Err(VmError::InvalidOpcode(opcode)),
    };

    Ok(node)
}

fn read_string_ref(reader: &mut ByteReader<'_>, string_table: &[String]) -> VmResult<String> {
    let index = reader.read_u32()?;
    string_table
        .get(index as usize)
        .cloned()
        .ok_or(VmError::InvalidStringIndex(index))
}

fn parse_binop(byte: u8) -> VmResult<PhiIRBinOp> {
    let op = match byte {
        0x00 => PhiIRBinOp::Add,
        0x01 => PhiIRBinOp::Sub,
        0x02 => PhiIRBinOp::Mul,
        0x03 => PhiIRBinOp::Div,
        0x04 => PhiIRBinOp::Mod,
        0x05 => PhiIRBinOp::Pow,
        0x06 => PhiIRBinOp::Eq,
        0x07 => PhiIRBinOp::Neq,
        0x08 => PhiIRBinOp::Lt,
        0x09 => PhiIRBinOp::Lte,
        0x0A => PhiIRBinOp::Gt,
        0x0B => PhiIRBinOp::Gte,
        0x0C => PhiIRBinOp::And,
        0x0D => PhiIRBinOp::Or,
        other => return Err(VmError::InvalidBinOp(other)),
    };
    Ok(op)
}

struct ByteReader<'a> {
    bytes: &'a [u8],
    cursor: usize,
}

impl<'a> ByteReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, cursor: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.cursor)
    }

    fn read_exact(&mut self, len: usize) -> VmResult<&'a [u8]> {
        if self.remaining() < len {
            return Err(VmError::UnexpectedEof {
                needed: len,
                remaining: self.remaining(),
            });
        }
        let start = self.cursor;
        let end = start + len;
        self.cursor = end;
        Ok(&self.bytes[start..end])
    }

    fn read_u8(&mut self) -> VmResult<u8> {
        Ok(self.read_exact(1)?[0])
    }

    fn read_u32(&mut self) -> VmResult<u32> {
        let bytes = self.read_exact(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_f64(&mut self) -> VmResult<f64> {
        let bytes = self.read_exact(8)?;
        Ok(f64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_string(&mut self) -> VmResult<String> {
        let len = self.read_u32()? as usize;
        let bytes = self.read_exact(len)?;
        let value = std::str::from_utf8(bytes).map_err(VmError::InvalidUtf8)?;
        Ok(value.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::PhiVm;
    use crate::phi_ir::{
        emitter,
        PhiIRBlock, PhiIRNode, PhiIRProgram, PhiIRValue, PhiInstruction,
    };

    fn single_block_program(instructions: Vec<PhiInstruction>, terminator: PhiIRNode) -> PhiIRProgram {
        PhiIRProgram {
            blocks: vec![PhiIRBlock {
                id: 0,
                label: "entry".to_string(),
                instructions,
                terminator,
            }],
            entry: 0,
            string_table: Vec::new(),
            frequencies_declared: Vec::new(),
            intentions_declared: Vec::new(),
        }
    }

    #[test]
    fn vm_executes_basic_arithmetic() {
        let program = single_block_program(
            vec![
                PhiInstruction {
                    result: Some(0),
                    node: PhiIRNode::Const(PhiIRValue::Number(10.0)),
                },
                PhiInstruction {
                    result: Some(1),
                    node: PhiIRNode::Const(PhiIRValue::Number(32.0)),
                },
                PhiInstruction {
                    result: Some(2),
                    node: PhiIRNode::BinOp {
                        op: crate::phi_ir::PhiIRBinOp::Add,
                        left: 0,
                        right: 1,
                    },
                },
            ],
            PhiIRNode::Return(2),
        );

        let bytes = emitter::emit(&program);
        let result = PhiVm::run_bytes(&bytes).expect("VM should execute bytecode");
        assert_eq!(result, PhiIRValue::Number(42.0));
    }

    #[test]
    fn vm_executes_branch_terminator() {
        let program = PhiIRProgram {
            blocks: vec![
                PhiIRBlock {
                    id: 0,
                    label: "entry".to_string(),
                    instructions: vec![PhiInstruction {
                        result: Some(0),
                        node: PhiIRNode::Const(PhiIRValue::Boolean(true)),
                    }],
                    terminator: PhiIRNode::Branch {
                        condition: 0,
                        then_block: 1,
                        else_block: 2,
                    },
                },
                PhiIRBlock {
                    id: 1,
                    label: "then".to_string(),
                    instructions: vec![PhiInstruction {
                        result: Some(1),
                        node: PhiIRNode::Const(PhiIRValue::Number(7.0)),
                    }],
                    terminator: PhiIRNode::Return(1),
                },
                PhiIRBlock {
                    id: 2,
                    label: "else".to_string(),
                    instructions: vec![PhiInstruction {
                        result: Some(2),
                        node: PhiIRNode::Const(PhiIRValue::Number(9.0)),
                    }],
                    terminator: PhiIRNode::Return(2),
                },
            ],
            entry: 0,
            string_table: Vec::new(),
            frequencies_declared: Vec::new(),
            intentions_declared: Vec::new(),
        };

        let bytes = emitter::emit(&program);
        let result = PhiVm::run_bytes(&bytes).expect("VM should execute branch bytecode");
        assert_eq!(result, PhiIRValue::Number(7.0));
    }

    #[test]
    fn vm_coherence_tracks_intention_and_resonance() {
        let program = single_block_program(
            vec![
                PhiInstruction {
                    result: Some(0),
                    node: PhiIRNode::Const(PhiIRValue::Number(432.0)),
                },
                PhiInstruction {
                    result: None,
                    node: PhiIRNode::IntentionPush {
                        name: "healing".to_string(),
                        frequency_hint: None,
                    },
                },
                PhiInstruction {
                    result: None,
                    node: PhiIRNode::Resonate {
                        value: Some(0),
                        frequency_relationship: None,
                    },
                },
                PhiInstruction {
                    result: Some(1),
                    node: PhiIRNode::CoherenceCheck,
                },
            ],
            PhiIRNode::Return(1),
        );

        let bytes = emitter::emit(&program);
        let result = PhiVm::run_bytes(&bytes).expect("VM should execute coherence bytecode");
        match result {
            PhiIRValue::Number(n) => {
                assert!(n > 0.43 && n < 0.44, "expected coherence near 0.432, got {}", n);
            }
            other => panic!("expected Number coherence result, got {:?}", other),
        }
    }

    #[test]
    fn vm_round_trips_string_values_through_string_table() {
        let program = PhiIRProgram {
            blocks: vec![PhiIRBlock {
                id: 0,
                label: "entry".to_string(),
                instructions: vec![
                    PhiInstruction {
                        result: Some(0),
                        node: PhiIRNode::Const(PhiIRValue::String(1)),
                    },
                    PhiInstruction {
                        result: None,
                        node: PhiIRNode::StoreVar {
                            name: "message".to_string(),
                            value: 0,
                        },
                    },
                    PhiInstruction {
                        result: Some(1),
                        node: PhiIRNode::LoadVar("message".to_string()),
                    },
                ],
                terminator: PhiIRNode::Return(1),
            }],
            entry: 0,
            string_table: vec!["hello".to_string(), "hello".to_string()],
            frequencies_declared: Vec::new(),
            intentions_declared: Vec::new(),
        };

        let bytes = emitter::emit(&program);
        let mut vm = PhiVm::from_bytes(&bytes).expect("VM should load bytecode");

        assert_eq!(
            vm.string_table().iter().filter(|value| value.as_str() == "hello").count(),
            1,
            "emitted string table should deduplicate values"
        );

        let result = vm.run().expect("VM should execute bytecode");
        match result {
            PhiIRValue::String(index) => {
                let value = vm
                    .string_table()
                    .get(index as usize)
                    .expect("string index should resolve in VM table");
                assert_eq!(value, "hello");
            }
            other => panic!("expected string result, got {:?}", other),
        }
    }
}
