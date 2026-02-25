//! PhiIR Bytecode Emitter
//!
//! Serializes a `PhiIRProgram` into a self-contained `.phivm` bytecode binary.
//!
//! # Format
//! ```text
//! [MAGIC: 4 bytes "PHIV"]
//! [VERSION: 1 byte]
//! [STRING COUNT: u32 LE]
//! For each string:
//!   [STRING LEN: u32 LE]
//!   [UTF-8 BYTES]
//! [BLOCK COUNT: u32 LE]
//! For each block:
//!   [BLOCK ID: u32 LE]
//!   [INSTR COUNT: u32 LE]
//!   For each instruction:
//!     [RESULT: u8 (0=None, 1=Some)] + [OPERAND: u32 LE if Some]
//!     [OPCODE+PAYLOAD via emit_node]
//!   [TERMINATOR: emit_node]
//! ```

use crate::phi_ir::{PhiIRBinOp, PhiIRNode, PhiIRProgram, PhiIRValue};
use std::collections::HashMap;

// --- Opcodes ---
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
const OP_STREAM: u8 = 0x37;
const OP_DOMAIN_CALL: u8 = 0x40;
// Terminators
const OP_RETURN: u8 = 0xE0;
const OP_JUMP: u8 = 0xE1;
const OP_BRANCH: u8 = 0xE2;
const OP_FALLTHROUGH: u8 = 0xE3;
const OP_BREAK_STREAM: u8 = 0xE4;

const MAGIC: &[u8; 4] = b"PHIV";
const VERSION: u8 = 1;

struct StringInterner {
    table: Vec<String>,
    index_map: HashMap<String, u32>,
}

impl StringInterner {
    fn new() -> Self {
        Self {
            table: Vec::new(),
            index_map: HashMap::new(),
        }
    }

    fn intern(&mut self, value: &str) -> u32 {
        if let Some(index) = self.index_map.get(value) {
            return *index;
        }

        let index = self.table.len() as u32;
        let owned = value.to_string();
        self.table.push(owned.clone());
        self.index_map.insert(owned, index);
        index
    }

    fn index_of(&self, value: &str) -> Option<u32> {
        self.index_map.get(value).copied()
    }
}

struct EmitContext<'a> {
    interner: &'a StringInterner,
    literal_remap: &'a [u32],
}

/// Emit a `PhiIRProgram` to bytecode bytes.
pub fn emit(program: &PhiIRProgram) -> Vec<u8> {
    let interner = build_string_interner(program);
    let literal_remap: Vec<u32> = program
        .string_table
        .iter()
        .map(|value| {
            interner
                .index_of(value)
                .expect("program string must exist in emitted string table")
        })
        .collect();

    let ctx = EmitContext {
        interner: &interner,
        literal_remap: &literal_remap,
    };

    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    out.push(VERSION);

    emit_u32(&mut out, interner.table.len() as u32);
    for value in &interner.table {
        emit_str_bytes(&mut out, value);
    }

    emit_u32(&mut out, program.blocks.len() as u32);

    for block in &program.blocks {
        emit_u32(&mut out, block.id);
        emit_u32(&mut out, block.instructions.len() as u32);

        for instr in &block.instructions {
            match instr.result {
                Some(reg) => {
                    out.push(1u8);
                    emit_u32(&mut out, reg);
                }
                None => out.push(0u8),
            }
            emit_node(&mut out, &instr.node, &ctx);
        }

        emit_node(&mut out, &block.terminator, &ctx);
    }

    out
}

/// Basic disassembly info.
pub fn disassemble(bytes: &[u8]) -> String {
    if bytes.len() < 9 || &bytes[..4] != MAGIC {
        return "Error: invalid magic bytes".to_string();
    }

    let version = bytes[4];
    let mut cursor = 5usize;

    let Some(string_count) = read_u32_at(bytes, &mut cursor) else {
        return "Error: truncated string table header".to_string();
    };

    for _ in 0..string_count {
        let Some(len) = read_u32_at(bytes, &mut cursor) else {
            return "Error: truncated string length".to_string();
        };

        let Some(next) = cursor.checked_add(len as usize) else {
            return "Error: string length overflow".to_string();
        };

        if next > bytes.len() {
            return "Error: truncated string payload".to_string();
        }
        cursor = next;
    }

    let Some(block_count) = read_u32_at(bytes, &mut cursor) else {
        return "Error: truncated block header".to_string();
    };

    format!(
        "PhiVM bytecode v{}\nStrings: {}\nBlocks: {}\nTotal bytes: {}",
        version,
        string_count,
        block_count,
        bytes.len()
    )
}

fn read_u32_at(bytes: &[u8], cursor: &mut usize) -> Option<u32> {
    if bytes.len().saturating_sub(*cursor) < 4 {
        return None;
    }

    let b0 = bytes[*cursor];
    let b1 = bytes[*cursor + 1];
    let b2 = bytes[*cursor + 2];
    let b3 = bytes[*cursor + 3];
    *cursor += 4;
    Some(u32::from_le_bytes([b0, b1, b2, b3]))
}

fn build_string_interner(program: &PhiIRProgram) -> StringInterner {
    let mut interner = StringInterner::new();

    for value in &program.string_table {
        interner.intern(value);
    }

    for block in &program.blocks {
        for instruction in &block.instructions {
            collect_node_strings(&instruction.node, &mut interner);
        }
        collect_node_strings(&block.terminator, &mut interner);
    }

    interner
}

fn collect_node_strings(node: &PhiIRNode, interner: &mut StringInterner) {
    match node {
        PhiIRNode::LoadVar(name)
        | PhiIRNode::Call { name, .. }
        | PhiIRNode::FuncDef { name, .. }
        | PhiIRNode::StreamPush(name)
        | PhiIRNode::IntentionPush { name, .. } => {
            interner.intern(name);
        }

        PhiIRNode::StoreVar { name, .. } => {
            interner.intern(name);
        }

        PhiIRNode::CreatePattern { params, .. } => {
            for (key, _) in params {
                interner.intern(key);
            }
        }

        PhiIRNode::DomainCall { string_args, .. } => {
            for value in string_args {
                interner.intern(value);
            }
        }

        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn emit_u32(out: &mut Vec<u8>, val: u32) {
    out.extend_from_slice(&val.to_le_bytes());
}

fn emit_f64(out: &mut Vec<u8>, val: f64) {
    out.extend_from_slice(&val.to_le_bytes());
}

fn emit_str_bytes(out: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    emit_u32(out, bytes.len() as u32);
    out.extend_from_slice(bytes);
}

fn emit_string_ref(out: &mut Vec<u8>, value: &str, ctx: &EmitContext<'_>) {
    let index = ctx
        .interner
        .index_of(value)
        .expect("string must be interned before emission");
    emit_u32(out, index);
}

fn emit_value(out: &mut Vec<u8>, val: &PhiIRValue, ctx: &EmitContext<'_>) {
    match val {
        PhiIRValue::Number(n) => {
            out.push(OP_CONST_NUM);
            emit_f64(out, *n);
        }
        PhiIRValue::String(idx) => {
            out.push(OP_CONST_STR);
            let remapped = ctx
                .literal_remap
                .get(*idx as usize)
                .copied()
                .unwrap_or(*idx);
            emit_u32(out, remapped);
        }
        PhiIRValue::Boolean(b) => {
            out.push(OP_CONST_BOOL);
            out.push(if *b { 1 } else { 0 });
        }
        PhiIRValue::Void => out.push(OP_CONST_VOID),
    }
}

fn binop_byte(op: &PhiIRBinOp) -> u8 {
    match op {
        PhiIRBinOp::Add => 0x00,
        PhiIRBinOp::Sub => 0x01,
        PhiIRBinOp::Mul => 0x02,
        PhiIRBinOp::Div => 0x03,
        PhiIRBinOp::Mod => 0x04,
        PhiIRBinOp::Pow => 0x05,
        PhiIRBinOp::Eq => 0x06,
        PhiIRBinOp::Neq => 0x07,
        PhiIRBinOp::Lt => 0x08,
        PhiIRBinOp::Lte => 0x09,
        PhiIRBinOp::Gt => 0x0A,
        PhiIRBinOp::Gte => 0x0B,
        PhiIRBinOp::And => 0x0C,
        PhiIRBinOp::Or => 0x0D,
    }
}

fn emit_node(out: &mut Vec<u8>, node: &PhiIRNode, ctx: &EmitContext<'_>) {
    match node {
        PhiIRNode::Nop => out.push(OP_NOP),

        PhiIRNode::Const(val) => emit_value(out, val, ctx),

        PhiIRNode::LoadVar(name) => {
            out.push(OP_LOAD_VAR);
            emit_string_ref(out, name, ctx);
        }

        PhiIRNode::StoreVar { name, value } => {
            out.push(OP_STORE_VAR);
            emit_string_ref(out, name, ctx);
            emit_u32(out, *value);
        }

        PhiIRNode::BinOp { op, left, right } => {
            out.push(OP_BINOP);
            out.push(binop_byte(op));
            emit_u32(out, *left);
            emit_u32(out, *right);
        }

        PhiIRNode::UnaryOp { operand, .. } => {
            out.push(OP_UNOP);
            emit_u32(out, *operand);
        }

        PhiIRNode::Call { name, args } => {
            out.push(OP_CALL);
            emit_string_ref(out, name, ctx);
            emit_u32(out, args.len() as u32);
            for arg in args {
                emit_u32(out, *arg);
            }
        }

        PhiIRNode::ListNew(ops) => {
            out.push(OP_LIST_NEW);
            emit_u32(out, ops.len() as u32);
            for op in ops {
                emit_u32(out, *op);
            }
        }

        PhiIRNode::ListGet { list, index } => {
            out.push(OP_LIST_GET);
            emit_u32(out, *list);
            emit_u32(out, *index);
        }

        PhiIRNode::FuncDef { name, body, .. } => {
            out.push(OP_FUNC_DEF);
            emit_string_ref(out, name, ctx);
            emit_u32(out, *body);
        }

        PhiIRNode::Witness { target, .. } => {
            out.push(OP_WITNESS);
            match target {
                Some(op) => {
                    out.push(1);
                    emit_u32(out, *op);
                }
                None => out.push(0),
            }
        }

        PhiIRNode::IntentionPush { name, .. } => {
            out.push(OP_INTENTION_PUSH);
            emit_string_ref(out, name, ctx);
        }

        PhiIRNode::IntentionPop => out.push(OP_INTENTION_POP),

        PhiIRNode::StreamPush(name) => {
            out.push(OP_STREAM);
            emit_string_ref(out, name, ctx);
        }

        PhiIRNode::StreamPop => out.push(OP_BREAK_STREAM),

        PhiIRNode::Resonate { value, .. } => {
            out.push(OP_RESONATE);
            match value {
                Some(op) => {
                    out.push(1);
                    emit_u32(out, *op);
                }
                None => out.push(0),
            }
        }

        PhiIRNode::CoherenceCheck => out.push(OP_COHERENCE_CHECK),

        PhiIRNode::Sleep { duration } => {
            out.push(OP_SLEEP);
            emit_u32(out, *duration);
        }

        PhiIRNode::CreatePattern {
            frequency, params, ..
        } => {
            out.push(OP_CREATE_PATTERN);
            emit_u32(out, *frequency);
            emit_u32(out, params.len() as u32);
            for (key, val) in params {
                emit_string_ref(out, key, ctx);
                emit_u32(out, *val);
            }
        }

        PhiIRNode::DomainCall {
            args, string_args, ..
        } => {
            out.push(OP_DOMAIN_CALL);
            emit_u32(out, args.len() as u32);
            for arg in args {
                emit_u32(out, *arg);
            }
            emit_u32(out, string_args.len() as u32);
            for value in string_args {
                emit_string_ref(out, value, ctx);
            }
        }

        PhiIRNode::Return(op) => {
            out.push(OP_RETURN);
            emit_u32(out, *op);
        }

        PhiIRNode::Jump(target) => {
            out.push(OP_JUMP);
            emit_u32(out, *target);
        }

        PhiIRNode::Branch {
            condition,
            then_block,
            else_block,
        } => {
            out.push(OP_BRANCH);
            emit_u32(out, *condition);
            emit_u32(out, *then_block);
            emit_u32(out, *else_block);
        }

        PhiIRNode::Fallthrough => out.push(OP_FALLTHROUGH),
    }
}
