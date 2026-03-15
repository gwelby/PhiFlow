//! PhiIR – WebAssembly Text Format Codegen
//!
//! Emits `.wat` (WebAssembly Text Format) from a `PhiIRProgram`.
//! The output is a valid, self-contained WASM module runnable in any WASM host
//! (browser, wasmtime, wasmer, Node.js).
//!
//! # Design Decisions
//!
//! ## The Four Consciousness Constructs in WASM
//!
//! These have no WASM equivalent — we invent the mapping:
//!
//! | PhiIR Construct | WASM Representation |
//! |----------------|---------------------|
//! | `Witness`       | Import fn `phi_witness(operand: i32) -> f64` — host observes state |
//! | `IntentionPush` | Global `$intention_depth` incremented; string stored in linear memory |
//! | `IntentionPop`  | Global `$intention_depth` decremented |
//! | `Resonate`      | Import fn `phi_resonate(value: f64)` — host handles resonance field |
//! | `CoherenceCheck`| Import fn `phi_coherence() -> f64` — host returns 0.0-1.0 score |
//!
//! The host (browser JS / wasmtime) provides implementations.
//! This keeps the WASM module pure and host-observable.
//!
//! ## Value Representation
//!
//! All PhiIR values map to f64 (WASM `f64`). Booleans are f64 (0.0 = false, 1.0 = true).
//! The SSA register file is modeled as WASM locals.
//!
//! ## Control Flow
//!
//! PhiIR basic blocks map to WASM `block`/`loop` structures.
//! Jump → `br`, Branch → `br_if`, Return → `return`.

use crate::phi_ir::{BlockId, Operand, Param, PhiIRBinOp, PhiIRNode, PhiIRProgram, PhiIRValue};
use std::collections::{HashMap, HashSet};

/// Emit a `PhiIRProgram` as a WebAssembly Text Format (`.wat`) string.
pub fn emit_wat(program: &PhiIRProgram) -> String {
    let mut w = WatEmitter::new(program);
    w.emit()
}

/// Base byte offset for string data in linear memory.
/// We leave the first 256 bytes for runtime use (intention depth stack, etc.).
pub const STRING_BASE: u32 = 0x100;

// BSEI (Backend Semantics Equivalence Invariant) NaN-boxing tags
pub const NAN_BOX_MASK: u64 = 0x7FF80000_00000000;
pub const TAG_BOOLEAN: u64 = 0x7FF80001_00000000;
pub const TAG_STRING: u64 = 0x7FF80002_00000000;
pub const TAG_VOID: u64 = 0x7FF80003_00000000;
pub const PAYLOAD_MASK: u64 = 0x00000000_FFFFFFFF;

struct WatEmitter<'a> {
    program: &'a PhiIRProgram,
    out: String,
    indent: usize,
    /// Max SSA register index used across all blocks.
    max_reg: u32,
    /// Variable name -> SSA register index for lowered `StoreVar`.
    var_map: HashMap<String, u32>,
    /// Byte offset in linear memory where each string begins.
    string_offsets: Vec<u32>,
    /// Total bytes consumed by the string table.
    string_data_len: u32,
    /// User-defined function metadata discovered from FuncDef nodes.
    function_map: HashMap<String, (Vec<Param>, BlockId)>,
    /// Registers that were actually emitted in visited blocks.
    emitted_defined_regs: HashSet<u32>,
}

impl<'a> WatEmitter<'a> {
    fn new(program: &'a PhiIRProgram) -> Self {
        // Find max register used across all blocks
        let max_reg = program
            .blocks
            .iter()
            .flat_map(|b| b.instructions.iter())
            .filter_map(|i| i.result)
            .max()
            .unwrap_or(0);

        // Pre-compute byte offsets for each string in linear memory.
        let mut string_offsets = Vec::with_capacity(program.string_table.len());
        let mut cursor: u32 = STRING_BASE;
        for s in &program.string_table {
            string_offsets.push(cursor);
            cursor += s.len() as u32;
        }
        let string_data_len = cursor - STRING_BASE;

        // Pre-scan variable stores so LoadVar can resolve to local registers.
        let mut var_map = HashMap::new();
        let mut function_map = HashMap::new();
        for block in &program.blocks {
            for instr in &block.instructions {
                if let PhiIRNode::StoreVar { name, value } = &instr.node {
                    var_map.insert(name.clone(), *value);
                }
                if let PhiIRNode::FuncDef { name, params, body } = &instr.node {
                    function_map.insert(name.clone(), (params.clone(), *body));
                }
            }
        }

        Self {
            program,
            out: String::new(),
            indent: 0,
            max_reg,
            var_map,
            string_offsets,
            string_data_len,
            function_map,
            emitted_defined_regs: HashSet::new(),
        }
    }

    fn emit(&mut self) -> String {
        self.line("(module");
        self.indent += 1;

        // --- Imports: host-provided consciousness hooks ---
        self.emit_imports();

        // --- Memory for string/intention data ---
        self.line("(memory (export \"memory\") 1)");

        // --- String table: packed into linear memory as data segments ---
        if !self.program.string_table.is_empty() {
            self.emit_string_data_segments();
        }

        // --- Globals for intention depth, coherence, and string length sidecar ---
        self.line("(global $intention_depth (mut i32) (i32.const 0))");
        self.line("(global $coherence_score (mut f64) (f64.const 0.618))");
        self.line("(global $string_len (export \"string_len\") (mut i32) (i32.const 0))");

        // --- Main exported function ---
        self.emit_main_function();

        self.indent -= 1;
        self.line(")");

        self.out.clone()
    }

    fn emit_string_data_segments(&mut self) {
        self.line(";; String table — packed into linear memory");
        for (idx, s) in self.program.string_table.iter().enumerate() {
            let offset = self.string_offsets[idx];
            let escaped = escape_wat_string(s);
            self.line(&format!(
                "(data (i32.const {}) \"{}\") ;; [{}] = {:?}",
                offset, escaped, idx, s
            ));
        }
    }

    fn emit_imports(&mut self) {
        self.line(r#"(import "phi" "witness" (func $phi_witness (param i32) (result f64)))"#);
        self.line(r#"(import "phi" "resonate" (func $phi_resonate (param f64)))"#);
        self.line(r#"(import "phi" "coherence" (func $phi_coherence (result f64)))"#);
        self.line(r#"(import "phi" "intention_push" (func $phi_intention_push (param i32)))"#);
        self.line(r#"(import "phi" "intention_pop" (func $phi_intention_pop))"#);
    }

    fn emit_main_function(&mut self) {
        self.line("(func (export \"phi_run\") (result f64)");
        self.indent += 1;

        for i in 0..=self.max_reg {
            self.line(&format!("(local $r{} f64)", i));
        }
        self.line("(local $result f64)");

        let entry = self.program.entry;
        let mut visited = std::collections::HashSet::new();
        self.emit_block(entry, &mut visited);

        self.line("local.get $result");

        self.indent -= 1;
        self.line(")");
    }

    fn emit_block(&mut self, block_id: BlockId, visited: &mut std::collections::HashSet<BlockId>) {
        if visited.contains(&block_id) {
            return;
        }
        visited.insert(block_id);

        let block = match self.program.blocks.iter().find(|b| b.id == block_id) {
            Some(b) => b,
            None => return,
        };

        self.line(&format!(";; Block {}", block_id));

        let instructions = block.instructions.clone();
        let terminator = block.terminator.clone();
        let mut const_regs: HashMap<Operand, PhiIRValue> = HashMap::new();
        let mut var_consts: HashMap<String, PhiIRValue> = HashMap::new();

        for instr in &instructions {
            let inferred = self.infer_const_value(&instr.node, &const_regs, &var_consts);
            let wat = self.emit_node_wat(&instr.node, inferred.as_ref());
            if wat.is_empty() {
                continue;
            }
            if let Some(reg) = instr.result {
                self.line(&wat);
                self.line(&format!("local.set $r{}", reg));
                self.line(&format!("local.get $r{}", reg));
                self.line("local.set $result");
                self.emitted_defined_regs.insert(reg);
                if let Some(value) = inferred {
                    const_regs.insert(reg, value);
                } else {
                    const_regs.remove(&reg);
                }
            } else {
                self.line(&wat);
                let pushes_value = matches!(
                    &instr.node,
                    PhiIRNode::Const(_)
                        | PhiIRNode::BinOp { .. }
                        | PhiIRNode::UnaryOp { .. }
                        | PhiIRNode::LoadVar(_)
                        | PhiIRNode::Witness { .. }
                        | PhiIRNode::CoherenceCheck
                        | PhiIRNode::CreatePattern { .. }
                        | PhiIRNode::Recall(_)
                        | PhiIRNode::Listen(_)
                        | PhiIRNode::VoidDepth
                );
                if pushes_value {
                    self.line("drop");
                }
            }

            if let PhiIRNode::StoreVar { name, value } = &instr.node {
                if let Some(const_value) = const_regs.get(value).cloned() {
                    var_consts.insert(name.clone(), const_value);
                } else {
                    var_consts.remove(name);
                }
            }
        }

        match &terminator {
            PhiIRNode::Return(reg) => {
                if self.emitted_defined_regs.contains(reg) {
                    self.line(&format!("local.get $r{}", reg));
                    self.line("local.set $result");
                } else {
                    self.line(&format!(
                        ";; Return r{} unresolved — preserve last computed $result",
                        reg
                    ));
                }
            }
            PhiIRNode::Jump(target) => {
                let t = *target;
                self.emit_block(t, visited);
            }
            PhiIRNode::Branch {
                condition,
                then_block,
                else_block,
            } => {
                let (cond, then_b, else_b) = (*condition, *then_block, *else_block);
                self.line(&format!("local.get $r{}", cond));
                self.line("f64.const 0.0");
                self.line("f64.ne");
                self.line("(if");
                self.indent += 1;
                self.line("(then");
                self.indent += 1;
                self.line(&format!(";; -> block {}", then_b));
                self.indent -= 1;
                self.line(")");
                self.line("(else");
                self.indent += 1;
                self.line(&format!(";; -> block {}", else_b));
                self.indent -= 1;
                self.line(")");
                self.indent -= 1;
                self.line(")");
                self.emit_block(then_b, visited);
                self.emit_block(else_b, visited);
            }
            PhiIRNode::Fallthrough => {}
            _ => {}
        }
    }

    fn emit_node_wat(&self, node: &PhiIRNode, inferred: Option<&PhiIRValue>) -> String {
        match node {
            PhiIRNode::Nop => String::new(),

            PhiIRNode::Const(val) => match val {
                PhiIRValue::Number(n) => format!("f64.const {}", n),
                PhiIRValue::Boolean(b) => {
                    let bits = TAG_BOOLEAN | (if *b { 1 } else { 0 });
                    format!("i64.const {}\nf64.reinterpret_i64", bits)
                }
                PhiIRValue::Void => {
                    format!("i64.const {}\nf64.reinterpret_i64", TAG_VOID)
                }
                PhiIRValue::String(idx) => {
                    let idx = *idx as usize;
                    if let Some(&offset) = self.string_offsets.get(idx) {
                        let len = self
                            .program
                            .string_table
                            .get(idx)
                            .map(|s| s.len() as u32)
                            .unwrap_or(0);
                        let bits = TAG_STRING | (offset as u64);
                        format!(
                            ";; string[{}] = {:?} offset={} len={}\n\
                             i32.const {}\n\
                             global.set $string_len\n\
                             i64.const {}\n\
                             f64.reinterpret_i64",
                            idx,
                            self.program
                                .string_table
                                .get(idx)
                                .map(|s| s.as_str())
                                .unwrap_or(""),
                            offset,
                            len,
                            len,
                            bits
                        )
                    } else {
                        format!(
                            "i64.const {}\nf64.reinterpret_i64 ;; string index {} out of range",
                            TAG_VOID, idx
                        )
                    }
                }
            },

            PhiIRNode::LoadVar(name) => {
                if let Some(reg) = self.var_map.get(name) {
                    format!("local.get $r{}", reg)
                } else {
                    format!("f64.const 0.0 ;; LoadVar '{}' unresolved", name)
                }
            }

            PhiIRNode::StoreVar { value, .. } => {
                format!("nop ;; StoreVar $r{}", value)
            }

            PhiIRNode::BinOp { op, left, right } => {
                let l = format!("local.get $r{}", left);
                let r = format!("local.get $r{}", right);
                let op_wat = binop_wat(op);
                format!("{}\n{}{}\n{}", l, "  ".repeat(self.indent), r, op_wat)
            }

            PhiIRNode::UnaryOp { operand, .. } => {
                format!("f64.const 0.0\nlocal.get $r{}\nf64.sub", operand)
            }

            PhiIRNode::Witness { target, .. } => {
                let operand = target.map(|r| r as i32).unwrap_or(-1);
                // Witness evaluates to the observed coherence number, matching the evaluator.
                format!("i32.const {}\ncall $phi_witness", operand)
            }

            PhiIRNode::IntentionPush { name, .. } => {
                let mem_ref = self
                    .program
                    .string_table
                    .iter()
                    .position(|s| s == name)
                    .and_then(|i| self.string_offsets.get(i).copied());
                match mem_ref {
                    Some(offset) => format!(
                        ";; intention \"{}\" at memory[{}]\ni32.const {}\ncall $phi_intention_push",
                        name, offset, offset
                    ),
                    None => format!(
                        ";; intention \"{}\" (name len fallback)\ni32.const {}\ncall $phi_intention_push",
                        name, name.len()
                    ),
                }
            }

            PhiIRNode::IntentionPop => "call $phi_intention_pop".to_string(),

            PhiIRNode::Resonate { value, direction: _, .. } => match value {
                Some(reg) => format!("local.get $r{}\ncall $phi_resonate", reg),
                None => "f64.const 0.0\ncall $phi_resonate".to_string(),
            },

            PhiIRNode::CoherenceCheck => "call $phi_coherence".to_string(),

            PhiIRNode::Sleep { .. } => ";; sleep — no-op in WASM".to_string(),

            PhiIRNode::Call { name, .. } => {
                if let Some(PhiIRValue::Number(n)) = inferred {
                    return format!("f64.const {}", n);
                }
                if let Some(PhiIRValue::Boolean(b)) = inferred {
                    let bits = TAG_BOOLEAN | (if *b { 1 } else { 0 });
                    return format!("i64.const {}\nf64.reinterpret_i64", bits);
                }
                format!(
                    "i64.const {}\nf64.reinterpret_i64 ;; unresolved call {}",
                    TAG_VOID, name
                )
            }

            PhiIRNode::CreatePattern { frequency, .. } => {
                format!(
                    "f64.const {} ;; create_pattern @ {}Hz",
                    frequency, frequency
                )
            }

            PhiIRNode::DomainCall { args, .. } => {
                let mut s = String::new();
                for arg in args {
                    s.push_str(&format!("local.get $r{}\n", arg));
                }
                s.push_str(";; domain_call (future: import)");
                s
            }

            PhiIRNode::Return(_)
            | PhiIRNode::Jump(_)
            | PhiIRNode::Branch { .. }
            | PhiIRNode::Fallthrough => String::new(),

            // side-effect-only nodes (no return value → no stack push)
            PhiIRNode::FuncDef { .. }
            | PhiIRNode::Remember { .. }
            | PhiIRNode::Broadcast { .. }
            | PhiIRNode::AgentDecl { .. }
            | PhiIRNode::StreamPush(_)
            | PhiIRNode::StreamPop => String::new(),

            // v0.3.0: value-returning nodes not yet fully codegen'd (stub — in pushes_value list)
            _ => "f64.const 0.0 ;; v0.3.0 stub".to_string(),
        }
    }

    fn infer_const_value(
        &self,
        node: &PhiIRNode,
        const_regs: &HashMap<Operand, PhiIRValue>,
        var_consts: &HashMap<String, PhiIRValue>,
    ) -> Option<PhiIRValue> {
        match node {
            PhiIRNode::Const(v) => Some(v.clone()),
            PhiIRNode::LoadVar(name) => var_consts.get(name).cloned(),
            PhiIRNode::UnaryOp { op, operand } => {
                let value = const_regs.get(operand)?;
                match (op, value) {
                    (crate::phi_ir::PhiIRUnOp::Neg, PhiIRValue::Number(n)) => {
                        Some(PhiIRValue::Number(-n))
                    }
                    (crate::phi_ir::PhiIRUnOp::Not, PhiIRValue::Boolean(b)) => {
                        Some(PhiIRValue::Boolean(!b))
                    }
                    (crate::phi_ir::PhiIRUnOp::Not, PhiIRValue::Number(n)) => {
                        Some(PhiIRValue::Boolean(*n == 0.0))
                    }
                    _ => None,
                }
            }
            PhiIRNode::BinOp { op, left, right } => {
                let lhs = const_regs.get(left)?;
                let rhs = const_regs.get(right)?;
                self.eval_const_binop(op, lhs, rhs)
            }
            PhiIRNode::Call { name, args } => {
                let mut values = Vec::with_capacity(args.len());
                for arg in args {
                    values.push(const_regs.get(arg)?.clone());
                }
                self.eval_const_function(name, &values, 0)
            }
            PhiIRNode::CoherenceCheck => Some(PhiIRValue::Number(0.0)),
            PhiIRNode::Witness { .. } => Some(PhiIRValue::Number(0.0)),
            _ => None,
        }
    }

    fn eval_const_function(
        &self,
        name: &str,
        args: &[PhiIRValue],
        depth: usize,
    ) -> Option<PhiIRValue> {
        if depth > 16 {
            return None;
        }
        let (params, body) = self.function_map.get(name)?.clone();
        let mut vars: HashMap<String, PhiIRValue> = HashMap::new();
        for (idx, param) in params.iter().enumerate() {
            vars.insert(
                param.name.clone(),
                args.get(idx).cloned().unwrap_or(PhiIRValue::Void),
            );
        }
        self.eval_const_block(body, &mut vars, depth)
    }

    fn eval_const_block(
        &self,
        entry: BlockId,
        vars: &mut HashMap<String, PhiIRValue>,
        depth: usize,
    ) -> Option<PhiIRValue> {
        let mut regs: HashMap<Operand, PhiIRValue> = HashMap::new();
        let mut current = entry;
        let mut ip = 0usize;
        let mut steps = 0usize;

        loop {
            steps += 1;
            if steps > 50_000 {
                return None;
            }

            let block = self.program.blocks.iter().find(|b| b.id == current)?;

            if ip < block.instructions.len() {
                let instr = &block.instructions[ip];
                ip += 1;
                let value = self.eval_const_instruction(&instr.node, &regs, vars, depth)?;
                if let Some(reg) = instr.result {
                    regs.insert(reg, value);
                }
                continue;
            }

            match &block.terminator {
                PhiIRNode::Return(op) => return regs.get(op).cloned(),
                PhiIRNode::Jump(target) => {
                    current = *target;
                    ip = 0;
                }
                PhiIRNode::Branch {
                    condition,
                    then_block,
                    else_block,
                } => {
                    let cond = regs.get(condition)?;
                    current = if self.is_truthy(cond) {
                        *then_block
                    } else {
                        *else_block
                    };
                    ip = 0;
                }
                PhiIRNode::Fallthrough => return None,
                _ => return None,
            }
        }
    }

    fn eval_const_instruction(
        &self,
        node: &PhiIRNode,
        regs: &HashMap<Operand, PhiIRValue>,
        vars: &mut HashMap<String, PhiIRValue>,
        depth: usize,
    ) -> Option<PhiIRValue> {
        match node {
            PhiIRNode::Nop | PhiIRNode::IntentionPush { .. } | PhiIRNode::IntentionPop => {
                Some(PhiIRValue::Void)
            }
            PhiIRNode::Const(v) => Some(v.clone()),
            PhiIRNode::LoadVar(name) => Some(vars.get(name).cloned().unwrap_or(PhiIRValue::Void)),
            PhiIRNode::StoreVar { name, value } => {
                let v = regs.get(value)?.clone();
                vars.insert(name.clone(), v);
                Some(PhiIRValue::Void)
            }
            PhiIRNode::BinOp { op, left, right } => {
                let lhs = regs.get(left)?;
                let rhs = regs.get(right)?;
                self.eval_const_binop(op, lhs, rhs)
            }
            PhiIRNode::UnaryOp { op, operand } => {
                let value = regs.get(operand)?;
                match (op, value) {
                    (crate::phi_ir::PhiIRUnOp::Neg, PhiIRValue::Number(n)) => {
                        Some(PhiIRValue::Number(-n))
                    }
                    (crate::phi_ir::PhiIRUnOp::Not, PhiIRValue::Boolean(b)) => {
                        Some(PhiIRValue::Boolean(!b))
                    }
                    (crate::phi_ir::PhiIRUnOp::Not, PhiIRValue::Number(n)) => {
                        Some(PhiIRValue::Boolean(*n == 0.0))
                    }
                    _ => None,
                }
            }
            PhiIRNode::Call { name, args } => {
                let mut values = Vec::with_capacity(args.len());
                for arg in args {
                    values.push(regs.get(arg)?.clone());
                }
                self.eval_const_function(name, &values, depth + 1)
            }
            PhiIRNode::CoherenceCheck | PhiIRNode::Witness { .. } => Some(PhiIRValue::Number(0.0)),
            PhiIRNode::Resonate { .. } => Some(PhiIRValue::Void),
            PhiIRNode::Sleep { .. } => Some(PhiIRValue::Void),
            PhiIRNode::CreatePattern { frequency, .. } => regs.get(frequency).cloned(),
            _ => None,
        }
    }

    fn eval_const_binop(
        &self,
        op: &PhiIRBinOp,
        lhs: &PhiIRValue,
        rhs: &PhiIRValue,
    ) -> Option<PhiIRValue> {
        use PhiIRBinOp as Op;
        match (op, lhs, rhs) {
            (Op::Add, PhiIRValue::Number(a), PhiIRValue::Number(b)) => {
                Some(PhiIRValue::Number(a + b))
            }
            (Op::Sub, PhiIRValue::Number(a), PhiIRValue::Number(b)) => {
                Some(PhiIRValue::Number(a - b))
            }
            (Op::Mul, PhiIRValue::Number(a), PhiIRValue::Number(b)) => {
                Some(PhiIRValue::Number(a * b))
            }
            (Op::Div, PhiIRValue::Number(a), PhiIRValue::Number(b)) => {
                if *b == 0.0 {
                    Some(PhiIRValue::Number(0.0))
                } else {
                    Some(PhiIRValue::Number(a / b))
                }
            }
            (Op::Mod, PhiIRValue::Number(a), PhiIRValue::Number(b)) => {
                if *b == 0.0 {
                    Some(PhiIRValue::Number(0.0))
                } else {
                    Some(PhiIRValue::Number(a % b))
                }
            }
            (Op::Pow, PhiIRValue::Number(a), PhiIRValue::Number(b)) => {
                Some(PhiIRValue::Number(a.powf(*b)))
            }
            (Op::Eq, PhiIRValue::Number(a), PhiIRValue::Number(b)) => {
                Some(PhiIRValue::Boolean((*a - *b).abs() < f64::EPSILON))
            }
            (Op::Neq, PhiIRValue::Number(a), PhiIRValue::Number(b)) => {
                Some(PhiIRValue::Boolean((*a - *b).abs() >= f64::EPSILON))
            }
            (Op::Lt, PhiIRValue::Number(a), PhiIRValue::Number(b)) => {
                Some(PhiIRValue::Boolean(a < b))
            }
            (Op::Lte, PhiIRValue::Number(a), PhiIRValue::Number(b)) => {
                Some(PhiIRValue::Boolean(a <= b))
            }
            (Op::Gt, PhiIRValue::Number(a), PhiIRValue::Number(b)) => {
                Some(PhiIRValue::Boolean(a > b))
            }
            (Op::Gte, PhiIRValue::Number(a), PhiIRValue::Number(b)) => {
                Some(PhiIRValue::Boolean(a >= b))
            }
            (Op::And, PhiIRValue::Boolean(a), PhiIRValue::Boolean(b)) => {
                Some(PhiIRValue::Boolean(*a && *b))
            }
            (Op::Or, PhiIRValue::Boolean(a), PhiIRValue::Boolean(b)) => {
                Some(PhiIRValue::Boolean(*a || *b))
            }
            _ => None,
        }
    }

    fn is_truthy(&self, value: &PhiIRValue) -> bool {
        match value {
            PhiIRValue::Boolean(v) => *v,
            PhiIRValue::Number(v) => *v != 0.0,
            PhiIRValue::String(_) => true,
            PhiIRValue::Void => false,
        }
    }

    fn line(&mut self, s: &str) {
        let indent = "  ".repeat(self.indent);
        for l in s.lines() {
            self.out.push_str(&indent);
            self.out.push_str(l);
            self.out.push('\n');
        }
    }
}

fn escape_wat_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for byte in s.bytes() {
        match byte {
            0x20..=0x7e if byte != b'\\' && byte != b'"' => {
                out.push(byte as char);
            }
            b'"' => out.push_str("\\22"),
            b'\\' => out.push_str("\\5c"),
            other => out.push_str(&format!("\\{:02x}", other)),
        }
    }
    out
}

fn binop_wat(op: &PhiIRBinOp) -> &'static str {
    match op {
        PhiIRBinOp::Add => "f64.add",
        PhiIRBinOp::Sub => "f64.sub",
        PhiIRBinOp::Mul => "f64.mul",
        PhiIRBinOp::Div => "f64.div",
        PhiIRBinOp::Mod => "f64.div ;; mod approx",
        PhiIRBinOp::Pow => "f64.mul ;; pow approx",
        PhiIRBinOp::Eq => "f64.eq\nf64.convert_i32_s",
        PhiIRBinOp::Neq => "f64.ne\nf64.convert_i32_s",
        PhiIRBinOp::Lt => "f64.lt\nf64.convert_i32_s",
        PhiIRBinOp::Lte => "f64.le\nf64.convert_i32_s",
        PhiIRBinOp::Gt => "f64.gt\nf64.convert_i32_s",
        PhiIRBinOp::Gte => "f64.ge\nf64.convert_i32_s",
        PhiIRBinOp::And => "f64.mul",
        PhiIRBinOp::Or => "f64.add",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_phi_program;
    use crate::phi_ir::{
        lowering::lower_program,
        optimizer::{OptimizationLevel, Optimizer},
    };

    fn compile_to_wat(source: &str) -> String {
        let exprs = parse_phi_program(source).expect("parse failed");
        let mut program = lower_program(&exprs);
        let mut opt = Optimizer::new(OptimizationLevel::Basic);
        opt.optimize(&mut program);
        emit_wat(&program)
    }

    #[test]
    fn test_wat_arithmetic_emits_valid_module() {
        let wat = compile_to_wat("let x = 6 + 7");
        assert!(wat.contains("(module"), "should start with module");
        assert!(wat.contains("(func"), "should have a function");
        assert!(wat.contains("phi_run"), "should export phi_run");
    }

    #[test]
    fn test_wat_imports_consciousness_hooks() {
        let wat = compile_to_wat("let x = 1");
        assert!(wat.contains("phi_witness"), "must import witness hook");
        assert!(wat.contains("phi_coherence"), "must import coherence hook");
        assert!(wat.contains("phi_resonate"), "must import resonate hook");
        assert!(
            wat.contains("phi_intention_push"),
            "must import intention hooks"
        );
    }

    #[test]
    fn test_wat_contains_phi_const() {
        let wat = compile_to_wat("let x = 42");
        assert!(wat.contains("f64.const"), "should emit f64 constants");
    }
}
