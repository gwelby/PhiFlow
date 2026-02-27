//! PhiIR Pretty Printer
//!
//! Visualizes the SSA-like IR structure, blocks, and instructions.

use crate::phi_ir::{PhiIRNode, PhiIRProgram};
use std::fmt::Write;

pub struct PhiIRPrinter;

impl PhiIRPrinter {
    pub fn print(program: &PhiIRProgram) -> String {
        let mut output = String::new();

        writeln!(&mut output, "PhiIR Program").unwrap();
        writeln!(&mut output, "=============").unwrap();
        writeln!(&mut output, "Entry Block: {}", program.entry).unwrap();

        if !program.intentions_declared.is_empty() {
            writeln!(&mut output, "Declared Intentions:").unwrap();
            for intent in &program.intentions_declared {
                writeln!(&mut output, "  - {}", intent).unwrap();
            }
        }

        writeln!(&mut output).unwrap();

        for block in &program.blocks {
            writeln!(&mut output, "Block {} ({}):", block.id, block.label).unwrap();

            for (i, instr) in block.instructions.iter().enumerate() {
                let instr_str = format_instr(&instr.node);
                if let Some(res) = instr.result {
                    writeln!(&mut output, "    %{:<3} = {}", res, instr_str).unwrap();
                } else {
                    writeln!(&mut output, "           {}", instr_str).unwrap();
                }
            }

            // Terminator
            writeln!(&mut output, "    ⟶ {}", format_instr(&block.terminator)).unwrap();
            writeln!(&mut output).unwrap();
        }

        output
    }
}

fn format_instr(node: &PhiIRNode) -> String {
    match node {
        PhiIRNode::Const(val) => format!("Const {:?}", val),
        PhiIRNode::LoadVar(name) => format!("LoadVar '{}'", name),
        PhiIRNode::StoreVar { name, value } => format!("StoreVar '{}' = %{}", name, value),
        PhiIRNode::BinOp { op, left, right } => format!("BinOp {:?} %{}, %{}", op, left, right),
        PhiIRNode::UnaryOp { op, operand } => format!("UnaryOp {:?} %{}", op, operand),
        PhiIRNode::Call { name, args } => {
            let arg_str = args
                .iter()
                .map(|a| format!("%{}", a))
                .collect::<Vec<_>>()
                .join(", ");
            format!("Call {}({})", name, arg_str)
        }
        PhiIRNode::Return(op) => format!("Return %{}", op),
        PhiIRNode::ListNew(items) => {
            let item_str = items
                .iter()
                .map(|a| format!("%{}", a))
                .collect::<Vec<_>>()
                .join(", ");
            format!("ListNew [{}]", item_str)
        }
        PhiIRNode::ListGet { list, index } => format!("ListGet %{}[%{}]", list, index),
        PhiIRNode::Witness {
            target,
            collapse_policy,
        } => {
            let t_str = target
                .map(|t| format!("%{}", t))
                .unwrap_or("ALL".to_string());
            format!("Witness target={} policy={:?}", t_str, collapse_policy)
        }
        PhiIRNode::IntentionPush { name, .. } => format!("IntentionPush \"{}\"", name),
        PhiIRNode::IntentionPop => "IntentionPop".to_string(),
        PhiIRNode::Resonate { value, .. } => {
            let v_str = value
                .map(|v| format!("%{}", v))
                .unwrap_or("Self".to_string());
            format!("Resonate value={}", v_str)
        }
        PhiIRNode::CreatePattern {
            kind, frequency, ..
        } => {
            format!("CreatePattern {:?} @ %{}", kind, frequency)
        }
        PhiIRNode::DomainCall { op, args, .. } => {
            let arg_str = args
                .iter()
                .map(|a| format!("%{}", a))
                .collect::<Vec<_>>()
                .join(", ");
            format!("DomainCall {:?}({})", op, arg_str)
        }
        PhiIRNode::Branch {
            condition,
            then_block,
            else_block,
        } => {
            format!(
                "Branch %{} ? Then:{} : Else:{}",
                condition, then_block, else_block
            )
        }
        PhiIRNode::Jump(block) => format!("Jump {}", block),
        PhiIRNode::Sleep { duration } => format!("Sleep %{}", duration),
        PhiIRNode::Fallthrough => "Fallthrough".to_string(),

        _ => format!("{:?}", node),
    }
}
