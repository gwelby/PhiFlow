//! **DEPRECATED** — This module has been moved to `src/_archive/`.
//! It is superseded by the PhiIR pipeline (`src/phi_ir/`) and is no longer compiled.
//! Kept for historical reference only.
//!
// PhiFlow Compiler Module
// Exports lexer, parser, and AST components

pub mod ast;
pub mod lexer;
pub mod parser;

pub use ast::{PhiFlowExpression, PhiFlowProgram, QuantumGate, QuantumGateType};
pub use lexer::{PhiFlowLexer, Token};
pub use parser::{ParseError, PhiFlowParser};
