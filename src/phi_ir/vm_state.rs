//! Serializable evaluator/VM state for yield/resume.
//!
//! This state is captured when execution yields at `witness`, and can be
//! serialized/deserialized across process boundaries before resuming.

use crate::phi_ir::{BlockId, Operand, PhiIRValue};
use std::collections::HashMap;

/// A snapshot of program state recorded each time `witness` executes.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct VmWitnessEvent {
    /// Active intention stack at the moment of observation (innermost last).
    pub intention_stack: Vec<String>,
    /// Phi-harmonic coherence score: 0.0 (no purpose) -> 1.0 (fully aligned).
    pub coherence: f64,
    /// Number of SSA registers live at this point.
    pub register_count: usize,
    /// Total values shared through the resonance field across all intentions.
    pub resonance_count: usize,
}

/// Serializable evaluator state, captured when the program yields.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct VmState {
    pub registers: HashMap<Operand, PhiIRValue>,
    pub variables: HashMap<String, PhiIRValue>,
    pub intention_stack: Vec<String>,
    pub active_streams: Vec<String>,
    pub resonance_field: HashMap<String, Vec<PhiIRValue>>,
    pub resonance_events: Vec<(String, PhiIRValue)>,
    pub ended_streams: Vec<String>,
    pub witness_log: Vec<VmWitnessEvent>,
    pub current_block: BlockId,
    pub instruction_ptr: usize,
}
