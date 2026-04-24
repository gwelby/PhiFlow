use crate::phi_ir::{Operand, PhiIRNode, PhiIRProgram, PhiIRValue};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum QuantumOverlayPlan {
    LegacyFreqChains(LegacyFreqChainPlan),
    ContradictionLadder(ContradictionLadderPlan),
}

#[derive(Debug, Clone, PartialEq)]
pub struct LegacyFreqChainPlan {
    pub frequencies: Vec<FrequencyChain>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FrequencyChain {
    pub frequency_hz: u32,
    pub operands: Vec<Operand>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ContradictionLadderPlan {
    pub left_lane: Vec<Operand>,
    pub right_lane: Vec<Operand>,
    pub final_merge: Operand,
    pub depth: usize,
    pub witness_target: Option<Operand>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantumOverlayError {
    UnsupportedCallShape(String),
    MixedQuantumAndNonQuantumOperand(Operand),
    MissingOperandProducer(Operand),
    NoContradictionPatternFound,
}

impl fmt::Display for QuantumOverlayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuantumOverlayError::UnsupportedCallShape(message) => f.write_str(message),
            QuantumOverlayError::MixedQuantumAndNonQuantumOperand(op) => {
                write!(f, "operand `{op}` does not resolve to a quantumizable numeric lineage")
            }
            QuantumOverlayError::MissingOperandProducer(op) => {
                write!(f, "missing producer for operand `{op}`")
            }
            QuantumOverlayError::NoContradictionPatternFound => {
                f.write_str("no contradiction ladder or legacy entangle chain found")
            }
        }
    }
}

impl Error for QuantumOverlayError {}

pub fn analyze_quantum_overlay(
    ir: &PhiIRProgram,
) -> Result<QuantumOverlayPlan, QuantumOverlayError> {
    let index = ProgramIndex::build(ir);

    if let Some(plan) = detect_contradiction_ladder(ir, &index)? {
        return Ok(QuantumOverlayPlan::ContradictionLadder(plan));
    }

    let legacy = detect_legacy_freq_chains(ir);
    if !legacy.frequencies.is_empty() {
        return Ok(QuantumOverlayPlan::LegacyFreqChains(legacy));
    }

    Err(QuantumOverlayError::NoContradictionPatternFound)
}

#[derive(Default)]
struct ProgramIndex {
    producers: HashMap<Operand, PhiIRNode>,
    variable_values: HashMap<String, Operand>,
}

impl ProgramIndex {
    fn build(ir: &PhiIRProgram) -> Self {
        let mut producers = HashMap::new();
        let mut variable_values = HashMap::new();

        for block in &ir.blocks {
            for instruction in &block.instructions {
                if let Some(result) = instruction.result {
                    producers.insert(result, instruction.node.clone());
                }
                if let PhiIRNode::StoreVar { name, value } = &instruction.node {
                    variable_values.insert(name.clone(), *value);
                }
            }
        }

        Self {
            producers,
            variable_values,
        }
    }

    fn producer(&self, operand: Operand) -> Result<&PhiIRNode, QuantumOverlayError> {
        self.producers
            .get(&operand)
            .ok_or(QuantumOverlayError::MissingOperandProducer(operand))
    }

    fn resolve_operand(&self, operand: Operand) -> Result<Operand, QuantumOverlayError> {
        let mut current = operand;
        let mut seen = HashSet::new();
        loop {
            if !seen.insert(current) {
                return Err(QuantumOverlayError::UnsupportedCallShape(format!(
                    "cyclic operand resolution while tracing operand `{operand}`"
                )));
            }
            match self.producer(current)? {
                PhiIRNode::LoadVar(name) => {
                    current = *self
                        .variable_values
                        .get(name)
                        .ok_or(QuantumOverlayError::MissingOperandProducer(current))?;
                }
                _ => return Ok(current),
            }
        }
    }

    fn resolve_numeric_operand(&self, operand: Operand) -> Result<Operand, QuantumOverlayError> {
        let resolved = self.resolve_operand(operand)?;
        match self.producer(resolved)? {
            PhiIRNode::Const(PhiIRValue::Number(_))
            | PhiIRNode::Call { .. } => Ok(resolved),
            _ => Err(QuantumOverlayError::MixedQuantumAndNonQuantumOperand(operand)),
        }
    }

    fn number_value(&self, operand: Operand) -> Result<f64, QuantumOverlayError> {
        let resolved = self.resolve_operand(operand)?;
        match self.producer(resolved)? {
            PhiIRNode::Const(PhiIRValue::Number(value)) => Ok(*value),
            _ => Err(QuantumOverlayError::MixedQuantumAndNonQuantumOperand(operand)),
        }
    }

    fn coherence_args(&self, operand: Operand) -> Result<(Operand, Operand), QuantumOverlayError> {
        let resolved = self.resolve_operand(operand)?;
        match self.producer(resolved)? {
            PhiIRNode::Call { name, args } if name == "coherence" && args.len() == 2 => Ok((
                self.resolve_numeric_operand(args[0])?,
                self.resolve_numeric_operand(args[1])?,
            )),
            PhiIRNode::Call { name, .. } => Err(QuantumOverlayError::UnsupportedCallShape(
                format!("unsupported call shape `{name}` while tracing operand `{operand}`"),
            )),
            other => Err(QuantumOverlayError::UnsupportedCallShape(format!(
                "operand `{operand}` does not resolve to a quantumizable call: {other:?}"
            ))),
        }
    }
}

fn detect_legacy_freq_chains(ir: &PhiIRProgram) -> LegacyFreqChainPlan {
    let mut chains = BTreeMap::<u32, Vec<Operand>>::new();
    for block in &ir.blocks {
        for instruction in &block.instructions {
            if let (Some(result), PhiIRNode::Entangle(freq)) = (instruction.result, &instruction.node)
            {
                chains
                    .entry(freq.round() as u32)
                    .or_default()
                    .push(result);
            }
        }
    }

    LegacyFreqChainPlan {
        frequencies: chains
            .into_iter()
            .map(|(frequency_hz, operands)| FrequencyChain {
                frequency_hz,
                operands,
            })
            .collect(),
    }
}

fn detect_contradiction_ladder(
    ir: &PhiIRProgram,
    index: &ProgramIndex,
) -> Result<Option<ContradictionLadderPlan>, QuantumOverlayError> {
    let mut left = BTreeMap::<usize, Operand>::new();
    let mut right = BTreeMap::<usize, Operand>::new();
    let mut final_merge = None;
    let mut witness_targets = Vec::new();

    for block in &ir.blocks {
        for instruction in &block.instructions {
            match &instruction.node {
                PhiIRNode::StoreVar { name, value } => {
                    if let Some(level) = parse_lane_name(name, 'l') {
                        left.insert(level, index.resolve_operand(*value)?);
                    } else if let Some(level) = parse_lane_name(name, 'f') {
                        right.insert(level, index.resolve_operand(*value)?);
                    } else if name == "final_state" {
                        final_merge = Some(index.resolve_operand(*value)?);
                    }
                }
                PhiIRNode::Witness { target, .. } => {
                    if let Some(target) = target
                        .as_ref()
                        .map(|operand| index.resolve_operand(*operand))
                        .transpose()?
                    {
                        witness_targets.push(target);
                    }
                }
                _ => {}
            }
        }
    }

    if left.is_empty() || right.is_empty() {
        return Ok(None);
    }

    let depth = left.len().min(right.len());
    if depth == 0 {
        return Ok(None);
    }

    for level in 1..=depth {
        let left_op = *left
            .get(&level)
            .ok_or(QuantumOverlayError::NoContradictionPatternFound)?;
        let right_op = *right
            .get(&level)
            .ok_or(QuantumOverlayError::NoContradictionPatternFound)?;

        let (left_a, left_b) = index.coherence_args(left_op)?;
        let (right_a, right_b) = index.coherence_args(right_op)?;

        if level == 1 {
            assert_literal_pair(index, left_a, left_b, 1.0, 0.0)?;
            assert_literal_pair(index, right_a, right_b, 0.0, 1.0)?;
        } else {
            let prev_left = *left
                .get(&(level - 1))
                .ok_or(QuantumOverlayError::NoContradictionPatternFound)?;
            let prev_right = *right
                .get(&(level - 1))
                .ok_or(QuantumOverlayError::NoContradictionPatternFound)?;

            if left_a != prev_left || left_b != prev_right {
                return Err(QuantumOverlayError::UnsupportedCallShape(format!(
                    "left lane level {level} is not a mirrored contradiction step"
                )));
            }
            if right_a != prev_right || right_b != prev_left {
                return Err(QuantumOverlayError::UnsupportedCallShape(format!(
                    "right lane level {level} is not a mirrored contradiction step"
                )));
            }
        }
    }

    let final_merge = final_merge.unwrap_or_else(|| {
        witness_targets
            .first()
            .copied()
            .and_then(|target| index.resolve_operand(target).ok())
            .unwrap_or(*left.get(&depth).expect("depth checked"))
    });
    let witness_target = witness_targets
        .iter()
        .copied()
        .find(|target| *target == final_merge)
        .or_else(|| witness_targets.first().copied());
    let (final_left, final_right) = index.coherence_args(final_merge)?;
    if final_left != *left.get(&depth).expect("depth checked")
        || final_right != *right.get(&depth).expect("depth checked")
    {
        return Err(QuantumOverlayError::UnsupportedCallShape(
            "final_state does not merge the deepest contradiction lanes".to_string(),
        ));
    }

    Ok(Some(ContradictionLadderPlan {
        left_lane: left.into_values().collect(),
        right_lane: right.into_values().collect(),
        final_merge,
        depth,
        witness_target,
    }))
}

fn parse_lane_name(name: &str, prefix: char) -> Option<usize> {
    let mut chars = name.chars();
    match chars.next() {
        Some(first) if first.to_ascii_lowercase() == prefix.to_ascii_lowercase() => {}
        _ => return None,
    }
    let suffix = chars.collect::<String>();
    if suffix.is_empty() {
        return None;
    }
    suffix.parse::<usize>().ok()
}

fn assert_literal_pair(
    index: &ProgramIndex,
    left: Operand,
    right: Operand,
    expected_left: f64,
    expected_right: f64,
) -> Result<(), QuantumOverlayError> {
    let actual_left = index.number_value(left)?;
    let actual_right = index.number_value(right)?;
    if !approx_eq(actual_left, expected_left) || !approx_eq(actual_right, expected_right) {
        return Err(QuantumOverlayError::UnsupportedCallShape(format!(
            "expected base contradiction pair ({expected_left}, {expected_right}), found ({actual_left}, {actual_right})"
        )));
    }
    Ok(())
}

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}
