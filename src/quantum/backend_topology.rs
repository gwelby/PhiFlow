use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessorFamily {
    Heron,
    Eagle,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeTwoQGate {
    Cz,
    Ecr,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct QubitCalibration {
    pub t1_s: Option<f64>,
    pub t2_s: Option<f64>,
    pub readout_error: Option<f64>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct EdgeCalibration {
    pub duration_s: Option<f64>,
    pub error: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BackendTopologyProfile {
    pub backend_name: String,
    pub family: ProcessorFamily,
    pub num_qubits: usize,
    pub coupling_map: Vec<(usize, usize)>,
    pub native_two_qubit_gate: NativeTwoQGate,
    pub qubits: HashMap<usize, QubitCalibration>,
    pub edges: HashMap<(usize, usize), EdgeCalibration>,
}

impl BackendTopologyProfile {
    pub fn normalized_coupling_map(&self) -> Vec<(usize, usize)> {
        let mut normalized = self
            .coupling_map
            .iter()
            .map(|&(a, b)| normalize_edge(a, b))
            .collect::<Vec<_>>();
        normalized.sort_unstable();
        normalized.dedup();
        normalized
    }

    pub fn has_edge(&self, a: usize, b: usize) -> bool {
        self.normalized_coupling_map()
            .contains(&normalize_edge(a, b))
    }
}

impl Default for BackendTopologyProfile {
    fn default() -> Self {
        Self {
            backend_name: "unknown".to_string(),
            family: ProcessorFamily::Unknown,
            num_qubits: 0,
            coupling_map: Vec::new(),
            native_two_qubit_gate: NativeTwoQGate::Cz,
            qubits: HashMap::new(),
            edges: HashMap::new(),
        }
    }
}

pub fn normalize_edge(a: usize, b: usize) -> (usize, usize) {
    if a <= b { (a, b) } else { (b, a) }
}
