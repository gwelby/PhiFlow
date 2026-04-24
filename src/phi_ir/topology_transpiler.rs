use crate::phi_ir::quantum_interaction::{ContradictionLadderPlan, QuantumOverlayPlan};
use crate::quantum::backend_topology::{
    normalize_edge, BackendTopologyProfile, EdgeCalibration, NativeTwoQGate,
};
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct LadderCorridor {
    pub left_path: Vec<usize>,
    pub right_path: Vec<usize>,
    pub rung_edges: Vec<(usize, usize)>,
    pub witness_qubit: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlacementScore {
    pub hop_cost: f64,
    pub edge_error_cost: f64,
    pub readout_cost: f64,
    pub total: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingStrategy {
    CalibrationWeightedShortestPath,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TopologyTranspileConfig {
    pub backend_name: String,
    pub strategy: RoutingStrategy,
    pub native_two_qubit_gate: NativeTwoQGate,
}

impl Default for TopologyTranspileConfig {
    fn default() -> Self {
        Self {
            backend_name: "ibm_fez".to_string(),
            strategy: RoutingStrategy::CalibrationWeightedShortestPath,
            native_two_qubit_gate: NativeTwoQGate::Cz,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TopologyTranspileError {
    NoUsableCorridor { required_depth: usize },
    DisconnectedEdge { from: usize, to: usize },
    UnsupportedOverlay(String),
}

impl fmt::Display for TopologyTranspileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TopologyTranspileError::NoUsableCorridor { required_depth } => {
                write!(
                    f,
                    "no SWAP-free ladder corridor found for required depth `{required_depth}`"
                )
            }
            TopologyTranspileError::DisconnectedEdge { from, to } => {
                write!(f, "physical edge q[{from}] -> q[{to}] is not adjacent on backend")
            }
            TopologyTranspileError::UnsupportedOverlay(message) => f.write_str(message),
        }
    }
}

impl Error for TopologyTranspileError {}

pub fn choose_ladder_corridor(
    plan: &ContradictionLadderPlan,
    profile: &BackendTopologyProfile,
    _config: &TopologyTranspileConfig,
) -> Result<LadderCorridor, TopologyTranspileError> {
    let required_depth = plan.depth;
    if required_depth == 0 {
        return Err(TopologyTranspileError::NoUsableCorridor { required_depth });
    }

    let normalized_coupling_map = profile.normalized_coupling_map();
    let adjacency = build_adjacency(&normalized_coupling_map);
    let mut best: Option<(LadderCorridor, PlacementScore)> = None;

    for &(left_start, right_start) in &normalized_coupling_map {
        let mut used = HashSet::from([left_start, right_start]);
        let mut left_path = vec![left_start];
        let mut right_path = vec![right_start];
        search_ladder_corridor(
            required_depth,
            &adjacency,
            profile,
            &mut left_path,
            &mut right_path,
            &mut used,
            &mut best,
        );
    }

    best.map(|(corridor, _)| corridor).ok_or(
        TopologyTranspileError::NoUsableCorridor {
            required_depth,
        },
    )
}

pub(crate) fn choose_frequency_chain_path(
    required_nodes: usize,
    profile: &BackendTopologyProfile,
) -> Result<Vec<usize>, TopologyTranspileError> {
    if required_nodes == 0 {
        return Ok(Vec::new());
    }

    let normalized_coupling_map = profile.normalized_coupling_map();
    let adjacency = build_adjacency(&normalized_coupling_map);
    let mut best: Option<(Vec<usize>, f64)> = None;

    for &start in adjacency.keys() {
        let mut path = vec![start];
        let mut used = HashSet::from([start]);
        search_simple_path(
            required_nodes,
            &adjacency,
            profile,
            &mut path,
            &mut used,
            &mut best,
        );
    }

    best.map(|(path, _)| path).ok_or(TopologyTranspileError::NoUsableCorridor {
        required_depth: required_nodes,
    })
}

pub(crate) fn score_ladder_corridor(
    corridor: &LadderCorridor,
    profile: &BackendTopologyProfile,
) -> PlacementScore {
    let mut edge_error_cost = 0.0;
    for edge in corridor
        .rung_edges
        .iter()
        .copied()
        .chain(corridor.left_path.windows(2).map(|w| (w[0], w[1])))
        .chain(corridor.right_path.windows(2).map(|w| (w[0], w[1])))
    {
        edge_error_cost += edge_error(profile, edge).error.unwrap_or(1.0);
    }

    let readout_cost = profile
        .qubits
        .get(&corridor.witness_qubit)
        .and_then(|cal| cal.readout_error)
        .unwrap_or(1.0);
    let hop_cost =
        (corridor.left_path.len().saturating_sub(1) + corridor.right_path.len().saturating_sub(1))
            as f64;
    let total = hop_cost * 10.0 + edge_error_cost * 5.0 + readout_cost * 2.0;

    PlacementScore {
        hop_cost,
        edge_error_cost,
        readout_cost,
        total,
    }
}

fn search_ladder_corridor(
    required_depth: usize,
    adjacency: &HashMap<usize, Vec<usize>>,
    profile: &BackendTopologyProfile,
    left_path: &mut Vec<usize>,
    right_path: &mut Vec<usize>,
    used: &mut HashSet<usize>,
    best: &mut Option<(LadderCorridor, PlacementScore)>,
) {
    if left_path.len() == required_depth {
        let left_last = *left_path.last().expect("non-empty");
        let right_last = *right_path.last().expect("non-empty");
        let left_readout = profile
            .qubits
            .get(&left_last)
            .and_then(|cal| cal.readout_error)
            .unwrap_or(f64::INFINITY);
        let right_readout = profile
            .qubits
            .get(&right_last)
            .and_then(|cal| cal.readout_error)
            .unwrap_or(f64::INFINITY);
        let witness_qubit = if left_readout <= right_readout {
            left_last
        } else {
            right_last
        };
        let corridor = LadderCorridor {
            left_path: left_path.clone(),
            right_path: right_path.clone(),
            rung_edges: left_path
                .iter()
                .copied()
                .zip(right_path.iter().copied())
                .collect(),
            witness_qubit,
        };
        let score = score_ladder_corridor(&corridor, profile);
        match best {
            Some((_, best_score)) if best_score.total <= score.total => {}
            _ => *best = Some((corridor, score)),
        }
        return;
    }

    let left_curr = *left_path.last().expect("non-empty");
    let right_curr = *right_path.last().expect("non-empty");

    if let Some(left_neighbors) = adjacency.get(&left_curr) {
        for &left_next in left_neighbors {
            if used.contains(&left_next) {
                continue;
            }

            if let Some(right_neighbors) = adjacency.get(&right_curr) {
                for &right_next in right_neighbors {
                    if right_next == left_next || used.contains(&right_next) {
                        continue;
                    }
                    if !are_adjacent(adjacency, left_next, right_next) {
                        continue;
                    }

                    used.insert(left_next);
                    used.insert(right_next);
                    left_path.push(left_next);
                    right_path.push(right_next);

                    search_ladder_corridor(
                        required_depth,
                        adjacency,
                        profile,
                        left_path,
                        right_path,
                        used,
                        best,
                    );

                    right_path.pop();
                    left_path.pop();
                    used.remove(&right_next);
                    used.remove(&left_next);
                }
            }
        }
    }
}

fn search_simple_path(
    required_nodes: usize,
    adjacency: &HashMap<usize, Vec<usize>>,
    profile: &BackendTopologyProfile,
    path: &mut Vec<usize>,
    used: &mut HashSet<usize>,
    best: &mut Option<(Vec<usize>, f64)>,
) {
    if path.len() == required_nodes {
        let mut edge_error_cost = 0.0;
        for edge in path.windows(2).map(|w| (w[0], w[1])) {
            edge_error_cost += edge_error(profile, edge).error.unwrap_or(1.0);
        }
        let score = edge_error_cost;
        match best {
            Some((_, best_score)) if *best_score <= score => {}
            _ => *best = Some((path.clone(), score)),
        }
        return;
    }

    let current = *path.last().expect("non-empty");
    if let Some(neighbors) = adjacency.get(&current) {
        for &next in neighbors {
            if used.contains(&next) {
                continue;
            }
            used.insert(next);
            path.push(next);
            search_simple_path(required_nodes, adjacency, profile, path, used, best);
            path.pop();
            used.remove(&next);
        }
    }
}

pub fn validate_overlay_support(
    overlay: &QuantumOverlayPlan,
) -> Result<(), TopologyTranspileError> {
    match overlay {
        QuantumOverlayPlan::ContradictionLadder(_) | QuantumOverlayPlan::LegacyFreqChains(_) => {
            Ok(())
        }
    }
}

fn build_adjacency(edges: &[(usize, usize)]) -> HashMap<usize, Vec<usize>> {
    let mut adjacency: HashMap<usize, Vec<usize>> = HashMap::new();
    for &(a, b) in edges {
        adjacency.entry(a).or_default().push(b);
        adjacency.entry(b).or_default().push(a);
    }
    adjacency
}

fn are_adjacent(adjacency: &HashMap<usize, Vec<usize>>, from: usize, to: usize) -> bool {
    adjacency
        .get(&from)
        .map(|neighbors| neighbors.contains(&to))
        .unwrap_or(false)
}

fn edge_error(profile: &BackendTopologyProfile, edge: (usize, usize)) -> EdgeCalibration {
    profile
        .edges
        .get(&normalize_edge(edge.0, edge.1))
        .cloned()
        .unwrap_or_default()
}
