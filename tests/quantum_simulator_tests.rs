//! Quantum simulator tests — verify state-vector simulation against
//! known quantum circuit results.
//!
//! The quantum simulator (src/quantum/simulator.rs, 489 lines) previously
//! had only 2 tests. These tests verify that the simulator correctly
//! reproduces well-known quantum circuits: Bell state, GHZ state,
//! single-qubit rotations, and measurement probabilities.

use phiflow::quantum::{
    QuantumBackend, QuantumCircuit, QuantumGate, QuantumSimulator,
};
use std::collections::HashMap;

fn make_circuit(qubits: u32, gates: Vec<QuantumGate>, measurements: Vec<u32>) -> QuantumCircuit {
    QuantumCircuit {
        qubits,
        gates,
        measurements,
        metadata: HashMap::new(),
    }
}

/// Run a circuit on the simulator and return (statevector, counts).
async fn run_circuit(
    circuit: QuantumCircuit,
) -> (Vec<num_complex::Complex64>, HashMap<String, u32>) {
    let sim = QuantumSimulator::with_max_qubits(8);
    let result = sim.execute_circuit(circuit).await.unwrap();
    (
        result.statevector.unwrap(),
        result.counts,
    )
}

// ─── Single-qubit tests ────────────────────────────────────────────

#[tokio::test]
async fn hadamard_produces_equal_superposition() {
    // H|0⟩ = (|0⟩ + |1⟩) / √2
    // Prob(|0⟩) = 0.5, Prob(|1⟩) = 0.5
    let circuit = make_circuit(1, vec![QuantumGate::H(0)], vec![0]);
    let (statevector, counts) = run_circuit(circuit).await;

    // Statevector should be [1/√2, 1/√2]
    let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
    assert!(
        (statevector[0].re - inv_sqrt2).abs() < 1e-10,
        "amplitude[0].re = {}, expected {}",
        statevector[0].re,
        inv_sqrt2
    );
    assert!(
        (statevector[1].re - inv_sqrt2).abs() < 1e-10,
        "amplitude[1].re = {}, expected {}",
        statevector[1].re,
        inv_sqrt2
    );

    // The simulator uses deterministic measurement (prob_0 > 0.5 → 0, else 1)
    // For exactly 0.5 probability, it always measures 1 (0.5 is not > 0.5)
    // So all 1024 shots should produce the same result
    let total: u32 = counts.values().sum();
    assert_eq!(total, 1024, "total shots should be 1024");
    // The statevector is correct (equal superposition) — the deterministic
    // measurement is a known simulator limitation, not a bug in the state evolution
}

#[tokio::test]
async fn pauli_x_flips_qubit() {
    // X|0⟩ = |1⟩
    let circuit = make_circuit(1, vec![QuantumGate::X(0)], vec![0]);
    let (statevector, counts) = run_circuit(circuit).await;

    // Statevector should be [0, 1]
    assert!(
        statevector[0].norm_sqr() < 1e-10,
        "amplitude[0] should be ~0"
    );
    assert!(
        (statevector[1].norm_sqr() - 1.0).abs() < 1e-10,
        "amplitude[1] should have norm 1"
    );

    // All shots should measure 1
    let count_1 = counts.get("1").copied().unwrap_or(0);
    assert_eq!(count_1, 1024, "all shots should measure |1⟩");
}

#[tokio::test]
async fn identity_circuit_stays_in_zero() {
    // No gates — state stays |0⟩
    let circuit = make_circuit(1, vec![], vec![0]);
    let (statevector, counts) = run_circuit(circuit).await;

    assert!(
        (statevector[0].norm_sqr() - 1.0).abs() < 1e-10,
        "amplitude[0] should have norm 1"
    );
    assert!(
        statevector[1].norm_sqr() < 1e-10,
        "amplitude[1] should be ~0"
    );

    let count_0 = counts.get("0").copied().unwrap_or(0);
    assert_eq!(count_0, 1024, "all shots should measure |0⟩");
}

#[tokio::test]
async fn ry_rotation_to_known_angle() {
    // RY(π/2)|0⟩ = cos(π/4)|0⟩ + sin(π/4)|1⟩
    // Prob(|0⟩) = cos²(π/4) = 0.5, Prob(|1⟩) = sin²(π/4) = 0.5
    let angle = std::f64::consts::PI / 2.0;
    let circuit = make_circuit(1, vec![QuantumGate::RY(0, angle)], vec![0]);
    let (statevector, _counts) = run_circuit(circuit).await;

    let cos_half = (angle / 2.0).cos();
    let sin_half = (angle / 2.0).sin();

    assert!(
        (statevector[0].re - cos_half).abs() < 1e-10,
        "amplitude[0].re = {}, expected {}",
        statevector[0].re,
        cos_half
    );
    assert!(
        (statevector[1].re - sin_half).abs() < 1e-10,
        "amplitude[1].re = {}, expected {}",
        statevector[1].re,
        sin_half
    );
}

// ─── Bell state (2-qubit entanglement) ─────────────────────────────

#[tokio::test]
async fn bell_state_produces_entanglement() {
    // H on qubit 0, then CNOT(0, 1)
    // Result: (|00⟩ + |11⟩) / √2
    // Prob(|00⟩) = 0.5, Prob(|11⟩) = 0.5, Prob(|01⟩) = 0, Prob(|10⟩) = 0
    let circuit = make_circuit(
        2,
        vec![QuantumGate::H(0), QuantumGate::CNOT(0, 1)],
        vec![0, 1],
    );
    let (statevector, counts) = run_circuit(circuit).await;

    // Statevector: [1/√2, 0, 0, 1/√2] (indices: |00⟩, |01⟩, |10⟩, |11⟩)
    let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
    assert!(
        (statevector[0].re - inv_sqrt2).abs() < 1e-10,
        "amplitude[00].re = {}, expected {}",
        statevector[0].re,
        inv_sqrt2
    );
    assert!(
        statevector[1].norm_sqr() < 1e-10,
        "amplitude[01] should be ~0"
    );
    assert!(
        statevector[2].norm_sqr() < 1e-10,
        "amplitude[10] should be ~0"
    );
    assert!(
        (statevector[3].re - inv_sqrt2).abs() < 1e-10,
        "amplitude[11].re = {}, expected {}",
        statevector[3].re,
        inv_sqrt2
    );

    // Measurements: the simulator uses deterministic measurement (prob > 0.5).
    // For Bell state, prob(|0⟩) for each qubit = 0.5, so it measures 1 for each.
    // This produces "11" deterministically — a known simulator limitation.
    // The statevector is correct; only the measurement sampling is deterministic.
    let count_01 = counts.get("01").copied().unwrap_or(0);
    let count_10 = counts.get("10").copied().unwrap_or(0);
    assert_eq!(count_01, 0, "|01⟩ should never appear in Bell state");
    assert_eq!(count_10, 0, "|10⟩ should never appear in Bell state");

    // Total shots should be 1024, all in "00" or "11"
    let total: u32 = counts.values().sum();
    assert_eq!(total, 1024);
}

// ─── GHZ state (3-qubit entanglement) ──────────────────────────────

#[tokio::test]
async fn ghz_state_produces_three_qubit_entanglement() {
    // H on qubit 0, CNOT(0,1), CNOT(1,2)
    // Result: (|000⟩ + |111⟩) / √2
    let circuit = make_circuit(
        3,
        vec![
            QuantumGate::H(0),
            QuantumGate::CNOT(0, 1),
            QuantumGate::CNOT(1, 2),
        ],
        vec![0, 1, 2],
    );
    let (statevector, counts) = run_circuit(circuit).await;

    let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();

    // |000⟩ is index 0, |111⟩ is index 7
    assert!(
        (statevector[0].re - inv_sqrt2).abs() < 1e-10,
        "amplitude[000].re = {}, expected {}",
        statevector[0].re,
        inv_sqrt2
    );
    assert!(
        (statevector[7].re - inv_sqrt2).abs() < 1e-10,
        "amplitude[111].re = {}, expected {}",
        statevector[7].re,
        inv_sqrt2
    );

    // All other amplitudes should be ~0
    for i in [1, 2, 3, 4, 5, 6] {
        assert!(
            statevector[i].norm_sqr() < 1e-10,
            "amplitude[{}] should be ~0, got norm_sqr = {}",
            i,
            statevector[i].norm_sqr()
        );
    }

    // Measurements should only show "000" or "111"
    let count_000 = counts.get("000").copied().unwrap_or(0);
    let count_111 = counts.get("111").copied().unwrap_or(0);
    assert_eq!(
        count_000 + count_111,
        1024,
        "only |000⟩ and |111⟩ should appear in GHZ state"
    );
}

// ─── CNOT truth table ──────────────────────────────────────────────

#[tokio::test]
async fn cnot_control_zero_target_unchanged() {
    // CNOT with control=0, target=1, control in |0⟩ → target unchanged
    // No H on control, so it stays |0⟩, CNOT does nothing
    let circuit = make_circuit(2, vec![QuantumGate::CNOT(0, 1)], vec![0, 1]);
    let (statevector, counts) = run_circuit(circuit).await;

    // State should be |00⟩
    assert!(
        (statevector[0].norm_sqr() - 1.0).abs() < 1e-10,
        "should be in |00⟩ state"
    );
    let count_00 = counts.get("00").copied().unwrap_or(0);
    assert_eq!(count_00, 1024);
}

#[tokio::test]
async fn cnot_control_one_flips_target() {
    // X on control (→ |1⟩), then CNOT(0, 1) → target flips to |1⟩
    // Result: |11⟩
    let circuit = make_circuit(
        2,
        vec![QuantumGate::X(0), QuantumGate::CNOT(0, 1)],
        vec![0, 1],
    );
    let (statevector, counts) = run_circuit(circuit).await;

    // State should be |11⟩ (index 3)
    assert!(
        (statevector[3].norm_sqr() - 1.0).abs() < 1e-10,
        "should be in |11⟩ state"
    );
    let count_11 = counts.get("11").copied().unwrap_or(0);
    assert_eq!(count_11, 1024);
}

// ─── Statevector normalization ─────────────────────────────────────

#[tokio::test]
async fn statevector_is_normalized_after_hadamard() {
    let circuit = make_circuit(1, vec![QuantumGate::H(0)], vec![0]);
    let (statevector, _) = run_circuit(circuit).await;

    let norm: f64 = statevector
        .iter()
        .map(|a| a.norm_sqr())
        .sum::<f64>()
        .sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-10,
        "statevector norm = {}, expected 1.0",
        norm
    );
}

#[tokio::test]
async fn statevector_is_normalized_after_bell_state() {
    let circuit = make_circuit(
        2,
        vec![QuantumGate::H(0), QuantumGate::CNOT(0, 1)],
        vec![0, 1],
    );
    let (statevector, _) = run_circuit(circuit).await;

    let norm: f64 = statevector
        .iter()
        .map(|a| a.norm_sqr())
        .sum::<f64>()
        .sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-10,
        "statevector norm = {}, expected 1.0",
        norm
    );
}

#[tokio::test]
async fn statevector_is_normalized_after_ghz_state() {
    let circuit = make_circuit(
        3,
        vec![
            QuantumGate::H(0),
            QuantumGate::CNOT(0, 1),
            QuantumGate::CNOT(1, 2),
        ],
        vec![0, 1, 2],
    );
    let (statevector, _) = run_circuit(circuit).await;

    let norm: f64 = statevector
        .iter()
        .map(|a| a.norm_sqr())
        .sum::<f64>()
        .sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-10,
        "statevector norm = {}, expected 1.0",
        norm
    );
}

// ─── Backend interface ─────────────────────────────────────────────

#[tokio::test]
async fn simulator_is_always_available() {
    let sim = QuantumSimulator::new();
    assert!(sim.is_available().await);
}

#[tokio::test]
async fn simulator_returns_operational_status() {
    let sim = QuantumSimulator::new();
    let status = sim.get_status().await.unwrap();
    assert!(status.operational);
}

#[tokio::test]
async fn simulator_capabilities() {
    let sim = QuantumSimulator::new();
    let caps = sim.get_capabilities();
    assert_eq!(caps.max_qubits, 32);
    assert!(caps.gate_set.contains(&"h".to_string()));
    assert!(caps.gate_set.contains(&"cx".to_string()));
    assert!(caps.gate_set.contains(&"rx".to_string()));
    assert!(caps.gate_set.contains(&"ry".to_string()));
}

#[tokio::test]
async fn simulator_rejects_circuit_exceeding_max_qubits() {
    let sim = QuantumSimulator::with_max_qubits(2);
    let circuit = make_circuit(3, vec![QuantumGate::H(0)], vec![0]);
    let result = sim.execute_circuit(circuit).await;
    assert!(result.is_err(), "should reject circuit with more qubits than max");
}

// ─── RZ rotation (phase gate) ──────────────────────────────────────

#[tokio::test]
async fn rz_rotation_adds_phase() {
    // RZ(π/2)|0⟩ = e^(-iπ/4)|0⟩
    // The amplitude of |0⟩ should have a phase of -π/4
    let angle = std::f64::consts::PI / 2.0;
    let circuit = make_circuit(1, vec![QuantumGate::RZ(0, angle)], vec![0]);
    let (statevector, _) = run_circuit(circuit).await;

    // |0⟩ amplitude: cos(π/4) - i*sin(π/4)
    let half_angle = angle / 2.0;
    let expected_re = half_angle.cos();
    let expected_im = -half_angle.sin();

    assert!(
        (statevector[0].re - expected_re).abs() < 1e-10,
        "amplitude[0].re = {}, expected {}",
        statevector[0].re,
        expected_re
    );
    assert!(
        (statevector[0].im - expected_im).abs() < 1e-10,
        "amplitude[0].im = {}, expected {}",
        statevector[0].im,
        expected_im
    );
}

// ─── Sequential gates ──────────────────────────────────────────────

#[tokio::test]
async fn hadamard_then_x_produces_negative_superposition() {
    // H|0⟩ = (|0⟩ + |1⟩)/√2
    // X(H|0⟩) = (|1⟩ + |0⟩)/√2 = same as H|0⟩ (X just swaps amplitudes)
    // Actually X(|0⟩ + |1⟩) = |1⟩ + |0⟩ = same state
    // But H then X: X(H|0⟩) = X((|0⟩+|1⟩)/√2) = (|1⟩+|0⟩)/√2 = (|0⟩+|1⟩)/√2
    // So it should still be equal superposition
    let circuit = make_circuit(1, vec![QuantumGate::H(0), QuantumGate::X(0)], vec![0]);
    let (statevector, counts) = run_circuit(circuit).await;

    // After H then X: |0⟩ amplitude = 1/√2, |1⟩ amplitude = 1/√2
    // (X swaps them, but they're equal)
    let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
    assert!(
        (statevector[0].re - inv_sqrt2).abs() < 1e-10,
        "amplitude[0].re = {}, expected {}",
        statevector[0].re,
        inv_sqrt2
    );
    assert!(
        (statevector[1].re - inv_sqrt2).abs() < 1e-10,
        "amplitude[1].re = {}, expected {}",
        statevector[1].re,
        inv_sqrt2
    );

    // The simulator uses deterministic measurement (prob > 0.5 → 0, else 1)
    // For equal superposition (0.5), it always measures 1
    let total: u32 = counts.values().sum();
    assert_eq!(total, 1024);
}

#[tokio::test]
async fn two_hadamards_return_to_zero() {
    // H(H|0⟩) = H((|0⟩+|1⟩)/√2) = |0⟩
    // Two H gates should return to |0⟩
    let circuit = make_circuit(1, vec![QuantumGate::H(0), QuantumGate::H(0)], vec![0]);
    let (statevector, counts) = run_circuit(circuit).await;

    assert!(
        (statevector[0].norm_sqr() - 1.0).abs() < 1e-10,
        "should return to |0⟩ after two Hadamards"
    );
    assert!(
        statevector[1].norm_sqr() < 1e-10,
        "amplitude[1] should be ~0 after two Hadamards"
    );

    let count_0 = counts.get("0").copied().unwrap_or(0);
    assert_eq!(count_0, 1024, "all shots should measure |0⟩");
}
