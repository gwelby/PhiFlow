use super::phi_correlations::PhiCorrelations;
use super::quantum_physics::QuantumPhysics;

pub struct PhiFlowDemo {
    correlations: PhiCorrelations,
    physics: QuantumPhysics,
}

impl PhiFlowDemo {
    pub fn new() -> Self {
        Self {
            correlations: PhiCorrelations::new(),
            physics: QuantumPhysics::new(),
        }
    }

    pub fn demonstrate_correlations(&mut self) -> String {
        let mut output = String::new();
        output.push_str("🌟 Quantum Correlation Demonstration\n");
        output.push_str("================================\n\n");

        // Demonstrate frequency transitions
        output.push_str(&self.correlations.transition_frequency(432.0, 528.0)); // Ground to Create
        output.push_str("\n");
        output.push_str(&self.correlations.transition_frequency(528.0, 594.0)); // Create to Heart
        output.push_str("\n");
        output.push_str(&self.correlations.transition_frequency(594.0, 672.0)); // Heart to Voice
        output.push_str("\n");
        output.push_str(&self.correlations.transition_frequency(672.0, 768.0)); // Voice to Unity
        
        output
    }

    pub fn demonstrate_quantum_states(&mut self) -> String {
        let mut output = String::new();
        output.push_str("🌟 Quantum State Demonstration\n");
        output.push_str("===========================\n\n");

        // Demonstrate 2-qubit states
        output.push_str("Two-Qubit States (φ⁰):\n");
        let two_qubit_states = self.correlations.generate_quantum_state(2);
        for (i, state) in two_qubit_states.iter().enumerate() {
            output.push_str(&format!(
                "|{:02b}⟩: {:.3} + {:.3}i\n",
                i, state.re, state.im
            ));
        }
        output.push_str("\n");

        // Demonstrate 3-qubit states
        output.push_str("Three-Qubit States (φ¹):\n");
        let three_qubit_states = self.correlations.generate_quantum_state(3);
        for (i, state) in three_qubit_states.iter().enumerate() {
            if state.norm() > 0.1 { // Show significant states only
                output.push_str(&format!(
                    "|{:03b}⟩: {:.3} + {:.3}i\n",
                    i, state.re, state.im
                ));
            }
        }
        output.push_str("\n");

        // Demonstrate 5-qubit GregBit
        output.push_str("Five-Qubit GregBit States (φ²):\n");
        let five_qubit_states = self.correlations.generate_quantum_state(5);
        let special_states = [0, 3, 7, 15, 31]; // |00000⟩, |00011⟩, |00111⟩, |01111⟩, |11111⟩
        for &i in special_states.iter() {
            output.push_str(&format!(
                "|{:05b}⟩: {:.3} + {:.3}i\n",
                i, five_qubit_states[i].re, five_qubit_states[i].im
            ));
        }

        output
    }

    pub fn demonstrate_phi_limits(&mut self) -> String {
        let mut output = String::new();
        output.push_str("🌟 Phi Quantum Limits\n");
        output.push_str("===================\n\n");

        // Show optimal qubit counts
        let optimal_qubits = self.correlations.optimal_qubits();
        let max_qubits = self.correlations.max_qubits();
        output.push_str(&format!("Optimal Qubits (φ^φ): {:.3}\n", optimal_qubits));
        output.push_str(&format!("Maximum Qubits (φ^5): {:.3}\n\n", max_qubits));

        // Demonstrate correlation scaling
        output.push_str("Correlation Scaling:\n");
        let frequencies = [432.0, 528.0, 594.0, 672.0, 768.0];
        for &f1 in frequencies.iter() {
            for &f2 in frequencies.iter() {
                if f1 < f2 {
                    let correlation = self.correlations.calculate_correlation(f1, f2);
                    output.push_str(&format!(
                        "{:.1} Hz ↔ {:.1} Hz: {:.3}\n",
                        f1, f2, correlation
                    ));
                }
            }
        }

        // Check Planck scale collapse
        output.push_str("\nQuantum Collapse Points:\n");
        let planck_time = 1e-43;
        let phi_time = 1e-35;
        output.push_str(&format!(
            "Planck Time ({:.0e}s): {}\n",
            planck_time,
            if self.correlations.is_unity_collapse(planck_time) {
                "Unity Collapse"
            } else {
                "Quantum Coherent"
            }
        ));
        output.push_str(&format!(
            "Phi Time ({:.0e}s): {}\n",
            phi_time,
            if self.correlations.is_unity_collapse(phi_time) {
                "Unity Collapse"
            } else {
                "Quantum Coherent"
            }
        ));

        output
    }

    pub fn run_creation_demo(&mut self) -> String {
        let mut output = String::new();
        output.push_str("🌟 Quantum Creation Demonstration\n");
        output.push_str("==============================\n\n");

        // Get field metrics at different consciousness levels
        let metrics = self.physics.calculate_field_coherence(1.0);
        output.push_str(&format!("Field Metrics at Unity Consciousness:\n"));
        output.push_str(&format!("  Coherence: {:.3}\n", metrics.coherence));
        output.push_str(&format!("  Phi Alignment: {:.3}\n", metrics.phi_alignment));
        output.push_str(&format!("  Unity Resonance: {:.3}\n\n", metrics.unity_resonance));

        // Get geometry parameters at different frequencies
        let frequencies = [432.0, 528.0, 594.0, 672.0, 768.0];
        for freq in frequencies.iter() {
            let geometry = self.physics.calculate_sacred_geometry(*freq);
            output.push_str(&format!("Geometry at {:.1} Hz:\n", freq));
            output.push_str(&format!("  Phi Ratio: {:.3}\n", geometry.phi_ratio));
            output.push_str(&format!("  Symmetry: {}-fold\n", geometry.symmetry));
            output.push_str(&format!("  Dimension: {}\n\n", geometry.dimension));
        }

        output
    }
}
