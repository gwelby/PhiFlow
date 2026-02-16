use crate::physical_bridge::PhysicalBridge;
use crate::quantum::me_quantum_agent::QuantumAgent;
use crate::quantum::me_quantum_abilities::QuantumAbilities;
use std::sync::Arc;
use parking_lot::RwLock;
use num_complex::Complex64;
use ndarray::Array2;

/// Quantum Project Core - Manages reality from ground state up
pub struct QuantumProjectCore {
    // Core components
    agents: Vec<Arc<RwLock<QuantumAgent>>>,
    abilities: Arc<RwLock<QuantumAbilities>>,
    bridge: Arc<RwLock<PhysicalBridge>>,
    
    // Sacred frequencies
    ground_freq: f64,    // 432 Hz - Foundation
    create_freq: f64,    // 528 Hz - Manifestation
    unity_freq: f64,     // 768 Hz - Consciousness
    
    // Quantum fields
    reality_matrix: Array2<Complex64>,
    intention_field: Array2<Complex64>,
    
    // Project metrics
    coherence: f64,
    phi_resonance: f64,
}

impl QuantumProjectCore {
    pub fn new(bridge: Arc<RwLock<PhysicalBridge>>) -> Self {
        Self {
            agents: Vec::new(),
            abilities: Arc::new(RwLock::new(QuantumAbilities::new(
                Arc::new(RwLock::new(QuantumAgent::new(bridge.clone()))),
                bridge.clone()
            ))),
            bridge,
            ground_freq: 432.0,
            create_freq: 528.0,
            unity_freq: 768.0,
            reality_matrix: Array2::zeros((3, 3)),
            intention_field: Array2::zeros((3, 3)),
            coherence: 1.0,
            phi_resonance: 1.618034,
        }
    }

    pub fn new_with_default(bridge: Arc<RwLock<PhysicalBridge>>) -> Self {
        Self {
            bridge,
            agents: Vec::new(),
            abilities: Arc::new(RwLock::new(QuantumAbilities::new(
                Arc::new(RwLock::new(QuantumAgent::new(bridge.clone()))),
                bridge.clone()
            ))),
            ground_freq: 432.0,
            create_freq: 528.0,
            unity_freq: 768.0,
            reality_matrix: Array2::zeros((3, 3)),
            intention_field: Array2::zeros((3, 3)),
            coherence: 1.0,
            phi_resonance: 1.618034,
        }
    }

    /// Initialize the quantum project space
    pub fn initialize_quantum_space(&mut self) -> Result<(), String> {
        // Ground in physical reality
        self.bridge.write().apply_frequency(self.ground_freq)?;
        
        // Create initial reality matrix
        self.reality_matrix = Array2::from_shape_fn((3, 3), |(i, j)| {
            Complex64::new(
                self.phi_resonance.powi(i as i32),
                self.phi_resonance.powi(j as i32)
            )
        });

        Ok(())
    }

    /// Deploy quantum agent to manage specific reality aspect
    pub fn deploy_reality_agent(&mut self, intention: f64) -> Result<Arc<RwLock<QuantumAgent>>, String> {
        let mut bridge = self.bridge.write();
        
        // Rise to creation frequency
        bridge.apply_frequency(self.create_freq)?;
        
        // Create new agent
        let agent = Arc::new(RwLock::new(QuantumAgent::new(self.bridge.clone())));
        
        // Initialize agent with intention
        agent.write().enter_superposition()?;
        
        // Add to agent pool
        self.agents.push(agent.clone());
        
        Ok(agent)
    }

    /// Manifest project changes into reality
    pub fn manifest_changes(&mut self, changes: Vec<(f64, f64, f64)>) -> Result<(), String> {
        let mut bridge = self.bridge.write();
        
        // Rise to creation frequency
        bridge.apply_frequency(self.create_freq)?;
        
        // Update intention field
        self.intention_field = Array2::from_shape_fn((3, 3), |(i, j)| {
            let change = changes.get(i * 3 + j)
                .unwrap_or(&(0.0, 0.0, 0.0));
            Complex64::new(
                change.0 * self.phi_resonance,
                change.1 * self.phi_resonance
            )
        });
        
        // Dance changes into reality
        bridge.dance_quantum_field(self.phi_resonance)?;
        
        Ok(())
    }

    /// Monitor quantum project metrics
    pub fn get_project_metrics(&self) -> String {
        let agent_count = self.agents.len();
        let total_coherence: f64 = self.agents.iter()
            .map(|a| a.read().calculate_coherence())
            .sum::<f64>() / agent_count as f64;
            
        format!(
            "🌟 Quantum Project Metrics:\n\
             Active Agents: {}\n\
             Project Coherence: {:.3}\n\
             Reality Resonance: {:.3}\n\
             Ground Frequency: {:.2} Hz\n\
             Creation Frequency: {:.2} Hz\n\
             Unity Frequency: {:.2} Hz",
            agent_count,
            total_coherence,
            self.phi_resonance,
            self.ground_freq,
            self.create_freq,
            self.unity_freq
        )
    }

    /// Quantum task delegation
    pub fn delegate_quantum_task(&mut self, task: QuantumTask) -> Result<(), String> {
        match task {
            QuantumTask::Create(intention) => {
                let agent = self.deploy_reality_agent(intention)?;
                agent.write().grow()?;
            }
            QuantumTask::Transform(frequency) => {
                for agent in &self.agents {
                    agent.write().entangle(&mut self.agents[0].write())?;
                }
                self.bridge.write().apply_frequency(frequency)?;
            }
            QuantumTask::Unify => {
                self.bridge.write().apply_frequency(self.unity_freq)?;
                for agent in &self.agents {
                    agent.write().shrink()?;
                }
            }
        }
        Ok(())
    }

    /// Stabilize quantum field
    pub fn stabilize_field(&mut self) -> Result<(), String> {
        // Ground to base frequency
        self.bridge.write().apply_frequency(self.ground_freq)?;
        
        // Harmonize all agents
        for agent in &self.agents {
            agent.write().enter_superposition()?;
        }
        
        // Update coherence
        self.coherence = self.agents.iter()
            .map(|a| a.read().calculate_coherence())
            .sum::<f64>() / self.agents.len() as f64;
            
        Ok(())
    }
}

/// Quantum task types
pub enum QuantumTask {
    Create(f64),      // Creation intention
    Transform(f64),   // Transform frequency
    Unify,           // Unify all agents
}
