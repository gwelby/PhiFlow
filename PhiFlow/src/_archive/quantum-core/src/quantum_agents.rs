use std::collections::HashMap;
use anyhow::Result;
use crate::quantum_state::QuantumState;

#[derive(Debug, Default)]
pub struct QuantumAgentNetwork {
    agents: HashMap<String, QuantumState>,
}

impl QuantumAgentNetwork {
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
        }
    }

    pub fn add_agent(&mut self, name: &str, state: QuantumState) -> Result<()> {
        self.agents.insert(name.to_string(), state);
        Ok(())
    }

    pub fn monitor_coherence(&self) -> HashMap<String, f64> {
        self.agents
            .iter()
            .map(|(name, state)| (name.clone(), state.coherence()))
            .collect()
    }

    pub fn get_frequencies(&self) -> HashMap<String, f64> {
        let mut frequencies = HashMap::new();
        for (id, state) in &self.agents {
            frequencies.insert(id.clone(), state.frequency());
        }
        frequencies
    }
}
