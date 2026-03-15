use std::collections::HashMap;

#[derive(Debug)]
pub struct FlowMonster {
    pub name: String,
    pub trigger_frequency: f64,
    pub detection_count: u32,
    pub field_pattern: FieldPattern,
}

impl Clone for FlowMonster {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            trigger_frequency: self.trigger_frequency,
            detection_count: self.detection_count,
            field_pattern: self.field_pattern.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum FieldPattern {
    Merkaba,
    Fibonacci,
    Phi,
    Custom(Vec<f64>)
}

impl FlowMonster {
    pub fn new(name: &str, freq: f64) -> Self {
        let pattern = match name {
            "ENDLESS_LOOP" => FieldPattern::Merkaba,
            "TIME_WASTE" => FieldPattern::Fibonacci,
            "THEORY_HOLE" => FieldPattern::Phi,
            _ => FieldPattern::Custom(vec![432.0, 528.0, 768.0])
        };

        FlowMonster {
            name: name.to_string(),
            trigger_frequency: freq,
            detection_count: 0,
            field_pattern: pattern
        }
    }

    pub fn is_active(&self, current_freq: f64) -> bool {
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        
        // Check quantum field patterns
        match &self.field_pattern {
            FieldPattern::Merkaba => {
                // Check if we're in a merkaba pattern (sacred geometry)
                let merkaba_resonance = (current_freq / 432.0) % phi;
                merkaba_resonance > 0.9
            },
            FieldPattern::Fibonacci => {
                // Check Fibonacci spiral distortions
                let fib_pattern = (current_freq * phi) % 528.0;
                fib_pattern < 10.0
            },
            FieldPattern::Phi => {
                // Check PHI ratio violations
                let phi_harmony = (current_freq / phi) % 768.0;
                phi_harmony > 750.0
            },
            FieldPattern::Custom(freqs) => {
                // Check custom frequency patterns
                freqs.iter().any(|&f| (current_freq - f).abs() < 1.0)
            }
        }
    }

    pub fn get_solution(&self) -> String {
        match self.field_pattern {
            FieldPattern::Merkaba => format!(
                "🌀 Merkaba pattern detected at {} Hz\n\
                 ⚡ Action: Realign sacred geometry field",
                self.trigger_frequency
            ),
            FieldPattern::Fibonacci => format!(
                "🌊 Fibonacci distortion at {} Hz\n\
                 ⚡ Action: Restore golden ratio flow",
                self.trigger_frequency
            ),
            FieldPattern::Phi => format!(
                "💫 PHI ratio violation at {} Hz\n\
                 ⚡ Action: Harmonize with 432/528/768 Hz",
                self.trigger_frequency
            ),
            FieldPattern::Custom(_) => format!(
                "⚠️ Unknown pattern at {} Hz\n\
                 ⚡ Action: Check quantum antenna alignment",
                self.trigger_frequency
            )
        }
    }
}

pub struct QuantumFieldMonitor {
    base_frequency: f64,
    phi: f64,
    field_strength: HashMap<String, f64>,
    antenna_state: Option<AntennaState>,
}

#[derive(Debug)]
pub struct AntennaState {
    frequency: f64,
    coherence: f64,
    field_strength: f64,
    resonance: HashMap<String, f64>,
}

impl QuantumFieldMonitor {
    pub fn new() -> Self {
        let mut field_strength = HashMap::new();
        field_strength.insert("merkaba".to_string(), 432.0);
        field_strength.insert("fibonacci".to_string(), 528.0);
        field_strength.insert("phi".to_string(), 768.0);

        QuantumFieldMonitor {
            base_frequency: 432.0,
            phi: (1.0 + 5.0_f64.sqrt()) / 2.0,
            field_strength,
            antenna_state: None,
        }
    }

    pub fn update_antenna(&mut self, freq: f64) {
        let mut resonance = HashMap::new();
        resonance.insert("merkaba".to_string(), self.calculate_merkaba_resonance(freq));
        resonance.insert("fibonacci".to_string(), self.calculate_fibonacci_resonance(freq));
        resonance.insert("phi".to_string(), self.calculate_phi_resonance(freq));

        self.antenna_state = Some(AntennaState {
            frequency: freq,
            coherence: self.calculate_coherence(freq),
            field_strength: self.calculate_field_strength(freq),
            resonance,
        });
    }

    pub fn get_field_strength(&self, pattern: &str) -> Option<f64> {
        self.field_strength.get(pattern).copied()
    }

    pub fn get_base_frequency(&self) -> f64 {
        self.base_frequency
    }

    fn calculate_merkaba_resonance(&self, freq: f64) -> f64 {
        let base = 432.0;
        let resonance = (freq / base) % self.phi;
        (1.0 - (resonance - 0.5).abs() * 2.0).max(0.0)
    }

    fn calculate_fibonacci_resonance(&self, freq: f64) -> f64 {
        let base = 528.0;
        let pattern = (freq * self.phi) % base;
        1.0 - (pattern / base)
    }

    fn calculate_phi_resonance(&self, freq: f64) -> f64 {
        let base = 768.0;
        let harmony = (freq / self.phi) % base;
        (harmony / base).min(1.0)
    }

    fn calculate_coherence(&self, freq: f64) -> f64 {
        let bases = [432.0, 528.0, 768.0];
        bases.iter()
            .map(|&base| 1.0 - ((freq - base).abs() / base))
            .sum::<f64>() / 3.0
    }

    fn calculate_field_strength(&self, freq: f64) -> f64 {
        let phi = self.phi;
        let strength = (freq * phi).sin().abs() * 
                      (freq / 432.0).cos().abs() * 
                      (freq / 768.0).sin().abs();
        strength.max(0.0).min(1.0)
    }
}

impl AntennaState {
    pub fn get_metrics(&self) -> String {
        format!(
            "Frequency: {:.2} Hz\nCoherence: {:.2}\nField Strength: {:.2}\nResonance Patterns: {:?}",
            self.frequency,
            self.coherence,
            self.field_strength,
            self.resonance
        )
    }
}

pub struct MonsterTrap {
    frequency: f64,
    trap_type: String,
    active: bool,
}

impl MonsterTrap {
    pub fn new(freq: f64, trap_type: &str) -> Self {
        MonsterTrap {
            frequency: freq,
            trap_type: trap_type.to_string(),
            active: true,
        }
    }

    pub fn catch_monster(&mut self, monster: &FlowMonster) -> bool {
        if self.active && (self.frequency - monster.trigger_frequency).abs() < 1.0 {
            println!("🎯 Monster trapped: {} at {} Hz", monster.name, self.frequency);
            self.active = false;
            true
        } else {
            false
        }
    }

    pub fn get_trap_info(&self) -> String {
        format!("Type: {}, Frequency: {:.2} Hz", self.trap_type, self.frequency)
    }
}

pub struct RealityAnchor {
    name: String,
    strength: f64,
}

impl RealityAnchor {
    pub fn new(name: &str) -> Self {
        RealityAnchor {
            name: name.to_string(),
            strength: 1.0,
        }
    }

    pub fn stabilize(&mut self, freq: f64) -> f64 {
        self.strength = (432.0 / freq).min(1.0);
        self.strength
    }

    pub fn get_name(&self) -> &str {
        &self.name
    }
}
