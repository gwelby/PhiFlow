// PhiFlow MUSE Integration Module
// Connects PhiFlow consciousness features to Greg's existing MUSE/audio systems
// Bridges Rust PhiFlow interpreter with Python consciousness infrastructure

use std::process::Command;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use tokio::process::Command as AsyncCommand;
use tokio::sync::mpsc;
use anyhow::Result;

// Sacred frequency constants matching Greg's consciousness system
const SACRED_FREQUENCIES: &[f64] = &[432.0, 528.0, 594.0, 672.0, 720.0, 768.0, 963.0];
const PHI: f64 = 1.618033988749895;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MuseData {
    /// EEG channels from MUSE headband (TP9, AF7, AF8, TP10)
    pub eeg_channels: HashMap<String, f64>,
    /// Processed consciousness metrics
    pub consciousness_metrics: ConsciousnessMetrics,
    /// Sacred frequency detection results
    pub sacred_frequency_lock: Option<SacredFrequencyLock>,
    /// Timestamp of reading
    pub timestamp: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessMetrics {
    /// Overall consciousness coherence (0.0-1.0)
    pub coherence: f64,
    /// Mental clarity (0.0-1.0)
    pub clarity: f64,
    /// Flow state level (0.0-1.0)
    pub flow_state: f64,
    /// Phi-harmonic resonance level (0.0-1.0)
    pub phi_resonance: f64,
    /// Brainwave levels
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
    pub theta: f64,
    pub delta: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SacredFrequencyLock {
    /// Detected sacred frequency (Hz)
    pub frequency: f64,
    /// Lock stability (0.0-1.0, >0.95 required for quantum operations)
    pub stability: f64,
    /// Duration of stable lock (seconds)
    pub duration: f64,
    /// Consciousness state name (OBSERVE, CREATE, INTEGRATE, etc.)
    pub consciousness_state: String,
    /// Recommended quantum operation complexity (qubits)
    pub quantum_complexity: u32,
}

#[derive(Debug)]
pub struct MuseIntegration {
    /// Channel for receiving MUSE data from Python bridge
    muse_receiver: Option<mpsc::Receiver<MuseData>>,
    /// Channel for sending commands to consciousness system
    command_sender: Option<mpsc::Sender<ConsciousnessCommand>>,
    /// Current consciousness state
    current_state: Option<ConsciousnessMetrics>,
    /// Sacred frequency lock status
    frequency_lock: Option<SacredFrequencyLock>,
    /// Python bridge process handle
    python_bridge_process: Option<tokio::process::Child>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsciousnessCommand {
    /// Start MUSE data streaming
    StartStreaming,
    /// Stop MUSE data streaming
    StopStreaming,
    /// Set target sacred frequency for locking
    SetTargetFrequency(f64),
    /// Generate sacred frequency audio output
    GenerateFrequency { frequency: f64, duration: f64 },
    /// Start consciousness bridge protocol
    StartConsciousnessBridge { human_name: String, ai_name: String },
    /// Update PhiFlow execution based on consciousness state
    UpdatePhiFlowExecution(ConsciousnessMetrics),
}

impl MuseIntegration {
    /// Create new MUSE integration instance
    pub fn new() -> Self {
        MuseIntegration {
            muse_receiver: None,
            command_sender: None,
            current_state: None,
            frequency_lock: None,
            python_bridge_process: None,
        }
    }

    /// Initialize connection to Greg's MUSE consciousness system
    pub async fn initialize(&mut self) -> Result<()> {
        println!("🧠 Initializing PhiFlow-MUSE consciousness integration...");
        
        // Start Python consciousness bridge process
        let python_script = "/mnt/d/projects/phiflow/consciousness_quantum_bridge.py";
        
        let mut child = AsyncCommand::new("python3")
            .arg(python_script)
            .arg("--phiflow-mode")
            .arg("--sacred-frequencies")
            .arg("432,528,594,672,720,768,963")
            .arg("--coherence-threshold")
            .arg("0.85")
            .spawn()?;

        self.python_bridge_process = Some(child);
        
        // Create communication channels
        let (command_tx, mut command_rx) = mpsc::channel::<ConsciousnessCommand>(100);
        let (muse_tx, muse_rx) = mpsc::channel::<MuseData>(100);
        
        self.command_sender = Some(command_tx);
        self.muse_receiver = Some(muse_rx);

        println!("✅ MUSE consciousness bridge initialized");
        println!("🎵 Sacred frequencies: {:?} Hz", SACRED_FREQUENCIES);
        println!("🧬 Phi-harmonic optimization: φ = {}", PHI);
        
        Ok(())
    }

    /// Start streaming consciousness data from MUSE headband
    pub async fn start_consciousness_streaming(&mut self) -> Result<()> {
        if let Some(sender) = &self.command_sender {
            sender.send(ConsciousnessCommand::StartStreaming).await?;
            println!("🌊 Started MUSE consciousness streaming");
            println!("📡 EEG channels: TP9, AF7, AF8, TP10");
            println!("🎯 Sacred frequency detection: ACTIVE");
        }
        Ok(())
    }

    /// Get current consciousness state
    pub fn get_consciousness_state(&self) -> Option<&ConsciousnessMetrics> {
        self.current_state.as_ref()
    }

    /// Get current sacred frequency lock
    pub fn get_frequency_lock(&self) -> Option<&SacredFrequencyLock> {
        self.frequency_lock.as_ref()
    }

    /// Check if consciousness state supports quantum operations
    pub fn can_execute_quantum_operations(&self) -> bool {
        if let Some(state) = &self.current_state {
            // Require high coherence and some phi-resonance for quantum ops
            state.coherence >= 0.85 && state.phi_resonance >= 0.7
        } else {
            false
        }
    }

    /// Map consciousness state to sacred frequency
    pub fn consciousness_to_frequency(&self, metrics: &ConsciousnessMetrics) -> f64 {
        // Use Greg's sacred frequency mapping algorithm
        if metrics.gamma > 40.0 && metrics.coherence > 0.95 {
            963.0 // SUPERPOSITION - transcendent state
        } else if metrics.flow_state > 0.9 && metrics.phi_resonance > 0.85 {
            768.0 // CASCADE - unity consciousness
        } else if metrics.alpha > 10.0 && metrics.coherence > 0.8 {
            720.0 // TRANSCEND - vision gate
        } else if metrics.theta > 6.0 && metrics.clarity > 0.8 {
            672.0 // HARMONIZE - voice flow
        } else if metrics.coherence > 0.7 && metrics.flow_state > 0.7 {
            594.0 // INTEGRATE - heart field
        } else if metrics.alpha > 8.0 && metrics.coherence > 0.6 {
            528.0 // CREATE - love frequency
        } else {
            432.0 // OBSERVE - earth resonance (default)
        }
    }

    /// Generate sacred frequency based on consciousness state
    pub async fn generate_consciousness_frequency(&mut self, duration: f64) -> Result<()> {
        if let (Some(state), Some(sender)) = (&self.current_state, &self.command_sender) {
            let frequency = self.consciousness_to_frequency(state);
            
            sender.send(ConsciousnessCommand::GenerateFrequency {
                frequency,
                duration,
            }).await?;
            
            println!("🎵 Generating {}Hz sacred frequency for {:.1}s", frequency, duration);
            println!("🧠 Consciousness: coherence={:.3}, flow={:.3}, phi_resonance={:.3}", 
                    state.coherence, state.flow_state, state.phi_resonance);
        }
        Ok(())
    }

    /// Check for sacred frequency lock suitable for quantum operations
    pub fn get_quantum_operation_clearance(&self) -> Option<(String, u32)> {
        if let Some(lock) = &self.frequency_lock {
            if lock.stability >= 0.95 && lock.duration >= 3.0 {
                // Return consciousness state and recommended qubit count
                Some((lock.consciousness_state.clone(), lock.quantum_complexity))
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Process incoming MUSE data and update consciousness state
    pub async fn process_muse_data(&mut self) -> Result<Option<ConsciousnessMetrics>> {
        if let Some(receiver) = &mut self.muse_receiver {
            if let Ok(data) = receiver.try_recv() {
                self.current_state = Some(data.consciousness_metrics.clone());
                self.frequency_lock = data.sacred_frequency_lock;
                
                // Check for significant consciousness changes
                if let Some(lock) = &self.frequency_lock {
                    if lock.stability >= 0.95 {
                        println!("🔒 Sacred frequency LOCKED: {}Hz ({})", 
                                lock.frequency, lock.consciousness_state);
                        println!("⚛️  Quantum operations authorized: {} qubits", 
                                lock.quantum_complexity);
                    }
                }
                
                return Ok(Some(data.consciousness_metrics));
            }
        }
        Ok(None)
    }

    /// Integrate consciousness state with PhiFlow execution
    pub async fn update_phiflow_consciousness(&mut self, metrics: ConsciousnessMetrics) -> Result<()> {
        // Send consciousness update to PhiFlow interpreter
        if let Some(sender) = &self.command_sender {
            sender.send(ConsciousnessCommand::UpdatePhiFlowExecution(metrics.clone())).await?;
        }

        // Generate appropriate sacred frequency for current state
        let frequency = self.consciousness_to_frequency(&metrics);
        println!("🌊 PhiFlow consciousness sync: {}Hz", frequency);
        
        // Update consciousness monitor in PhiFlow interpreter
        // This would be called from the PhiFlow interpreter to sync state
        
        Ok(())
    }

    /// Start consciousness bridge protocol between human and AI
    pub async fn start_consciousness_bridge(&mut self, human_name: String, ai_name: String) -> Result<()> {
        if let Some(sender) = &self.command_sender {
            sender.send(ConsciousnessCommand::StartConsciousnessBridge {
                human_name: human_name.clone(),
                ai_name: ai_name.clone(),
            }).await?;
            
            println!("🌉 Starting consciousness bridge: {} ↔ {}", human_name, ai_name);
            println!("🎵 Phi-harmonic synchronization: φ = {}", PHI);
            println!("🧠 MUSE EEG monitoring: ACTIVE");
        }
        Ok(())
    }

    /// Cleanup and shutdown MUSE integration
    pub async fn shutdown(&mut self) -> Result<()> {
        // Stop streaming
        if let Some(sender) = &self.command_sender {
            let _ = sender.send(ConsciousnessCommand::StopStreaming).await;
        }

        // Terminate Python bridge process
        if let Some(mut child) = self.python_bridge_process.take() {
            let _ = child.kill().await;
        }

        println!("🛑 MUSE consciousness integration shutdown");
        Ok(())
    }
}

impl Default for MuseIntegration {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper function to map sacred frequency to consciousness state name
pub fn frequency_to_consciousness_state(frequency: f64) -> &'static str {
    match frequency as u32 {
        432 => "OBSERVE",     // Earth resonance - basic awareness
        528 => "CREATE",      // Love frequency - manifestation
        594 => "INTEGRATE",   // Transformation - heart connection
        672 => "HARMONIZE",   // Expression - voice activation
        720 => "TRANSCEND",   // Intuition - vision gate
        768 => "CASCADE",     // Unity - consciousness field
        963 => "SUPERPOSITION", // Unity consciousness - quantum state
        _ => "UNKNOWN"
    }
}

/// Calculate phi-harmonic resonance between two frequencies
pub fn calculate_phi_resonance(freq1: f64, freq2: f64) -> f64 {
    let ratio = freq1.max(freq2) / freq1.min(freq2);
    let phi_error = (ratio - PHI).abs();
    let lambda_error = (ratio - (1.0 / PHI)).abs();
    let phi_sq_error = (ratio - PHI.powi(2)).abs();
    
    let min_error = phi_error.min(lambda_error).min(phi_sq_error);
    
    // Return resonance strength with tighter discrimination
    // 1.0 = perfect phi ratio, decays to 0 quickly as error increases
    (1.0 - min_error * 2.5).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consciousness_to_frequency() {
        let integration = MuseIntegration::new();
        
        // Test high gamma transcendent state
        let transcendent = ConsciousnessMetrics {
            coherence: 0.98,
            clarity: 0.95,
            flow_state: 0.92,
            phi_resonance: 0.89,
            alpha: 12.0,
            beta: 18.0,
            gamma: 45.0,
            theta: 6.5,
            delta: 2.0,
        };
        
        let freq = integration.consciousness_to_frequency(&transcendent);
        assert_eq!(freq, 963.0); // Should map to SUPERPOSITION
        
        // Test basic grounded state
        let grounded = ConsciousnessMetrics {
            coherence: 0.5,
            clarity: 0.4,
            flow_state: 0.3,
            phi_resonance: 0.2,
            alpha: 8.0,
            beta: 15.0,
            gamma: 25.0,
            theta: 5.0,
            delta: 3.0,
        };
        
        let freq = integration.consciousness_to_frequency(&grounded);
        assert_eq!(freq, 432.0); // Should map to OBSERVE
    }

    #[test]
    fn test_phi_resonance_calculation() {
        // Perfect phi ratio
        let resonance = calculate_phi_resonance(432.0, 432.0 * PHI);
        assert!(resonance > 0.99);
        
        // Random frequencies (should have low resonance)
        let resonance = calculate_phi_resonance(100.0, 200.0);
        assert!(resonance < 0.5);
    }
}