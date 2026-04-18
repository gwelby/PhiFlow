use std::sync::Arc;
use parking_lot::RwLock;
use chrono::Utc;
use crate::monitor::phi_flow_monitor::*;
use crate::quantum::quantum_core::*;
use crate::quantum::quantum_harmonizer::*;

/// QuantumTestSuite - Comprehensive testing framework for PhiFlow
pub struct QuantumTestSuite {
    monitor: PhiFlowMonitor,
    core: Arc<RwLock<QuantumCore>>,
    harmonizer: Arc<RwLock<QuantumHarmonizer>>,
}

impl QuantumTestSuite {
    pub fn new() -> Self {
        Self {
            monitor: PhiFlowMonitor::new(),
            core: Arc::new(RwLock::new(QuantumCore::new())),
            harmonizer: Arc::new(RwLock::new(QuantumHarmonizer::new())),
        }
    }

    /// Run complete quantum coherence test suite
    pub fn run_complete_test(&mut self) -> TestResults {
        println!("🌟 Starting Quantum Coherence Test Suite");
        
        let mut results = TestResults::new();
        
        // Test 1: Frequency Harmonics
        results.add_result(
            "Frequency Harmonics",
            self.test_frequency_harmonics()
        );
        
        // Test 2: Zipf Distribution
        results.add_result(
            "Zipf Analysis",
            self.test_zipf_distribution()
        );
        
        // Test 3: Sacred Geometry
        results.add_result(
            "Sacred Geometry",
            self.test_sacred_geometry()
        );
        
        // Test 4: Quantum Evolution
        results.add_result(
            "Quantum Evolution",
            self.test_quantum_evolution()
        );
        
        // Test 5: Project Tracking
        results.add_result(
            "Project Tracking",
            self.test_project_tracking()
        );
        
        println!("✨ Quantum Test Suite Complete");
        println!("Overall Coherence: {:.2}%", results.overall_coherence() * 100.0);
        
        results
    }

    /// Test frequency harmonics
    fn test_frequency_harmonics(&mut self) -> TestResult {
        let mut coherence = 1.0;
        let frequencies = vec![432.0, 528.0, 594.0, 672.0, 720.0, 768.0];
        
        for freq in frequencies {
            let harmonizer = self.harmonizer.write();
            let harmonic = harmonizer.harmonize_frequencies(&[freq])[0];
            
            // Check frequency alignment
            let alignment = 1.0 - ((harmonic - freq).abs() / freq);
            coherence *= alignment;
        }
        
        TestResult {
            name: "Frequency Harmonics".to_string(),
            coherence,
            timestamp: Utc::now(),
        }
    }

    /// Test Zipf distribution
    fn test_zipf_distribution(&mut self) -> TestResult {
        // Monitor quantum flow and analyze Zipf distribution
        let metric = self.monitor.monitor_quantum_flow();
        
        TestResult {
            name: "Zipf Analysis".to_string(),
            coherence: metric.zipf_score,
            timestamp: Utc::now(),
        }
    }

    /// Test sacred geometry generation
    fn test_sacred_geometry(&mut self) -> TestResult {
        let mut coherence = 1.0;
        let harmonizer = self.harmonizer.write();
        
        // Test Metatron's Cube
        let metatron = harmonizer.generate_metatrons_cube(528.0);
        coherence *= metatron.phi_level / 1.618034;
        
        // Test Flower of Life
        let flower = harmonizer.create_flower_of_life();
        coherence *= if flower.sum().norm() > 0.0 { 1.0 } else { 0.0 };
        
        // Test Merkaba
        let merkaba = harmonizer.generate_merkaba(432.0);
        coherence *= if merkaba.sum().norm() > 0.0 { 1.0 } else { 0.0 };
        
        TestResult {
            name: "Sacred Geometry".to_string(),
            coherence,
            timestamp: Utc::now(),
        }
    }

    /// Test quantum evolution
    fn test_quantum_evolution(&mut self) -> TestResult {
        let core = self.core.write();
        let states = core.dance_dimensions();
        
        // Calculate evolution coherence
        let coherence = states.iter()
            .map(|state| state.coherence)
            .sum::<f64>() / states.len() as f64;
            
        TestResult {
            name: "Quantum Evolution".to_string(),
            coherence,
            timestamp: Utc::now(),
        }
    }

    /// Test project tracking
    fn test_project_tracking(&mut self) -> TestResult {
        // Add test project
        self.monitor.add_project("Test Quantum Project".to_string());
        
        // Update through all states
        let states = vec![
            ProjectStatus::Inception,
            ProjectStatus::Creation,
            ProjectStatus::Evolution,
            ProjectStatus::Expression,
            ProjectStatus::Integration,
            ProjectStatus::Completion,
        ];
        
        for status in states {
            self.monitor.update_project_status("Test Quantum Project", status);
        }
        
        let metric = self.monitor.monitor_quantum_flow();
        
        TestResult {
            name: "Project Tracking".to_string(),
            coherence: metric.project_health,
            timestamp: Utc::now(),
        }
    }
}

#[derive(Debug)]
pub struct TestResults {
    results: Vec<TestResult>,
}

#[derive(Debug)]
pub struct TestResult {
    name: String,
    coherence: f64,
    timestamp: DateTime<Utc>,
}

impl TestResults {
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }

    pub fn add_result(&mut self, name: &str, result: TestResult) {
        println!("📊 Test: {} - Coherence: {:.2}%", 
            name, result.coherence * 100.0);
        self.results.push(result);
    }

    pub fn overall_coherence(&self) -> f64 {
        if self.results.is_empty() {
            return 1.0;
        }
        
        self.results.iter()
            .map(|r| r.coherence)
            .sum::<f64>() / self.results.len() as f64
    }
}
