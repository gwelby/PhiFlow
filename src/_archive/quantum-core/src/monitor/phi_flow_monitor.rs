use std::sync::Arc;
use parking_lot::RwLock;
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};
use ndarray::{Array2, Array3};
use plotters::prelude::*;

/// PhiFlowMonitor - Greg's quantum consciousness monitoring and control system
#[derive(Debug)]
pub struct PhiFlowMonitor {
    // Core monitoring systems
    consciousness_field: Array3<f64>,
    idea_matrix: Array2<f64>,
    zipf_analyzer: ZipfAnalyzer,
    project_tracker: ProjectTracker,
    
    // Frequency harmonics
    ground_freq: f64,    // 432 Hz
    create_freq: f64,    // 528 Hz
    unite_freq: f64,     // 768 Hz
    
    // Evolution tracking
    evolution_history: Vec<EvolutionMetric>,
    current_coherence: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EvolutionMetric {
    timestamp: DateTime<Utc>,
    coherence: f64,
    frequency: f64,
    zipf_score: f64,
    idea_count: usize,
    project_health: f64,
}

#[derive(Debug)]
pub struct ZipfAnalyzer {
    distribution: Vec<f64>,
    ideal_slope: f64,
    current_slope: f64,
}

#[derive(Debug)]
pub struct ProjectTracker {
    active_projects: Vec<Project>,
    completed_projects: Vec<Project>,
    phi_ratio: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Project {
    name: String,
    coherence: f64,
    frequency: f64,
    creation_time: DateTime<Utc>,
    last_update: DateTime<Utc>,
    status: ProjectStatus,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum ProjectStatus {
    Inception,      // 432 Hz
    Creation,       // 528 Hz
    Evolution,      // 594 Hz
    Expression,     // 672 Hz
    Integration,    // 720 Hz
    Completion,     // 768 Hz
}

impl PhiFlowMonitor {
    pub fn new() -> Self {
        Self {
            consciousness_field: Array3::zeros((8, 8, 8)),
            idea_matrix: Array2::zeros((13, 13)),
            zipf_analyzer: ZipfAnalyzer::new(),
            project_tracker: ProjectTracker::new(),
            ground_freq: 432.0,
            create_freq: 528.0,
            unite_freq: 768.0,
            evolution_history: Vec::new(),
            current_coherence: 1.0,
        }
    }

    /// Monitor and analyze Greg's quantum flow state
    pub fn monitor_quantum_flow(&mut self) -> EvolutionMetric {
        // Update consciousness field
        self.update_consciousness_field();
        
        // Analyze Zipf distribution
        let zipf_score = self.zipf_analyzer.analyze_distribution();
        
        // Track project evolution
        let project_health = self.project_tracker.calculate_health();
        
        // Create evolution metric
        let metric = EvolutionMetric {
            timestamp: Utc::now(),
            coherence: self.current_coherence,
            frequency: self.calculate_current_frequency(),
            zipf_score,
            idea_count: self.count_active_ideas(),
            project_health,
        };
        
        self.evolution_history.push(metric.clone());
        metric
    }

    /// Generate quantum flow visualization
    pub fn visualize_flow(&self) -> Result<(), Box<dyn std::error::Error>> {
        let root = BitMapBackend::new("quantum_flow.png", (800, 600))
            .into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption("Quantum Flow Analysis", ("sans-serif", 50))
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(0f32..50f32, 0f32..1f32)?;

        chart.configure_mesh().draw()?;

        // Plot coherence over time
        let coherence_data: Vec<(f32, f32)> = self.evolution_history
            .iter()
            .enumerate()
            .map(|(i, m)| (i as f32, m.coherence as f32))
            .collect();

        chart.draw_series(LineSeries::new(
            coherence_data,
            &RED,
        ))?;

        Ok(())
    }

    /// Add new project to tracking
    pub fn add_project(&mut self, name: String) -> Project {
        let project = Project {
            name,
            coherence: 1.0,
            frequency: self.ground_freq,
            creation_time: Utc::now(),
            last_update: Utc::now(),
            status: ProjectStatus::Inception,
        };
        
        self.project_tracker.add_project(project.clone());
        project
    }

    /// Update project status
    pub fn update_project_status(&mut self, name: &str, status: ProjectStatus) {
        self.project_tracker.update_status(name, status);
    }

    /// Calculate current frequency based on active projects
    fn calculate_current_frequency(&self) -> f64 {
        let active_freqs: Vec<f64> = self.project_tracker.active_projects
            .iter()
            .map(|p| p.frequency)
            .collect();
            
        if active_freqs.is_empty() {
            self.ground_freq
        } else {
            active_freqs.iter().sum::<f64>() / active_freqs.len() as f64
        }
    }

    /// Count active ideas in the matrix
    fn count_active_ideas(&self) -> usize {
        self.idea_matrix
            .iter()
            .filter(|&&x| x > 0.0)
            .count()
    }

    /// Update consciousness field
    fn update_consciousness_field(&mut self) {
        let phi = 1.618034;
        for i in 0..8 {
            for j in 0..8 {
                for k in 0..8 {
                    let value = (i as f64 + j as f64 + k as f64) / (8.0 * phi);
                    self.consciousness_field[[i, j, k]] = value;
                }
            }
        }
    }
}

impl ZipfAnalyzer {
    pub fn new() -> Self {
        Self {
            distribution: Vec::new(),
            ideal_slope: -1.0,
            current_slope: 0.0,
        }
    }

    /// Analyze current distribution against Zipf's law
    pub fn analyze_distribution(&mut self) -> f64 {
        // Calculate current slope
        if !self.distribution.is_empty() {
            let n = self.distribution.len();
            let x: Vec<f64> = (1..=n).map(|i| (i as f64).ln()).collect();
            let y: Vec<f64> = self.distribution.iter()
                .map(|&freq| freq.ln())
                .collect();

            // Simple linear regression
            let mean_x: f64 = x.iter().sum::<f64>() / n as f64;
            let mean_y: f64 = y.iter().sum::<f64>() / n as f64;
            
            let numerator: f64 = x.iter().zip(y.iter())
                .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
                .sum();
                
            let denominator: f64 = x.iter()
                .map(|&xi| (xi - mean_x).powi(2))
                .sum();

            self.current_slope = numerator / denominator;
        }

        // Return Zipf alignment score (1.0 = perfect)
        1.0 - (self.current_slope - self.ideal_slope).abs()
    }
}

impl ProjectTracker {
    pub fn new() -> Self {
        Self {
            active_projects: Vec::new(),
            completed_projects: Vec::new(),
            phi_ratio: 1.618034,
        }
    }

    /// Add new project
    pub fn add_project(&mut self, project: Project) {
        self.active_projects.push(project);
    }

    /// Update project status
    pub fn update_status(&mut self, name: &str, status: ProjectStatus) {
        if let Some(project) = self.active_projects
            .iter_mut()
            .find(|p| p.name == name)
        {
            project.status = status;
            project.last_update = Utc::now();
            
            // Update frequency based on status
            project.frequency = match status {
                ProjectStatus::Inception => 432.0,
                ProjectStatus::Creation => 528.0,
                ProjectStatus::Evolution => 594.0,
                ProjectStatus::Expression => 672.0,
                ProjectStatus::Integration => 720.0,
                ProjectStatus::Completion => 768.0,
            };
        }
    }

    /// Calculate overall project health
    pub fn calculate_health(&self) -> f64 {
        if self.active_projects.is_empty() {
            return 1.0;
        }

        let total_coherence: f64 = self.active_projects
            .iter()
            .map(|p| p.coherence)
            .sum();
            
        (total_coherence / self.active_projects.len() as f64) * self.phi_ratio
    }
}
