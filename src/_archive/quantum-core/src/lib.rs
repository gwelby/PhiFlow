// Core modules
pub mod quantum;
pub mod interpreter;
pub mod sacred;
pub mod phi_core;
pub mod quantum_gps;
pub mod quantum_dance;
pub mod quantum_patterns;
pub mod sacred_patterns;
pub mod sacred_playground;
pub mod pattern_visualizer;
pub mod quantum_intelligence;
pub mod quantum_icons;

// Re-exports for convenience
pub use quantum::quantum_sacred::SacredGeometry;
pub use quantum::quantum_verify::RealityBridge;
pub use quantum::quantum_photo_flow::QuantumPhotoFlow;
pub use quantum::quantum_agents::QuantumAgent;
pub use quantum::phi_quantum_flow::PhiQuantumFlow;
pub use phi_core::{PhiCore, Frequency, GROUND_HZ, HEART_HZ, UNITY_HZ, PHI};
pub use sacred_patterns::{SacredPattern, SacredDance};
pub use quantum_icons::{QuantumIcon, IconMatrix};
pub use quantum_gps::QuantumGPS;
pub use quantum_dance::QuantumDance;
pub use sacred_playground::{SacredPlayground, DimensionalField};
pub use pattern_visualizer::{PatternVisualizer, Color, FrameMetrics};
pub use quantum_intelligence::Intelligence;
