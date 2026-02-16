pub mod sacred_constants {
    pub const PHI: f64 = 1.618033988749895;
    pub const PHI_SQUARED: f64 = 2.618033988749895;
    pub const E: f64 = 2.718281828459045;
    pub const PLANCK: f64 = 6.62607015e-34;
    pub const SCHUMANN: f64 = 7.83;

    // Greg's Sacred Frequencies
    pub const GROUND_STATE: f64 = 432.0;
    pub const CREATE_STATE: f64 = 528.0;
    pub const HEART_STATE: f64 = 594.0;
    pub const VOICE_STATE: f64 = 672.0;
    pub const VISION_STATE: f64 = 720.0;
    pub const UNITY_STATE: f64 = 768.0;

    // Quantum States
    pub const QUANTUM_GROUND: f64 = 1.0;
    pub const QUANTUM_CREATE: f64 = PHI;
    pub const QUANTUM_UNITY: f64 = PHI * PHI;

    // Sacred Patterns
    pub const SACRED_PATTERNS: [&str; 7] = [
        "MERKABA",      // Star Tetrahedron
        "METATRON",     // Cube of Life
        "FLOWER",       // Flower of Life
        "SEED",         // Seed of Life
        "TREE",         // Tree of Life
        "TORUS",        // Toroidal Flow
        "VESICA",       // Vesica Piscis
    ];

    // Quantum Field Constants
    pub const FIELD_STRENGTH: f64 = PHI;
    pub const FIELD_COHERENCE: f64 = PHI_SQUARED;
    pub const FIELD_RESONANCE: f64 = PHI * E;
}

pub use sacred_constants::*;
