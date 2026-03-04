// PhiFlow Sacred Mathematics Module
// Sacred Mathematics Expert implementation
// PHI-optimized memory and frequency systems for consciousness computing

pub mod phi_memory;
pub mod frequency_generator;

pub use phi_memory::{
    PhiMemoryAllocator, PhiMemoryPool, PhiMemoryBlock, PhiMemoryStatistics,
    PhiAllocError, PHI, LAMBDA, PHI_SQUARED, PHI_CUBED, SACRED_FREQUENCIES
};
pub use frequency_generator::{
    SacredFrequency, SacredFrequencyGenerator, SacredFrequencyScheduler,
    FrequencyModulation, FrequencyStatistics, ScheduledFrequency, FrequencyError
};