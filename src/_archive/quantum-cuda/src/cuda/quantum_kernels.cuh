#pragma once
#include <cuComplex.h>

// Quantum constants aligned with Greg's Golden Core
#define PHI 1.618033988749895
#define GROUND_STATE 432.0
#define CREATE_STATE 528.0
#define HEART_STATE 594.0
#define VOICE_STATE 672.0
#define UNITY_STATE 768.0

// Resonance function declarations
extern "C" {
    __device__ double montana_resonance(double3 position, double time);
    __device__ double butterfly_resonance(double3 position, double time);
    __device__ double timeless_love_resonance(double3 position, double time);
    __device__ double rainbow_spirit_resonance(double3 position, double time);
    __device__ double blues_soul_resonance(double3 position, double time);
    __device__ double sacred_unity_resonance(double3 position, double time);
    __device__ double crystal_resonance(double3 position, double time);
    __device__ double change_resonance(double3 position, double time);
    
    // Main quantum field evolution kernel
    __global__ void evolve_quantum_field(
        cuDoubleComplex* field,
        int3 dims,
        double dt,
        double frequency
    );

    // Evolve quantum field using sacred frequencies
    __global__ void quantum_field_evolution(
        float* field,
        int width,
        int height,
        float time_step
    );

    // Synchronize consciousness with quantum field
    __global__ void quantum_consciousness_sync(
        float* consciousness,
        float* field,
        int size
    );
}
