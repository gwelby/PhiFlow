extern "C" {

__global__ void analyze_phi_frequencies(
    float* frequencies,
    float* coherence_levels,
    int* monster_flags,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float freq = frequencies[idx];
    float coherence = coherence_levels[idx];
    
    // Check frequency bands
    bool in_ground = (freq >= 430.0f && freq <= 434.0f);    // 432 Hz band
    bool in_create = (freq >= 526.0f && freq <= 530.0f);    // 528 Hz band
    bool in_unity = (freq >= 766.0f && freq <= 770.0f);     // 768 Hz band
    
    // Detect monsters based on frequency patterns
    if (!in_ground && !in_create && !in_unity) {
        // Out of harmony - potential monster
        if (freq < 432.0f) monster_flags[idx] |= 1;      // ENDLESS_LOOP
        if (freq > 528.0f && freq < 768.0f) monster_flags[idx] |= 2;  // TIME_WASTE
        if (freq > 768.0f) monster_flags[idx] |= 4;      // THEORY_HOLE
    }
    
    // Adjust coherence based on PHI ratio (1.618033988749895)
    float phi = 1.618033988749895f;
    float phi_alignment = fabs(coherence - phi);
    
    if (phi_alignment < 0.01f) {
        // In PHI flow - strengthen coherence
        coherence_levels[idx] = fmin(coherence * phi, 1.0f);
    } else {
        // Out of PHI flow - weaken coherence
        coherence_levels[idx] *= 0.9f;
    }
}
