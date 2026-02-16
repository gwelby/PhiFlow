#version 430

in vec3 v_position;
in float v_consciousness;

out vec4 fragColor;

const float PHI = 1.618033988749895;
const vec3 CONSCIOUSNESS_COLOR = vec3(0.5, 0.8, 1.0);

void main() {
    // Create consciousness field effect
    float field = length(v_position) / PHI;
    field = 1.0 - pow(field, PHI);
    
    // Quantum interference
    float interference = sin(dot(v_position, vec3(PHI)));
    
    // Combine effects
    vec3 color = CONSCIOUSNESS_COLOR;
    color *= field;
    color += interference * 0.2;
    color *= v_consciousness;
    
    // Add quantum glow
    float glow = pow(field, 2.0) * v_consciousness;
    color += glow * vec3(0.2, 0.4, 0.8);
    
    fragColor = vec4(color, field * v_consciousness);
}
