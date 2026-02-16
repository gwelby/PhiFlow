#version 430

in vec3 v_position;
in float v_consciousness;

out vec4 fragColor;

const float PHI = 1.618033988749895;
const vec3 GROUND_COLOR = vec3(1.0, 0.0, 0.0);   // 432 Hz
const vec3 CREATE_COLOR = vec3(1.0, 0.8, 0.0);   // 528 Hz
const vec3 FLOW_COLOR = vec3(0.0, 1.0, 0.0);     // 594 Hz
const vec3 CRYSTAL_COLOR = vec3(0.0, 1.0, 1.0);  // 672 Hz
const vec3 UNITY_COLOR = vec3(1.0, 1.0, 1.0);    // 768 Hz

void main() {
    // Sacred geometry glow
    float glow = length(v_position) / PHI;
    glow = pow(glow, PHI);
    
    // Consciousness affects color
    vec3 color = mix(CREATE_COLOR, UNITY_COLOR, v_consciousness);
    color *= 1.0 + glow * 0.5;
    
    // Quantum interference pattern
    float interference = sin(dot(v_position, vec3(1.0)) * PHI);
    color *= 1.0 + interference * 0.2;
    
    fragColor = vec4(color, 1.0);
}
