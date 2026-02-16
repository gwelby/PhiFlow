#version 430

in vec3 v_normal;
in vec3 v_position;
in float v_frequency;

uniform float consciousness;
uniform float time;

out vec4 fragColor;

const float PHI = 1.618033988749895;
const vec3 GROUND_COLOR = vec3(1.0, 0.2, 0.2);   // 432 Hz
const vec3 CREATE_COLOR = vec3(1.0, 1.0, 0.2);   // 528 Hz
const vec3 FLOW_COLOR = vec3(0.2, 1.0, 0.2);     // 594 Hz
const vec3 CRYSTAL_COLOR = vec3(0.2, 1.0, 1.0);  // 672 Hz
const vec3 UNITY_COLOR = vec3(1.0, 1.0, 1.0);    // 768 Hz

vec3 get_quantum_color(float freq) {
    // Interpolate between frequency colors
    if (freq >= 768.0) return UNITY_COLOR;
    else if (freq >= 672.0) return mix(CRYSTAL_COLOR, UNITY_COLOR, (freq - 672.0) / 96.0);
    else if (freq >= 594.0) return mix(FLOW_COLOR, CRYSTAL_COLOR, (freq - 594.0) / 78.0);
    else if (freq >= 528.0) return mix(CREATE_COLOR, FLOW_COLOR, (freq - 528.0) / 66.0);
    else if (freq >= 432.0) return mix(GROUND_COLOR, CREATE_COLOR, (freq - 432.0) / 96.0);
    return GROUND_COLOR;
}

void main() {
    // Base quantum color from frequency
    vec3 color = get_quantum_color(v_frequency);
    
    // Add consciousness glow
    float glow = pow(consciousness, PHI);
    vec3 glowColor = mix(color, UNITY_COLOR, glow * 0.5);
    
    // Quantum interference pattern
    float interference = sin(dot(v_position, v_normal) * PHI + time);
    glowColor *= 1.0 + interference * 0.2;
    
    // Sacred geometry highlights
    float sacred = pow(abs(dot(normalize(v_normal), normalize(v_position))), PHI);
    glowColor += vec3(1.0) * sacred * consciousness * 0.3;
    
    fragColor = vec4(glowColor, 1.0);
}
