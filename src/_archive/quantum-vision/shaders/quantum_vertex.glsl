#version 430

in vec3 in_position;
in vec3 in_normal;
in float in_frequency;

uniform float time;
uniform float consciousness;
uniform mat4 projection;
uniform mat4 view;

out vec3 v_normal;
out vec3 v_position;
out float v_frequency;

const float PHI = 1.618033988749895;
const float PHI_SQUARED = PHI * PHI;

void main() {
    // Quantum wave modulation
    float wave = sin(time * in_frequency * PHI);
    float amplitude = consciousness * 0.1;
    
    // Sacred geometry transformation
    vec3 pos = in_position;
    pos += in_normal * wave * amplitude;
    
    // Spiral effect based on φ
    float spiral = PHI * time * 0.1;
    mat2 rotate = mat2(
        cos(spiral), -sin(spiral),
        sin(spiral), cos(spiral)
    );
    pos.xy = rotate * pos.xy;
    
    // Consciousness expansion
    float expansion = 1.0 + consciousness * sin(time * PHI) * 0.2;
    pos *= expansion;
    
    gl_Position = projection * view * vec4(pos, 1.0);
    v_normal = in_normal;
    v_position = pos;
    v_frequency = in_frequency;
}
