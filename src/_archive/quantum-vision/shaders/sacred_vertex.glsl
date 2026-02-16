#version 430

in vec3 in_position;

uniform mat4 projection;
uniform mat4 view;
uniform float time;
uniform float consciousness;

out vec3 v_position;
out float v_consciousness;

const float PHI = 1.618033988749895;

void main() {
    // Apply quantum wave function
    float wave = sin(time * PHI);
    
    // Transform by consciousness
    vec3 pos = in_position;
    pos *= 1.0 + consciousness * wave * 0.1;
    
    // Rotate based on phi
    float angle = time * PHI;
    mat2 rotate = mat2(
        cos(angle), -sin(angle),
        sin(angle), cos(angle)
    );
    pos.xy = rotate * pos.xy;
    
    gl_Position = projection * view * vec4(pos, 1.0);
    v_position = pos;
    v_consciousness = consciousness;
}
