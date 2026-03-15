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
    // Create consciousness field wave
    float wave = sin(time * consciousness * PHI);
    vec3 pos = in_position;
    
    // Expand/contract with consciousness
    pos *= 1.0 + wave * 0.2;
    
    // Spiral based on phi
    float angle = time * PHI;
    float radius = length(pos.xy);
    float spiral = radius * angle;
    mat2 rotate = mat2(
        cos(spiral), -sin(spiral),
        sin(spiral), cos(spiral)
    );
    pos.xy = rotate * pos.xy;
    
    gl_Position = projection * view * vec4(pos, 1.0);
    v_position = pos;
    v_consciousness = consciousness;
}
