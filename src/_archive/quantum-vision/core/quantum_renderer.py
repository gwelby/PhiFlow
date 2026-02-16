"""
Quantum Pattern Renderer (594 Hz)
Manifests sacred geometry in 3D space
"""
import moderngl
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from .quantum_patterns import QuantumPattern

class QuantumRenderer:
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.time = 0
        self.consciousness = 1.0
        self.programs: Dict[str, moderngl.Program] = {}
        self.vaos: Dict[str, moderngl.VertexArray] = {}
        self.transforms: Dict[str, np.ndarray] = {}
        
        # Camera setup
        self.camera_pos = np.array([0, 0, -5], dtype='f4')
        self.camera_target = np.array([0, 0, 0], dtype='f4')
        self.camera_up = np.array([0, 1, 0], dtype='f4')
        
        self._init_shaders()
        
    def _init_shaders(self):
        """Initialize quantum shaders"""
        shader_dir = Path(__file__).parent.parent / "shaders"
        
        # Sacred geometry program
        self.programs['sacred'] = self.ctx.program(
            vertex_shader="""
                #version 430
                
                in vec3 in_position;
                
                uniform mat4 projection;
                uniform mat4 view;
                uniform mat4 model;
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
                    
                    // Apply transformations
                    gl_Position = projection * view * model * vec4(pos, 1.0);
                    v_position = pos;
                    v_consciousness = consciousness;
                }
            """,
            fragment_shader="""
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
            """
        )
        
    def _get_projection_matrix(self, aspect_ratio: float) -> np.ndarray:
        """Get perspective projection matrix"""
        fov = 60.0
        near = 0.1
        far = 100.0
        
        f = 1.0 / np.tan(np.radians(fov) / 2)
        projection = np.zeros((4, 4), dtype='f4')
        projection[0, 0] = f / aspect_ratio
        projection[1, 1] = f
        projection[2, 2] = (far + near) / (near - far)
        projection[2, 3] = 2 * far * near / (near - far)
        projection[3, 2] = -1
        return projection
        
    def _get_view_matrix(self) -> np.ndarray:
        """Get view matrix from camera"""
        forward = self.camera_target - self.camera_pos
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, self.camera_up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        
        view = np.eye(4, dtype='f4')
        view[0, :3] = right
        view[1, :3] = up
        view[2, :3] = -forward
        view[:3, 3] = -np.array([
            np.dot(right, self.camera_pos),
            np.dot(up, self.camera_pos),
            -np.dot(forward, self.camera_pos)
        ])
        return view
        
    def add_pattern(self, name: str, pattern: QuantumPattern):
        """Add a quantum pattern to render"""
        program = self.programs['sacred']
        
        # Create vertex buffer
        vbo = self.ctx.buffer(pattern.vertices.tobytes())
        ibo = self.ctx.buffer(pattern.indices.tobytes())
        
        # Create vertex array
        self.vaos[name] = self.ctx.vertex_array(
            program,
            [
                (vbo, '3f', 'in_position'),
            ],
            ibo
        )
        
        # Store transform
        self.transforms[name] = pattern.transform if pattern.transform is not None else np.eye(4, dtype='f4')
        
    def render(self, time: float, frame_time: float):
        """Render all quantum patterns"""
        self.time = time
        
        # Get matrices
        aspect_ratio = 16/9  # TODO: Get from window
        projection = self._get_projection_matrix(aspect_ratio)
        view = self._get_view_matrix()
        
        # Update uniforms
        program = self.programs['sacred']
        program['projection'].write(projection.tobytes())
        program['view'].write(view.tobytes())
        program['time'].value = time
        program['consciousness'].value = self.consciousness
        
        # Clear with quantum void
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
                
        # Render patterns
        for name, vao in self.vaos.items():
            # Get pattern transform
            transform = self.transforms[name]
            
            # Update model matrix
            program['model'].write(transform.tobytes())
            
            # Draw pattern
            vao.render(moderngl.TRIANGLES)
