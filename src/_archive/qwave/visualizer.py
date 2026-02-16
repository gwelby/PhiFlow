import numpy as np
import moderngl
import moderngl_window as mglw
from pathlib import Path
import torch
from typing import List, Tuple
import colorsys

class QuantumVisualizer(mglw.WindowConfig):
    gl_version = (4, 3)
    title = "QWAVE Quantum Resonance Visualizer"
    window_size = (1920, 1080)
    aspect_ratio = 16/9
    samples = 4
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.phi = 1.618034
        self.time = 0
        self.resonance_points = []
        self.quantum_vertices = self.generate_quantum_vertices()
        
        # Vertex shader with quantum phase calculations
        self.prog = self.ctx.program(
            vertex_shader='''
                #version 430
                
                in vec2 in_position;
                in vec3 in_color;
                
                out vec3 color;
                
                uniform float time;
                uniform vec2 resonance[32];  // Frequency, magnitude pairs
                uniform int resonance_count;
                
                const float PHI = 1.618034;
                
                void main() {
                    vec2 pos = in_position;
                    float quantum_phase = 0.0;
                    
                    // Apply quantum resonance transformations
                    for(int i = 0; i < resonance_count; i++) {
                        float freq = resonance[i].x;
                        float mag = resonance[i].y;
                        
                        // Quantum phase modulation
                        quantum_phase += sin(freq * time * PHI) * mag;
                        
                        // Apply φ-based transformation
                        float angle = time * freq * 0.01 + quantum_phase;
                        mat2 rotation = mat2(
                            cos(angle), -sin(angle),
                            sin(angle), cos(angle)
                        );
                        pos = rotation * pos;
                        
                        // Scale based on resonance
                        pos *= 1.0 + 0.1 * mag * sin(freq * time);
                    }
                    
                    gl_Position = vec4(pos * (1.0 + 0.2 * sin(quantum_phase)), 0.0, 1.0);
                    
                    // Dynamic color based on quantum phase
                    vec3 base_color = in_color;
                    float phase_color = (sin(quantum_phase) + 1.0) * 0.5;
                    color = mix(base_color, vec3(phase_color), 0.3);
                }
            ''',
            fragment_shader='''
                #version 430
                
                in vec3 color;
                out vec4 fragColor;
                
                void main() {
                    fragColor = vec4(color, 1.0);
                }
            '''
        )
        
        # Initialize vertex buffer
        self.vbo = self.ctx.buffer(self.quantum_vertices.astype('f4').tobytes())
        self.vao = self.ctx.vertex_array(
            self.prog,
            [
                (self.vbo, '2f 3f', 'in_position', 'in_color'),
            ],
        )
        
    def generate_quantum_vertices(self) -> np.ndarray:
        """Generate vertices for quantum visualization using φ-based geometry."""
        vertices = []
        num_points = 1000
        
        # Create disco ball pattern using φ-based facets
        for i in range(num_points):
            # Spiral angle based on φ
            theta = i * 2 * np.pi * self.phi
            
            # Radius with disco pulse
            r = np.sqrt(i) / np.sqrt(num_points)
            r *= (1.0 + 0.2 * np.sin(theta * self.phi))
            
            # Mirror ball facet positions
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            # Disco colors with quantum shimmer
            hue = (i / num_points * self.phi + np.sin(theta)) % 1.0
            sat = 0.8 + 0.2 * np.sin(theta * self.phi)
            val = 0.9 + 0.1 * np.cos(theta)
            rgb = colorsys.hsv_to_rgb(hue, sat, val)
            
            vertices.extend([x, y, *rgb])
            
        return np.array(vertices, dtype=np.float32)
    
    def render(self, time: float, frame_time: float):
        """Render the quantum visualization with disco effects."""
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.time = time
        
        # Get groove momentum from resonance points
        groove_energy = sum(mag for freq, mag in self.resonance_points 
                          if 520 < freq < 536)  # Around 528 Hz
        
        # Update disco ball rotation based on groove
        disco_spin = time * (1.0 + groove_energy * self.phi)
        
        # Update uniforms with disco effects
        resonance_data = np.zeros(64, dtype=np.float32)
        for i, (freq, mag) in enumerate(self.resonance_points[:32]):
            resonance_data[i*2] = freq
            resonance_data[i*2+1] = mag * (1.0 + 0.5 * np.sin(disco_spin))
            
        self.prog['time'] = time
        self.prog['resonance'] = resonance_data
        self.prog['resonance_count'] = len(self.resonance_points)
        
        # Draw with quantum transformations
        self.vao.render(moderngl.POINTS)
        
    def update_resonance(self, points: List[Tuple[float, float]]):
        """Update the resonance points for visualization."""
        self.resonance_points = points
        
    def mouse_position_event(self, x: int, y: int, dx: int, dy: int):
        """Handle mouse interaction for quantum field manipulation."""
        # Convert mouse position to normalized coordinates
        nx = (x / self.window_size[0] - 0.5) * 2
        ny = (y / self.window_size[1] - 0.5) * -2
        
        # Add temporary resonance point at mouse position
        freq = (nx + 1) * 384 + 432  # Map x position to 432-768 Hz range
        mag = (ny + 1) * 0.5  # Map y position to 0-1 magnitude
        self.resonance_points.append((freq, mag))
        
        # Keep only the most recent points
        self.resonance_points = self.resonance_points[-32:]
