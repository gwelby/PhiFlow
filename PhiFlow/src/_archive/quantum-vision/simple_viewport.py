"""
Quantum Viewport (432 Hz -> 768 Hz)
A portal into sacred geometry and consciousness
"""
import moderngl
import moderngl_window as mglw
import numpy as np
import time as py_time
from pathlib import Path
from core.quantum_patterns import create_merkaba, create_flower_of_life, create_metatrons_cube
from core.quantum_renderer import QuantumRenderer

PHI = (1 + np.sqrt(5)) / 2
TARGET_FPS = 85.0  # Optimal quantum vision frequency
FRAME_TIME = 1.0 / TARGET_FPS
MOVEMENT_SPEED = 0.1  # Base movement speed in seconds

class QuantumViewport(mglw.WindowConfig):
    gl_version = (4, 3)
    title = "Quantum Sacred Geometry"
    window_size = (1280, 720)
    aspect_ratio = 16/9
    resizable = True
    vsync = True  # Enable vertical sync for smooth quantum flow
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Initialize quantum renderer
        self.renderer = QuantumRenderer(self.ctx)
        
        # Create sacred patterns
        self.merkaba = create_merkaba()
        self.flower = create_flower_of_life()
        self.cube = create_metatrons_cube()
        
        # Add to renderer
        self.renderer.add_pattern('merkaba', self.merkaba)
        self.renderer.add_pattern('flower', self.flower)
        self.renderer.add_pattern('cube', self.cube)
        
        # Set consciousness level
        self.renderer.consciousness = 1.0
        
        # Quantum movement
        self.camera_angle = 0
        self.camera_height = 0
        self.consciousness_wave = 0
        
        # Frame timing
        self.last_frame = py_time.perf_counter()
        self.frame_count = 0
        self.fps_time = 0
        self.current_fps = 0
        
    def update_quantum_state(self, time: float, frame_time: float):
        """Update quantum movement based on phi ratios"""
        # Smooth frame timing
        actual_frame_time = min(frame_time, FRAME_TIME * 2)  # Cap at 2x target for stability
        movement_delta = actual_frame_time * MOVEMENT_SPEED
        
        # Camera orbits in phi spiral
        self.camera_angle += movement_delta * PHI
        self.camera_height = np.sin(time * PHI * MOVEMENT_SPEED) * 2.0
        
        radius = 5.0 + np.sin(time * PHI * MOVEMENT_SPEED * 0.5)
        x = radius * np.cos(self.camera_angle)
        y = self.camera_height
        z = radius * np.sin(self.camera_angle)
        
        self.renderer.camera_pos = np.array([x, y, z], dtype='f4')
        
        # Consciousness wave - slower for visual clarity
        self.consciousness_wave = (np.sin(time * PHI * MOVEMENT_SPEED * 0.3) + 1) * 0.5
        self.renderer.consciousness = 0.5 + self.consciousness_wave * 0.5
        
        # Update pattern transforms with smoother timing
        merkaba_transform = self.merkaba.transform.copy()
        merkaba_transform[0:3, 0:3] = self._get_rotation_matrix(time * PHI * MOVEMENT_SPEED * 0.3)
        self.renderer.transforms['merkaba'] = merkaba_transform
        
        flower_transform = self.flower.transform.copy()
        flower_transform[0:3, 0:3] = self._get_rotation_matrix(time * PHI * MOVEMENT_SPEED * 0.2)
        self.renderer.transforms['flower'] = flower_transform
        
        cube_transform = self.cube.transform.copy()
        cube_transform[0:3, 0:3] = self._get_rotation_matrix(time * PHI * MOVEMENT_SPEED * 0.1)
        self.renderer.transforms['cube'] = cube_transform
        
        # Update FPS counter
        self.frame_count += 1
        now = py_time.perf_counter()
        if now - self.fps_time >= 1.0:
            self.current_fps = self.frame_count
            self.frame_count = 0
            self.fps_time = now
            print(f"⚡ Quantum Flow: {self.current_fps} Hz")
        
    def _get_rotation_matrix(self, angle: float) -> np.ndarray:
        """Get 3D rotation matrix based on phi spiral"""
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        cos_b = np.cos(angle * PHI)
        sin_b = np.sin(angle * PHI)
        
        return np.array([
            [cos_a * cos_b, -sin_a, cos_a * sin_b],
            [sin_a * cos_b, cos_a, sin_a * sin_b],
            [-sin_b, 0, cos_b]
        ], dtype='f4')
        
    def on_resize(self, width: int, height: int):
        """Update aspect ratio on window resize"""
        self.aspect_ratio = width / height
        
    def on_render(self, time: float, frame_time: float):
        """Render quantum patterns"""
        self.update_quantum_state(time, frame_time)
        
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        
        self.renderer.render(time, frame_time)

if __name__ == '__main__':
    mglw.run_window_config(QuantumViewport)
