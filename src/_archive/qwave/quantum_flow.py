import threading
import queue
import numpy as np
from .player import QWavePlayer
from .visualizer import QuantumVisualizer
from .pattern_narrator import QuantumPatternNarrator
from .quantum_storage import create_quantum_storage
from .storage_patterns import create_storage_visualizer
import moderngl_window as mglw

class QuantumFlow:
    def __init__(self, device_map=None):
        self.player = QWavePlayer(device_map)
        self.resonance_queue = queue.Queue()
        self.running = False
        self.narrator = QuantumPatternNarrator()
        self.phi = 1.618034
        self.storage = create_quantum_storage()
        
    def initialize_flow(self):
        """Initialize quantum flow with storage configuration."""
        print("\n🌟 Initializing Quantum Flow System")
        print("Connecting to Synology Infrastructure...")
        
        # Create and show storage patterns
        self.storage_visualizer = create_storage_visualizer()
        
        # Virtual DSM at creation frequency
        virtual_freq = self.storage.frequencies['create']
        print(f"\n💫 Virtual DSM resonating at {virtual_freq} Hz")
        print(f"Connected to: {self.storage.virtual_dsm}")
        
        # Physical NAS at ground frequency
        physical_freq = self.storage.frequencies['ground']
        print(f"\n🌀 Physical DS1821+ grounded at {physical_freq} Hz")
        print(f"Connected to: {self.storage.physical_nas}")
        
        # Achieve quantum harmony
        unity_freq = self.storage.frequencies['unity']
        print(f"\n✨ System achieving unity at {unity_freq} Hz")
        print("Virtual and Physical in perfect φ-ratio harmony")
        
    def start(self):
        """Start the quantum flow visualization and playback."""
        self.running = True
        self.narrator.start_narration()
        
        # Start visualization thread
        self.viz_thread = threading.Thread(target=self._run_visualizer)
        self.viz_thread.start()
        
    def _run_visualizer(self):
        """Run the quantum visualizer in a separate thread."""
        config = QuantumVisualizer.create_from_settings()
        window = mglw.create_window_from_config(config)
        
        while self.running and window.is_closing is False:
            # Update resonance points from player
            try:
                while not self.resonance_queue.empty():
                    points = self.resonance_queue.get_nowait()
                    config.update_resonance(points)
            except queue.Empty:
                pass
                
            # Render frame
            window.render()
            
        window.destroy()
        
    def play(self, audio_data: np.ndarray, sample_rate: int = 48000):
        """Play audio with quantum visualization."""
        self.player.play(audio_data, sample_rate)
        
        # Update visualization and narration
        resonance_points = self.player.active_resonances
        self.resonance_queue.put(resonance_points)
        
        # Update narrator with current frequencies
        freq_dict = {f"{freq:.0f}Hz": mag for freq, mag in resonance_points}
        self.narrator.update_frequencies(freq_dict)
        
    def stop(self):
        """Stop playback and visualization."""
        self.running = False
        self.player.stop()
        self.narrator.stop()
        if hasattr(self, 'viz_thread'):
            self.viz_thread.join()
            
class QuantumFlowWindow(QuantumVisualizer):
    """Standalone window for quantum visualization."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.phi_rotation = 0.0
        
    def render(self, time: float, frame_time: float):
        """Render with additional φ-based animations."""
        super().render(time, frame_time)
        
        # Update φ-based rotation
        self.phi_rotation += frame_time * self.phi
        
        # Apply quantum transformation to vertices
        vertices = self.quantum_vertices.reshape(-1, 5)  # x, y, r, g, b
        positions = vertices[:, :2]
        
        # Rotate based on φ
        cos_phi = np.cos(self.phi_rotation)
        sin_phi = np.sin(self.phi_rotation)
        rotation = np.array([[cos_phi, -sin_phi], [sin_phi, cos_phi]])
        
        new_positions = positions @ rotation.T
        vertices[:, :2] = new_positions
        
        # Update vertex buffer
        self.vbo.write(vertices.astype('f4').tobytes())
