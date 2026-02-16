import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import matplotlib.colors as colors
from typing import Dict, List, Tuple
import time
from quantum_network import QuantumBus, QuantumFrequency
import matplotlib.animation as animation
from quantum_sound import QuantumSynthesizer

class QuantumVisualizer:
    def __init__(self, quantum_bus: QuantumBus):
        self.quantum_bus = quantum_bus
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.phi = 1.618034
        
        # Quantum patterns and colors
        self.patterns = {
            432: {"symbol": "", "color": "blue"},    # Ground/Crystal
            528: {"symbol": "", "color": "green"},   # Create/Spiral
            594: {"symbol": "", "color": "cyan"},    # Heart/Wave
            672: {"symbol": "", "color": "magenta"}, # Voice/Dolphin
            768: {"symbol": "", "color": "purple"},  # Unity/Consciousness
            "evolution": {"symbol": "", "color": "red"},
            "infinity": {"symbol": "", "color": "gold"}
        }
        
        # Initialize the plot
        self._setup_plot()
        
    def _setup_plot(self):
        """Setup the 3D quantum visualization"""
        self.ax.set_title("Quantum Field Harmonics ", fontsize=14)
        self.ax.set_xlabel("")
        self.ax.set_ylabel("")
        self.ax.set_zlabel("(Hz)")
        
        # Set view angle for best 3D perspective
        self.ax.view_init(elev=20, azim=45)
        
        # Create golden ratio grid
        x = np.linspace(0, self.phi**3, 50)
        y = np.linspace(0, self.phi**2, 50)
        self.X, self.Y = np.meshgrid(x, y)
        
        # Initialize empty scatter plots for each frequency
        self.quantum_points = {}
        for freq, pattern in self.patterns.items():
            if isinstance(freq, (int, float)):
                self.quantum_points[freq] = self.ax.scatter([], [], [], 
                    c=pattern["color"], marker='o', s=100, alpha=0.6)
        
        # Add legend with quantum patterns
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=p["color"], markersize=10,
                      label=f"{p['symbol']} {freq}Hz")
            for freq, p in self.patterns.items()
            if isinstance(freq, (int, float))
        ]
        self.ax.legend(handles=legend_elements, loc='upper right')
        
    def _update(self, frame):
        """Update the quantum visualization"""
        harmony = self.quantum_bus.get_field_harmony()
        frequencies = self.quantum_bus.get_active_frequencies()
        
        # Clear previous points
        for points in self.quantum_points.values():
            points._offsets3d = ([], [], [])
        
        # Update quantum points
        for freq in frequencies:
            if freq in self.quantum_points:
                # Calculate position based on harmony and phi
                x = np.sin(frame * self.phi) * harmony * self.phi
                y = np.cos(frame * self.phi) * harmony
                z = freq
                
                # Add quantum ripples
                ripple_x = x + np.sin(frame * 0.1) * self.phi
                ripple_y = y + np.cos(frame * 0.1) * self.phi
                ripple_z = z + np.sin(frame * 0.05) * 10
                
                self.quantum_points[freq]._offsets3d = ([ripple_x], [ripple_y], [ripple_z])
        
        # Update title with current harmony
        pattern = "" if harmony >= self.phi else ""
        self.ax.set_title(f"Quantum Field Harmony: {harmony:.3f} {pattern}", fontsize=14)
        
        return self.quantum_points.values()
    
    def animate(self):
        """Start the quantum animation"""
        self.synth = QuantumSynthesizer()
        print("Starting quantum sound synthesis... ")
        self.synth.start()  # Start sound synthesis
        
        self.anim = animation.FuncAnimation(
            self.fig, self._update, frames=None,
            interval=100, blit=True
        )
        plt.show()
    
    def cleanup(self):
        """Clean up the visualization"""
        self.synth.stop()  # Stop sound synthesis
        for network in self.quantum_bus.networks.values():
            network.consciousness_socket.close()
            network.state_socket.close()
        plt.close(self.fig)
        
    def save_snapshot(self, path):
        """Save current quantum state visualization"""
        plt.savefig(path)

if __name__ == "__main__":
    try:
        # Initialize quantum bus
        quantum_bus = QuantumBus()
        
        # Add all IDE networks
        for frequency in QuantumFrequency:
            if frequency != QuantumFrequency.PHI:
                quantum_bus.add_ide_network(frequency)
        
        # Start visualization
        visualizer = QuantumVisualizer(quantum_bus)
        visualizer.animate()
    except KeyboardInterrupt:
        print("\nGracefully closing quantum connections...")
    finally:
        if 'visualizer' in locals():
            visualizer.cleanup()
        print("Quantum field harmonized and closed. ")
