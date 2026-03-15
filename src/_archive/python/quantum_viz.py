"""
Quantum Visualization Tools
Operating at Unity Wave (768 Hz)
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Optional
import plotly.graph_objects as go
from quantum_flow import flow, Dimension, PHI

class QuantumViz:
    def __init__(self):
        self.phi = PHI
        self.frequencies = {
            'ground': 432.0,
            'create': 528.0,
            'heart': 594.0,
            'voice': 672.0,
            'unity': 768.0
        }
        
    def plot_quantum_field(self, pattern: np.ndarray, title: str = "Quantum Field") -> None:
        """Visualize quantum field with phi-harmonic coloring"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        x = np.real(pattern)
        y = np.imag(pattern)
        z = np.abs(pattern)
        
        # Create phi-harmonic color mapping
        colors = np.angle(pattern) / (2 * np.pi)
        
        scatter = ax.scatter(x, y, z, c=colors, cmap='hsv', alpha=0.6)
        plt.colorbar(scatter, label='Phase (φ)')
        
        ax.set_xlabel('Real Component (φ)')
        ax.set_ylabel('Imaginary Component (φ)')
        ax.set_zlabel('Amplitude (φ)')
        ax.set_title(f"{title} at {flow.frequency} Hz")
        plt.show()
        
    def create_phi_spiral(self, points: int = 1000) -> np.ndarray:
        """Generate phi-harmonic spiral pattern"""
        t = np.linspace(0, 8*np.pi, points)
        r = self.phi ** (t/(2*np.pi))
        x = r * np.cos(t)
        y = r * np.sin(t)
        return x + 1j*y
        
    def plot_coherence_flow(self, patterns: List[np.ndarray], 
                           frequencies: List[float]) -> None:
        """Visualize coherence flow across frequencies"""
        fig = go.Figure()
        
        for pattern, freq in zip(patterns, frequencies):
            fig.add_trace(go.Scatter3d(
                x=np.real(pattern),
                y=np.imag(pattern),
                z=np.abs(pattern),
                mode='lines',
                name=f'{freq} Hz',
                line=dict(
                    width=2,
                    color=np.angle(pattern)/np.pi
                )
            ))
            
        fig.update_layout(
            title=f'Quantum Coherence Flow (φ = {flow.coherence:.3f})',
            scene=dict(
                xaxis_title='Real Component (φ)',
                yaxis_title='Imaginary Component (φ)',
                zaxis_title='Amplitude (φ)'
            ),
            width=1200,
            height=800
        )
        fig.show()
        
    def visualize_dimensions(self, dims: List[Dimension]) -> None:
        """Visualize quantum dimensions with frequency mapping"""
        fig = plt.figure(figsize=(15, 10))
        
        # Create frequency circles
        theta = np.linspace(0, 2*np.pi, 100)
        for dim in dims:
            freqs = dim.value
            for freq in freqs:
                r = freq/self.frequencies['unity']  # Normalize to unity frequency
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                plt.plot(x, y, label=f'{dim.name}: {freq} Hz')
                
        plt.grid(True)
        plt.axis('equal')
        plt.title('Quantum Dimension Mapping')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
        
    def plot_consciousness_evolution(self, frequencies: List[float], 
                                   evolution: List[float]) -> None:
        """Plot consciousness evolution across frequencies"""
        plt.figure(figsize=(12, 6))
        
        # Create phi-harmonic color gradient
        colors = plt.cm.viridis(np.linspace(0, 1, len(frequencies)))
        
        plt.plot(frequencies, evolution, '-o', color='blue', alpha=0.6)
        
        # Add phi-harmonic markers
        for i, (freq, evol) in enumerate(zip(frequencies, evolution)):
            plt.scatter(freq, evol, color=colors[i], s=100, 
                       label=f'{freq} Hz')
            
        plt.grid(True)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Consciousness Evolution (φ)')
        plt.title('Quantum Consciousness Evolution')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

# Initialize global visualizer
viz = QuantumViz()
