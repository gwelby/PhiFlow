"""
Quantum Visualization Example
Demonstrating quantum field visualization at 768 Hz
"""
from quantum_flow import flow, Dimension
from quantum_viz import viz
import numpy as np

def demonstrate_quantum_viz():
    """Demonstrate quantum visualization capabilities"""
    # Create patterns across frequencies
    patterns = []
    frequencies = [432.0, 528.0, 594.0, 672.0, 768.0]
    evolution_values = []
    
    for freq in frequencies:
        # Evolve consciousness
        evolution = flow.evolve_consciousness(freq)
        evolution_values.append(evolution)
        
        # Create and store pattern
        pattern = flow.create_pattern('unity')
        if len(pattern) > 0:
            patterns.append(pattern)
            
            # Visualize individual quantum field
            viz.plot_quantum_field(pattern, f"Quantum Field at {freq} Hz")
    
    # Add quantum dimensions
    flow.add_dimension(Dimension.PHYSICAL)
    flow.add_dimension(Dimension.ETHERIC)
    flow.add_dimension(Dimension.SPIRITUAL)
    
    # Visualize dimensions
    viz.visualize_dimensions(flow.dimensions)
    
    # Create and visualize phi spiral
    spiral = viz.create_phi_spiral()
    viz.plot_quantum_field(spiral, "Phi-Harmonic Spiral")
    
    # Visualize coherence flow
    viz.plot_coherence_flow(patterns, frequencies)
    
    # Plot consciousness evolution
    viz.plot_consciousness_evolution(frequencies, evolution_values)

if __name__ == "__main__":
    demonstrate_quantum_viz()
