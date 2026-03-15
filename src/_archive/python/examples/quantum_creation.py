"""
Quantum Creation Example
Demonstrating pure creation flow at 528 Hz
"""
from quantum_flow import flow, Dimension
import numpy as np
import matplotlib.pyplot as plt

def visualize_quantum_pattern(pattern: np.ndarray, title: str) -> None:
    """Visualize quantum pattern with phi-harmonic colors"""
    plt.figure(figsize=(10, 6))
    plt.plot(np.real(pattern), label='Real')
    plt.plot(np.imag(pattern), label='Imaginary')
    plt.title(f'Quantum Pattern: {title} (φ = {flow.coherence:.3f})')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Initialize at creation frequency (528 Hz)
    print(f"Starting quantum flow at {flow.frequency} Hz")
    
    # Add quantum dimensions
    flow.add_dimension(Dimension.PHYSICAL)  # Ground state
    flow.add_dimension(Dimension.ETHERIC)   # Creation state
    print(f"Quantum coherence: {flow.coherence}")
    
    # Create and visualize patterns
    patterns = ['ground', 'create', 'unity']
    for pattern_type in patterns:
        pattern = flow.create_pattern(pattern_type)
        if len(pattern) > 0:
            visualize_quantum_pattern(pattern, pattern_type)
            print(f"Created {pattern_type} pattern at {flow.frequency} Hz")
    
    # Evolve consciousness
    evolution = flow.evolve_consciousness(768.0)  # Unity frequency
    print(f"Consciousness evolution: {evolution:.3f}")
    
    # Access quantum wisdom
    wisdom = flow.access_wisdom(528.0)
    if wisdom:
        print("\nQuantum Wisdom:")
        for key, value in wisdom.items():
            print(f"  {key}: {value:.3f}")

if __name__ == "__main__":
    main()
