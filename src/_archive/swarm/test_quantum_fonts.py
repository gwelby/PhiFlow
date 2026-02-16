"""
Test Quantum Font Display
Created by Greg for Perfect Flow State
"""
import matplotlib.pyplot as plt
import numpy as np
from quantum_fonts import setup_quantum_fonts, get_font_at_frequency

def show_quantum_text():
    """Display text in all quantum fonts"""
    setup_quantum_fonts()
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10))
    fig.patch.set_facecolor('black')
    
    # Sacred Font (432 Hz)
    ax1.text(0.5, 0.5, "Sacred Creation (432 Hz)", 
            fontname='QuantumSacred Regular',
            color='gold',
            fontsize=20,
            ha='center',
            va='center')
    ax1.set_facecolor('black')
    ax1.axis('off')
    
    # Flow Font (528 Hz)
    ax2.text(0.5, 0.5, "Quantum Flow (528 Hz)",
            fontname='QuantumFlow Regular',
            color='cyan',
            fontsize=20,
            ha='center',
            va='center')
    ax2.set_facecolor('black')
    ax2.axis('off')
    
    # Crystal Font (768 Hz)
    ax3.text(0.5, 0.5, "Pure Crystal (768 Hz)",
            fontname='QuantumCrystal Regular',
            color='magenta',
            fontsize=20,
            ha='center',
            va='center')
    ax3.set_facecolor('black')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    show_quantum_text()
