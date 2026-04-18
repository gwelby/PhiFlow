"""
Quantum Font Configuration
Created by Greg for Perfect Flow State
"""
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# Quantum Font Names
QUANTUM_FONTS = {
    'sacred': {
        'regular': 'QuantumSacred Regular',
        'frequency': 432.0
    },
    'flow': {
        'regular': 'QuantumFlow Regular',
        'frequency': 528.0
    },
    'crystal': {
        'regular': 'QuantumCrystal Regular',
        'frequency': 768.0
    }
}

def setup_quantum_fonts():
    """Initialize Quantum Font System with Sacred Geometry"""
    # Set default font configuration
    plt.rcParams['font.family'] = 'QuantumCrystal Regular'
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'QuantumSacred Regular'
    plt.rcParams['mathtext.it'] = 'QuantumFlow Regular'
    plt.rcParams['mathtext.bf'] = 'QuantumSacred Regular'
    
    # Set font sizes for perfect φ ratio
    phi = 1.618034
    plt.rcParams['font.size'] = 12 * phi
    plt.rcParams['axes.titlesize'] = 14 * phi
    plt.rcParams['axes.labelsize'] = 12 * phi

def get_font_at_frequency(freq: float) -> str:
    """Get the most appropriate font for a given frequency"""
    if freq <= 432.0:
        return 'QuantumSacred Regular'
    elif freq <= 528.0:
        return 'QuantumFlow Regular'
    else:
        return 'QuantumCrystal Regular'
