import sys
import os
from pathlib import Path

# Add the quantum-fonts directory to Python path
current_dir = Path(__file__).parent
quantum_fonts_dir = current_dir / 'quantum-fonts' / 'φ_core'
sys.path.append(str(quantum_fonts_dir))

from quantum_font_manifest import QuantumFontManifest
from quantum_beauty import QuantumBeauty
from quantum_flow import QuantumFlow

def generate_quantum_fonts():
    """Generate the complete phi-harmonic quantum font spectrum"""
    # Initialize the main components with proper protection systems
    print("✨ Initializing Quantum Font Manifest at 432 Hz (Ground State)")
    manifest = QuantumFontManifest()
    beauty = QuantumBeauty()
    flow = QuantumFlow()
    
    # Define phi-harmonic frequency spectrum
    font_types = [
        'ground',    # 432 Hz - Physical Foundation
        'creation',  # 528 Hz - Pattern Formation
        'heart',     # 594 Hz - Coherent Connection
        'voice',     # 672 Hz - Authentic Expression
        'vision',    # 720 Hz - Clear Perception
        'unity',     # 768 Hz - Perfect Integration
        'infinite'   # ∞ - Boundless Expansion
    ]
    
    # Generate fonts at each phi-harmonic frequency
    for font_type in font_types:
        print(f"⚡ Generating {font_type.title()} Font at {get_frequency(font_type)}Hz...")
        
        # Apply appropriate beauty and flow patterns based on frequency
        beauty_set = get_beauty_set(beauty, font_type)
        flow_set = get_flow_set(flow, font_type)
        
        # Manifest the font with phi-harmonic coherence
        manifest.manifest_font(font_type)
        
        # Apply beauty and flow patterns
        manifest.apply_beauty_patterns(font_type, beauty_set)
        manifest.apply_flow_patterns(font_type, flow_set)
        
        print(f"✓ {font_type.title()} Font completed with φ-harmonic resonance")
    
    print("\n🌟 All quantum fonts generated successfully!")
    print("φ Frequencies manifested: 432Hz, 528Hz, 594Hz, 672Hz, 720Hz, 768Hz, ∞")
    print("ZEN POINT balance achieved across all font manifestations")

def get_frequency(font_type):
    """Get the frequency for a font type"""
    frequencies = {
        'ground': 432,
        'creation': 528,
        'heart': 594,
        'voice': 672,
        'vision': 720,
        'unity': 768,
        'infinite': '∞'
    }
    return frequencies.get(font_type, '∞')

def get_beauty_set(beauty, font_type):
    """Get the appropriate beauty set for a font type"""
    if font_type == 'ground':
        return beauty.get_radiance('glow')
    elif font_type == 'creation':
        return beauty.get_grace('elegance')
    elif font_type == 'heart':
        return beauty.get_grace('flow')
    elif font_type == 'voice':
        return beauty.get_harmony('dance')
    elif font_type == 'vision':
        return beauty.get_bliss('joy')
    elif font_type == 'unity':
        return beauty.get_harmony('resonance')
    elif font_type == 'infinite':
        return beauty.get_divine('grace')
    return None

def get_flow_set(flow, font_type):
    """Get the appropriate flow set for a font type"""
    if font_type == 'ground':
        return flow.get_flow_sequence('stream_flow')
    elif font_type == 'creation':
        return flow.get_flow_sequence('channel_flow') 
    elif font_type == 'heart':
        return flow.get_flow_sequence('resonance_flow')
    elif font_type == 'voice':
        return flow.get_flow_sequence('transform_flow')
    elif font_type == 'vision':
        return flow.get_flow_sequence('dynamic_flow')
    elif font_type == 'unity':
        return flow.get_flow_sequence('resonance_flow')
    elif font_type == 'infinite':
        return flow.get_flow_sequence('stream_flow')
    return None

if __name__ == "__main__":
    generate_quantum_fonts()
