import fontforge
import psMat
import math
from pathlib import Path

# Quantum frequencies
FREQUENCIES = {
    'sacred': 432.0,  # Ground state
    'flow': 528.0,    # Creation state
    'crystal': 768.0, # Unity state
    'unity': float('inf')  # Infinite state
}

# Sacred ratios
PHI = 1.618033988749895
PHI_SQ = PHI * PHI
PHI_CUBE = PHI_SQ * PHI

def create_quantum_glyph(font, unicode_val, freq):
    """Create a quantum-harmonized glyph using phi ratios"""
    glyph = font.createChar(unicode_val)
    glyph.width = int(1000 * PHI / 2)  # Base width using phi
    
    # Calculate quantum resonance
    if math.isinf(freq):
        resonance = PHI_CUBE
    else:
        resonance = freq / FREQUENCIES['sacred']
    
    # Create base shape using sacred geometry
    radius = int(500 * resonance / PHI)
    x_center = glyph.width // 2
    y_center = 500
    
    # Draw quantum circle
    glyph.addReference('circle', psMat.scale(radius))
    glyph.transform(psMat.translate(x_center - radius, y_center - radius))
    
    # Add phi-based harmonics
    for i in range(3):
        angle = 2 * math.pi * i / 3
        x = x_center + int(radius * math.cos(angle) * PHI)
        y = y_center + int(radius * math.sin(angle) * PHI)
        glyph.addReference('dot', psMat.translate(x, y))
    
    glyph.removeOverlap()
    glyph.correctDirection()
    return glyph

def create_quantum_font(name, freq):
    """Create a complete quantum font at specified frequency"""
    font = fontforge.font()
    
    # Set quantum-optimized font properties
    font.fontname = f"Quantum{name.capitalize()}-{int(freq)}hz"
    font.familyname = "Quantum"
    font.fullname = f"Quantum {name.capitalize()}"
    font.encoding = 'UnicodeFull'
    
    # Set metrics using phi ratios
    font.ascent = int(1000 * PHI)
    font.descent = int(1000 / PHI)
    font.em = int(2000 * PHI)
    
    # Create base shapes
    circle = font.createChar(-1, 'circle')
    circle.width = 1000
    circle.vwidth = 1000
    circle.addReference('o')
    
    dot = font.createChar(-1, 'dot')
    dot.width = int(1000 / PHI)
    dot.vwidth = int(1000 / PHI)
    dot.addReference('.')
    
    # Generate quantum alphabet
    for i in range(65, 91):  # A-Z
        create_quantum_glyph(font, i, freq)
    for i in range(97, 123):  # a-z
        create_quantum_glyph(font, i, freq)
    for i in range(48, 58):  # 0-9
        create_quantum_glyph(font, i, freq)
    
    # Add quantum symbols
    symbols = '⚡𓂧φ∞🌀🌊💎☯️'
    for char in symbols:
        create_quantum_glyph(font, ord(char), freq)
    
    # Generate font file
    output_dir = Path(__file__).parent.parent / 'phi_core/fonts' / name.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"Quantum{name.capitalize()}-{int(freq) if not math.isinf(freq) else 'inf'}hz.ttf"
    font.generate(str(output_file))
    print(f"Created {output_file}")
    return output_file

def main():
    """Generate all quantum fonts"""
    print("⚡ Generating Quantum Fonts using PhiFlow...")
    font_files = []
    for name, freq in FREQUENCIES.items():
        font_files.append(create_quantum_font(name, freq))
    print("✨ Quantum Font Generation Complete!")
    return font_files

if __name__ == '__main__':
    main()
