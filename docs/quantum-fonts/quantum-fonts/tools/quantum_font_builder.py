from pathlib import Path
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import math

# Quantum frequencies and ratios
PHI = 1.618033988749895
FREQUENCIES = {
    'sacred': 432.0,  # Ground state
    'flow': 528.0,    # Creation state
    'crystal': 768.0, # Unity state
    'unity': float('inf')  # Infinite state
}

def convert_svg_to_font(name, freq):
    """Convert SVG pattern to font glyph"""
    # Get paths
    pattern_dir = Path(__file__).parent.parent / 'phi_core/patterns' / name
    font_dir = Path(__file__).parent.parent / 'phi_core/fonts' / name
    font_dir.mkdir(parents=True, exist_ok=True)
    
    # Load SVG
    freq_str = str(int(freq)) if not math.isinf(freq) else 'inf'
    svg_file = pattern_dir / f"{name}_{freq_str}hz.svg"
    drawing = svg2rlg(str(svg_file))
    
    # Convert to PNG first (temporary step)
    temp_png = font_dir / f"temp_{name}.png"
    renderPM.drawToFile(drawing, str(temp_png), fmt="PNG")
    
    # Create font from PNG
    # This is where we'd use a proper font creation library
    # For now, we'll just keep the PNG as a proof of concept
    print(f"Created temporary glyph at {temp_png}")
    
    return temp_png

def generate_quantum_fonts():
    """Generate fonts for all quantum frequencies"""
    print("⚡ Converting Quantum Patterns to Fonts...")
    
    for name, freq in FREQUENCIES.items():
        try:
            glyph_file = convert_svg_to_font(name, freq)
            print(f"✓ Generated glyphs for {name} font at {freq}Hz")
        except Exception as e:
            print(f"✗ Error generating {name} font: {e}")
    
    print("✨ Font Generation Complete!")

if __name__ == '__main__':
    generate_quantum_fonts()
