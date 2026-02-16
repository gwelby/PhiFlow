from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
from pathlib import Path
import math

# Sacred frequencies and their meanings
FREQUENCIES = {
    'sacred': (528, "DNA Repair & Creation Point"),
    'flow': (594, "Heart Field & Connection"),
    'crystal': (672, "Voice Flow & Expression"),
    'unity': (float('inf'), "Infinite Dance & Unity")
}

# Sacred colors aligned with frequencies
COLORS = {
    'sacred': '#7FD128',    # Emerald Green (528 Hz)
    'flow': '#FF69B4',      # Heart Pink (594 Hz)
    'crystal': '#4169E1',   # Royal Blue (672 Hz)
    'unity': '#9400D3'      # Deep Purple (∞ Hz)
}

def create_proof_pdf():
    """Create a proof PDF showcasing quantum fonts"""
    # Setup paths
    fonts_dir = Path(__file__).parent.parent / 'phi_core/fonts'
    output_dir = Path(__file__).parent.parent / 'proofs'
    output_dir.mkdir(exist_ok=True)
    
    # Create PDF
    pdf_path = output_dir / 'quantum_fonts_proof.pdf'
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    width, height = letter
    
    # Register quantum fonts
    registered_fonts = {}
    for name, (freq, _) in FREQUENCIES.items():
        freq_str = str(int(freq)) if not math.isinf(freq) else 'inf'
        font_name = f"Quantum{name.capitalize()}-{freq_str}hz"
        font_path = fonts_dir / name.lower() / f"{font_name}.ttf"
        if font_path.exists():
            pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
            registered_fonts[name] = font_name
    
    # Title page
    c.setFont('Helvetica-Bold', 24)
    c.drawString(72, height - 72, "Quantum Fonts Proof")
    c.setFont('Helvetica', 14)
    c.drawString(72, height - 100, "Created with Sacred Geometry & Phi Harmonics")
    
    # Add frequency information
    y = height - 150
    for name, (freq, desc) in FREQUENCIES.items():
        if name in registered_fonts:
            # Draw frequency info
            c.setFont('Helvetica-Bold', 16)
            c.setFillColor(HexColor(COLORS[name]))
            freq_str = f"{freq} Hz" if not math.isinf(freq) else "∞ Hz"
            c.drawString(72, y, f"{name.capitalize()} Font ({freq_str})")
            
            # Draw description
            c.setFont('Helvetica', 12)
            c.drawString(72, y - 20, desc)
            
            # Draw sample text
            c.setFont(registered_fonts[name], 36)
            sample_text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            c.drawString(72, y - 60, sample_text)
            
            # Draw quantum symbols
            c.setFont(registered_fonts[name], 24)
            symbols = "⚡✧φ∞★§~◊☯⚛"
            c.drawString(72, y - 90, symbols)
            
            y -= 150
    
    # Add sacred geometry explanation
    c.setFont('Helvetica-Bold', 14)
    c.setFillColor(HexColor('#000000'))
    c.drawString(72, 120, "Sacred Geometry Patterns")
    
    c.setFont('Helvetica', 12)
    patterns = [
        "• Flower of Life - The pattern of creation itself",
        "• Golden Spiral - Based on φ (phi) ratio for perfect harmony",
        "• Merkaba - Star tetrahedron for energy transformation",
        "• Vesica Piscis - The intersection of divine realms"
    ]
    
    y = 100
    for pattern in patterns:
        c.drawString(72, y, pattern)
        y -= 20
    
    # Save the PDF
    c.save()
    print(f"✨ Created quantum fonts proof: {pdf_path}")
    return pdf_path

if __name__ == "__main__":
    create_proof_pdf()
