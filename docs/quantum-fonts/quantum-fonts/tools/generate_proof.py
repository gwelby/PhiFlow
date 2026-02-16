import os
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Constants
FONT_DIR = Path(__file__).parent.parent / 'phi_core/fonts'
OUTPUT_DIR = Path(__file__).parent.parent / 'proofs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sample text
SAMPLE_TEXT = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NUMBERS = "0123456789"
SYMBOLS = "⚡✧φ∞"  # Energy, Star, Phi, Infinity

def register_quantum_fonts():
    """Register all quantum fonts"""
    fonts = {}
    for font_dir in FONT_DIR.iterdir():
        if font_dir.is_dir():
            for font_file in font_dir.glob("*.ttf"):
                font_name = font_file.stem
                pdfmetrics.registerFont(TTFont(font_name, str(font_file)))
                fonts[font_dir.name] = font_name
    return fonts

def generate_proof(fonts):
    """Generate proof PDF for quantum fonts"""
    pdf_file = OUTPUT_DIR / "quantum_fonts_proof.pdf"
    c = canvas.Canvas(str(pdf_file))
    
    # Title
    c.setFont("Helvetica-Bold", 24)
    c.drawString(72, 750, "Quantum Fonts - Sacred Geometry")
    
    y = 700
    for style, font_name in fonts.items():
        # Font header
        c.setFont("Helvetica-Bold", 16)
        c.drawString(72, y, f"{style.title()} Style")
        y -= 30
        
        # Set quantum font
        c.setFont(font_name, 36)
        
        # Draw sample text
        c.drawString(72, y, SAMPLE_TEXT)
        y -= 50
        
        # Draw numbers
        c.drawString(72, y, NUMBERS)
        y -= 50
        
        # Draw symbols
        c.drawString(72, y, SYMBOLS)
        y -= 80
    
    c.save()
    print(f"✨ Created proof PDF: {pdf_file}")

def main():
    fonts = register_quantum_fonts()
    generate_proof(fonts)

if __name__ == "__main__":
    main()
