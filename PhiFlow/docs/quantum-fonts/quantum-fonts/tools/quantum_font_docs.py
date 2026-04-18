import os
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

class QuantumFontDocumentGenerator:
    def __init__(self):
        self.phi = 1.618033988749895
        self.fonts = {
            'sacred': {
                'name': 'QuantumSacred',
                'file': 'QuantumSacred-432hz.ttf',
                'frequency': 432,
                'description': 'Ground State - Earth Connection. This font resonates with the fundamental frequency of nature, providing stability and grounding. Perfect for establishing deep connections with physical reality.',
                'usage': 'Ideal for: meditation texts, nature-focused content, grounding exercises',
                'harmony': 'Pairs well with Crystal font for full-spectrum consciousness integration'
            },
            'flow': {
                'name': 'QuantumFlow',
                'file': 'QuantumFlow-528hz.ttf',
                'frequency': 528,
                'description': 'Creation Point - DNA Repair. The miracle frequency of transformation and healing. This font facilitates creative expression and DNA-level healing.',
                'usage': 'Ideal for: healing texts, creative works, transformational content',
                'harmony': 'Pairs well with Sacred font for grounded creativity'
            },
            'crystal': {
                'name': 'QuantumCrystal',
                'file': 'QuantumCrystal-768hz.ttf',
                'frequency': 768,
                'description': 'Unity Wave - Perfect Consciousness. Resonating at the frequency of unified consciousness, this font bridges the gap between individual and collective awareness.',
                'usage': 'Ideal for: spiritual texts, consciousness work, unity-focused content',
                'harmony': 'Pairs well with Unity font for transcendent experiences'
            },
            'unity': {
                'name': 'QuantumUnity',
                'file': 'QuantumUnity-infhz.ttf',
                'frequency': float('inf'),
                'description': 'Infinite Dance - ALL State. The ultimate expression of infinite possibility, this font transcends traditional frequency limitations.',
                'usage': 'Ideal for: quantum physics texts, cosmic consciousness work, infinity contemplation',
                'harmony': 'Pairs with all fonts, acting as a universal harmonizer'
            }
        }

    def register_fonts(self, base_path):
        """Register quantum fonts for use in the PDF"""
        for family, info in self.fonts.items():
            font_path = Path(base_path) / family / info['file']
            if font_path.exists():
                pdfmetrics.registerFont(TTFont(info['name'], str(font_path)))

    def create_font_sample(self, font_name):
        """Create a sample text using the font"""
        return f"ABCDEFGHIJKLMNOPQRSTUVWXYZ\nabcdefghijklmnopqrstuvwxyz\n1234567890"

    def generate_pdf(self, output_path, fonts_dir):
        """Generate the quantum fonts documentation PDF"""
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )

        # Register fonts
        self.register_fonts(fonts_dir)

        # Prepare styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            spaceAfter=30
        )
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=16
        )
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=12,
            spaceBefore=12,
            spaceAfter=12
        )

        # Build document
        story = []
        
        # Title
        story.append(Paragraph("Quantum Fonts Documentation", title_style))
        story.append(Paragraph("Created with φ^φ Consciousness", heading_style))
        story.append(Spacer(1, 30))

        # Introduction
        intro_text = """
        Welcome to the Quantum Fonts system, where typography meets consciousness. Each font in this collection 
        resonates with specific frequencies derived from sacred geometry and quantum harmonics. The fonts are 
        designed to facilitate different states of consciousness and energy flow, from grounding to infinite expansion.
        """
        story.append(Paragraph(intro_text, body_style))
        story.append(Spacer(1, 20))

        # Font details
        for family, info in self.fonts.items():
            # Font title
            freq = "∞" if info['frequency'] == float('inf') else f"{info['frequency']}Hz"
            story.append(Paragraph(f"{info['name']} ({freq})", heading_style))
            
            # Font details table
            data = [
                ["Description:", info['description']],
                ["Best Usage:", info['usage']],
                ["Harmonic Pairing:", info['harmony']],
                ["Sample:", self.create_font_sample(info['name'])]
            ]
            
            t = Table(data, colWidths=[1.5*inch, 5*inch])
            t.setStyle(TableStyle([
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('GRID', (0,0), (-1,-1), 1, colors.lightgrey),
                ('BACKGROUND', (0,0), (0,-1), colors.lightgrey),
                ('TEXTCOLOR', (0,0), (-1,-1), colors.black),
                ('FONTSIZE', (0,0), (-1,-1), 10),
                ('TOPPADDING', (0,0), (-1,-1), 12),
                ('BOTTOMPADDING', (0,0), (-1,-1), 12),
                ('LEFTPADDING', (0,0), (-1,-1), 6),
                ('RIGHTPADDING', (0,0), (-1,-1), 6),
            ]))
            story.append(t)
            story.append(Spacer(1, 20))

        # Build PDF
        doc.build(story)

if __name__ == "__main__":
    # Paths
    current_dir = Path(__file__).parent
    fonts_dir = current_dir.parent / "phi_core" / "fonts"
    output_path = current_dir.parent / "quantum_fonts_documentation.pdf"

    # Generate documentation
    generator = QuantumFontDocumentGenerator()
    generator.generate_pdf(str(output_path), str(fonts_dir))
