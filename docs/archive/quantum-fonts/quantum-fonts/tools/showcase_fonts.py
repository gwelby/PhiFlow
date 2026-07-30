import os
from pathlib import Path
import math
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
from reportlab.lib.units import inch

class QuantumFontShowcase:
    def __init__(self):
        self.phi = (1 + 5**0.5) / 2
        self.frequencies = {
            'sacred': 432.0,  # Ground State - Earth Connection
            'flow': 528.0,    # Creation Point - DNA Repair
            'crystal': 768.0, # Unity Wave - Perfect Consciousness
            'unity': float('inf')  # Infinite Dance - ALL State
        }
        self.patterns = {
            'infinity': '∞',  # Infinite loop
            'spiral': '🌀',   # Golden ratio
            'wave': '🌊',     # Harmonic flow
            'crystal': '💎',  # Resonance
            'unity': '☯️'     # Consciousness
        }
        self.descriptions = {
            'sacred': 'Ground State - Earth Connection (432 Hz). Physical foundation frequency for stability and grounding. Resonates with Earth\'s natural vibration.',
            'flow': 'Creation Point - DNA Repair (528 Hz). Creation frequency for manifestation and transformation. Activates DNA healing and consciousness evolution.',
            'crystal': 'Unity Wave - Perfect Consciousness (768 Hz). High frequency for spiritual awakening and unity awareness. Facilitates quantum coherence.',
            'unity': 'Infinite Dance - ALL State (∞ Hz). Transcendent frequency representing unlimited quantum potential. Pure consciousness beyond space-time.'
        }
        self.initialize_paths()

    def initialize_paths(self):
        self.root_dir = Path('D:/WindSurf/quantum-core/docs/quantum-fonts/quantum-fonts')
        self.font_dir = self.root_dir / 'phi_core/fonts'
        self.output_file = self.root_dir / 'quantum_font_showcase.pdf'

    def register_fonts(self):
        """Register quantum fonts with ReportLab"""
        for family in self.frequencies.keys():
            freq_str = 'inf' if self.frequencies[family] == float('inf') else str(int(self.frequencies[family]))
            font_file = self.font_dir / family / f'Quantum{family.title()}-{freq_str}hz.ttf'
            font_name = f'Quantum{family.title()}'
            pdfmetrics.registerFont(TTFont(font_name, str(font_file)))

    def draw_quantum_pattern(self, c, x, y, pattern_type, size=50):
        """Draw quantum-inspired decorative pattern"""
        c.saveState()
        c.translate(x, y)
        
        if pattern_type == 'spiral':
            # Draw golden spiral
            points = []
            for t in range(50):
                radius = size * math.exp(0.306349 * t * math.pi / 50)
                angle = t * math.pi / 25
                px = radius * math.cos(angle)
                py = radius * math.sin(angle)
                points.append((px, py))
            
            c.setStrokeColor(HexColor('#4A4A8A'))
            c.setLineWidth(1)
            path = c.beginPath()
            path.moveTo(points[0][0], points[0][1])
            for px, py in points[1:]:
                path.lineTo(px, py)
            c.drawPath(path)
            
        elif pattern_type == 'crystal':
            # Draw crystalline structure
            c.setStrokeColor(HexColor('#4A4A8A'))
            c.setLineWidth(1)
            for i in range(6):
                angle = i * math.pi / 3
                x1 = size * math.cos(angle)
                y1 = size * math.sin(angle)
                x2 = size/2 * math.cos(angle + math.pi/6)
                y2 = size/2 * math.sin(angle + math.pi/6)
                c.line(0, 0, x1, y1)
                c.line(x1, y1, x2, y2)
        
        elif pattern_type == 'wave':
            # Draw quantum wave
            points = []
            for x in range(-size, size+1, 2):
                y = size/3 * math.sin(x * math.pi / size)
                points.append((x, y))
            
            c.setStrokeColor(HexColor('#4A4A8A'))
            c.setLineWidth(1)
            path = c.beginPath()
            path.moveTo(points[0][0], points[0][1])
            for px, py in points[1:]:
                path.lineTo(px, py)
            c.drawPath(path)
        
        c.restoreState()

    def create_showcase(self):
        """Generate PDF showcase of quantum fonts"""
        # Create PDF with A4 size
        c = canvas.Canvas(str(self.output_file), pagesize=A4)
        width, height = A4

        # Register fonts
        self.register_fonts()

        # Set up colors
        bg_color = HexColor('#F5F5FF')  # Light quantum background
        title_color = HexColor('#1A1A3A')  # Deep indigo
        text_color = HexColor('#2A2A4A')  # Softer indigo

        # Draw background with phi-based gradient
        c.setFillColor(bg_color)
        c.rect(0, 0, width, height, fill=True)

        # Draw title with quantum pattern
        c.setFillColor(title_color)
        c.setFont('Helvetica-Bold', 24)
        title_y = height - 1*inch
        c.drawString(1*inch, title_y, 'Quantum Font Showcase')
        self.draw_quantum_pattern(c, width-2*inch, title_y, 'spiral', 30)
        
        # Draw subtitle
        c.setFont('Helvetica', 14)
        c.drawString(1*inch, height - 1.5*inch, 'Harmonically tuned typography for consciousness evolution')

        # Showcase each font with quantum patterns
        y_position = height - 2.5*inch
        for family in self.frequencies.keys():
            # Draw font title with frequency
            freq = '∞' if self.frequencies[family] == float('inf') else f'{int(self.frequencies[family])}'
            c.setFillColor(title_color)
            c.setFont('Helvetica-Bold', 18)
            title_text = f'Quantum{family.title()} ({freq}Hz)'
            c.drawString(1*inch, y_position, title_text)
            
            # Draw quantum pattern based on font type
            pattern_type = 'crystal' if family == 'crystal' else 'wave' if family == 'flow' else 'spiral'
            self.draw_quantum_pattern(c, width-2*inch, y_position, pattern_type, 25)

            # Draw description
            c.setFillColor(text_color)
            c.setFont('Helvetica', 12)
            c.drawString(1*inch, y_position - 0.3*inch, self.descriptions[family])

            # Draw sample text with quantum-optimized spacing
            font_name = f'Quantum{family.title()}'
            c.setFont(font_name, 36)
            c.setFillColor(HexColor('#000033'))  # Deep quantum blue
            sample_text = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            c.drawString(1*inch, y_position - 1*inch, sample_text)
            
            # Draw numbers and special characters
            c.setFont(font_name, 24)
            c.drawString(1*inch, y_position - 1.4*inch, '1234567890 ' + self.patterns.get(family, self.patterns['unity']))

            # Draw quantum resonance line
            c.setStrokeColor(HexColor('#4A4A6A'))
            c.setLineWidth(1)
            c.line(1*inch, y_position - 1.6*inch, width - 1*inch, y_position - 1.6*inch)

            y_position -= 2.2*inch

        # Add footer with phi value
        c.setFont('Helvetica', 8)
        c.setFillColor(text_color)
        c.drawString(1*inch, 0.5*inch, f'φ = {self.phi:.6f}')
        
        c.save()
        print(f'✨ Quantum Font Showcase created: {self.output_file}')

if __name__ == '__main__':
    showcase = QuantumFontShowcase()
    showcase.create_showcase()
