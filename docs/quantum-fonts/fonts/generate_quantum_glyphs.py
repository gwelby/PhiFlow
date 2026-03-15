import math
from pathlib import Path
import xml.etree.ElementTree as ET
import copy

class QuantumGlyphGenerator:
    def __init__(self):
        self.φ = (1 + 5**0.5) / 2  # Golden ratio
        self.frequencies = {
            'ground': 432,    # Physical foundation
            'create': 528,    # DNA repair & creation
            'heart': 594,     # Heart field resonance
            'voice': 672,     # Voice flow frequency
            'vision': 720,    # Vision gate frequency
            'unity': 768,     # Unity consciousness
            'infinite': float('inf')  # Infinite state
        }
        self.template_path = Path('sources/svg/quantum_template.svg')
        self.output_dir = Path('sources/svg/letters')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_template(self):
        """Load the quantum template SVG"""
        tree = ET.parse(self.template_path)
        return tree
        
    def generate_letter(self, char: str, frequency: float):
        """Generate a quantum letter with specific frequency"""
        tree = self.load_template()
        root = tree.getroot()
        
        # Get the letter template group
        template = root.find(".//*[@id='letter_template']")
        
        # Adjust the frequency-based attributes
        freq_factor = frequency / self.frequencies['ground']
        
        # Scale patterns based on frequency
        for pattern in root.findall(".//pattern"):
            w = float(pattern.get('width', '100'))
            h = float(pattern.get('height', '100'))
            pattern.set('width', str(w * freq_factor))
            pattern.set('height', str(h * freq_factor))
        
        # Adjust glow intensity
        glow = root.find(".//filter/feGaussianBlur")
        if glow is not None:
            glow.set('stdDeviation', str(2 * freq_factor))
        
        # Save the modified SVG
        output_path = self.output_dir / f'{char}.svg'
        tree.write(output_path)
        
    def generate_alphabet(self):
        """Generate entire alphabet with quantum frequencies"""
        chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        frequencies = list(self.frequencies.values())[:-1]  # Exclude infinite
        
        for i, char in enumerate(chars):
            # Cycle through frequencies for different letters
            freq = frequencies[i % len(frequencies)]
            self.generate_letter(char, freq)
            print(f'Generated {char} with frequency {freq} Hz')

if __name__ == '__main__':
    generator = QuantumGlyphGenerator()
    generator.generate_alphabet()
