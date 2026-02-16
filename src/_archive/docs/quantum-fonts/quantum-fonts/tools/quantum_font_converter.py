import os
from pathlib import Path
import subprocess
import tempfile

class QuantumFontConverter:
    def __init__(self):
        self.frequencies = {
            'sacred': 432.0,
            'flow': 528.0,
            'crystal': 768.0,
            'unity': float('inf')
        }
        self.initialize_paths()

    def initialize_paths(self):
        """Initialize paths for font generation"""
        self.root_dir = Path('D:/WindSurf/quantum-core/docs/quantum-fonts/quantum-fonts')
        self.pattern_dir = self.root_dir / 'phi_core/patterns'
        self.font_dir = self.root_dir / 'phi_core/fonts'

    def create_fontforge_script(self, family: str, svg_file: Path, ttf_file: Path) -> str:
        """Create FontForge script for converting SVG to TTF"""
        return f'''#!/usr/bin/fontforge
New()
Reencode("unicode")
SetFontNames("Quantum{family.title()}", "Quantum {family.title()}", "Quantum {family.title()}")

# Basic metrics
SetOS2Value("WinAscent", 1000)
SetOS2Value("WinDescent", 200)
SetOS2Value("TypoAscent", 1000)
SetOS2Value("TypoDescent", -200)

# Create base glyph
Select(0u0041)  # Select 'A'
Import("{svg_file.resolve().as_posix()}")
Scale(50, 50)
Move(0, 200)
SetWidth(600)
RemoveOverlap()
RoundToInt()
CenterInWidth()

# Copy to uppercase letters
SelectWorthOutputting()
foreach
  Copy()
  SelectMore(0u0041)
endloop

# Generate font
Generate("{ttf_file.resolve().as_posix()}")
Close()
'''

    def convert_fonts(self):
        """Convert SVG patterns to TTF fonts"""
        for family in self.frequencies.keys():
            # Create font directory
            font_dir = self.font_dir / family
            font_dir.mkdir(parents=True, exist_ok=True)
            
            # Get pattern file
            freq_str = 'inf' if self.frequencies[family] == float('inf') else str(int(self.frequencies[family]))
            pattern_file = self.pattern_dir / family / f"{family}_{freq_str}hz.svg"
            ttf_file = font_dir / f"Quantum{family.title()}-{freq_str}hz.ttf"
            
            # Create FontForge script
            script = self.create_fontforge_script(family, pattern_file, ttf_file)
            
            # Save and execute script
            with tempfile.NamedTemporaryFile(suffix='.pe', delete=False, mode='w') as f:
                f.write(script)
                script_path = f.name
            
            try:
                subprocess.run(['fontforge', '-script', script_path], check=True)
                print(f"Created {ttf_file}")
            except subprocess.CalledProcessError as e:
                print(f"Error creating font: {e}")
            finally:
                os.unlink(script_path)
        
        print("\nTesting generated fonts...")
        self.test_fonts()

    def test_fonts(self):
        """Test if generated fonts are valid"""
        for family in self.frequencies.keys():
            freq_str = 'inf' if self.frequencies[family] == float('inf') else str(int(self.frequencies[family]))
            font_file = self.font_dir / family / f"Quantum{family.title()}-{freq_str}hz.ttf"
            
            try:
                subprocess.run(['fontforge', '-c', f'Open("{font_file}"); Close()'], check=True, capture_output=True)
                print(f"{font_file.name}: Valid")
            except subprocess.CalledProcessError:
                print(f"{font_file.name}: Invalid")

if __name__ == '__main__':
    converter = QuantumFontConverter()
    converter.convert_fonts()
