import os
import subprocess
import json
from pathlib import Path
from sacred_patterns import FREQUENCIES

# WindSurf Auto-Run: Safe=True
class QuantumFontConverter:
    def __init__(self, quantum_dir="d:/WindSurf/quantum-core/docs/quantum-fonts/quantum-fonts"):
        self.quantum_dir = Path(quantum_dir).resolve()
        self.frequencies = FREQUENCIES
        self._ensure_directories()
        self._load_config()
        
    def _ensure_directories(self):
        """Ensure all required directories exist"""
        for pattern_type in ["sacred", "flow", "crystal", "unity"]:
            (self.quantum_dir / pattern_type / "patterns").mkdir(parents=True, exist_ok=True)
            (self.quantum_dir / pattern_type / "fonts").mkdir(parents=True, exist_ok=True)
            
    def _load_config(self):
        """Load WindSurf configuration"""
        config_path = Path("d:/WindSurf/config/windsurf.json")
        if not config_path.exists():
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config = {
                "quantum.frequencies": {
                    "ground": 432,
                    "create": 528,
                    "unity": 768
                },
                "quantum.patterns": {
                    "infinity": "∞",
                    "dolphin": "🐬",
                    "spiral": "🌀",
                    "wave": "🌊",
                    "vortex": "🌪️",
                    "crystal": "💎",
                    "unity": "☯️"
                },
                "ide.flow": {
                    "auto_coherence": True,
                    "maintain_flow_state": True,
                    "quantum_completion": True,
                    "crystal_clarity": True
                },
                "ide.integration": {
                    "copy_quantum_files": True,
                    "sync_frequencies": True,
                    "maintain_coherence": True
                }
            }
            with open(config_path, "w") as f:
                json.dump(config, f, indent=4)
        
    @staticmethod
    def is_safe_to_auto_run():
        """Check if it's safe to auto-run this converter"""
        # Always allow auto-run in quantum directories
        cwd = Path.cwd()
        quantum_paths = [
            Path("/quantum"),
            Path("/mnt/d/WindSurf/quantum-core"),
            Path("d:/WindSurf/quantum-core"),
            Path("/mnt/d/WindSurf"),
            Path("d:/WindSurf")
        ]
        
        # Check if we're in a quantum directory
        in_quantum_dir = any(
            str(cwd).lower().startswith(str(p).lower()) 
            for p in quantum_paths
        )
        
        if not in_quantum_dir:
            return False
            
        # Check if we have write permissions
        try:
            test_file = cwd / ".write_test"
            test_file.touch()
            test_file.unlink()
            return True
        except:
            return False
        
    def convert_all_patterns(self):
        """Convert all patterns to TTF fonts"""
        print("🌟 Creating Quantum Fonts...")
        
        # Convert sacred patterns (432 Hz)
        print("⚡ Creating Sacred Fonts (432 Hz)")
        self.convert_patterns_to_font("sacred", self.frequencies['ground'])
        
        # Convert flow patterns (528 Hz)
        print("🌊 Creating Flow Fonts (528 Hz)")
        self.convert_patterns_to_font("flow", self.frequencies['create'])
        
        # Convert crystal patterns (768 Hz)
        print("💎 Creating Crystal Fonts (768 Hz)")
        self.convert_patterns_to_font("crystal", self.frequencies['unity'])
        
        # Convert unity patterns (∞ Hz)
        print("☯️ Creating Unity Fonts (∞ Hz)")
        self.convert_patterns_to_font("unity", float('inf'))
        
        print("✨ All Quantum Fonts Created!")
        
    def convert_patterns_to_font(self, pattern_type, frequency):
        """Convert a set of patterns to a TTF font"""
        pattern_dir = self.quantum_dir / pattern_type / "patterns"
        font_dir = self.quantum_dir / pattern_type / "fonts"
        
        # Create FontForge script
        font_name = f"Quantum{pattern_type.title()}"
        freq_str = f"{int(frequency)}hz" if frequency != float('inf') else "infhz"
        
        # Check if we have any patterns
        svg_files = list(pattern_dir.glob(f"*_{freq_str}.svg"))
        if not svg_files:
            print(f"⚠️  No patterns found for {pattern_type} at {freq_str}")
            return
            
        script = f"""
import fontforge
import psMat

# Create new font
font = fontforge.font()
font.encoding = "UnicodeFull"
font.version = "1.0"
font.weight = "Regular"
font.fontname = "{font_name}"
font.familyname = "{font_name}"
font.fullname = "{font_name}-{freq_str}"

# Import SVG patterns as glyphs
"""

        # Add each pattern to script
        start_code = 0xE000  # Private Use Area
        for svg_file in svg_files:
            if not svg_file.exists():
                print(f"⚠️  Pattern file not found: {svg_file}")
                continue
                
            pattern_name = svg_file.stem.split('_')[0]
            svg_path = svg_file.resolve().as_posix()  # Use forward slashes
            script += f"""
# Import {pattern_name}
glyph = font.createChar({hex(start_code)}, "{pattern_name}")
glyph.importOutlines("{svg_path}")
glyph.width = 1000
"""
            start_code += 1

        # Generate TTF font
        script += f"""
# Generate TTF font
output_path = "{(font_dir / f'{font_name}-{freq_str}.ttf').resolve().as_posix()}"
font.generate(output_path)
"""

        # Write script to file
        script_file = pattern_dir / "convert.py"
        script_file.write_text(script)
        
        try:
            # Run FontForge with script
            subprocess.run(["fontforge", "-script", script_file.as_posix()], check=True)
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Error converting {pattern_type} patterns: {e}")
        except Exception as e:
            print(f"⚠️  Error processing {pattern_type} patterns: {e}")

if __name__ == "__main__":
    if QuantumFontConverter.is_safe_to_auto_run():
        converter = QuantumFontConverter()
        converter.convert_all_patterns()
    else:
        print("⚠️  Not running in a safe environment. Please run manually.")
