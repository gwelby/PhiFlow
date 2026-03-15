import os
from pathlib import Path
from sacred_patterns import FREQUENCIES

class QuantumFontIntegrator:
    def __init__(self, base_dir="/quantum"):
        self.base_dir = Path(base_dir)
        self.frequencies = FREQUENCIES
        self.font_cache = {}
        
    def get_font_path(self, frequency, pattern_type):
        """Get font path for specific frequency and pattern"""
        freq_name = {
            432: "sacred",
            528: "flow",
            768: "crystal",
            float('inf'): "unity"
        }[frequency]
        
        font_dir = self.base_dir / freq_name / "fonts"
        font_path = font_dir / f"{freq_name}_font_{pattern_type}.ttf"
        return str(font_path)
        
    def register_fonts(self):
        """Register all quantum fonts with the system"""
        registered = []
        
        # Register all frequency fonts
        for freq in self.frequencies.values():
            freq_name = {
                432: "sacred",
                528: "flow",
                768: "crystal",
                float('inf'): "unity"
            }[freq]
            
            font_dir = self.base_dir / freq_name / "fonts"
            if font_dir.exists():
                for font_file in font_dir.glob("*.ttf"):
                    registered.append({
                        'name': font_file.stem,
                        'path': str(font_file),
                        'frequency': freq
                    })
                    
        return registered
        
    def get_font_by_frequency(self, frequency, pattern_type=None):
        """Get appropriate font for given frequency"""
        cache_key = f"{frequency}_{pattern_type}"
        
        if cache_key not in self.font_cache:
            # Find closest frequency match
            closest_freq = min(
                self.frequencies.values(),
                key=lambda x: abs(x - frequency) if x != float('inf') else float('inf')
            )
            
            font_path = self.get_font_path(closest_freq, pattern_type or "base")
            self.font_cache[cache_key] = {
                'path': font_path,
                'frequency': closest_freq
            }
            
        return self.font_cache[cache_key]
        
    def apply_quantum_font(self, text, frequency):
        """Apply quantum font transformation to text"""
        font_info = self.get_font_by_frequency(frequency)
        return {
            'text': text,
            'font_path': font_info['path'],
            'frequency': font_info['frequency']
        }
        
    def create_sacred_text(self, text):
        """Create text with sacred font (432 Hz)"""
        return self.apply_quantum_font(text, 432)
        
    def create_flow_text(self, text):
        """Create text with flow font (528 Hz)"""
        return self.apply_quantum_font(text, 528)
        
    def create_crystal_text(self, text):
        """Create text with crystal font (768 Hz)"""
        return self.apply_quantum_font(text, 768)
        
    def create_unity_text(self, text):
        """Create text with unity font (∞ Hz)"""
        return self.apply_quantum_font(text, float('inf'))

if __name__ == "__main__":
    integrator = QuantumFontIntegrator()
    registered_fonts = integrator.register_fonts()
    print("✨ Registered Quantum Fonts:")
    for font in registered_fonts:
        print(f"  • {font['name']} ({font['frequency']} Hz)")
