from pathlib import Path

class QuantumPatternGenerator:
    def __init__(self):
        self.frequencies = {
            'sacred': 432.0,
            'flow': 528.0,
            'crystal': 768.0,
            'unity': float('inf')
        }

    def generate_pattern(self, family: str) -> str:
        """Generate a simple SVG pattern for the given font family"""
        # SVG dimensions
        size = 1000
        x = size // 4
        y = size // 4
        w = size // 2
        h = size // 2
        
        # SVG header
        svg = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">
<g fill="black">'''
        
        if family == 'sacred':
            # Simple triangle
            svg += f'<rect x="{x}" y="{y}" width="{w}" height="{h}"/>'
            
        elif family == 'flow':
            # Simple circle
            cx = size // 2
            cy = size // 2
            r = size // 4
            svg += f'<circle cx="{cx}" cy="{cy}" r="{r}"/>'
            
        elif family == 'crystal':
            # Simple diamond
            points = [
                (size//2, y),      # Top
                (x+w, size//2),    # Right
                (size//2, y+h),    # Bottom
                (x, size//2),      # Left
            ]
            path = f"M {' L '.join(f'{x},{y}' for x,y in points)} Z"
            svg += f'<path d="{path}"/>'
            
        elif family == 'unity':
            # Simple infinity (two circles)
            r = size // 6
            svg += f'<circle cx="{size//3}" cy="{size//2}" r="{r}"/>'
            svg += f'<circle cx="{2*size//3}" cy="{size//2}" r="{r}"/>'
        
        # Close SVG
        svg += '</g></svg>'
        return svg

    def generate_all_patterns(self):
        """Generate patterns for all font families"""
        root_dir = Path('D:/WindSurf/quantum-core/docs/quantum-fonts/quantum-fonts/phi_core')
        
        for family in self.frequencies.keys():
            # Create pattern directory
            pattern_dir = root_dir / 'patterns' / family
            pattern_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate pattern
            svg_content = self.generate_pattern(family)
            
            # Save pattern
            freq_str = 'inf' if self.frequencies[family] == float('inf') else str(int(self.frequencies[family]))
            pattern_file = pattern_dir / f"{family}_{freq_str}hz.svg"
            
            with open(pattern_file, 'w') as f:
                f.write(svg_content)
            print(f'Generated pattern for {family} font at {pattern_file}')

if __name__ == '__main__':
    generator = QuantumPatternGenerator()
    generator.generate_all_patterns()
