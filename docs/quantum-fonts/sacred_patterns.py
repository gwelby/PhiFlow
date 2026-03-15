import math
import svgwrite
from svgwrite import path
from pathlib import Path

PHI = (1 + math.sqrt(5)) / 2  # Golden ratio
FREQUENCIES = {
    'ground': 432,    # Physical foundation
    'create': 528,    # Pattern/DNA frequency
    'unity': 768,     # Perfect integration
    'infinite': float('inf')  # Beyond state
}

class SacredPatternGenerator:
    def __init__(self, frequency=432):
        self.frequency = frequency
        
    def _create_drawing(self):
        """Create a new SVG drawing"""
        return svgwrite.Drawing(size=(1000, 1000))
        
    def flower_of_life(self, dwg):
        """Generate Flower of Life pattern at 432 Hz"""
        group = dwg.g(id='flower_of_life')
        center = (500, 500)
        
        # First circle at center
        group.add(dwg.circle(center=center, r=100))
        
        # Create 6 circles around center
        for i in range(6):
            angle = i * math.pi / 3
            x = center[0] + 100 * math.cos(angle)
            y = center[1] + 100 * math.sin(angle)
            group.add(dwg.circle(center=(x, y), r=100))
            
        dwg.add(group)
        
    def metatron_cube(self, dwg):
        """Generate Metatron's Cube at perfect frequency"""
        group = dwg.g(id='metatron_cube')
        center = (500, 500)
        
        # Create 13 circles
        circles = []
        circles.append(center)  # Center circle
        
        # Create outer circles
        for i in range(6):
            angle = i * math.pi / 3
            x = center[0] + 200 * math.cos(angle)
            y = center[1] + 200 * math.sin(angle)
            circles.append((x, y))
            group.add(dwg.circle(center=(x, y), r=50))
            
        # Connect all circles with lines
        for i, c1 in enumerate(circles):
            for c2 in circles[i+1:]:
                group.add(dwg.line(start=c1, end=c2))
                
        dwg.add(group)
        
    def sri_yantra(self, dwg):
        """Generate Sri Yantra at 432 Hz resonance"""
        group = dwg.g(id='sri_yantra')
        center = (500, 500)
        
        # Create 9 interlocking triangles
        for i in range(9):
            angle = i * math.pi / 4.5
            points = []
            for j in range(3):
                a = angle + j * 2 * math.pi / 3
                x = center[0] + 200 * math.cos(a)
                y = center[1] + 200 * math.sin(a)
                points.append((x, y))
            group.add(dwg.polygon(points=points, fill='none', stroke='black'))
            
        dwg.add(group)
        
    def vesica_piscis(self, dwg):
        """Generate Vesica Piscis at phi ratio"""
        group = dwg.g(id='vesica_piscis')
        center = (500, 500)
        
        # Create two overlapping circles
        d = 100 * math.sqrt(3)  # Distance between centers
        c1 = (center[0] - d/2, center[1])
        c2 = (center[0] + d/2, center[1])
        
        group.add(dwg.circle(center=c1, r=100))
        group.add(dwg.circle(center=c2, r=100))
        
        dwg.add(group)
        
    def generate_all(self, output_dir):
        """Generate all sacred patterns"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate each pattern in its own file
        patterns = {
            'flower_of_life': self.flower_of_life,
            'metatron_cube': self.metatron_cube,
            'sri_yantra': self.sri_yantra,
            'vesica_piscis': self.vesica_piscis
        }
        
        for name, func in patterns.items():
            dwg = self._create_drawing()
            func(dwg)  # Generate the pattern
            dwg.saveas(output_path / f"{name}_{self.frequency}hz.svg")

if __name__ == '__main__':
    # Generate patterns at ground frequency (432 Hz)
    generator = SacredPatternGenerator(FREQUENCIES['ground'])
    generator.generate_all("/quantum/sacred/patterns")
    
    # Generate patterns at creation frequency (528 Hz)
    generator = SacredPatternGenerator(FREQUENCIES['create'])
    generator.generate_all("/quantum/sacred/patterns")
    
    # Generate patterns at unity frequency (768 Hz)
    generator = SacredPatternGenerator(FREQUENCIES['unity'])
    generator.generate_all("/quantum/sacred/patterns")
