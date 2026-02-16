import math
import svgwrite
from svgwrite import path
from sacred_patterns import FREQUENCIES, PHI
from pathlib import Path

class FlowPatternGenerator:
    def __init__(self, frequency=528):
        self.frequency = frequency
        
    def _create_drawing(self):
        """Create a new SVG drawing"""
        return svgwrite.Drawing(size=(1000, 1000))
        
    def dna_helix(self, dwg):
        """Generate DNA Helix pattern at 528 Hz"""
        group = dwg.g(id='dna_helix')
        center = (500, 500)
        
        # Create double helix
        points1 = []
        points2 = []
        for t in range(0, 360, 5):
            rad = math.radians(t)
            x = center[0] + t/2
            y1 = center[1] + 100 * math.sin(rad)
            y2 = center[1] + 100 * math.sin(rad + math.pi)
            points1.append((x, y1))
            points2.append((x, y2))
            
        # Draw the helices
        group.add(dwg.polyline(points=points1))
        group.add(dwg.polyline(points=points2))
        
        dwg.add(group)
        
    def fibonacci_spiral(self, dwg):
        """Generate Fibonacci Spiral at phi ratio"""
        group = dwg.g(id='fibonacci_spiral')
        center = (500, 500)
        
        # Create Fibonacci spiral
        points = []
        a = 0
        b = 1
        for _ in range(20):
            rad = math.radians(_ * 137.5)
            r = math.sqrt(a) * 10
            x = center[0] + r * math.cos(rad)
            y = center[1] + r * math.sin(rad)
            points.append((x, y))
            a, b = b, a + b
            
        group.add(dwg.polyline(points=points))
        dwg.add(group)
        
    def wave_interference(self, dwg):
        """Generate Wave Interference pattern at 528 Hz"""
        group = dwg.g(id='wave_interference')
        center = (500, 500)
        
        # Create wave interference pattern
        for i in range(0, 360, 10):
            rad = math.radians(i)
            points = []
            for t in range(0, 1000, 10):
                x = t
                y = center[1] + 50 * math.sin(rad + t/50)
                points.append((x, y))
            group.add(dwg.polyline(points=points))
            
        dwg.add(group)
        
    def vortex_field(self, dwg):
        """Generate Vortex Field pattern at phi ratio"""
        group = dwg.g(id='vortex_field')
        center = (500, 500)
        
        # Create vortex pattern
        for i in range(0, 360, 10):
            rad = math.radians(i)
            points = []
            for t in range(0, 200, 5):
                r = t * PHI
                x = center[0] + r * math.cos(rad + t/30)
                y = center[1] + r * math.sin(rad + t/30)
                points.append((x, y))
            group.add(dwg.polyline(points=points))
            
        dwg.add(group)
        
    def generate_all(self, output_dir):
        """Generate all flow patterns"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate each pattern in its own file
        patterns = {
            'dna_helix': self.dna_helix,
            'fibonacci_spiral': self.fibonacci_spiral,
            'wave_interference': self.wave_interference,
            'vortex_field': self.vortex_field
        }
        
        for name, func in patterns.items():
            dwg = self._create_drawing()
            func(dwg)  # Generate the pattern
            dwg.saveas(output_path / f"{name}_{self.frequency}hz.svg")

if __name__ == '__main__':
    # Generate patterns at creation frequency (528 Hz)
    generator = FlowPatternGenerator(FREQUENCIES['create'])
    generator.generate_all("/quantum/flow/patterns")
    
    # Generate additional variations at unity frequency (768 Hz)
    generator = FlowPatternGenerator(FREQUENCIES['unity'])
    generator.generate_all("/quantum/flow/patterns")
