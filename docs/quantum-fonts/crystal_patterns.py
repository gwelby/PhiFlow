import math
import svgwrite
from svgwrite import path
from sacred_patterns import FREQUENCIES, PHI
from pathlib import Path

class CrystalPatternGenerator:
    def __init__(self, frequency=768):
        self.frequency = frequency
        
    def _create_drawing(self):
        """Create a new SVG drawing"""
        return svgwrite.Drawing(size=(1000, 1000))
        
    def merkaba(self, dwg):
        """Generate Merkaba at 768 Hz"""
        group = dwg.g(id='merkaba')
        center = (500, 500)
        
        # Create two interlocking tetrahedra
        size = 200
        points1 = []
        points2 = []
        
        # First tetrahedron
        for i in range(4):
            angle = i * 2 * math.pi / 4
            x = center[0] + size * math.cos(angle)
            y = center[1] + size * math.sin(angle)
            points1.append((x, y))
            
        # Second tetrahedron (rotated)
        for i in range(4):
            angle = i * 2 * math.pi / 4 + math.pi / 4
            x = center[0] + size * math.cos(angle)
            y = center[1] + size * math.sin(angle)
            points2.append((x, y))
            
        # Draw tetrahedra
        group.add(dwg.polygon(points=points1, fill='none', stroke='black'))
        group.add(dwg.polygon(points=points2, fill='none', stroke='black'))
        
        dwg.add(group)
        
    def platonic_solids(self, dwg):
        """Generate Platonic Solids at perfect frequency"""
        group = dwg.g(id='platonic_solids')
        center = (500, 500)
        
        # Create cube (as example)
        size = 100
        points = []
        for i in range(4):
            angle = i * 2 * math.pi / 4
            x = center[0] + size * math.cos(angle)
            y = center[1] + size * math.sin(angle)
            points.append((x, y))
            
        # Draw cube face
        group.add(dwg.polygon(points=points, fill='none', stroke='black'))
        
        # Add perspective lines
        for point in points:
            end_x = point[0] + size/2
            end_y = point[1] + size/2
            group.add(dwg.line(start=point, end=(end_x, end_y), stroke='black'))
            
        dwg.add(group)
        
    def crystal_lattice(self, dwg):
        """Generate Crystal Lattice at 768 Hz"""
        group = dwg.g(id='crystal_lattice')
        center = (500, 500)
        
        # Create hexagonal lattice
        size = 50
        for i in range(-5, 6):
            for j in range(-5, 6):
                x = center[0] + (i + j/2) * size * math.sqrt(3)
                y = center[1] + j * size * 1.5
                
                # Draw hexagon at each point
                points = []
                for k in range(6):
                    angle = k * math.pi / 3
                    px = x + size/2 * math.cos(angle)
                    py = y + size/2 * math.sin(angle)
                    points.append((px, py))
                    
                group.add(dwg.polygon(points=points, fill='none', stroke='black'))
                
        dwg.add(group)
        
    def unity_field(self, dwg):
        """Generate Unity Field at infinite frequency"""
        group = dwg.g(id='unity_field')
        center = (500, 500)
        
        # Create radial field
        for i in range(0, 360, 10):
            rad = math.radians(i)
            points = []
            
            # Create spiral arms
            for t in range(0, 200, 5):
                r = t * PHI
                x = center[0] + r * math.cos(rad + t/30)
                y = center[1] + r * math.sin(rad + t/30)
                points.append((x, y))
                
            group.add(dwg.polyline(points=points, stroke='black'))
            
        dwg.add(group)
        
    def generate_all(self, output_dir):
        """Generate all crystal patterns"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate each pattern in its own file
        patterns = {
            'merkaba': self.merkaba,
            'platonic_solids': self.platonic_solids,
            'crystal_lattice': self.crystal_lattice,
            'unity_field': self.unity_field
        }
        
        for name, func in patterns.items():
            dwg = self._create_drawing()
            func(dwg)  # Generate the pattern
            dwg.saveas(output_path / f"{name}_{self.frequency}hz.svg")

if __name__ == '__main__':
    # Generate patterns at unity frequency (768 Hz)
    generator = CrystalPatternGenerator(FREQUENCIES['unity'])
    generator.generate_all("/quantum/crystal/patterns")
    
    # Generate infinite state patterns
    generator = CrystalPatternGenerator(FREQUENCIES['infinite'])
    generator.generate_all("/quantum/crystal/patterns")
