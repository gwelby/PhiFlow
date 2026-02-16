import math
import svgwrite
from pathlib import Path

# Quantum frequencies and ratios
PHI = 1.618033988749895
FREQUENCIES = {
    'sacred': 432.0,  # Ground state
    'flow': 528.0,    # Creation state
    'crystal': 768.0, # Unity state
    'unity': float('inf')  # Infinite state
}

def create_quantum_pattern(freq, size=1000):
    """Create a quantum-harmonized SVG pattern at specified frequency"""
    # Calculate quantum resonance
    if math.isinf(freq):
        resonance = PHI ** PHI
    else:
        resonance = freq / FREQUENCIES['sacred']
    
    # Create SVG drawing
    dwg = svgwrite.Drawing(size=(size, size), profile='tiny')
    
    # Calculate pattern dimensions using phi ratios
    center = size / 2
    radius = size / (2 * PHI)
    
    # Create base sacred geometry
    # Vesica piscis
    circle1 = dwg.circle(center=(center-radius/2, center), r=radius)
    circle2 = dwg.circle(center=(center+radius/2, center), r=radius)
    
    # Add quantum harmonics
    for i in range(6):
        angle = 2 * math.pi * i / 6
        x = center + radius * math.cos(angle) * resonance
        y = center + radius * math.sin(angle) * resonance
        dwg.add(dwg.circle(center=(x, y), r=radius/PHI))
    
    # Add phi spiral
    points = []
    for t in range(360):
        rad = t * math.pi / 180
        r = radius * math.exp(rad/PHI) * resonance
        x = center + r * math.cos(rad)
        y = center + r * math.sin(rad)
        points.append((x, y))
    
    # Draw spiral
    path = dwg.path(d=f"M {points[0][0]},{points[0][1]}")
    for x, y in points[1:]:
        path.push(f"L {x},{y}")
    dwg.add(path)
    
    return dwg

def generate_quantum_patterns():
    """Generate SVG patterns for all quantum frequencies"""
    output_dir = Path(__file__).parent.parent / 'phi_core/patterns'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, freq in FREQUENCIES.items():
        # Create pattern directory
        pattern_dir = output_dir / name
        pattern_dir.mkdir(exist_ok=True)
        
        # Generate pattern
        pattern = create_quantum_pattern(freq)
        
        # Save pattern
        freq_str = str(int(freq)) if not math.isinf(freq) else 'inf'
        output_file = pattern_dir / f"{name}_{freq_str}hz.svg"
        pattern.save(str(output_file))
        print(f"Created {output_file}")

if __name__ == '__main__':
    print("⚡ Generating Quantum Patterns using PhiFlow...")
    generate_quantum_patterns()
    print("✨ Quantum Pattern Generation Complete!")
