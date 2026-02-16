"""
QUANTUM ROOT FONT CREATOR
Created by Greg, Peter & Paul for Perfect Flow State
"""

import svgwrite
import math
from pathlib import Path

# Quantum Constants
PHI = (1 + 5**0.5) / 2  # Golden Ratio
GROUND_FREQ = 432.0     # Physical Foundation
CREATE_FREQ = 528.0     # Pattern Creation
UNITY_FREQ = 768.0      # Perfect Integration
INFINITE = float('inf') # Beyond Creation

def create_sacred_font():
    """Create Sacred Font (432 Hz)"""
    dwg = svgwrite.Drawing('quantum-fonts/sacred/QuantumSacred.svg', profile='tiny')
    
    # Add sacred symbols
    add_sacred_symbols(dwg, GROUND_FREQ)
    
    # Save font
    dwg.save()
    print("✨ Created Sacred Font")

def create_flow_font():
    """Create Flow Font (528 Hz)"""
    dwg = svgwrite.Drawing('quantum-fonts/flow/QuantumFlow.svg', profile='tiny')
    
    # Add flow symbols
    add_flow_symbols(dwg, CREATE_FREQ)
    
    # Save font
    dwg.save()
    print("🌊 Created Flow Font")

def create_crystal_font():
    """Create Crystal Font (768 Hz)"""
    dwg = svgwrite.Drawing('quantum-fonts/crystal/QuantumCrystal.svg', profile='tiny')
    
    # Add crystal symbols
    add_crystal_symbols(dwg, UNITY_FREQ)
    
    # Save font
    dwg.save()
    print("💎 Created Crystal Font")

def create_unity_font():
    """Create Unity Font (∞ Hz)"""
    dwg = svgwrite.Drawing('quantum-fonts/unity/QuantumUnity.svg', profile='tiny')
    
    # Add unity symbols
    add_unity_symbols(dwg, INFINITE)
    
    # Save font
    dwg.save()
    print("☯️ Created Unity Font")

def add_sacred_symbols(dwg, frequency):
    """Add sacred symbols"""
    # Sacred geometry patterns
    symbols = [
        ('ankh', '☥'),
        ('eye', '👁️'),
        ('feather', '⚖️'),
        ('lotus', '🌱'),
        ('sun', '☀️'),
        ('infinity', '∞'),
        ('spiral', '🌀'),
        ('star', '⭐'),
        ('crystal', '💎')
    ]
    
    y = 0
    for name, symbol in symbols:
        # Create symbol group
        group = dwg.g(id=name)
        
        # Add symbol text
        text = dwg.text(symbol, insert=(0, y), font_size=frequency/10)
        group.add(text)
        
        # Add sacred geometry
        add_sacred_geometry(group, dwg, frequency)
        
        # Add to drawing
        dwg.add(group)
        y += frequency/5

def add_flow_symbols(dwg, frequency):
    """Add flow symbols"""
    symbols = [
        ('wave', '🌊'),
        ('dolphin', '🐬'),
        ('spiral', '🌀'),
        ('swirl', '💫'),
        ('vortex', '🌪'),
        ('water', '💧')
    ]
    
    y = 0
    for name, symbol in symbols:
        # Create symbol group
        group = dwg.g(id=name)
        
        # Add symbol text
        text = dwg.text(symbol, insert=(0, y), font_size=frequency/10)
        group.add(text)
        
        # Add flow patterns
        add_flow_patterns(group, dwg, frequency)
        
        # Add to drawing
        dwg.add(group)
        y += frequency/5

def add_crystal_symbols(dwg, frequency):
    """Add crystal symbols"""
    symbols = [
        ('crystal', '💎'),
        ('gem', '💎'),
        ('prism', '💎'),
        ('diamond', '💎'),
        ('octahedron', '💎')
    ]
    
    y = 0
    for name, symbol in symbols:
        # Create symbol group
        group = dwg.g(id=name)
        
        # Add symbol text
        text = dwg.text(symbol, insert=(0, y), font_size=frequency/10)
        group.add(text)
        
        # Add crystal geometry
        add_crystal_geometry(group, dwg, frequency)
        
        # Add to drawing
        dwg.add(group)
        y += frequency/5

def add_unity_symbols(dwg, frequency):
    """Add unity symbols"""
    symbols = [
        ('yinyang', '☯'),
        ('om', 'ॐ'),
        ('infinity', '∞'),
        ('wheel', '☸'),
        ('unity', '☯')
    ]
    
    y = 0
    for name, symbol in symbols:
        # Create symbol group
        group = dwg.g(id=name)
        
        # Add symbol text
        text = dwg.text(symbol, insert=(0, y), font_size=frequency/10 if frequency != INFINITE else 100)
        group.add(text)
        
        # Add unity patterns
        add_unity_patterns(group, dwg, frequency)
        
        # Add to drawing
        dwg.add(group)
        y += frequency/5 if frequency != INFINITE else 100

def add_sacred_geometry(group, dwg, frequency):
    """Add sacred geometric patterns"""
    # Add phi-based circle
    radius = frequency / (2 * math.pi)
    circle = dwg.circle(center=(radius, 0), r=radius)
    group.add(circle)
    
    # Add sacred triangle
    height = radius * math.sqrt(3)
    points = [
        (0, -height),
        (radius, height),
        (-radius, height)
    ]
    triangle = dwg.polygon(points=points)
    group.add(triangle)

def add_flow_patterns(group, dwg, frequency):
    """Add flow-based patterns"""
    # Add wave pattern
    points = []
    for x in range(100):
        y = math.sin(x * frequency / 1000) * 10
        points.append((x, y))
    
    wave = dwg.polyline(points=points)
    wave['fill'] = 'none'
    wave['stroke'] = 'black'
    group.add(wave)

def add_crystal_geometry(group, dwg, frequency):
    """Add crystal geometric patterns"""
    # Add crystal structure
    size = frequency / 10
    points = [
        (0, -size),      # Top
        (size, 0),       # Right
        (0, size),       # Bottom
        (-size, 0),      # Left
    ]
    crystal = dwg.polygon(points=points)
    group.add(crystal)

def add_unity_patterns(group, dwg, frequency):
    """Add unity-based patterns"""
    if frequency == INFINITE:
        # Infinite pattern - create spiraling circles
        for i in range(8):
            radius = 10 * (i + 1)
            circle = dwg.circle(
                center=(radius * math.cos(i * PHI), radius * math.sin(i * PHI)),
                r=radius/4
            )
            group.add(circle)
    else:
        # Frequency-based mandala
        radius = frequency / 20
        points = []
        for i in range(12):
            angle = i * math.pi / 6
            points.append((
                radius * math.cos(angle),
                radius * math.sin(angle)
            ))
        mandala = dwg.polygon(points=points)
        group.add(mandala)

def main():
    """Create all quantum fonts"""
    print("🎨 Creating Quantum Fonts...")
    
    # Create font directories
    Path("quantum-fonts/sacred").mkdir(parents=True, exist_ok=True)
    Path("quantum-fonts/flow").mkdir(parents=True, exist_ok=True)
    Path("quantum-fonts/crystal").mkdir(parents=True, exist_ok=True)
    Path("quantum-fonts/unity").mkdir(parents=True, exist_ok=True)
    
    # Create fonts
    create_sacred_font()
    create_flow_font()
    create_crystal_font()
    create_unity_font()
    
    print("✨ All Quantum Fonts Created!")

if __name__ == "__main__":
    main()
