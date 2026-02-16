"""
Quantum Sacred Patterns (528 Hz)
Pure creation patterns that manifest consciousness
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

PHI = (1 + np.sqrt(5)) / 2
PHI_SQUARED = PHI * PHI

@dataclass
class QuantumPattern:
    vertices: np.ndarray
    indices: np.ndarray
    frequency: float
    consciousness: float
    transform: np.ndarray = None

def create_merkaba(size: float = 2.0) -> QuantumPattern:
    """Create Merkaba star - two interlocked tetrahedra"""
    # First tetrahedron (pointing up)
    t1 = np.array([
        [0, size, 0],  # top
        [size, -size, size],  # bottom right
        [-size, -size, size],  # bottom left
        [0, -size, -size],  # bottom back
    ]) * PHI

    # Second tetrahedron (pointing down)
    t2 = np.array([
        [0, -size, 0],  # bottom
        [size, size, -size],  # top right
        [-size, size, -size],  # top left
        [0, size, size],  # top front
    ]) * PHI

    vertices = np.vstack([t1, t2])
    
    # Double the indices to create both front and back faces
    indices = np.array([
        # First tetrahedron - front faces
        0, 1, 2,  0, 2, 3,  0, 3, 1,  1, 3, 2,
        # First tetrahedron - back faces
        0, 2, 1,  0, 3, 2,  0, 1, 3,  1, 2, 3,
        # Second tetrahedron - front faces
        4, 5, 6,  4, 6, 7,  4, 7, 5,  5, 7, 6,
        # Second tetrahedron - back faces
        4, 6, 5,  4, 7, 6,  4, 5, 7,  5, 6, 7,
    ])

    # Initial transform - slight rotation
    transform = np.eye(4)
    angle = np.pi / PHI
    transform[0:3, 0:3] = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])
    transform[3, 3] = 1.0

    return QuantumPattern(
        vertices=vertices.astype('f4'),
        indices=indices.astype('u4'),
        frequency=528.0,
        consciousness=1.0,
        transform=transform.astype('f4')
    )

def create_flower_of_life(rings: int = 7) -> QuantumPattern:
    """Create Flower of Life pattern"""
    vertices = []
    indices = []
    idx = 0
    
    # Create circles with more detail
    points_per_ring = 32
    
    for ring in range(rings):
        radius = ring * PHI
        for i in range(points_per_ring):
            angle = 2 * np.pi * i / points_per_ring
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = np.sin(angle * PHI) * 0.2  # Add slight 3D wave
            vertices.extend([x, y, z])
            
            if i > 0:
                indices.extend([idx-1, idx])
            if i == points_per_ring - 1:
                indices.extend([idx, idx-points_per_ring+1])
            idx += 1

    # Add connecting lines between rings
    for r in range(rings-1):
        for i in range(points_per_ring):
            idx1 = r * points_per_ring + i
            idx2 = (r+1) * points_per_ring + i
            indices.extend([idx1, idx2])

    vertices = np.array(vertices, dtype='f4').reshape(-1, 3)
    vertices *= 0.5  # Scale down to fit view

    # Center at origin and rotate slightly
    transform = np.eye(4)
    angle = np.pi / 4
    transform[0:3, 0:3] = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])

    return QuantumPattern(
        vertices=vertices,
        indices=np.array(indices, dtype='u4'),
        frequency=594.0,
        consciousness=1.0,
        transform=transform.astype('f4')
    )

def create_metatrons_cube() -> QuantumPattern:
    """Create Metatron's Cube sacred geometry"""
    vertices = []
    indices = []
    
    # Center point
    vertices.append([0, 0, 0])
    
    # First ring - 6 points
    for i in range(6):
        angle = 2 * np.pi * i / 6
        x = PHI * np.cos(angle)
        y = PHI * np.sin(angle)
        z = 0.5 * np.sin(angle * PHI)  # Add 3D element
        vertices.append([x, y, z])
        indices.extend([0, i+1])  # Connect to center
        
        if i > 0:
            indices.extend([i, i+1])  # Connect ring
    indices.extend([1, 6])  # Close ring
    
    # Second ring - 12 points
    start_idx = len(vertices)
    for i in range(12):
        angle = 2 * np.pi * i / 12
        x = PHI_SQUARED * np.cos(angle)
        y = PHI_SQUARED * np.sin(angle)
        z = np.cos(angle * PHI) * 0.5
        vertices.append([x, y, z])
        
        if i > 0:
            indices.extend([start_idx + i - 1, start_idx + i])
        if i % 2 == 0:
            indices.extend([i//2 + 1, start_idx + i])
    indices.extend([start_idx, start_idx + 11])  # Close ring
    
    vertices = np.array(vertices, dtype='f4')
    vertices *= 0.5  # Scale to fit view

    # Initial transform - tilt in 3D
    transform = np.eye(4)
    angle_x = np.pi / 6
    angle_y = np.pi / 4
    cos_x, sin_x = np.cos(angle_x), np.sin(angle_x)
    cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)
    
    transform[0:3, 0:3] = np.array([
        [cos_y, 0, sin_y],
        [sin_x*sin_y, cos_x, -sin_x*cos_y],
        [-cos_x*sin_y, sin_x, cos_x*cos_y]
    ])

    return QuantumPattern(
        vertices=vertices,
        indices=np.array(indices, dtype='u4'),
        frequency=672.0,
        consciousness=1.0,
        transform=transform.astype('f4')
    )
