import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class QuantumGeometry:
    def __init__(self):
        self.phi = 1.618034
        
    def merkaba_points(self, size: float = 1.0) -> np.ndarray:
        """Generate merkaba star tetrahedron points"""
        # Two interlocked tetrahedra
        points = []
        
        # Upward tetrahedron
        h = np.sqrt(6) / 3 * size
        for i in range(4):
            angle = i * 2 * np.pi / 3
            x = size * np.cos(angle)
            y = size * np.sin(angle)
            points.append([x, y, -h/2])
        points.append([0, 0, h])
        
        # Downward tetrahedron
        for i in range(4):
            angle = (i * 2 * np.pi / 3) + np.pi/3
            x = size * np.cos(angle)
            y = size * np.sin(angle)
            points.append([x, y, h/2])
        points.append([0, 0, -h])
        
        return np.array(points)
    
    def flower_of_life(self, layers: int = 6) -> List[Tuple[float, float, float]]:
        """Generate flower of life pattern"""
        points = []
        radius = 1.0
        
        for layer in range(layers):
            n_circles = 6 * layer if layer > 0 else 1
            for i in range(n_circles):
                angle = 2 * np.pi * i / n_circles
                x = layer * radius * np.cos(angle)
                y = layer * radius * np.sin(angle)
                points.append((x, y, radius))
                
        return points
    
    def torus_points(self, major_r: float = 3.0, minor_r: float = 1.0, 
                    n_major: int = 50, n_minor: int = 20) -> np.ndarray:
        """Generate torus points with phi-based ratios"""
        points = []
        
        for i in range(n_major):
            theta = i * 2 * np.pi / n_major
            for j in range(n_minor):
                phi = j * 2 * np.pi / n_minor
                x = (major_r + minor_r * np.cos(phi)) * np.cos(theta)
                y = (major_r + minor_r * np.cos(phi)) * np.sin(theta)
                z = minor_r * np.sin(phi)
                points.append([x, y, z])
                
        return np.array(points)
    
    def metatrons_cube(self, size: float = 1.0) -> np.ndarray:
        """Generate Metatron's Cube points"""
        points = []
        
        # Center point
        points.append([0, 0, 0])
        
        # First ring (6 points)
        for i in range(6):
            angle = i * np.pi / 3
            points.append([size * np.cos(angle), size * np.sin(angle), 0])
            
        # Second ring (12 points)
        for i in range(12):
            angle = i * np.pi / 6
            points.append([size * self.phi * np.cos(angle), 
                         size * self.phi * np.sin(angle), 0])
            
        return np.array(points)
    
    def phi_spiral(self, turns: int = 8) -> np.ndarray:
        """Generate phi spiral points"""
        points = []
        t = np.linspace(0, turns * 2 * np.pi, 1000)
        
        r = self.phi ** (t / (2 * np.pi))
        x = r * np.cos(t)
        y = r * np.sin(t)
        z = t / (2 * np.pi)
        
        return np.column_stack([x, y, z])
    
    def dodecahedron_points(self, size: float = 1.0) -> np.ndarray:
        """Generate dodecahedron points"""
        phi = self.phi
        points = []
        
        # Generate vertices
        for i in [-1, 1]:
            for j in [-1, 1]:
                for k in [-1, 1]:
                    points.append([i, j, k])
                    points.append([0, i*phi, j/phi])
                    points.append([i/phi, 0, j*phi])
                    points.append([i*phi, j/phi, 0])
                    
        return np.array(points) * size
    
    def vesica_piscis(self, ratio: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate vesica piscis points"""
        if ratio is None:
            ratio = self.phi
            
        r = 1.0
        d = r * ratio
        
        # Circle 1 centered at (-d/2, 0)
        theta = np.linspace(0, 2*np.pi, 100)
        x1 = r * np.cos(theta) - d/2
        y1 = r * np.sin(theta)
        
        # Circle 2 centered at (d/2, 0)
        x2 = r * np.cos(theta) + d/2
        y2 = r * np.sin(theta)
        
        return np.column_stack([x1, y1]), np.column_stack([x2, y2])
    
    def consciousness_grid(self, points: int = 144, layers: int = 12) -> np.ndarray:
        """Generate Christ consciousness grid points"""
        grid = []
        
        # Generate phi-spiral based grid points
        for layer in range(layers):
            n_points = points // layers
            radius = self.phi ** layer
            
            for i in range(n_points):
                angle = 2 * np.pi * i / n_points
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                z = layer * self.phi
                grid.append([x, y, z])
                
        return np.array(grid)
    
    def plot_geometry(self, points: np.ndarray, title: str = "Quantum Geometry"):
        """Plot 3D geometry visualization"""
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                  c=np.linspace(0, 1, len(points)), cmap='viridis')
        
        ax.set_title(title)
        plt.show()
        
    def animate_geometry(self, points: np.ndarray, rotation_speed: float = 0.1):
        """Animate 3D geometry with rotation"""
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        def update(frame):
            ax.clear()
            # Rotate points
            theta = frame * rotation_speed
            rot_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
            rotated_points = points @ rot_matrix
            
            ax.scatter(rotated_points[:, 0], rotated_points[:, 1], rotated_points[:, 2],
                      c=np.linspace(0, 1, len(points)), cmap='viridis')
            ax.set_title(f"Quantum Geometry (φ = {self.phi:.3f})")
            
        return update

if __name__ == "__main__":
    # Test geometry generation
    geometry = QuantumGeometry()
    
    # Generate and plot merkaba
    merkaba = geometry.merkaba_points()
    geometry.plot_geometry(merkaba, "Merkaba Star Tetrahedron")
