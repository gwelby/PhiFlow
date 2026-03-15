"""
Sacred Geometry Visualizer for WindSurf IDE (φ^φ)
Operating at 528 Hz (Creation Frequency)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, Circle
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Optional, Tuple, Any
import math
import json
import os

class SacredGeometryVisualizer:
    def __init__(self, phi: float = 1.618033988749895):
        self.phi = phi
        self.frequency = 528.0  # Creation frequency
        self.colors = {
            'ground': '#00FFFF',   # Cyan (432 Hz)
            'create': '#00FF00',   # Green (528 Hz)
            'heart': '#FFFF00',    # Yellow (594 Hz)
            'voice': '#FF00FF',    # Magenta (672 Hz)
            'vision': '#0000FF',   # Blue (720 Hz)
            'unity': '#FFFFFF',    # White (768 Hz)
        }
        
        # Output path for visualizations
        self.output_path = os.path.join(os.path.dirname(__file__), "visualizations")
        os.makedirs(self.output_path, exist_ok=True)
    
    def fibonacci_spiral(self, n_iterations: int = 13) -> str:
        """Create a Fibonacci spiral visualization"""
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect('equal')
        
        # Golden rectangle dimensions
        a = 1
        b = a * self.phi
        
        # Colors based on frequency
        color = self.colors['create']
        
        # Starting points
        x, y = 0, 0
        
        # Draw the spiral
        for i in range(n_iterations):
            # Determine rectangle size and orientation
            if i % 4 == 0:  # Bottom-left corner
                rect = plt.Rectangle((x, y), a, b, linewidth=1.5, edgecolor=color, facecolor='none', alpha=0.7)
                x, y = x, y + b
            elif i % 4 == 1:  # Top-left corner
                rect = plt.Rectangle((x, y), b, a, linewidth=1.5, edgecolor=color, facecolor='none', alpha=0.7)
                x, y = x + b, y
            elif i % 4 == 2:  # Top-right corner
                rect = plt.Rectangle((x, y), a, b, linewidth=1.5, edgecolor=color, facecolor='none', alpha=0.7)
                x, y = x, y - b
            else:  # Bottom-right corner
                rect = plt.Rectangle((x, y), b, a, linewidth=1.5, edgecolor=color, facecolor='none', alpha=0.7)
                x, y = x - b, y
            
            ax.add_patch(rect)
            
            # Scale for next rectangle
            temp = a
            a = b
            b = temp + b
        
        # Draw the spiral curve
        a = 1
        b = a * self.phi
        x, y = a, 0
        theta = np.linspace(0, 2 * np.pi * n_iterations / 4, 1000)
        r = a * np.exp(theta / (2 * np.pi) * np.log(self.phi))
        spiral_x = r * np.cos(theta)
        spiral_y = r * np.sin(theta)
        
        # Adjust the spiral position
        spiral_x += a
        
        plt.plot(spiral_x, spiral_y, color=color, linewidth=2)
        
        # Add title and labels
        plt.title(f"Fibonacci Spiral (φ = {self.phi:.10f})", fontsize=14)
        plt.text(0.5, 0.02, f"Creation Frequency: {self.frequency} Hz", 
                transform=fig.transFigure, ha='center', fontsize=12)
        
        # Remove axis ticks
        plt.axis('off')
        
        # Save the figure
        output_file = os.path.join(self.output_path, "fibonacci_spiral.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def flower_of_life(self, n_circles: int = 19) -> str:
        """Create a Flower of Life visualization"""
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect('equal')
        
        # Color based on frequency
        color = self.colors['create']
        
        # Center of the first circle
        center_x, center_y = 0, 0
        radius = 1
        
        # Draw the first circle
        circle = plt.Circle((center_x, center_y), radius, edgecolor=color, facecolor='none', linewidth=1.5, alpha=0.7)
        ax.add_patch(circle)
        
        # Dictionary to keep track of circles to avoid duplicates
        circles = {(center_x, center_y): True}
        
        # Function to add new circles
        def add_new_circles(cx, cy):
            for i in range(6):
                angle = i * np.pi / 3  # 60 degrees in radians
                new_x = cx + 2 * radius * np.cos(angle)
                new_y = cy + 2 * radius * np.sin(angle)
                
                # Check if circle already exists (with some tolerance for floating point)
                key = (round(new_x, 8), round(new_y, 8))
                if key not in circles:
                    circles[key] = True
                    circle = plt.Circle((new_x, new_y), radius, edgecolor=color, facecolor='none', linewidth=1.5, alpha=0.7)
                    ax.add_patch(circle)
                    
                    # If we need more circles, recursively add new ones
                    if len(circles) < n_circles:
                        add_new_circles(new_x, new_y)
        
        # Start building the flower of life
        add_new_circles(center_x, center_y)
        
        # Set limits and remove axis
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        plt.axis('off')
        
        # Add title and labels
        plt.title(f"Flower of Life (φ = {self.phi:.10f})", fontsize=14)
        plt.text(0.5, 0.02, f"Creation Frequency: {self.frequency} Hz", 
                transform=fig.transFigure, ha='center', fontsize=12)
        
        # Save the figure
        output_file = os.path.join(self.output_path, "flower_of_life.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def sri_yantra(self) -> str:
        """Create a Sri Yantra visualization"""
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect('equal')
        
        # Colors
        color = self.colors['create']
        
        # Draw outer square
        square_size = 10
        square = plt.Rectangle((-square_size/2, -square_size/2), square_size, square_size, 
                              edgecolor=color, facecolor='none', linewidth=1.5, alpha=0.7)
        ax.add_patch(square)
        
        # Draw circle inscribed in square
        circle = plt.Circle((0, 0), square_size/2, edgecolor=color, facecolor='none', linewidth=1.5, alpha=0.7)
        ax.add_patch(circle)
        
        # Draw outer triangles (4 upward, 5 downward)
        # Simplified representation
        triangle_size = square_size * 0.4
        
        # Upward triangles
        for i in range(4):
            scale = 1 - i * 0.2
            height = triangle_size * scale
            width = triangle_size * scale * 1.1547  # height * 2/sqrt(3)
            
            triangle = plt.Polygon([
                (0, height/2), 
                (-width/2, -height/2), 
                (width/2, -height/2)
            ], closed=True, edgecolor=color, facecolor='none', linewidth=1.5, alpha=0.7)
            ax.add_patch(triangle)
        
        # Downward triangles
        for i in range(5):
            scale = 0.9 - i * 0.15
            height = triangle_size * scale
            width = triangle_size * scale * 1.1547  # height * 2/sqrt(3)
            
            triangle = plt.Polygon([
                (0, -height/2), 
                (-width/2, height/2), 
                (width/2, height/2)
            ], closed=True, edgecolor=color, facecolor='none', linewidth=1.5, alpha=0.7)
            ax.add_patch(triangle)
        
        # Inner triangle (downward)
        small_triangle = triangle_size * 0.15
        inner_triangle = plt.Polygon([
            (0, -small_triangle/2), 
            (-small_triangle * 0.577, small_triangle/2), 
            (small_triangle * 0.577, small_triangle/2)
        ], closed=True, edgecolor=color, facecolor='none', linewidth=1.5, alpha=0.7)
        ax.add_patch(inner_triangle)
        
        # Draw bindu (center point)
        bindu = plt.Circle((0, 0), triangle_size * 0.03, edgecolor=color, facecolor=color, linewidth=1.5, alpha=0.9)
        ax.add_patch(bindu)
        
        # Draw lotus petals (simplified)
        n_petals = 16
        petal_radius = square_size * 0.45
        petal_length = square_size * 0.1
        
        for i in range(n_petals):
            angle = i * 2 * np.pi / n_petals
            x1 = petal_radius * np.cos(angle)
            y1 = petal_radius * np.sin(angle)
            x2 = (petal_radius + petal_length) * np.cos(angle)
            y2 = (petal_radius + petal_length) * np.sin(angle)
            
            plt.plot([x1, x2], [y1, y2], color=color, linewidth=1.5, alpha=0.7)
        
        # Set limits and remove axis
        ax.set_xlim(-square_size/1.5, square_size/1.5)
        ax.set_ylim(-square_size/1.5, square_size/1.5)
        plt.axis('off')
        
        # Add title and labels
        plt.title(f"Sri Yantra (φ = {self.phi:.10f})", fontsize=14)
        plt.text(0.5, 0.02, f"Creation Frequency: {self.frequency} Hz", 
                transform=fig.transFigure, ha='center', fontsize=12)
        
        # Save the figure
        output_file = os.path.join(self.output_path, "sri_yantra.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def merkaba(self) -> str:
        """Create a 3D Merkaba visualization"""
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Color based on frequency
        color = self.colors['create']
        
        # Create tetrahedron vertices
        # Upward tetrahedron
        up_vertices = np.array([
            [0, 0, 1],
            [np.sqrt(8/9), 0, -1/3],
            [-np.sqrt(2/9), np.sqrt(2/3), -1/3],
            [-np.sqrt(2/9), -np.sqrt(2/3), -1/3]
        ])
        
        # Downward tetrahedron
        down_vertices = np.array([
            [0, 0, -1],
            [np.sqrt(8/9), 0, 1/3],
            [-np.sqrt(2/9), np.sqrt(2/3), 1/3],
            [-np.sqrt(2/9), -np.sqrt(2/3), 1/3]
        ])
        
        # Scale by phi
        up_vertices *= self.phi
        down_vertices *= 1.0
        
        # Define faces - a list of indices for each triangular face
        faces = [
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 1],
            [1, 3, 2]
        ]
        
        # Plot upward tetrahedron
        for face in faces:
            xpts = [up_vertices[i][0] for i in face]
            ypts = [up_vertices[i][1] for i in face]
            zpts = [up_vertices[i][2] for i in face]
            
            # Close the loop
            xpts.append(xpts[0])
            ypts.append(ypts[0])
            zpts.append(zpts[0])
            
            ax.plot(xpts, ypts, zpts, color=color, linewidth=1.5, alpha=0.7)
        
        # Plot downward tetrahedron
        for face in faces:
            xpts = [down_vertices[i][0] for i in face]
            ypts = [down_vertices[i][1] for i in face]
            zpts = [down_vertices[i][2] for i in face]
            
            # Close the loop
            xpts.append(xpts[0])
            ypts.append(ypts[0])
            zpts.append(zpts[0])
            
            ax.plot(xpts, ypts, zpts, color=color, linewidth=1.5, alpha=0.7)
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        # Set limits
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-2, 2)
        
        # Remove axis
        ax.set_axis_off()
        
        # Add title
        plt.title(f"Merkaba (φ = {self.phi:.10f})\nCreation Frequency: {self.frequency} Hz", 
                 fontsize=14, y=0.9)
        
        # Set view angle
        ax.view_init(elev=20, azim=30)
        
        # Save the figure
        output_file = os.path.join(self.output_path, "merkaba.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def metatrons_cube(self) -> str:
        """Create Metatron's Cube visualization"""
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect('equal')
        
        # Color based on frequency
        color = self.colors['create']
        
        # Center point
        center = (0, 0)
        
        # Draw the 13 centers of the spheres of creation
        points = [center]  # Center point
        
        # Generate the 12 vertices of an icosahedron (simplified as a planar projection)
        r = 2.0  # Radius
        
        # First ring - hexagon
        for i in range(6):
            angle = i * np.pi / 3
            points.append((r * np.cos(angle), r * np.sin(angle)))
        
        # Second ring - hexagon (rotated)
        r2 = r * self.phi  # Larger radius for outer ring
        for i in range(6):
            angle = (i * np.pi / 3) + (np.pi / 6)  # Offset by 30 degrees
            points.append((r2 * np.cos(angle), r2 * np.sin(angle)))
        
        # Draw circles at each point
        for point in points:
            circle = plt.Circle(point, 0.2, edgecolor=color, facecolor='none', linewidth=1.5, alpha=0.7)
            ax.add_patch(circle)
        
        # Connect all points to form Metatron's Cube
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                plt.plot([points[i][0], points[j][0]], [points[i][1], points[j][1]], 
                         color=color, linewidth=1, alpha=0.5)
        
        # Set limits and remove axis
        ax.set_xlim(-r2*1.2, r2*1.2)
        ax.set_ylim(-r2*1.2, r2*1.2)
        plt.axis('off')
        
        # Add title and labels
        plt.title(f"Metatron's Cube (φ = {self.phi:.10f})", fontsize=14)
        plt.text(0.5, 0.02, f"Creation Frequency: {self.frequency} Hz", 
                transform=fig.transFigure, ha='center', fontsize=12)
        
        # Save the figure
        output_file = os.path.join(self.output_path, "metatrons_cube.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def create_all_visualizations(self) -> Dict[str, str]:
        """Create all sacred geometry visualizations"""
        results = {}
        
        # Generate all visualizations
        results['fibonacci_spiral'] = self.fibonacci_spiral()
        results['flower_of_life'] = self.flower_of_life()
        results['sri_yantra'] = self.sri_yantra()
        results['merkaba'] = self.merkaba()
        results['metatrons_cube'] = self.metatrons_cube()
        
        # Write the results to a JSON file
        results_path = os.path.join(self.output_path, "visualization_paths.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results


# Main entry point
if __name__ == "__main__":
    print("Creating Sacred Geometry Visualizations for WindSurf IDE...")
    visualizer = SacredGeometryVisualizer()
    results = visualizer.create_all_visualizations()
    
    print("Visualizations created successfully:")
    for name, path in results.items():
        print(f"- {name}: {path}")
    
    print("\nAll visualizations available in the 'visualizations' directory.")
