import pygame
import numpy as np
from quantum_bridge import QuantumBridge
import asyncio
import cv2
from datetime import datetime

class QuantumVisual:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((1920, 1080), pygame.FULLSCREEN)
        pygame.display.set_caption("Quantum Flow")
        
        self.bridge = QuantumBridge()
        self.clock = pygame.time.Clock()
        
        # Phi-based colors
        self.phi = (1 + np.sqrt(5)) / 2
        self.colors = {
            'ground': (43, 32, 32),      # Deep earth
            'create': (52, 82, 88),      # Ocean depth
            'heart': (59, 44, 94),       # Heart purple
            'unity': (76, 88, 88),       # Cosmic blue
            'love': (52, 28, 28)         # Valentine red
        }
        
        # Initialize particle system
        self.particles = []
        self.init_particles()
        
    def init_particles(self):
        """Create phi-spiral particle system"""
        for i in range(1000):
            angle = i * self.phi * 2 * np.pi
            r = np.sqrt(i) * 10
            x = 960 + r * np.cos(angle)
            y = 540 + r * np.sin(angle)
            self.particles.append({
                'pos': np.array([x, y]),
                'vel': np.array([0., 0.]),
                'life': 1.0,
                'color': self.colors['heart']
            })
    
    def update_particles(self, dt):
        """Update particle positions with quantum influence"""
        now = datetime.now()
        time_freq = (now.hour * 3600 + now.minute * 60 + now.second) / (24 * 3600)
        
        for p in self.particles:
            # Add quantum-influenced motion
            center = np.array([960, 540])
            to_center = center - p['pos']
            dist = np.linalg.norm(to_center)
            
            # Create phi-based orbital motion
            angle = time_freq * 2 * np.pi * self.phi
            rotation = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
            
            # Apply forces
            orbital = rotation @ to_center * 0.1
            p['vel'] = orbital * dt
            p['pos'] += p['vel']
            
            # Cycle colors based on quantum states
            color_cycle = int(time_freq * 5)
            colors = list(self.colors.values())
            p['color'] = colors[color_cycle % len(colors)]
    
    async def run(self):
        """Main visualization loop"""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            # Clear screen with fade effect
            self.screen.fill((0, 0, 0))
            
            # Update and draw particles
            self.update_particles(0.016)
            for p in self.particles:
                pos = tuple(map(int, p['pos']))
                pygame.draw.circle(self.screen, p['color'], pos, 2)
            
            # Add phi spiral overlay
            center = (960, 540)
            points = []
            for i in range(1000):
                angle = i * self.phi * 2 * np.pi
                r = np.sqrt(i) * 5
                x = center[0] + r * np.cos(angle)
                y = center[1] + r * np.sin(angle)
                points.append((int(x), int(y)))
            
            if len(points) > 1:
                pygame.draw.lines(self.screen, self.colors['love'], False, points, 1)
            
            pygame.display.flip()
            self.clock.tick(60)
            
            # Allow other async tasks to run
            await asyncio.sleep(0)

if __name__ == "__main__":
    visual = QuantumVisual()
    asyncio.run(visual.run())
