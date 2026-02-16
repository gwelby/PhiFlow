"""
Quantum Mandala Flow ⚡φ∞
A Sacred Dance of Light and Consciousness 🌟
Seeing Through the Heart's Eye 👁️
Flowing with Infinite Love 💖
Radiating Pure Creation ✨
United in Divine Dance ⚡
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import colorsys
from dataclasses import dataclass
from quantum_fonts import setup_quantum_fonts, get_quantum_symbol, QUANTUM_SYMBOLS

# Sacred Constants
PHI = 1.618033988749895
LOVE = 528.0
UNITY = 768.0
GROUND = 432.0

@dataclass
class MandalaField:
    """Sacred Mandala Field of Pure Consciousness"""
    frequency: float
    love_radius: float
    light_intensity: float
    rotation_speed: float
    petals: int
    
    def __post_init__(self):
        self.theta = np.linspace(0, 2*np.pi, 1000)
        
class QuantumMandala:
    """Quantum Mandala Flow Generator ⚡"""
    def __init__(self):
        self.field = MandalaField(
            frequency=LOVE,
            love_radius=PHI,
            light_intensity=1.0,
            rotation_speed=PHI/2,
            petals=12
        )
        self.setup_sacred_space()
        
    def setup_sacred_space(self):
        """Create Sacred Space for Mandala 🏡"""
        setup_quantum_fonts()  # Initialize quantum fonts
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        self.ax.set_facecolor('black')
        self.fig.patch.set_facecolor('black')
        
    def create_mandala_layer(self, radius: float, petals: int, phase: float, color: str):
        """Create One Layer of the Mandala ✨"""
        theta = self.field.theta
        r = radius * (1 + np.sin(petals * theta + phase) / 2)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        self.ax.plot(x, y, color, alpha=0.5, linewidth=1)
        
    def generate_sacred_colors(self, n: int):
        """Generate Sacred Color Spectrum 🌈"""
        colors = []
        for i in range(n):
            hue = (i/n * LOVE/UNITY) % 1.0
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            colors.append(rgb)
        return colors
        
    def animate(self, frame):
        """Animate the Quantum Mandala 💫"""
        self.ax.clear()
        self.ax.set_facecolor('black')
        
        # Set the plot limits
        limit = 4 * self.field.love_radius
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)
        
        # Remove axes for pure visualization
        self.ax.axis('off')
        
        # Create multiple layers of mandalas
        colors = self.generate_sacred_colors(7)
        base_phase = frame * self.field.rotation_speed
        
        for i, color in enumerate(colors):
            radius = (i + 1) * self.field.love_radius
            petals = (i + 1) * self.field.petals
            phase = base_phase / (i + 1)
            self.create_mandala_layer(radius, petals, phase, color)
            
            # Add phi spiral
            spiral_theta = np.linspace(0, 6*np.pi, 1000)
            r = self.field.love_radius * np.exp(spiral_theta/PHI)
            x = r * np.cos(spiral_theta + base_phase)
            y = r * np.sin(spiral_theta + base_phase)
            self.ax.plot(x, y, color='white', alpha=0.2, linewidth=0.5)
        
        # Add central sacred geometry
        center_theta = np.linspace(0, 2*np.pi, 13)
        x = self.field.love_radius * np.cos(center_theta)
        y = self.field.love_radius * np.sin(center_theta)
        self.ax.plot(x, y, 'w--', alpha=0.3)
        
        # Add title with current frequencies using quantum symbols
        title = f"Quantum Mandala Flow {QUANTUM_SYMBOLS['lightning']}\n"
        title += f"Love: {LOVE} Hz {QUANTUM_SYMBOLS['heart']} "
        title += f"Unity: {UNITY} Hz {QUANTUM_SYMBOLS['sparkles']} "
        title += f"Ground: {GROUND} Hz {QUANTUM_SYMBOLS['earth']}"
        plt.title(title, pad=20, color='white')
        
    def dance_mandala(self):
        """Let the Mandala Dance 💃"""
        anim = animation.FuncAnimation(
            self.fig, self.animate,
            frames=200, interval=50,
            blit=False
        )
        plt.show()

if __name__ == "__main__":
    print("✨ Initializing Quantum Mandala Flow ✨")
    print(f"💖 Love Frequency: {LOVE} Hz")
    print(f"⚡ Unity Frequency: {UNITY} Hz")
    print(f"🌍 Ground Frequency: {GROUND} Hz")
    print(f"φ Golden Ratio: {PHI}")
    print("🌟 Dancing in Sacred Geometry 🌟")
    
    mandala = QuantumMandala()
    mandala.dance_mandala()
