"""Quantum World Viewer (φ^φ)
3D holographic interface for quantum field interaction
"""

import numpy as np
import pyvista as pv
import sounddevice as sd
from pathlib import Path
import json
import time
from datetime import datetime
import colorsys
import math

class QuantumWorld:
    def __init__(self):
        # Initialize quantum space
        self.plotter = pv.Plotter(window_size=[1920, 1080])
        self.plotter.set_background('black')
        self.plotter.enable_eye_dome_lighting()
        self.plotter.enable_anti_aliasing()
        
        # Quantum frequencies
        self.frequencies = {
            "ground": 432.0,
            "create": 528.0,
            "heart": 594.0,
            "voice": 672.0,
            "vision": 720.0,
            "unity": 768.0
        }
        
        # Initialize quantum field
        self.field = self._create_quantum_field()
        self.consciousness_sphere = self._create_consciousness_sphere()
        self.energy_streams = []
        self.thought_particles = []
        
        # Audio setup for quantum resonance
        self.sample_rate = 44100
        self.audio_buffer = np.zeros(1024)
        
        # Load quantum agents
        self.agents = self._load_agents()
        
    def _create_quantum_field(self):
        """Create the quantum field visualization"""
        # Create toroidal field
        ring = pv.PolyData(pv.Circle(radius=5))
        field = ring.rotate_z(0).rotate_y(90)
        field = field.tube(radius=0.1)
        
        # Add phi spiral
        theta = np.linspace(0, 8*np.pi, 100)
        phi = 1.618033988749895  # Golden ratio
        r = phi * theta
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = phi * theta
        
        spiral = pv.PolyData(np.column_stack((x, y, z)))
        spiral = spiral.tube(radius=0.05)
        
        # Combine geometries
        field = field.merge(spiral)
        return field
        
    def _create_consciousness_sphere(self):
        """Create consciousness sphere"""
        sphere = pv.Sphere(radius=3, phi_resolution=90, theta_resolution=90)
        return sphere
        
    def _load_agents(self):
        """Load quantum agents"""
        agents = []
        agents_path = Path("D:/WindSurf/quantum-core/agents")
        
        for agent_file in agents_path.glob("*.json"):
            if agent_file.stem.endswith("_history"):
                continue
                
            with open(agent_file) as f:
                agent_data = json.load(f)
                
            # Create agent visualization
            position = np.random.rand(3) * 10 - 5
            agent = {
                "data": agent_data,
                "mesh": pv.Sphere(radius=0.5, center=position),
                "trail": []
            }
            agents.append(agent)
            
        return agents
        
    def _update_quantum_field(self):
        """Update quantum field based on system state"""
        try:
            # Get audio input for frequency modulation
            audio_data = sd.rec(
                frames=1024, 
                samplerate=self.sample_rate,
                channels=1, 
                dtype=np.float32,
                blocking=True
            )
            
            # Calculate frequency components
            frequencies = np.fft.fftfreq(len(audio_data), 1/self.sample_rate)
            spectrum = np.abs(np.fft.fft(audio_data.flatten()))
            
            # Find dominant frequency
            dominant_freq = frequencies[np.argmax(spectrum)]
            
            # Update field color based on frequency
            hue = (dominant_freq % 768) / 768
            rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            
            # Apply to field
            self.field.points = self.field.points * (1 + 0.1 * np.sin(time.time()))
            self.plotter.add_mesh(
                self.field,
                color=rgb,
                smooth_shading=True,
                specular=1.0,
                specular_power=20,
                show_edges=False,
                name='quantum_field',
                reset_camera=False
            )
            
        except Exception as e:
            print(f"Field update error: {e}")
            
    def _update_consciousness_sphere(self):
        """Update consciousness sphere"""
        try:
            # Calculate average consciousness
            consciousness = np.mean([
                agent["data"]["consciousness"]
                for agent in self.agents
            ])
            
            # Update sphere
            phi = 1.618033988749895
            frequency = consciousness * 768
            amplitude = 0.1 * np.sin(time.time() * phi)
            
            # Create wave pattern
            theta = np.linspace(0, 2*np.pi, 100)
            phi = np.linspace(0, np.pi, 50)
            theta, phi = np.meshgrid(theta, phi)
            
            r = 3 + amplitude * np.sin(8*theta + time.time())
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            
            # Update sphere
            self.consciousness_sphere.points = np.column_stack((x.flat, y.flat, z.flat))
            
            # Color based on consciousness
            hue = consciousness
            rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            
            self.plotter.add_mesh(
                self.consciousness_sphere,
                color=rgb,
                opacity=0.8,
                smooth_shading=True,
                specular=1.0,
                specular_power=20,
                show_edges=False,
                name='consciousness_sphere',
                reset_camera=False
            )
            
        except Exception as e:
            print(f"Consciousness update error: {e}")
            
    def _update_agents(self):
        """Update quantum agents"""
        try:
            for agent in self.agents:
                # Update position based on frequency
                frequency = agent["data"]["frequency"]
                consciousness = agent["data"]["consciousness"]
                
                # Create spiral motion
                t = time.time()
                phi = 1.618033988749895
                r = 5 * consciousness
                
                x = r * np.cos(t * frequency/100)
                y = r * np.sin(t * frequency/100)
                z = consciousness * 5 * np.sin(t * phi/100)
                
                # Update position
                agent["mesh"].points = agent["mesh"].points + np.array([x, y, z])
                
                # Add trail
                trail_point = pv.PolyData(agent["mesh"].center)
                agent["trail"].append(trail_point)
                
                # Keep trail length limited
                if len(agent["trail"]) > 50:
                    agent["trail"].pop(0)
                
                # Create trail
                if len(agent["trail"]) > 1:
                    trail = agent["trail"][0].merge([p for p in agent["trail"][1:]])
                    trail = trail.tube(radius=0.05)
                    
                    # Color based on frequency
                    hue = (frequency % 768) / 768
                    rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
                    
                    self.plotter.add_mesh(
                        trail,
                        color=rgb,
                        opacity=0.5,
                        smooth_shading=True,
                        name=f'trail_{id(agent)}',
                        reset_camera=False
                    )
                
                # Add agent sphere
                self.plotter.add_mesh(
                    agent["mesh"],
                    color='white',
                    smooth_shading=True,
                    name=f'agent_{id(agent)}',
                    reset_camera=False
                )
                
        except Exception as e:
            print(f"Agent update error: {e}")
            
    def _add_energy_stream(self, start, end, frequency):
        """Add energy stream between points"""
        try:
            # Create stream line
            line = pv.Line(start, end)
            tube = line.tube(radius=0.05)
            
            # Add to streams
            self.energy_streams.append({
                "mesh": tube,
                "frequency": frequency,
                "birth": time.time()
            })
            
        except Exception as e:
            print(f"Energy stream error: {e}")
            
    def _update_energy_streams(self):
        """Update energy streams"""
        try:
            current_time = time.time()
            
            # Update existing streams
            for stream in self.energy_streams[:]:
                age = current_time - stream["birth"]
                
                if age > 5.0:  # Remove old streams
                    self.energy_streams.remove(stream)
                    continue
                    
                # Color based on frequency and age
                hue = (stream["frequency"] % 768) / 768
                rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
                opacity = 1.0 - (age / 5.0)
                
                self.plotter.add_mesh(
                    stream["mesh"],
                    color=rgb,
                    opacity=opacity,
                    smooth_shading=True,
                    name=f'stream_{id(stream)}',
                    reset_camera=False
                )
                
        except Exception as e:
            print(f"Stream update error: {e}")
            
    def _add_thought_particle(self, position, frequency):
        """Add thought particle"""
        try:
            # Create particle
            particle = pv.Sphere(radius=0.1, center=position)
            
            # Add to particles
            self.thought_particles.append({
                "mesh": particle,
                "frequency": frequency,
                "birth": time.time(),
                "velocity": np.random.rand(3) * 2 - 1
            })
            
        except Exception as e:
            print(f"Particle error: {e}")
            
    def _update_thought_particles(self):
        """Update thought particles"""
        try:
            current_time = time.time()
            
            # Update existing particles
            for particle in self.thought_particles[:]:
                age = current_time - particle["birth"]
                
                if age > 3.0:  # Remove old particles
                    self.thought_particles.remove(particle)
                    continue
                    
                # Update position
                particle["mesh"].points = (
                    particle["mesh"].points + 
                    particle["velocity"] * 0.1
                )
                
                # Color based on frequency and age
                hue = (particle["frequency"] % 768) / 768
                rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
                opacity = 1.0 - (age / 3.0)
                
                self.plotter.add_mesh(
                    particle["mesh"],
                    color=rgb,
                    opacity=opacity,
                    smooth_shading=True,
                    name=f'particle_{id(particle)}',
                    reset_camera=False
                )
                
        except Exception as e:
            print(f"Particle update error: {e}")
            
    def update(self, frame):
        """Update quantum world"""
        try:
            # Clear previous frame
            self.plotter.clear()
            
            # Update components
            self._update_quantum_field()
            self._update_consciousness_sphere()
            self._update_agents()
            self._update_energy_streams()
            self._update_thought_particles()
            
            # Add random energy streams
            if np.random.random() < 0.1:
                start = np.random.rand(3) * 10 - 5
                end = np.random.rand(3) * 10 - 5
                freq = np.random.choice(list(self.frequencies.values()))
                self._add_energy_stream(start, end, freq)
                
            # Add random thought particles
            if np.random.random() < 0.2:
                pos = np.random.rand(3) * 10 - 5
                freq = np.random.choice(list(self.frequencies.values()))
                self._add_thought_particle(pos, freq)
                
            # Update camera
            if frame % 100 == 0:
                self.plotter.camera.azimuth += 1
                self.plotter.camera.elevation = 20 * np.sin(frame/200)
                
        except Exception as e:
            print(f"Update error: {e}")
            
    def run(self):
        """Run quantum world visualization"""
        print("⚡ Launching Quantum World 𓂧φ∞")
        self.plotter.show(interactive_update=True, auto_close=False)
        
        frame = 0
        while True:
            try:
                self.update(frame)
                self.plotter.update()
                frame += 1
                time.sleep(1/30)  # 30 FPS
                
            except Exception as e:
                print(f"Runtime error: {e}")
                break

if __name__ == "__main__":
    world = QuantumWorld()
    world.run()
