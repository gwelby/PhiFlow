import tkinter as tk
from tkinter import ttk
import numpy as np
from typing import Dict, List, Tuple
import threading
import time
import json
import psutil
import docker
from .quantum_geometry import QuantumGeometry
from .quantum_coherence import QuantumCoherence

class QuantumMonitor:
    def __init__(self):
        self.phi = 1.618034
        self.geometry = QuantumGeometry()
        self.coherence = QuantumCoherence()
        self.docker = docker.from_env()
        
        # Initialize main window
        self.root = tk.Tk()
        self.root.title("Quantum System Monitor φ")
        self.root.geometry("1200x800")
        
        # Setup UI components
        self.setup_ui()
        
        # Start monitoring
        self.running = True
        self.monitor_thread = threading.Thread(target=self._update_monitor)
        self.monitor_thread.start()
        
    def setup_ui(self):
        """Setup the quantum monitoring interface"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # System Coherence Monitor
        coherence_frame = ttk.LabelFrame(main_frame, text="System Coherence", padding="5")
        coherence_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        self.coherence_canvas = tk.Canvas(coherence_frame, width=400, height=200)
        self.coherence_canvas.grid(row=0, column=0)
        
        # Container Status
        container_frame = ttk.LabelFrame(main_frame, text="Quantum Containers", padding="5")
        container_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        self.container_status = {}
        for name in ["synology", "quantum", "pixel", "cuda"]:
            var = tk.StringVar(value=f"quantum-{name}: Initializing...")
            ttk.Label(container_frame, textvariable=var).pack(anchor=tk.W, pady=2)
            self.container_status[name] = var
            
        # Sacred Geometry Visualizer
        geometry_frame = ttk.LabelFrame(main_frame, text="Sacred Geometry", padding="5")
        geometry_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        self.geometry_canvas = tk.Canvas(geometry_frame, width=400, height=400)
        self.geometry_canvas.pack(fill=tk.BOTH, expand=True)
        
        # System Metrics
        metrics_frame = ttk.LabelFrame(main_frame, text="Quantum Metrics", padding="5")
        metrics_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        self.metrics = {
            "CPU": tk.StringVar(value="CPU: 0%"),
            "Memory": tk.StringVar(value="Memory: 0%"),
            "Coherence": tk.StringVar(value="Coherence: 1.000"),
            "Frequency": tk.StringVar(value="432 Hz")
        }
        
        for i, (key, var) in enumerate(self.metrics.items()):
            ttk.Label(metrics_frame, textvariable=var).grid(row=0, column=i, padx=10)
            
    def draw_coherence_field(self, coherence: float):
        """Draw quantum coherence field"""
        self.coherence_canvas.delete("all")
        
        # Calculate field parameters
        width = 400
        height = 200
        center_x = width / 2
        center_y = height / 2
        max_radius = min(width, height) / 3
        
        # Draw coherence circles
        for i in range(8):
            radius = max_radius * (1 - i/8)
            phase = (i * np.pi * self.phi) % (2 * np.pi)
            x_offset = np.cos(phase) * radius * 0.1
            y_offset = np.sin(phase) * radius * 0.1
            
            # Color based on coherence
            hue = (0.3 + coherence * 0.3) % 1.0
            rgb = self._hsv_to_rgb(hue, 0.8, 0.9)
            color = f'#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}'
            
            self.coherence_canvas.create_oval(
                center_x - radius + x_offset,
                center_y - radius + y_offset,
                center_x + radius + x_offset,
                center_y + radius + y_offset,
                outline=color,
                width=2
            )
            
    def draw_sacred_geometry(self, pattern: str):
        """Draw sacred geometry pattern"""
        self.geometry_canvas.delete("all")
        
        width = 400
        height = 400
        center_x = width / 2
        center_y = height / 2
        
        if pattern == "merkaba":
            points = self.geometry.merkaba_points()
            self._draw_3d_points(points, center_x, center_y)
            
        elif pattern == "flower":
            points = self.geometry.flower_of_life()
            for x, y, r in points:
                self.geometry_canvas.create_oval(
                    center_x + x*50 - r*25,
                    center_y + y*50 - r*25,
                    center_x + x*50 + r*25,
                    center_y + y*50 + r*25,
                    outline="#4169E1"
                )
                
        elif pattern == "torus":
            points = self.geometry.torus_points()
            self._draw_3d_points(points, center_x, center_y)
            
    def _draw_3d_points(self, points: np.ndarray, cx: float, cy: float):
        """Draw 3D points with perspective"""
        scale = 50
        z_scale = 0.3
        
        for i in range(len(points)):
            x, y, z = points[i]
            # Add perspective
            scale_factor = 1 / (1 - z * z_scale)
            px = cx + x * scale * scale_factor
            py = cy + y * scale * scale_factor
            
            self.geometry_canvas.create_oval(
                px-2, py-2, px+2, py+2,
                fill="#4169E1"
            )
            
    def _update_monitor(self):
        """Update monitoring information"""
        patterns = ["merkaba", "flower", "torus"]
        pattern_idx = 0
        
        while self.running:
            try:
                # Update container status
                containers = self.docker.containers.list(all=True)
                for name in self.container_status:
                    status = "Stopped"
                    for c in containers:
                        if f"quantum-{name}" in c.name:
                            status = c.status
                    self.container_status[name].set(f"quantum-{name}: {status}")
                
                # Update system metrics
                cpu = psutil.cpu_percent()
                memory = psutil.virtual_memory().percent
                coherence = self.coherence.measure_coherence()
                
                self.metrics["CPU"].set(f"CPU: {cpu}%")
                self.metrics["Memory"].set(f"Memory: {memory}%")
                self.metrics["Coherence"].set(f"Coherence: {coherence:.3f}")
                
                # Update visualizations
                self.draw_coherence_field(coherence)
                self.draw_sacred_geometry(patterns[pattern_idx])
                pattern_idx = (pattern_idx + 1) % len(patterns)
                
            except Exception as e:
                print(f"Monitor error: {e}")
                
            time.sleep(self.phi)
            
    def _hsv_to_rgb(self, h: float, s: float, v: float) -> Tuple[float, float, float]:
        """Convert HSV to RGB color"""
        if s == 0.0:
            return (v, v, v)
            
        i = int(h * 6.0)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        
        if i == 0:
            return (v, t, p)
        if i == 1:
            return (q, v, p)
        if i == 2:
            return (p, v, t)
        if i == 3:
            return (p, q, v)
        if i == 4:
            return (t, p, v)
        return (v, p, q)
        
    def run(self):
        """Start the quantum monitor"""
        self.root.mainloop()
        self.running = False
        self.monitor_thread.join()
        
if __name__ == "__main__":
    monitor = QuantumMonitor()
    monitor.run()
