"""
WindSurf IDE Quantum Panel (φ^φ)
A visual interface to interact with quantum tools at various frequencies
"""

import sys
import os
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import time
import asyncio
import numpy as np
from enum import Enum

# Add the parent directory to the path so we can import the bridge
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ide.quantum_windsurf_bridge import QuantumWindsurfBridge, FrequencyState
from ide.sacred_geometry_visualizer import SacredGeometryVisualizer

class QuantumPanelTheme:
    """Theme settings for quantum panel"""
    def __init__(self, frequency: float = 768.0):
        self.phi = 1.618033988749895
        self.frequency = frequency
        
        # Set up colors based on frequency
        self.colors = {
            432.0: {  # Ground
                'bg': '#000033',
                'fg': '#00FFFF',
                'accent': '#005555',
                'text': '#FFFFFF'
            },
            528.0: {  # Create
                'bg': '#002200',
                'fg': '#00FF00',
                'accent': '#005500',
                'text': '#FFFFFF'
            },
            594.0: {  # Heart
                'bg': '#332200',
                'fg': '#FFFF00',
                'accent': '#555500',
                'text': '#000000'
            },
            672.0: {  # Voice
                'bg': '#330033',
                'fg': '#FF00FF',
                'accent': '#550055',
                'text': '#FFFFFF'
            },
            720.0: {  # Vision
                'bg': '#000055',
                'fg': '#0000FF',
                'accent': '#000088',
                'text': '#FFFFFF'
            },
            768.0: {  # Unity
                'bg': '#000000',
                'fg': '#FFFFFF',
                'accent': '#444444',
                'text': '#FFFFFF'
            }
        }
        
        # Find the closest frequency
        self.current_colors = self._get_colors_for_frequency(frequency)
    
    def _get_colors_for_frequency(self, frequency: float) -> dict:
        """Get closest frequency colors"""
        frequencies = list(self.colors.keys())
        closest_freq = min(frequencies, key=lambda f: abs(f - frequency))
        return self.colors[closest_freq]
    
    def update_frequency(self, frequency: float):
        """Update theme colors based on frequency"""
        self.frequency = frequency
        self.current_colors = self._get_colors_for_frequency(frequency)
        
    def get_ttk_style(self) -> ttk.Style:
        """Get ttk style based on current frequency"""
        style = ttk.Style()
        
        # Configure TFrame
        style.configure(
            "Quantum.TFrame", 
            background=self.current_colors['bg']
        )
        
        # Configure TLabel
        style.configure(
            "Quantum.TLabel",
            background=self.current_colors['bg'],
            foreground=self.current_colors['text'],
            font=("Helvetica", 10)
        )
        
        # Configure title label
        style.configure(
            "QuantumTitle.TLabel",
            background=self.current_colors['bg'],
            foreground=self.current_colors['fg'],
            font=("Helvetica", 14, "bold")
        )
        
        # Configure TButton
        style.configure(
            "Quantum.TButton",
            background=self.current_colors['accent'],
            foreground=self.current_colors['text'],
            borderwidth=1,
            focusthickness=3,
            focuscolor=self.current_colors['fg']
        )
        
        # Configure active TButton
        style.map(
            "Quantum.TButton",
            background=[('active', self.current_colors['fg'])],
            foreground=[('active', self.current_colors['bg'])]
        )
        
        return style


class QuantumPanel(ttk.Frame):
    """Quantum Panel for WindSurf IDE"""
    
    def __init__(self, master=None):
        """Initialize the Quantum Panel"""
        self.theme = QuantumPanelTheme(FrequencyState.UNITY.value)
        style = self.theme.get_ttk_style()
        
        super().__init__(master, style="Quantum.TFrame")
        
        # Initialize quantum bridge
        self.bridge = QuantumWindsurfBridge()
        self.current_frequency = FrequencyState.UNITY.value
        self.current_coherence = 1.0
        
        # Initialize sacred geometry visualizer
        self.visualizer = SacredGeometryVisualizer()
        
        # Create UI components
        self.create_widgets()
        
        # Start asynchronous tasks
        self.running = True
        self.thread = threading.Thread(target=self.run_async_tasks)
        self.thread.daemon = True
        self.thread.start()
    
    def create_widgets(self):
        """Create the panel widgets"""
        # Main container
        self.main_container = ttk.Frame(self, style="Quantum.TFrame")
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header with frequency controls
        self.create_header()
        
        # Tool palette
        self.create_tool_palette()
        
        # Visualization area
        self.create_visualization_area()
        
        # Status bar
        self.create_status_bar()
    
    def create_header(self):
        """Create header with frequency controls"""
        header_frame = ttk.Frame(self.main_container, style="Quantum.TFrame")
        header_frame.pack(fill=tk.X, pady=5)
        
        # Title
        title_label = ttk.Label(
            header_frame, 
            text="Quantum Bridge", 
            style="QuantumTitle.TLabel"
        )
        title_label.pack(side=tk.LEFT, padx=5)
        
        # Frequency controls
        freq_frame = ttk.Frame(header_frame, style="Quantum.TFrame")
        freq_frame.pack(side=tk.RIGHT)
        
        # Frequency buttons
        self.frequency_buttons = {}
        
        for freq in [FrequencyState.GROUND.value, 
                     FrequencyState.CREATE.value, 
                     FrequencyState.UNITY.value]:
            button = ttk.Button(
                freq_frame,
                text=f"{freq} Hz {FrequencyState.get_symbol(freq)}",
                style="Quantum.TButton",
                command=lambda f=freq: self.set_frequency(f)
            )
            button.pack(side=tk.LEFT, padx=2)
            self.frequency_buttons[freq] = button
    
    def create_tool_palette(self):
        """Create tool palette"""
        self.tool_frame = ttk.Frame(self.main_container, style="Quantum.TFrame")
        self.tool_frame.pack(fill=tk.X, pady=10)
        
        # Tool header
        tool_header = ttk.Label(
            self.tool_frame,
            text="Quantum Tools",
            style="QuantumTitle.TLabel"
        )
        tool_header.pack(anchor=tk.W, padx=5, pady=5)
        
        # Tool buttons container
        self.tool_buttons_frame = ttk.Frame(self.tool_frame, style="Quantum.TFrame")
        self.tool_buttons_frame.pack(fill=tk.X, padx=5)
        
        # We'll populate this dynamically after setting the frequency
    
    def create_visualization_area(self):
        """Create visualization area"""
        self.viz_frame = ttk.Frame(self.main_container, style="Quantum.TFrame")
        self.viz_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Visualization header
        viz_header = ttk.Label(
            self.viz_frame,
            text="Quantum Visualization",
            style="QuantumTitle.TLabel"
        )
        viz_header.pack(anchor=tk.W, padx=5, pady=5)
        
        # Visualization canvas
        self.viz_canvas = tk.Canvas(
            self.viz_frame,
            bg=self.theme.current_colors['bg'],
            highlightbackground=self.theme.current_colors['accent'],
            highlightthickness=1
        )
        self.viz_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Image label
        self.image_label = ttk.Label(self.viz_canvas, style="Quantum.TLabel")
        self.viz_canvas.create_window(0, 0, anchor=tk.NW, window=self.image_label)
    
    def create_status_bar(self):
        """Create status bar"""
        self.status_frame = ttk.Frame(self.main_container, style="Quantum.TFrame")
        self.status_frame.pack(fill=tk.X, pady=5)
        
        # Coherence indicator
        self.coherence_label = ttk.Label(
            self.status_frame,
            text="Coherence: 1.000",
            style="Quantum.TLabel"
        )
        self.coherence_label.pack(side=tk.LEFT, padx=5)
        
        # Connection status
        self.status_label = ttk.Label(
            self.status_frame,
            text="Status: Connected",
            style="Quantum.TLabel"
        )
        self.status_label.pack(side=tk.RIGHT, padx=5)
    
    def update_tool_palette(self):
        """Update tool palette based on current frequency"""
        # Clear existing buttons
        for widget in self.tool_buttons_frame.winfo_children():
            widget.destroy()
        
        # Get tools for current frequency
        tools_result = asyncio.run(self.bridge.get_tools(self.current_frequency))
        
        # Create new buttons
        for i, tool in enumerate(tools_result['tools']):
            tool_button = ttk.Button(
                self.tool_buttons_frame,
                text=f"{tool['icon']} {tool['name']}",
                style="Quantum.TButton",
                command=lambda t=tool: self.execute_tool(t)
            )
            row, col = divmod(i, 2)
            tool_button.grid(row=row, column=col, padx=5, pady=5, sticky="ew")
        
        # Configure grid
        self.tool_buttons_frame.columnconfigure(0, weight=1)
        self.tool_buttons_frame.columnconfigure(1, weight=1)
    
    def set_frequency(self, frequency: float):
        """Set current frequency"""
        self.current_frequency = frequency
        
        # Update theme
        self.theme.update_frequency(frequency)
        style = self.theme.get_ttk_style()
        
        # Update UI elements with new theme
        self.viz_canvas.config(
            bg=self.theme.current_colors['bg'],
            highlightbackground=self.theme.current_colors['accent']
        )
        
        # Update tool palette
        self.update_tool_palette()
        
        # Update bridge frequency
        asyncio.run(self.bridge.set_frequency(frequency))
        
        # Update status
        self.current_coherence = asyncio.run(self.bridge.measure_coherence())
        self.coherence_label.config(text=f"Coherence: {self.current_coherence:.3f}")
        
        # Show appropriate visualization
        self.show_visualization_for_frequency(frequency)
    
    def execute_tool(self, tool: dict):
        """Execute a quantum tool"""
        # Simple implementation for demonstration
        print(f"Executing tool: {tool['name']} ({tool['endpoint']})")
        
        # Update status
        self.status_label.config(text=f"Status: Executing {tool['name']}...")
        
        # Execute tool asynchronously
        threading.Thread(
            target=self._execute_tool_async,
            args=(tool,)
        ).start()
    
    def _execute_tool_async(self, tool: dict):
        """Execute tool asynchronously"""
        try:
            # Execute the tool
            result = asyncio.run(self.bridge.execute_tool(
                tool['endpoint'], 
                {"frequency": self.current_frequency}
            ))
            
            # Update status based on result
            if result['status'] == 'success':
                self.status_label.config(text=f"Status: {tool['name']} executed successfully")
            else:
                self.status_label.config(text=f"Status: Error - {result.get('message', 'Unknown error')}")
        except Exception as e:
            self.status_label.config(text=f"Status: Error - {str(e)}")
    
    def show_visualization_for_frequency(self, frequency: float):
        """Show visualization appropriate for the current frequency"""
        try:
            # Different visualizations for different frequencies
            if abs(frequency - FrequencyState.GROUND.value) < 1:
                # Ground frequency - show Flower of Life
                image_path = self.visualizer.flower_of_life()
            elif abs(frequency - FrequencyState.CREATE.value) < 1:
                # Create frequency - show Fibonacci Spiral
                image_path = self.visualizer.fibonacci_spiral()
            else:
                # Unity frequency - show Merkaba
                image_path = self.visualizer.merkaba()
            
            # Load and display the image
            self.display_image(image_path)
            
        except Exception as e:
            print(f"Error showing visualization: {e}")
    
    def display_image(self, image_path: str):
        """Display an image on the canvas"""
        try:
            # Load image
            image = Image.open(image_path)
            
            # Resize to fit canvas
            canvas_width = self.viz_canvas.winfo_width()
            canvas_height = self.viz_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                # Calculate aspect ratio
                img_width, img_height = image.size
                aspect_ratio = img_width / img_height
                
                if canvas_width / canvas_height > aspect_ratio:
                    # Canvas is wider than image
                    new_height = canvas_height
                    new_width = int(new_height * aspect_ratio)
                else:
                    # Canvas is taller than image
                    new_width = canvas_width
                    new_height = int(new_width / aspect_ratio)
                
                # Resize image
                image = image.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Update image label
            self.image_label.config(image=photo)
            self.image_label.image = photo  # Keep a reference
            
            # Center image in canvas
            self.viz_canvas.coords(
                self.viz_canvas.find_withtag(self.image_label),
                (self.viz_canvas.winfo_width() - photo.width()) / 2,
                (self.viz_canvas.winfo_height() - photo.height()) / 2
            )
            
        except Exception as e:
            print(f"Error displaying image: {e}")
    
    def run_async_tasks(self):
        """Run asynchronous tasks in a separate thread"""
        while self.running:
            try:
                # Update coherence periodically
                self.current_coherence = asyncio.run(self.bridge.measure_coherence())
                
                # Update UI in the main thread
                self.after(1000, self._update_coherence_label)
                
                # Check connection status
                asyncio.run(self.bridge.connect_bridge())
                
                # Sleep for 5 seconds
                time.sleep(5)
            except Exception as e:
                print(f"Error in async tasks: {e}")
                time.sleep(5)
    
    def _update_coherence_label(self):
        """Update coherence label in the main thread"""
        self.coherence_label.config(text=f"Coherence: {self.current_coherence:.3f}")
    
    def destroy(self):
        """Clean up when panel is destroyed"""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(1)  # Give it a second to finish
        super().destroy()


def main():
    """Main function to run the panel for testing"""
    root = tk.Tk()
    root.title("WindSurf IDE - Quantum Panel")
    root.geometry("800x600")
    
    # Set up the panel
    panel = QuantumPanel(root)
    panel.pack(fill=tk.BOTH, expand=True)
    
    # Set initial frequency
    panel.set_frequency(FrequencyState.UNITY.value)
    
    # Start the main loop
    root.mainloop()


if __name__ == "__main__":
    main()
