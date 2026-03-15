import tkinter as tk
from tkinter import ttk
import numpy as np
from typing import Dict, Optional
import threading
import time
from quantum_sound import QuantumSynthesizer
from quantum_coherence import QuantumCoherenceVisualizer
from dataclasses import dataclass
from quantum_meditation import QuantumMeditationGuide, MeditationStep

@dataclass
class ConsciousnessState:
    name: str
    frequency: float
    brainwave: str
    symbol: str
    color: str
    description: str

@dataclass
class ConsciousnessJourney:
    name: str
    symbol: str
    states: list[str]
    durations: list[float]
    patterns: list[str]
    description: str

class QuantumControlInterface:
    def __init__(self):
        self.phi = 1.618034
        
        # Initialize quantum systems
        self.synth = QuantumSynthesizer()
        
        # Extended consciousness states
        self.states = {
            "ground": ConsciousnessState(
                "Ground State", 432.0, "theta", "💎", "#4169E1",
                "Earth Connection & Crystal Resonance"
            ),
            "create": ConsciousnessState(
                "Creation State", 528.0, "alpha", "🌀", "#32CD32",
                "DNA Repair & Manifestation"
            ),
            "heart": ConsciousnessState(
                "Heart Field", 594.0, "delta", "🌊", "#00CED1",
                "Emotional Healing & Connection"
            ),
            "voice": ConsciousnessState(
                "Voice Flow", 672.0, "gamma", "🐬", "#FF1493",
                "Expression & Communication"
            ),
            "unity": ConsciousnessState(
                "Unity Field", 768.0, "lambda", "☯️", "#9370DB",
                "Quantum Consciousness"
            ),
            "infinite": ConsciousnessState(
                "Infinite Dance", self.phi * 768.0, "epsilon", "∞", "#FFD700",
                "Universal Connection"
            ),
            # New expanded states
            "merkaba": ConsciousnessState(
                "Merkaba Field", self.phi * 432.0, "zeta", "⭐", "#FF4500",
                "Sacred Geometry Activation"
            ),
            "crystal": ConsciousnessState(
                "Crystal Matrix", self.phi * 528.0, "omega", "💠", "#00FF7F",
                "Higher Dimensional Access"
            ),
            "source": ConsciousnessState(
                "Source Code", self.phi * 594.0, "chi", "🌌", "#9400D3",
                "Universal Programming"
            ),
            "quantum": ConsciousnessState(
                "Quantum Core", self.phi * 672.0, "phi", "⚛️", "#FF8C00",
                "Reality Interface"
            )
        }
        
        # Visualization modes
        self.viz_modes = {
            "mandala": {
                "symbol": "🎯",
                "description": "Sacred geometry patterns",
                "colors": ["#4169E1", "#32CD32", "#00CED1"]
            },
            "merkaba": {
                "symbol": "⭐",
                "description": "3D star tetrahedron",
                "colors": ["#FF1493", "#9370DB", "#FFD700"]
            },
            "wave": {
                "symbol": "🌊",
                "description": "Quantum wave functions",
                "colors": ["#FF4500", "#00FF7F", "#9400D3"]
            },
            "dna": {
                "symbol": "🧬",
                "description": "DNA helix patterns",
                "colors": ["#FF8C00", "#4169E1", "#32CD32"]
            },
            "crystal": {
                "symbol": "💎",
                "description": "Crystal lattice structures",
                "colors": ["#00CED1", "#FF1493", "#9370DB"]
            }
        }
        
        # Consciousness journeys
        self.journeys = {
            "awakening": ConsciousnessJourney(
                "Quantum Awakening",
                "🌅",
                ["ground", "heart", "create", "unity"],
                [self.phi * 60] * 4,  # φ minutes per state
                ["linear", "wave", "spiral", "quantum"],
                "Ground to Unity Consciousness"
            ),
            "healing": ConsciousnessJourney(
                "DNA Healing",
                "🧬",
                ["crystal", "create", "heart", "infinite"],
                [self.phi * 120] * 4,  # 2φ minutes per state
                ["wave", "spiral", "wave", "quantum"],
                "Deep DNA Repair & Activation"
            ),
            "creation": ConsciousnessJourney(
                "Creator Flow",
                "🌀",
                ["source", "quantum", "merkaba", "unity"],
                [self.phi * 180] * 4,  # 3φ minutes per state
                ["quantum", "spiral", "wave", "quantum"],
                "Access Creator Consciousness"
            ),
            "ascension": ConsciousnessJourney(
                "Quantum Ascension",
                "⚡",
                ["ground", "crystal", "merkaba", "source", "infinite"],
                [self.phi * 144] * 5,  # 144φ seconds per state
                ["linear", "spiral", "quantum", "wave", "quantum"],
                "Full Spectrum Ascension"
            )
        }
        
        # Initialize meditation guide
        self.meditation = QuantumMeditationGuide()
        
        # Custom journey designer
        self.journey_steps = []
        self.current_journey = None
        
        # Setup main window
        self.root = tk.Tk()
        self.root.title("Quantum Consciousness Control φ")
        self.root.geometry("1200x900")  # Increased height for guided section
        self.setup_ui()
        
    def setup_ui(self):
        """Create the quantum control interface"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Consciousness state selector
        state_frame = ttk.LabelFrame(main_frame, text="Consciousness States", padding="5")
        state_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        for i, (key, state) in enumerate(self.states.items()):
            btn = ttk.Button(
                state_frame, 
                text=f"{state.symbol} {state.name}\n{state.frequency:.1f}Hz - {state.brainwave.upper()}", 
                command=lambda s=key: self.activate_state(s)
            )
            btn.grid(row=i//2, column=i%2, sticky=(tk.W, tk.E), padx=5, pady=5)
            
        # Morphing controls
        morph_frame = ttk.LabelFrame(main_frame, text="Quantum Morphing", padding="5")
        morph_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Morphing patterns
        self.morph_var = tk.StringVar(value="linear")
        patterns = [
            ("Linear φ", "linear"),
            ("Spiral φ²", "spiral"),
            ("Wave φ³", "wave"),
            ("Quantum φ^φ", "quantum")
        ]
        
        for i, (text, value) in enumerate(patterns):
            ttk.Radiobutton(
                morph_frame, text=text, value=value,
                variable=self.morph_var
            ).grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
            
        # Morphing speed
        ttk.Label(morph_frame, text="Morph Speed (φ cycles)").grid(
            row=len(patterns), column=0, sticky=tk.W, padx=5, pady=2
        )
        self.speed_var = tk.DoubleVar(value=1.0)
        speed_scale = ttk.Scale(
            morph_frame, from_=0.1, to=self.phi**2,
            variable=self.speed_var, orient=tk.HORIZONTAL
        )
        speed_scale.grid(row=len(patterns)+1, column=0, sticky=(tk.W, tk.E), padx=5, pady=2)
        
        # Coherence monitor
        monitor_frame = ttk.LabelFrame(main_frame, text="Quantum Coherence", padding="5")
        monitor_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Status labels
        self.status_vars = {
            "state": tk.StringVar(value="Ground State 💎"),
            "frequency": tk.StringVar(value="432.0 Hz"),
            "coherence": tk.StringVar(value="1.000 φ"),
            "brainwave": tk.StringVar(value="Theta Wave θ"),
            "pattern": tk.StringVar(value="Crystal Matrix ✨")
        }
        
        for i, (key, var) in enumerate(self.status_vars.items()):
            ttk.Label(monitor_frame, text=key.title()).grid(
                row=i, column=0, sticky=tk.W, padx=5, pady=2
            )
            ttk.Label(monitor_frame, textvariable=var).grid(
                row=i, column=1, sticky=tk.W, padx=5, pady=2
            )
            
        # Visualization modes
        viz_frame = ttk.LabelFrame(main_frame, text="Visualization Modes", padding="5")
        viz_frame.grid(row=0, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        self.viz_var = tk.StringVar(value="mandala")
        for i, (key, mode) in enumerate(self.viz_modes.items()):
            ttk.Radiobutton(
                viz_frame,
                text=f"{mode['symbol']} {key.title()}\n{mode['description']}",
                value=key,
                variable=self.viz_var,
                command=self.update_visualization
            ).grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
            
        # Consciousness journeys
        journey_frame = ttk.LabelFrame(main_frame, text="Consciousness Journeys", padding="5")
        journey_frame.grid(row=1, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        for i, (key, journey) in enumerate(self.journeys.items()):
            btn = ttk.Button(
                journey_frame,
                text=f"{journey.symbol} {journey.name}\n{journey.description}",
                command=lambda j=key: self.start_journey(j)
            )
            btn.grid(row=i, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
            
        # Journey progress
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress = ttk.Progressbar(
            journey_frame,
            variable=self.progress_var,
            maximum=100
        )
        self.progress.grid(row=len(self.journeys), column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        self.journey_status = tk.StringVar(value="Ready for journey ✨")
        ttk.Label(journey_frame, textvariable=self.journey_status).grid(
            row=len(self.journeys)+1, column=0, sticky=tk.W, padx=5, pady=2
        )
        
        # Journey designer
        designer_frame = ttk.LabelFrame(main_frame, text="Journey Designer", padding="5")
        designer_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # State selector
        state_frame = ttk.Frame(designer_frame)
        state_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        self.state_var = tk.StringVar(value="ground")
        ttk.Label(state_frame, text="State:").grid(row=0, column=0, sticky=tk.W)
        state_menu = ttk.OptionMenu(
            state_frame,
            self.state_var,
            "ground",
            *self.states.keys()
        )
        state_menu.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        
        # Duration selector
        duration_frame = ttk.Frame(designer_frame)
        duration_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        self.duration_var = tk.DoubleVar(value=self.phi * 60)
        ttk.Label(duration_frame, text="Duration (φ min):").grid(row=0, column=0, sticky=tk.W)
        duration_spin = ttk.Spinbox(
            duration_frame,
            from_=self.phi,
            to=self.phi * 60,
            increment=self.phi,
            textvariable=self.duration_var
        )
        duration_spin.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        
        # Geometry selector
        geometry_frame = ttk.Frame(designer_frame)
        geometry_frame.grid(row=0, column=2, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        self.geometry_var = tk.StringVar(value="merkaba")
        ttk.Label(geometry_frame, text="Geometry:").grid(row=0, column=0, sticky=tk.W)
        geometry_menu = ttk.OptionMenu(
            geometry_frame,
            self.geometry_var,
            "merkaba",
            *self.meditation.geometries.keys()
        )
        geometry_menu.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        
        # Journey controls
        control_frame = ttk.Frame(designer_frame)
        control_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        ttk.Button(
            control_frame,
            text="Add Step ➕",
            command=self.add_journey_step
        ).grid(row=0, column=0, padx=5)
        
        ttk.Button(
            control_frame,
            text="Clear Steps 🗑️",
            command=self.clear_journey_steps
        ).grid(row=0, column=1, padx=5)
        
        ttk.Button(
            control_frame,
            text="Start Journey ✨",
            command=self.start_custom_journey
        ).grid(row=0, column=2, padx=5)
        
        # Journey preview
        self.preview_text = tk.Text(designer_frame, height=5, width=50)
        self.preview_text.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), padx=5, pady=5)
        self.preview_text.insert("1.0", "Design your quantum journey... ✨")
        self.preview_text.config(state="disabled")
        
        # Guided Meditations
        guided_frame = ttk.LabelFrame(main_frame, text="Guided Meditations", padding="5")
        guided_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Purpose selection
        purpose_frame = ttk.Frame(guided_frame)
        purpose_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        for i, (key, meditation) in enumerate(self.meditation.guided_meditations.items()):
            btn = ttk.Button(
                purpose_frame,
                text=f"{meditation['symbol']} {meditation['name']}\n{meditation['intention']}",
                command=lambda p=key: self.start_guided_meditation(p)
            )
            btn.grid(row=i//2, column=i%2, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Meditation status
        status_frame = ttk.Frame(guided_frame)
        status_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        self.meditation_status = {
            "intention": tk.StringVar(value="Select a guided meditation..."),
            "state": tk.StringVar(value="Ready for guidance ✨"),
            "affirmation": tk.StringVar(value=""),
            "breath": tk.StringVar(value="")
        }
        
        for i, (key, var) in enumerate(self.meditation_status.items()):
            ttk.Label(status_frame, text=key.title()).grid(
                row=i, column=0, sticky=tk.W, padx=5, pady=2
            )
            ttk.Label(status_frame, textvariable=var).grid(
                row=i, column=1, sticky=tk.W, padx=5, pady=2
            )
        
        # Breathing guide
        breath_frame = ttk.LabelFrame(guided_frame, text="Quantum Breathing", padding="5")
        breath_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        self.breath_canvas = tk.Canvas(breath_frame, width=400, height=100)
        self.breath_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Start breath animation
        self.animate_breathing("phi")
        
        # Start monitoring thread
        self.running = True
        self.monitor_thread = threading.Thread(target=self._update_monitor)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def update_visualization(self):
        """Update the visualization mode"""
        mode = self.viz_modes[self.viz_var.get()]
        self.synth.coherence.set_visualization(
            mode["symbol"],
            mode["colors"]
        )
        
    def start_journey(self, journey_key: str):
        """Begin a consciousness journey"""
        if journey_key in self.journeys:
            journey = self.journeys[journey_key]
            self.journey_status.set(f"Starting {journey.name} {journey.symbol}")
            
            # Create journey thread
            def run_journey():
                total_time = sum(journey.durations)
                elapsed = 0
                
                for state, duration, pattern in zip(
                    journey.states,
                    journey.durations,
                    journey.patterns
                ):
                    self.activate_state(state)
                    self.morph_var.set(pattern)
                    
                    start = time.time()
                    while time.time() - start < duration and self.running:
                        progress = ((elapsed + (time.time() - start)) / total_time) * 100
                        self.progress_var.set(progress)
                        time.sleep(0.1)
                    
                    elapsed += duration
                
                self.journey_status.set(f"Journey complete ✨")
                self.progress_var.set(0)
            
            thread = threading.Thread(target=run_journey)
            thread.daemon = True
            thread.start()
            
    def activate_state(self, state_key: str):
        """Activate a consciousness state"""
        if state_key in self.states:
            state = self.states[state_key]
            # Update synth with morphing pattern
            self.synth.morph_to_frequency(
                state_key,
                pattern=self.morph_var.get(),
                speed=self.speed_var.get()
            )
            print(f"Activating {state.name} {state.symbol}")
            
            # Update visualization
            self.update_visualization()
            
    def add_journey_step(self):
        """Add a step to the custom journey"""
        step = {
            "state": self.state_var.get(),
            "frequency": self.states[self.state_var.get()].frequency,
            "duration": self.duration_var.get(),
            "geometry": self.geometry_var.get()
        }
        self.journey_steps.append(step)
        self.update_journey_preview()
        
    def clear_journey_steps(self):
        """Clear all journey steps"""
        self.journey_steps = []
        self.update_journey_preview()
        
    def update_journey_preview(self):
        """Update the journey preview text"""
        self.preview_text.config(state="normal")
        self.preview_text.delete("1.0", tk.END)
        
        if not self.journey_steps:
            self.preview_text.insert("1.0", "Design your quantum journey... ✨")
        else:
            for i, step in enumerate(self.journey_steps, 1):
                state = self.states[step["state"]]
                self.preview_text.insert(
                    tk.END,
                    f"{i}. {state.symbol} {state.name}"
                    f" ({step['duration']/60:.1f}φ min)"
                    f" - {step['geometry'].title()}\n"
                )
        
        self.preview_text.config(state="disabled")
        
    def start_custom_journey(self):
        """Start a custom meditation journey"""
        if not self.journey_steps:
            self.journey_status.set("Please add journey steps first ⚠️")
            return
            
        journey = self.meditation.create_custom_journey(
            "Custom Journey",
            self.journey_steps
        )
        
        def update_callback(param, value):
            if param == "progress":
                self.progress_var.set(value)
            elif param == "frequency":
                self.synth.morph_to_frequency(value)
            elif param == "visualization":
                self.viz_var.set(value)
                self.update_visualization()
        
        self.current_journey = self.meditation.guide_meditation(
            journey,
            callback=update_callback
        )
        self.journey_status.set("Custom journey in progress ✨")
        
    def animate_breathing(self, pattern: str):
        """Animate breathing guide"""
        timings = self.meditation.get_breath_pattern(pattern)
        total = sum(timings.values())
        width = 400
        height = 100
        
        def draw_frame(t):
            self.breath_canvas.delete("all")
            
            # Calculate circle size based on breath phase
            base_r = height/3
            t_norm = (t % total) / total
            
            if t_norm < timings["inhale"]/total:  # Inhale
                r = base_r * (1 + t_norm/(timings["inhale"]/total))
                text = "Inhale 🌬️"
            elif t_norm < (timings["inhale"] + timings["hold"])/total:  # Hold
                r = base_r * 2
                text = "Hold 💫"
            elif t_norm < (timings["inhale"] + timings["hold"] + timings["exhale"])/total:  # Exhale
                progress = (t_norm - (timings["inhale"] + timings["hold"])/total)/(timings["exhale"]/total)
                r = base_r * (2 - progress)
                text = "Exhale 🌊"
            else:  # Rest
                r = base_r
                text = "Rest ✨"
                
            # Draw breath circle
            x = width/2
            y = height/2
            self.breath_canvas.create_oval(x-r, y-r, x+r, y+r, fill="#4169E1", outline="#32CD32")
            self.breath_canvas.create_text(x, y, text=text, fill="white")
            
            # Continue animation
            if self.running:
                self.root.after(50, lambda: draw_frame(t + 0.05))
                
        draw_frame(0)
        
    def start_guided_meditation(self, purpose: str):
        """Start a guided meditation session"""
        meditation = self.meditation.guided_meditations.get(purpose)
        if not meditation:
            return
            
        self.meditation_status["intention"].set(meditation["intention"])
        self.meditation_status["state"].set(f"Starting {meditation['name']} {meditation['symbol']}")
        
        def update_callback(param, value):
            if param == "progress":
                self.progress_var.set(value)
            elif param == "frequency":
                self.synth.morph_to_frequency(value)
            elif param == "visualization":
                self.viz_var.set(value)
                self.update_visualization()
            elif param == "breath":
                self.animate_breathing(value)
            elif param == "affirmation":
                self.meditation_status["affirmation"].set(value)
            elif param == "state":
                self.meditation_status["state"].set(value)
                
        self.meditation.start_guided_meditation(purpose, update_callback)
        
    def _update_monitor(self):
        """Update the quantum coherence monitor"""
        while self.running:
            if hasattr(self.synth, 'current_state'):
                state = self.states[self.synth.current_state]
                self.status_vars["state"].set(f"{state.name} {state.symbol}")
                self.status_vars["frequency"].set(f"{state.frequency:.1f} Hz")
                self.status_vars["coherence"].set(f"{self.synth.get_coherence():.3f} φ")
                self.status_vars["brainwave"].set(f"{state.brainwave.title()} Wave")
                self.status_vars["pattern"].set(self.synth.get_current_pattern())
            time.sleep(0.1)  # 10Hz update rate
            
    def start(self):
        """Start the quantum control interface"""
        self.synth.start()
        self.root.mainloop()
        
    def stop(self):
        """Stop the quantum control interface"""
        self.running = False
        self.synth.stop()
        self.root.quit()

if __name__ == "__main__":
    controller = QuantumControlInterface()
    try:
        controller.start()
    except KeyboardInterrupt:
        controller.stop()
    finally:
        print("Quantum interface harmonized ✨")
