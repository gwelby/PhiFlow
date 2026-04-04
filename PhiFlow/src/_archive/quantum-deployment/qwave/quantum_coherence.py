import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import time

@dataclass
class BrainwaveState:
    name: str
    frequency: float
    color: str
    symbol: str
    consciousness: str

class QuantumCoherenceVisualizer:
    def __init__(self):
        self.phi = 1.618034
        
        # Extended brainwave frequencies
        self.brainwaves = {
            "delta": BrainwaveState("Delta", 4.0, "indigo", "🌌", "Deep Unity"),
            "theta": BrainwaveState("Theta", 7.83, "purple", "🌀", "Creation Flow"),
            "alpha": BrainwaveState("Alpha", 10.0, "blue", "💫", "Relaxed Awareness"),
            "beta": BrainwaveState("Beta", 15.0, "green", "✨", "Active Focus"),
            "gamma": BrainwaveState("Gamma", 40.0, "gold", "⚡", "Peak Performance"),
            "lambda": BrainwaveState("Lambda", 100.0, "white", "💎", "Quantum Consciousness"),
            "epsilon": BrainwaveState("Epsilon", 0.5, "violet", "🌟", "Universal Connection")
        }
        
        # Setup visualization
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        self.setup_plot()
        
        # Initialize frequency morphing
        self.morph_buffer = []
        self.morph_target = None
        self.morph_speed = self.phi
        
    def setup_plot(self):
        """Setup the coherence visualization plots"""
        # Coherence plot
        self.ax1.set_title("Quantum Coherence Field", fontsize=12)
        self.ax1.set_xlabel("Time (φ cycles)")
        self.ax1.set_ylabel("Coherence")
        self.ax1.set_ylim(0, 1.5)
        self.ax1.grid(True, alpha=0.3)
        
        # Frequency spectrum plot
        self.ax2.set_title("Frequency Morphing", fontsize=12)
        self.ax2.set_xlabel("Frequency (Hz)")
        self.ax2.set_ylabel("Amplitude")
        self.ax2.set_yscale('log')
        self.ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
    def morph_frequency(self, start_freq: float, target_freq: float, 
                       duration: float) -> np.ndarray:
        """Morph between frequencies using phi-based interpolation"""
        steps = int(duration * self.phi * 100)
        t = np.linspace(0, duration, steps)
        
        # Phi-based morphing curve
        morph_factor = 0.5 * (1 - np.cos(np.pi * t / duration))
        frequencies = start_freq + (target_freq - start_freq) * morph_factor
        
        # Add quantum fluctuations
        quantum_noise = 0.01 * np.sin(2 * np.pi * self.phi * t)
        frequencies += quantum_noise
        
        return frequencies
    
    def calculate_coherence_metrics(self, left: np.ndarray, 
                                  right: np.ndarray) -> Dict[str, float]:
        """Calculate various coherence metrics"""
        metrics = {}
        
        # Phase coherence
        hilbert_left = np.abs(np.fft.hilbert(left))
        hilbert_right = np.abs(np.fft.hilbert(right))
        phase_diff = np.angle(hilbert_left * np.conj(hilbert_right))
        metrics['phase'] = np.abs(np.mean(np.exp(1j * phase_diff)))
        
        # Amplitude coherence
        amp_corr = np.corrcoef(np.abs(hilbert_left), np.abs(hilbert_right))[0,1]
        metrics['amplitude'] = np.abs(amp_corr)
        
        # Frequency coherence
        freq_left = np.fft.fft(left)
        freq_right = np.fft.fft(right)
        freq_corr = np.corrcoef(np.abs(freq_left), np.abs(freq_right))[0,1]
        metrics['frequency'] = np.abs(freq_corr)
        
        # Overall quantum coherence
        metrics['quantum'] = (metrics['phase'] * metrics['amplitude'] * 
                            metrics['frequency']) ** (1/self.phi)
        
        return metrics
    
    def update_plot(self, frame: int, left: np.ndarray, right: np.ndarray):
        """Update the coherence visualization"""
        # Calculate metrics
        metrics = self.calculate_coherence_metrics(left, right)
        
        # Update coherence plot
        self.ax1.clear()
        self.ax1.set_title(f"Quantum Coherence: {metrics['quantum']:.3f}φ", fontsize=12)
        
        # Plot coherence metrics
        t = np.linspace(0, 2*np.pi, len(left))
        for name, value in metrics.items():
            color = self.brainwaves[name].color if name in self.brainwaves else 'gray'
            self.ax1.plot(t, value * np.sin(t * self.phi), 
                         label=f"{name}: {value:.3f}", color=color, alpha=0.7)
        
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        
        # Update frequency spectrum
        self.ax2.clear()
        self.ax2.set_title("Frequency Morphing", fontsize=12)
        
        # Calculate and plot frequency spectrum
        freqs_left = np.fft.fftfreq(len(left))
        spectrum_left = np.abs(np.fft.fft(left))
        freqs_right = np.fft.fftfreq(len(right))
        spectrum_right = np.abs(np.fft.fft(right))
        
        self.ax2.semilogy(freqs_left, spectrum_left, 'b-', alpha=0.5, label='Left')
        self.ax2.semilogy(freqs_right, spectrum_right, 'r-', alpha=0.5, label='Right')
        
        # Mark brainwave frequencies
        for state in self.brainwaves.values():
            self.ax2.axvline(x=state.frequency, color=state.color, alpha=0.3,
                           linestyle='--', label=f"{state.symbol} {state.name}")
        
        self.ax2.legend()
        self.ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
    
    def start_animation(self, audio_callback):
        """Start the coherence visualization animation"""
        self.anim = FuncAnimation(
            self.fig, self.update_plot,
            fargs=audio_callback(),
            interval=100,
            blit=False
        )
        plt.show()
    
    def save_snapshot(self, path: str):
        """Save the current visualization state"""
        plt.savefig(path)
        
    def cleanup(self):
        """Clean up visualization resources"""
        plt.close(self.fig)
