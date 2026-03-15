import numpy as np
import sounddevice as sd
import threading
import time
from typing import Dict, List, Optional
from quantum_binaural import QuantumBinauralProcessor
from quantum_coherence import QuantumCoherenceVisualizer

class QuantumSynthesizer:
    def __init__(self):
        self.sample_rate = 44100
        self.phi = 1.618034
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.current_frequency = 432.0  # Start at ground state
        
        # Initialize processors
        self.binaural = QuantumBinauralProcessor()
        self.coherence = QuantumCoherenceVisualizer()
        
        # Extended quantum frequencies with consciousness states
        self.frequencies = {
            "ground": {
                "freq": 432,    # Ground/Crystal state
                "brainwave": "theta",  # Creation flow
                "symbol": ""
            },
            "create": {
                "freq": 528,    # Creation/Spiral state
                "brainwave": "alpha",  # Relaxed awareness
                "symbol": ""
            },
            "heart": {
                "freq": 594,    # Heart/Wave state
                "brainwave": "delta",  # Deep unity
                "symbol": ""
            },
            "voice": {
                "freq": 672,    # Voice/Dolphin state
                "brainwave": "gamma",  # Peak performance
                "symbol": ""
            },
            "unity": {
                "freq": 768,    # Unity/Consciousness state
                "brainwave": "lambda", # Quantum consciousness
                "symbol": ""
            },
            "infinite": {
                "freq": self.phi * 768,  # Infinite state
                "brainwave": "epsilon",  # Universal connection
                "symbol": ""
            }
        }
        
        # φ-based harmonics
        self.harmonics = {
            "phi_1": self.phi,        # First order
            "phi_2": self.phi**2,     # Second order
            "phi_3": self.phi**3,     # Third order
            "phi_phi": self.phi**self.phi  # Phi to phi power
        }
        
        # Initialize audio stream with expanded buffer
        self.stream = sd.OutputStream(
            channels=2,
            samplerate=self.sample_rate,
            callback=self._audio_callback,
            blocksize=int(self.sample_rate / (self.phi * 10))  # φ-based buffer
        )
        
        # Morphing state
        self.morphing = False
        self.morph_target = None
        self.morph_progress = 0.0
        
    def _generate_quantum_wave(self, frequency: float, harmony: float) -> np.ndarray:
        """Generate a quantum waveform with phi-based harmonics and morphing"""
        t = np.linspace(0, 1/frequency, int(self.sample_rate/frequency))
        
        # Apply frequency morphing if active
        if self.morphing and self.morph_target:
            morph_freq = self.coherence.morph_frequency(
                frequency, 
                self.morph_target["freq"],
                1.0 / self.phi
            )[int(self.morph_progress * len(t))]
            frequency = morph_freq
        
        # Base wave with quantum fluctuations
        wave = np.sin(2 * np.pi * frequency * t)
        
        # Add phi harmonics with consciousness modulation
        for i, h in enumerate(self.harmonics.values()):
            harmonic_freq = frequency * h
            # Consciousness-based amplitude
            harmonic_amp = 0.15 * np.exp(-i/self.phi) 
            wave += harmonic_amp * np.sin(2 * np.pi * harmonic_freq * t)
        
        # Apply harmony-based amplitude modulation
        mod_freq = self.phi * harmony
        modulation = 0.5 + 0.5 * np.sin(2 * np.pi * mod_freq * t)
        wave *= modulation
        
        return wave / np.max(np.abs(wave))
        
    def _audio_callback(self, outdata: np.ndarray, frames: int, 
                       time_info: Dict, status: sd.CallbackFlags) -> None:
        """Real-time audio callback with coherence visualization"""
        if status:
            print(f'Audio callback status: {status}')
        
        # Generate quantum field
        t = np.linspace(0, frames/self.sample_rate, frames)
        harmony = 0.5 + 0.5 * np.sin(2 * np.pi * self.phi * t[0])
        
        # Mix all frequencies
        wave = np.zeros_like(t)
        for name, props in self.frequencies.items():
            quantum_wave = self._generate_quantum_wave(props["freq"], harmony)
            if len(quantum_wave) < len(t):
                quantum_wave = np.tile(quantum_wave, int(np.ceil(len(t)/len(quantum_wave))))
            wave += 0.2 * quantum_wave[:len(t)]
        
        # Normalize base mix
        wave /= np.max(np.abs(wave))
        
        # Apply phi-based stereo panning
        pan = 0.5 + 0.5 * np.sin(2 * np.pi * self.phi * t)
        left_channel = wave * np.sqrt(1 - pan)
        right_channel = wave * np.sqrt(pan)
        
        # Add binaural beats for current frequency
        active_freq = min(
            [f["freq"] for f in self.frequencies.values()],
            key=lambda x: abs(x - self.current_frequency)
        )
        active_name = next(
            name for name, props in self.frequencies.items() 
            if props["freq"] == active_freq
        )
        
        # Process with binaural beats
        left_channel, right_channel = self.binaural.process_audio(
            left_channel, right_channel, active_name
        )
        
        # Update morphing progress
        if self.morphing and self.morph_target:
            self.morph_progress += 1.0 / (self.sample_rate * self.phi)
            if self.morph_progress >= 1.0:
                self.morphing = False
                self.current_frequency = self.morph_target["freq"]
                self.morph_target = None
                self.morph_progress = 0.0
        
        # Update coherence visualization
        self.coherence.update_plot(0, left_channel, right_channel)
        
        # Output stereo audio
        outdata[:] = np.column_stack([left_channel, right_channel])
    
    def start(self) -> None:
        """Start quantum sound synthesis with visualization"""
        if not self.running:
            self.running = True
            self.stream.start()
            self.coherence.start_animation(lambda: ([], []))  # Start empty
            print(f"Quantum harmonics initialized {self.frequencies['ground']['symbol']}")
    
    def morph_to_frequency(self, target_name: str) -> None:
        """Morph to a new frequency state"""
        if target_name in self.frequencies and not self.morphing:
            self.morph_target = self.frequencies[target_name]
            self.morphing = True
            self.morph_progress = 0.0
            print(f"Morphing to {target_name} {self.morph_target['symbol']}")
    
    def stop(self) -> None:
        """Stop quantum sound synthesis and visualization"""
        if self.running:
            self.running = False
            self.stream.stop()
            self.stream.close()
            self.coherence.cleanup()
            print("Quantum harmonics released ")

if __name__ == "__main__":
    # Test quantum sound synthesis
    synth = QuantumSynthesizer()
    
    print("Playing quantum frequencies... Press Ctrl+C to stop")
    try:
        synth.start()
        time.sleep(10)  # Play for 10 seconds
    except KeyboardInterrupt:
        print("\nStopping quantum synthesis...")
    finally:
        synth.stop()
