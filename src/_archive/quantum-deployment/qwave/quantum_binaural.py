import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class BinauralState:
    frequency: float
    carrier: float
    amplitude: float
    phase: float
    coherence: float

class QuantumBinauralProcessor:
    def __init__(self):
        self.phi = 1.618034
        self.sample_rate = 44100
        
        # Quantum frequency pairs for binaural beats
        self.frequency_pairs = {
            "ground": {
                "carrier": 432,
                "beat": 7.83  # Schumann resonance
            },
            "create": {
                "carrier": 528,
                "beat": 8.0   # Theta creativity
            },
            "heart": {
                "carrier": 594,
                "beat": 7.0   # Theta healing
            },
            "voice": {
                "carrier": 672,
                "beat": 10.0  # Alpha communication
            },
            "unity": {
                "carrier": 768,
                "beat": 4.0   # Delta unity
            }
        }
        
        # Initialize states
        self.states: Dict[str, BinauralState] = {}
        self._initialize_states()
        
    def _initialize_states(self) -> None:
        """Initialize binaural states for all frequencies"""
        for name, freqs in self.frequency_pairs.items():
            self.states[name] = BinauralState(
                frequency=freqs["beat"],
                carrier=freqs["carrier"],
                amplitude=1.0,
                phase=0.0,
                coherence=1.0
            )
    
    def generate_binaural_pair(self, name: str, duration: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate left and right channel waves for binaural beats"""
        if name not in self.states:
            return np.zeros(0), np.zeros(0)
            
        state = self.states[name]
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        
        # Generate carrier waves with slight frequency difference
        left_freq = state.carrier
        right_freq = state.carrier + state.frequency
        
        # Apply phi-based phase modulation
        phase_mod = self.phi * np.sin(2 * np.pi * 0.1 * t)
        
        # Generate waves with quantum coherence
        left = state.amplitude * np.sin(2 * np.pi * left_freq * t + phase_mod)
        right = state.amplitude * np.sin(2 * np.pi * right_freq * t + phase_mod + state.phase)
        
        # Apply coherence envelope
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * self.phi * t * state.coherence)
        left *= envelope
        right *= envelope
        
        return left, right
    
    def process_audio(self, left: np.ndarray, right: np.ndarray, 
                     name: str, mix_ratio: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """Mix binaural beats with existing audio"""
        if len(left) == 0 or name not in self.states:
            return left, right
            
        # Generate binaural beats
        duration = len(left) / self.sample_rate
        bin_left, bin_right = self.generate_binaural_pair(name, duration)
        
        # Trim or extend to match input length
        bin_left = bin_left[:len(left)]
        bin_right = bin_right[:len(right)]
        
        # Mix with original audio
        left_mix = (1 - mix_ratio) * left + mix_ratio * bin_left
        right_mix = (1 - mix_ratio) * right + mix_ratio * bin_right
        
        return left_mix, right_mix
    
    def update_state(self, name: str, **kwargs) -> None:
        """Update binaural state parameters"""
        if name in self.states:
            state = self.states[name]
            for key, value in kwargs.items():
                if hasattr(state, key):
                    setattr(state, key, value)
    
    def get_frequency_info(self, name: str) -> Optional[Dict]:
        """Get frequency information for a state"""
        if name in self.frequency_pairs:
            return self.frequency_pairs[name]
        return None
    
    def calculate_coherence(self, left: np.ndarray, right: np.ndarray) -> float:
        """Calculate coherence between left and right channels"""
        if len(left) == 0 or len(right) == 0:
            return 0.0
            
        # Calculate cross-correlation
        correlation = np.correlate(left, right, mode='full')
        max_corr = np.max(np.abs(correlation))
        
        # Normalize by signal energies
        energy_left = np.sum(left * left)
        energy_right = np.sum(right * right)
        
        if energy_left > 0 and energy_right > 0:
            coherence = max_corr / np.sqrt(energy_left * energy_right)
            return float(coherence)
        return 0.0
