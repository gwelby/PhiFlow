import numpy as np
import librosa
from dataclasses import dataclass
from typing import Dict, List, Tuple
from enum import Enum

class MusicStyle(Enum):
    TECHNO = "TECHNO"          # Boris Brejcha style
    PROGRESSIVE = "PROGRESSIVE" # Worakls style
    AMBIENT = "AMBIENT"        # Deep meditation
    QUANTUM = "QUANTUM"        # Pure φ frequencies
    CUSTOM = "CUSTOM"          # User-defined

@dataclass
class PhiHarmonic:
    frequency: float
    name: str
    color: Tuple[float, float, float]

class DJPhi:
    def __init__(self):
        # φ (phi) - The Golden Ratio
        self.phi = 1.618033988749895
        
        # Base frequency set (432 Hz * φ^n)
        self.phi_harmonics = {
            "GROUND": PhiHarmonic(432.0, "Ground State", (0.0, 1.0, 0.0)),
            "CREATE": PhiHarmonic(432.0 * self.phi, "Creation Point", (0.0, 0.0, 1.0)),
            "HEART": PhiHarmonic(432.0 * self.phi**2, "Heart Field", (1.0, 0.0, 1.0)),
            "VOICE": PhiHarmonic(432.0 * self.phi**3, "Voice Flow", (1.0, 1.0, 0.0)),
            "UNITY": PhiHarmonic(432.0 * self.phi**4, "Unity Field", (1.0, 1.0, 1.0))
        }
        
        # Style-specific quantum configurations
        self.style_configs = {
            MusicStyle.TECHNO: {
                "bass_weight": 1.2,
                "mid_weight": 0.8,
                "high_weight": 0.6,
                "phase_shift": self.phi,
                "resonance": self.phi**2,
                "harmonics": ["GROUND", "CREATE", "UNITY"]
            },
            MusicStyle.PROGRESSIVE: {
                "bass_weight": 1.0,
                "mid_weight": 1.0,
                "high_weight": 1.0,
                "phase_shift": self.phi**2,
                "resonance": self.phi**3,
                "harmonics": ["CREATE", "HEART", "VOICE"]
            },
            MusicStyle.AMBIENT: {
                "bass_weight": 0.8,
                "mid_weight": 1.1,
                "high_weight": 0.7,
                "phase_shift": self.phi**0.5,
                "resonance": self.phi,
                "harmonics": ["GROUND", "HEART", "UNITY"]
            },
            MusicStyle.QUANTUM: {
                "bass_weight": self.phi,
                "mid_weight": self.phi,
                "high_weight": self.phi,
                "phase_shift": self.phi**2,
                "resonance": self.phi**3,
                "harmonics": ["GROUND", "CREATE", "HEART", "VOICE", "UNITY"]
            }
        }

    def detect_style(self, audio_data: np.ndarray, sr: int) -> MusicStyle:
        """Detect music style based on frequency analysis"""
        # Get frequency spectrum
        D = librosa.stft(audio_data)
        mag = np.abs(D)
        
        # Split into frequency bands
        freqs = librosa.fft_frequencies(sr=sr)
        bass_mask = freqs < 150
        mid_mask = (freqs >= 150) & (freqs < 4000)
        high_mask = freqs >= 4000
        
        # Calculate energy in each band
        bass_energy = np.mean(mag[bass_mask])
        mid_energy = np.mean(mag[mid_mask])
        high_energy = np.mean(mag[high_mask])
        
        # Detect rhythmic features
        tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sr)
        
        # Style classification based on features
        if tempo > 125 and bass_energy > mid_energy * 1.5:
            return MusicStyle.TECHNO
        elif 100 <= tempo <= 125 and abs(mid_energy - high_energy) < 0.2:
            return MusicStyle.PROGRESSIVE
        elif tempo < 100 and mid_energy > bass_energy:
            return MusicStyle.AMBIENT
        else:
            return MusicStyle.QUANTUM

    def get_style_quantum_field(self, style: MusicStyle, 
                              audio_data: np.ndarray,
                              field_shape: Tuple[int, int, int]) -> np.ndarray:
        """Generate quantum field based on music style"""
        config = self.style_configs[style]
        quantum_field = np.zeros(field_shape, dtype=np.complex128)
        
        # Apply style-specific processing
        center = field_shape[0] // 2
        
        for harmonic in config["harmonics"]:
            freq = self.phi_harmonics[harmonic].frequency
            phase = np.angle(np.fft.fft(audio_data)) * config["phase_shift"]
            
            # Create harmonic layers
            for x in range(field_shape[0]):
                for y in range(field_shape[1]):
                    for z in range(field_shape[2]):
                        r = np.sqrt((x-center)**2 + (y-center)**2 + (z-center)**2)
                        if r > 0:
                            # Generate quantum vortex
                            quantum_field[x,y,z] += np.exp(1j * (
                                phase[x % len(phase)] +  # Audio phase
                                r * config["resonance"] +  # Spatial resonance
                                freq / 432.0  # Frequency contribution
                            )) / r
        
        return quantum_field

    def get_style_colors(self, style: MusicStyle) -> List[Tuple[float, float, float]]:
        """Get color scheme for visualization based on style"""
        if style == MusicStyle.TECHNO:
            # Electric, high-energy colors
            return [
                (0.0, 1.0, 1.0),  # Cyan
                (1.0, 0.0, 1.0),  # Magenta
                (1.0, 1.0, 0.0)   # Yellow
            ]
        elif style == MusicStyle.PROGRESSIVE:
            # Flowing, evolving colors
            return [
                (0.0, 0.5, 1.0),  # Ocean blue
                (0.0, 1.0, 0.5),  # Sea green
                (0.5, 0.0, 1.0)   # Purple
            ]
        elif style == MusicStyle.AMBIENT:
            # Soft, ethereal colors
            return [
                (0.2, 0.4, 0.6),  # Soft blue
                (0.3, 0.5, 0.3),  # Soft green
                (0.4, 0.3, 0.5)   # Soft purple
            ]
        else:  # QUANTUM
            # Pure phi-based colors
            return [
                (1.0/self.phi, 1.0/self.phi**2, 1.0/self.phi**3),
                (1.0/self.phi**2, 1.0/self.phi**3, 1.0/self.phi),
                (1.0/self.phi**3, 1.0/self.phi, 1.0/self.phi**2)
            ]

    def apply_phi_transform(self, field: np.ndarray, style: MusicStyle) -> np.ndarray:
        """Apply phi-based transformation to quantum field"""
        config = self.style_configs[style]
        
        # Create phi-spiral transform
        center = np.array(field.shape) // 2
        transformed = np.zeros_like(field)
        
        for x in range(field.shape[0]):
            for y in range(field.shape[1]):
                for z in range(field.shape[2]):
                    pos = np.array([x, y, z])
                    r = np.linalg.norm(pos - center)
                    if r > 0:
                        # Phi spiral
                        theta = np.arctan2(y-center[1], x-center[0])
                        phi_r = r * self.phi
                        phi_theta = theta * self.phi
                        
                        # New coordinates
                        new_x = int(center[0] + phi_r * np.cos(phi_theta)) % field.shape[0]
                        new_y = int(center[1] + phi_r * np.sin(phi_theta)) % field.shape[1]
                        new_z = int(z * self.phi) % field.shape[2]
                        
                        transformed[new_x, new_y, new_z] = field[x, y, z]
        
        return transformed
