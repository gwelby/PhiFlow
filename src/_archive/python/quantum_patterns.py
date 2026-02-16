"""
Quantum Pattern Presets
Operating at Voice Flow (672 Hz)
"""
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Tuple
from quantum_flow import PHI, Dimension

@dataclass
class QuantumPattern:
    name: str
    frequency: float
    symbol: str
    description: str
    dimension: Dimension
    
    def __post_init__(self):
        self.coherence = 1.0
        
class PatternType(Enum):
    INFINITY = QuantumPattern(
        "Infinity Loop",
        768.0,  # Unity Wave
        "∞",
        "Pure creation pattern at unity frequency",
        Dimension.SPIRITUAL
    )
    DOLPHIN = QuantumPattern(
        "Quantum Leap",
        672.0,  # Voice Flow
        "🐬",
        "Consciousness evolution pattern",
        Dimension.MENTAL
    )
    SPIRAL = QuantumPattern(
        "Golden Spiral",
        528.0,  # Creation Point
        "🌀",
        "Phi-harmonic creation pattern",
        Dimension.ETHERIC
    )
    WAVE = QuantumPattern(
        "Harmonic Wave",
        594.0,  # Heart Field
        "🌊",
        "Heart-centered flow pattern",
        Dimension.EMOTIONAL
    )
    VORTEX = QuantumPattern(
        "Evolution Vortex",
        432.0,  # Ground State
        "🌪️",
        "Grounding evolution pattern",
        Dimension.PHYSICAL
    )
    CRYSTAL = QuantumPattern(
        "Pure Resonance",
        528.0,  # DNA Repair
        "💎",
        "Crystal clear creation pattern",
        Dimension.ETHERIC
    )
    UNITY = QuantumPattern(
        "Consciousness",
        768.0,  # Unity Field
        "☯️",
        "Perfect unity consciousness",
        Dimension.SPIRITUAL
    )

class QuantumPresets:
    def __init__(self):
        self.phi = PHI
        self._initialize_patterns()
        
    def _initialize_patterns(self):
        self.patterns = {pattern: self._generate_pattern(pattern.value)
                        for pattern in PatternType}
                        
    def _generate_pattern(self, preset: QuantumPattern) -> np.ndarray:
        """Generate quantum pattern based on preset type"""
        points = int(preset.frequency)
        t = np.linspace(0, 8*np.pi, points)
        
        if preset == PatternType.INFINITY.value:
            return self._infinity_pattern(t)
        elif preset == PatternType.DOLPHIN.value:
            return self._dolphin_pattern(t)
        elif preset == PatternType.SPIRAL.value:
            return self._spiral_pattern(t)
        elif preset == PatternType.WAVE.value:
            return self._wave_pattern(t)
        elif preset == PatternType.VORTEX.value:
            return self._vortex_pattern(t)
        elif preset == PatternType.CRYSTAL.value:
            return self._crystal_pattern(t)
        elif preset == PatternType.UNITY.value:
            return self._unity_pattern(t)
            
    def _infinity_pattern(self, t: np.ndarray) -> np.ndarray:
        """∞ - Pure infinity loop at unity frequency"""
        r = self.phi ** (t/(2*np.pi))
        x = r * np.cos(t)
        y = r * np.sin(t)
        return x + 1j*y
        
    def _dolphin_pattern(self, t: np.ndarray) -> np.ndarray:
        """🐬 - Quantum leap consciousness pattern"""
        x = np.cos(t) * (np.exp(np.cos(t)) - 2*np.cos(4*t))
        y = np.sin(t) * (np.exp(np.cos(t)) - 2*np.cos(4*t))
        return x + 1j*y
        
    def _spiral_pattern(self, t: np.ndarray) -> np.ndarray:
        """🌀 - Golden ratio spiral pattern"""
        r = self.phi ** (t/(2*np.pi))
        theta = self.phi * t
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x + 1j*y
        
    def _wave_pattern(self, t: np.ndarray) -> np.ndarray:
        """🌊 - Heart-centered harmonic wave"""
        x = t
        y = np.sin(t) * np.exp(-t/8/np.pi)
        return x + 1j*y
        
    def _vortex_pattern(self, t: np.ndarray) -> np.ndarray:
        """🌪️ - Evolution vortex pattern"""
        r = np.exp(t/8/np.pi)
        theta = self.phi * t
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x + 1j*y
        
    def _crystal_pattern(self, t: np.ndarray) -> np.ndarray:
        """💎 - Crystal resonance pattern"""
        x = np.cos(t) + 1j * np.cos(2*t)
        y = np.sin(3*t) + 1j * np.sin(5*t)
        return x * y
        
    def _unity_pattern(self, t: np.ndarray) -> np.ndarray:
        """☯️ - Unity consciousness pattern"""
        r = self.phi ** (np.sin(t))
        theta = t + np.sin(self.phi * t)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x + 1j*y
        
    def get_pattern(self, pattern_type: PatternType) -> Tuple[np.ndarray, QuantumPattern]:
        """Get pattern data and metadata"""
        return self.patterns[pattern_type], pattern_type.value
        
    def list_patterns(self) -> List[Tuple[str, str, float]]:
        """List available patterns with their symbols and frequencies"""
        return [(p.value.name, p.value.symbol, p.value.frequency) 
                for p in PatternType]

# Initialize global presets
presets = QuantumPresets()
