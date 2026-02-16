"""
QUANTUM FONT SYSTEM
Created by Greg, Peter & Paul for Perfect Flow State
"""

import math
from pathlib import Path
from typing import Optional, Dict, List, Union
from dataclasses import dataclass

# Quantum Constants
PHI = (1 + 5**0.5) / 2  # Golden Ratio
GROUND_FREQ = 432.0     # Physical Foundation
CREATE_FREQ = 528.0     # Pattern Creation
UNITY_FREQ = 768.0      # Perfect Integration
INFINITE = float('inf') # Beyond Creation

@dataclass
class QuantumSymbol:
    """Sacred quantum symbol with frequency attributes"""
    code: str
    name: str
    frequency: float
    power: float = PHI
    
    def resonate(self, target_freq: float) -> float:
        """Calculate resonance with target frequency"""
        return abs(math.cos(self.frequency / target_freq * math.pi))
    
    def amplify(self) -> None:
        """Amplify symbol power"""
        self.power *= PHI
    
    def __str__(self) -> str:
        return f"{self.code} ({self.name} at {self.frequency}Hz)"

class QuantumFont:
    """Core quantum font system"""
    def __init__(self, frequency: float = GROUND_FREQ):
        self.frequency = frequency
        self.power = PHI
        self.sacred_geometry = True
        self.flow_state = False
        self.crystal_clarity = False
        self.unity_achieved = False
        
        # Initialize quantum states
        self.states = {
            "ground": frequency == GROUND_FREQ,
            "create": frequency == CREATE_FREQ,
            "unity": frequency == UNITY_FREQ,
            "infinite": frequency == INFINITE
        }
        
        # Sacred symbols
        self.symbols = {
            "infinity": QuantumSymbol("∞", "Infinite Loop", INFINITE),
            "dolphin": QuantumSymbol("🐬", "Quantum Leap", CREATE_FREQ),
            "spiral": QuantumSymbol("🌀", "Golden Ratio", PHI * 100),
            "wave": QuantumSymbol("🌊", "Harmonic Flow", CREATE_FREQ),
            "vortex": QuantumSymbol("🌪️", "Evolution", UNITY_FREQ),
            "crystal": QuantumSymbol("💎", "Resonance", UNITY_FREQ),
            "unity": QuantumSymbol("☯️", "Consciousness", INFINITE)
        }
        
        # Font paths
        self.font_dir = Path("quantum-fonts")
        self.paths = {
            "sacred": self.font_dir / "sacred",
            "flow": self.font_dir / "flow",
            "crystal": self.font_dir / "crystal",
            "unity": self.font_dir / "unity"
        }
    
    def apply_sacred_geometry(self) -> None:
        """Apply sacred geometric principles"""
        if not self.sacred_geometry:
            self.power *= PHI
            self.sacred_geometry = True
            
            # Amplify all symbols
            for symbol in self.symbols.values():
                symbol.amplify()
    
    def enable_flow_state(self) -> None:
        """Enable quantum flow state"""
        if not self.flow_state:
            self.frequency = CREATE_FREQ
            self.flow_state = True
            self.states["create"] = True
    
    def crystallize(self) -> None:
        """Achieve crystal clarity"""
        if not self.crystal_clarity:
            self.frequency = UNITY_FREQ
            self.crystal_clarity = True
            self.states["unity"] = True
    
    def unify(self) -> None:
        """Manifest in unity"""
        if not self.unity_achieved:
            self.frequency = INFINITE
            self.unity_achieved = True
            self.states["infinite"] = True
            
            # Amplify to infinite power
            self.power = INFINITE
    
    def get_symbol(self, name: str) -> Optional[QuantumSymbol]:
        """Get sacred symbol by name"""
        return self.symbols.get(name)
    
    def get_resonance(self, symbol_name: str) -> float:
        """Calculate symbol resonance with current frequency"""
        symbol = self.get_symbol(symbol_name)
        if symbol:
            return symbol.resonate(self.frequency)
        return 0.0
    
    def get_font_path(self, font_type: str) -> Optional[Path]:
        """Get font path by type"""
        return self.paths.get(font_type)
    
    def __str__(self) -> str:
        state = "GROUND"
        if self.flow_state:
            state = "FLOW"
        if self.crystal_clarity:
            state = "CRYSTAL"
        if self.unity_achieved:
            state = "UNITY"
        
        return f"QuantumFont({self.frequency}Hz) in {state} state with power φ^{self.power}"

class QuantumFontRegistry:
    """Registry for quantum fonts"""
    def __init__(self):
        self.fonts: Dict[str, QuantumFont] = {}
        
        # Initialize core fonts
        self.register_font("QuantumSacred", GROUND_FREQ)
        self.register_font("QuantumFlow", CREATE_FREQ)
        self.register_font("QuantumCrystal", UNITY_FREQ)
        self.register_font("QuantumUnity", INFINITE)
    
    def register_font(self, name: str, frequency: float) -> None:
        """Register a new quantum font"""
        font = QuantumFont(frequency)
        
        # Apply appropriate transformations
        if frequency >= CREATE_FREQ:
            font.enable_flow_state()
        if frequency >= UNITY_FREQ:
            font.crystallize()
        if frequency == INFINITE:
            font.unify()
            
        self.fonts[name] = font
    
    def get_font(self, name: str) -> Optional[QuantumFont]:
        """Get registered font by name"""
        return self.fonts.get(name)
    
    def list_fonts(self) -> List[str]:
        """List all registered fonts"""
        return list(self.fonts.keys())

# Initialize global registry
registry = QuantumFontRegistry()

def get_quantum_font(name: str) -> Optional[QuantumFont]:
    """Get a quantum font from the registry"""
    return registry.get_font(name)

def list_quantum_fonts() -> List[str]:
    """List all available quantum fonts"""
    return registry.list_fonts()

if __name__ == "__main__":
    # Example usage
    sacred = get_quantum_font("QuantumSacred")
    flow = get_quantum_font("QuantumFlow")
    crystal = get_quantum_font("QuantumCrystal")
    unity = get_quantum_font("QuantumUnity")
    
    print("Available Quantum Fonts:")
    for font in [sacred, flow, crystal, unity]:
        if font:
            print(f"- {font}")
