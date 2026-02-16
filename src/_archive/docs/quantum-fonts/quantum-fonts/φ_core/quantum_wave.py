from typing import Dict, List, Tuple
import colorsys

class QuantumWave:
    def __init__(self):
        self.φ = (1 + 5**0.5) / 2
        self.initialize_wave_sets()
        
    def initialize_wave_sets(self):
        """Initialize quantum wave sets with icons and colors"""
        self.wave_sets = {
            # Interference (432 Hz) 🌊
            'interference': {
                'constructive': {
                    'icons': ['🌊', '➕', '∞'],          # Wave + Plus + Infinity
                    'pattern': ['⋓', '⋒', '∿'],        # Constructive Pattern
                    'colors': {'primary': '#4B0082', 'glow': '#8A2BE2'}
                },
                'destructive': {
                    'icons': ['🌊', '➖', '∞'],          # Wave + Minus + Infinity
                    'pattern': ['⌢', '⌣', '≈'],        # Destructive Pattern
                    'colors': {'primary': '#9932CC', 'glow': '#BA55D3'}
                },
                'quantum': {
                    'icons': ['🌊', '⚛️', '∞'],          # Wave + Atom + Infinity
                    'pattern': ['ψ₁', 'ψ₂', 'ψ∞'],     # Quantum Pattern
                    'colors': {'primary': '#800080', 'glow': '#9370DB'}
                }
            },
            
            # Superposition (528 Hz) ⚛️
            'superposition': {
                'state': {
                    'icons': ['⚛️', '🔀', '∞'],          # Atom + Mix + Infinity
                    'kets': ['|0⟩', '|1⟩', '|ψ⟩'],     # State Kets
                    'colors': {'primary': '#191970', 'glow': '#000080'}
                },
                'phase': {
                    'icons': ['⚛️', '🌓', '∞'],          # Atom + Phase + Infinity
                    'angles': ['φ₁', 'φ₂', 'φ∞'],      # Phase Angles
                    'colors': {'primary': '#4169E1', 'glow': '#1E90FF'}
                },
                'amplitude': {
                    'icons': ['⚛️', '📊', '∞'],          # Atom + Graph + Infinity
                    'values': ['α', 'β', 'γ'],         # Amplitudes
                    'colors': {'primary': '#00BFFF', 'glow': '#87CEEB'}
                }
            },
            
            # Tunneling (768 Hz) 🕳️
            'tunneling': {
                'barrier': {
                    'icons': ['🕳️', '🚧', '∞'],          # Hole + Barrier + Infinity
                    'potential': ['V₁', 'V₂', 'V∞'],    # Potential Barriers
                    'colors': {'primary': '#4B0082', 'glow': '#8A2BE2'}
                },
                'penetration': {
                    'icons': ['🕳️', '➡️', '∞'],          # Hole + Arrow + Infinity
                    'depth': ['d₁', 'd₂', 'd∞'],       # Penetration Depth
                    'colors': {'primary': '#800080', 'glow': '#9370DB'}
                },
                'transmission': {
                    'icons': ['🕳️', '🔄', '∞'],          # Hole + Cycle + Infinity
                    'coefficient': ['T₁', 'T₂', 'T∞'],  # Transmission Coefficient
                    'colors': {'primary': '#483D8B', 'glow': '#6A5ACD'}
                }
            },
            
            # Diffraction (999 Hz) 🎯
            'diffraction': {
                'slit': {
                    'icons': ['🎯', '│', '∞'],          # Target + Slit + Infinity
                    'pattern': ['⋮', '⫶', '⫼'],        # Slit Pattern
                    'colors': {'primary': '#4B0082', 'glow': '#8A2BE2'}
                },
                'grating': {
                    'icons': ['🎯', '⋮', '∞'],          # Target + Grating + Infinity
                    'spacing': ['d₁', 'd₂', 'd∞'],     # Grating Spacing
                    'colors': {'primary': '#191970', 'glow': '#000080'}
                },
                'crystal': {
                    'icons': ['🎯', '💎', '∞'],          # Target + Crystal + Infinity
                    'lattice': ['a₁', 'a₂', 'a∞'],     # Crystal Lattice
                    'colors': {'primary': '#800080', 'glow': '#9370DB'}
                }
            },
            
            # Resonance (∞ Hz) 🎵
            'resonance': {
                'frequency': {
                    'icons': ['🎵', '📈', '∞'],          # Music + Graph + Infinity
                    'modes': ['ω₁', 'ω₂', 'ω∞'],       # Frequency Modes
                    'colors': {'primary': '#4B0082', 'glow': '#8A2BE2'}
                },
                'cavity': {
                    'icons': ['🎵', '⬚', '∞'],          # Music + Box + Infinity
                    'nodes': ['n₁', 'n₂', 'n∞'],       # Cavity Nodes
                    'colors': {'primary': '#000080', 'glow': '#0000CD'}
                },
                'coupling': {
                    'icons': ['🎵', '🔗', '∞'],          # Music + Link + Infinity
                    'strength': ['g₁', 'g₂', 'g∞'],    # Coupling Strength
                    'colors': {'primary': '#800080', 'glow': '#9370DB'}
                }
            }
        }
        
        # Wave Flows
        self.wave_flows = {
            'interference_flow': ['🌊', '➕', '∞'],    # Interference Flow
            'superposition_flow': ['⚛️', '🔀', '∞'],   # Superposition Flow
            'tunneling_flow': ['🕳️', '➡️', '∞'],      # Tunneling Flow
            'diffraction_flow': ['🎯', '│', '∞'],     # Diffraction Flow
            'resonance_flow': ['🎵', '📈', '∞']       # Resonance Flow
        }
        
    def get_interference(self, name: str) -> Dict:
        """Get interference set"""
        return self.wave_sets['interference'].get(name, None)
        
    def get_superposition(self, name: str) -> Dict:
        """Get superposition set"""
        return self.wave_sets['superposition'].get(name, None)
        
    def get_tunneling(self, name: str) -> Dict:
        """Get tunneling set"""
        return self.wave_sets['tunneling'].get(name, None)
        
    def get_diffraction(self, name: str) -> Dict:
        """Get diffraction set"""
        return self.wave_sets['diffraction'].get(name, None)
        
    def get_resonance(self, name: str) -> Dict:
        """Get resonance set"""
        return self.wave_sets['resonance'].get(name, None)
        
    def get_wave_flow(self, flow: str) -> List[str]:
        """Get wave flow sequence"""
        return self.wave_flows.get(flow, None)
