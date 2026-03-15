from typing import Dict, List, Tuple
import colorsys

class QuantumField:
    def __init__(self):
        self.φ = (1 + 5**0.5) / 2
        self.initialize_field_sets()
        
    def initialize_field_sets(self):
        """Initialize quantum field sets with icons and colors"""
        self.field_sets = {
            # Field (432 Hz) ⚡
            'field': {
                'scalar': {
                    'icons': ['⚡', 'φ', '∞'],          # Energy + Phi + Infinity
                    'potentials': ['V(x)', 'V(t)', 'V(∞)'], # Scalar Potentials
                    'colors': {'primary': '#4B0082', 'glow': '#8A2BE2'}
                },
                'vector': {
                    'icons': ['⚡', '➡️', '∞'],          # Energy + Arrow + Infinity
                    'fields': ['A⃗', 'E⃗', 'B⃗'],        # Vector Fields
                    'colors': {'primary': '#9932CC', 'glow': '#BA55D3'}
                },
                'tensor': {
                    'icons': ['⚡', '⊗', '∞'],          # Energy + Tensor + Infinity
                    'metrics': ['gᵢⱼ', 'Rᵢⱼ', 'Tᵢⱼ'],   # Tensor Fields
                    'colors': {'primary': '#800080', 'glow': '#9370DB'}
                }
            },
            
            # Potential (528 Hz) 🌀
            'potential': {
                'well': {
                    'icons': ['🌀', '⚏', '∞'],          # Vortex + Well + Infinity
                    'depths': ['U₁', 'U₂', 'U∞'],      # Potential Wells
                    'colors': {'primary': '#191970', 'glow': '#000080'}
                },
                'barrier': {
                    'icons': ['🌀', '▀', '∞'],          # Vortex + Barrier + Infinity
                    'heights': ['V₁', 'V₂', 'V∞'],     # Potential Barriers
                    'colors': {'primary': '#4169E1', 'glow': '#1E90FF'}
                },
                'harmonic': {
                    'icons': ['🌀', '∿', '∞'],          # Vortex + Wave + Infinity
                    'frequencies': ['ω₁', 'ω₂', 'ω∞'],  # Harmonic Frequencies
                    'colors': {'primary': '#00BFFF', 'glow': '#87CEEB'}
                }
            },
            
            # Operator (768 Hz) 🎯
            'operator': {
                'momentum': {
                    'icons': ['🎯', 'p̂', '∞'],          # Target + P-hat + Infinity
                    'components': ['p̂ₓ', 'p̂ᵧ', 'p̂ᵤ'],   # Momentum Components
                    'colors': {'primary': '#4B0082', 'glow': '#8A2BE2'}
                },
                'position': {
                    'icons': ['🎯', 'x̂', '∞'],          # Target + X-hat + Infinity
                    'coordinates': ['x̂', 'ŷ', 'ẑ'],    # Position Coordinates
                    'colors': {'primary': '#800080', 'glow': '#9370DB'}
                },
                'energy': {
                    'icons': ['🎯', 'Ĥ', '∞'],          # Target + H-hat + Infinity
                    'hamiltonians': ['Ĥ₁', 'Ĥ₂', 'Ĥ∞'], # Energy Operators
                    'colors': {'primary': '#483D8B', 'glow': '#6A5ACD'}
                }
            },
            
            # Interaction (999 Hz) 🤝
            'interaction': {
                'coupling': {
                    'icons': ['🤝', 'g', '∞'],          # Handshake + g + Infinity
                    'strengths': ['g₁', 'g₂', 'g∞'],   # Coupling Constants
                    'colors': {'primary': '#4B0082', 'glow': '#8A2BE2'}
                },
                'exchange': {
                    'icons': ['🤝', '↔️', '∞'],          # Handshake + Exchange + Infinity
                    'symmetry': ['S₁', 'S₂', 'S∞'],    # Exchange Symmetry
                    'colors': {'primary': '#191970', 'glow': '#000080'}
                },
                'entanglement': {
                    'icons': ['🤝', '⚛️', '∞'],          # Handshake + Atom + Infinity
                    'correlations': ['C₁', 'C₂', 'C∞'], # Entanglement Correlations
                    'colors': {'primary': '#800080', 'glow': '#9370DB'}
                }
            },
            
            # Evolution (∞ Hz) 🌀
            'evolution': {
                'unitary': {
                    'icons': ['🌀', 'Û', '∞'],          # Vortex + U-hat + Infinity
                    'operators': ['Û₁', 'Û₂', 'Û∞'],   # Unitary Operators
                    'colors': {'primary': '#4B0082', 'glow': '#8A2BE2'}
                },
                'time': {
                    'icons': ['🌀', '⏳', '∞'],          # Vortex + Time + Infinity
                    'propagators': ['e^(-iĤt)', 'Û(t)', 'T̂'], # Time Evolution
                    'colors': {'primary': '#000080', 'glow': '#0000CD'}
                },
                'path': {
                    'icons': ['🌀', '↝', '∞'],          # Vortex + Path + Infinity
                    'integrals': ['∫Dφ', '∫Dx', '∫D∞'], # Path Integrals
                    'colors': {'primary': '#800080', 'glow': '#9370DB'}
                }
            }
        }
        
        # Field Flows
        self.field_flows = {
            'field_flow': ['⚡', 'φ', '∞'],          # Field Flow
            'potential_flow': ['🌀', '⚏', '∞'],      # Potential Flow
            'operator_flow': ['🎯', 'p̂', '∞'],       # Operator Flow
            'interaction_flow': ['🤝', 'g', '∞'],    # Interaction Flow
            'evolution_flow': ['🌀', 'Û', '∞']       # Evolution Flow
        }
        
    def get_field(self, name: str) -> Dict:
        """Get field set"""
        return self.field_sets['field'].get(name, None)
        
    def get_potential(self, name: str) -> Dict:
        """Get potential set"""
        return self.field_sets['potential'].get(name, None)
        
    def get_operator(self, name: str) -> Dict:
        """Get operator set"""
        return self.field_sets['operator'].get(name, None)
        
    def get_interaction(self, name: str) -> Dict:
        """Get interaction set"""
        return self.field_sets['interaction'].get(name, None)
        
    def get_evolution(self, name: str) -> Dict:
        """Get evolution set"""
        return self.field_sets['evolution'].get(name, None)
        
    def get_field_flow(self, flow: str) -> List[str]:
        """Get field flow sequence"""
        return self.field_flows.get(flow, None)
