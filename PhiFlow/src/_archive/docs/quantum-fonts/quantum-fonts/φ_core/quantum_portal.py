from typing import Dict, List, Tuple
import colorsys

class QuantumPortal:
    def __init__(self):
        self.φ = (1 + 5**0.5) / 2
        self.initialize_portal_sets()
        
    def initialize_portal_sets(self):
        """Initialize quantum portal sets with icons and colors"""
        self.portal_sets = {
            # Portal (432 Hz) 🌀
            'portal': {
                'vortex': {
                    'icons': ['🌀', '💫', '∞'],          # Spiral + Sparkle + Infinity
                    'spin': ['↺', '↻', '⟳'],          # Vortex Spin
                    'colors': {'primary': '#4B0082', 'glow': '#8A2BE2'}
                },
                'tunnel': {
                    'icons': ['🌀', '🕳️', '∞'],          # Spiral + Hole + Infinity
                    'depth': ['⚫', '◎', '○'],         # Tunnel Depth
                    'colors': {'primary': '#9932CC', 'glow': '#BA55D3'}
                },
                'bridge': {
                    'icons': ['🌀', '🌈', '∞'],          # Spiral + Rainbow + Infinity
                    'paths': ['↝', '⇝', '⟿'],         # Bridge Paths
                    'colors': {'primary': '#800080', 'glow': '#9370DB'}
                }
            },
            
            # Gateway (528 Hz) 🚪
            'gateway': {
                'doorway': {
                    'icons': ['🚪', '🔮', '∞'],          # Door + Crystal + Infinity
                    'frames': ['⊏', '⊐', '⊓'],         # Door Frames
                    'colors': {'primary': '#191970', 'glow': '#000080'}
                },
                'stargate': {
                    'icons': ['🚪', '🌟', '∞'],          # Door + Star + Infinity
                    'rings': ['◌', '◎', '⊕'],         # Star Rings
                    'colors': {'primary': '#4169E1', 'glow': '#1E90FF'}
                },
                'threshold': {
                    'icons': ['🚪', '✨', '∞'],          # Door + Sparkle + Infinity
                    'boundaries': ['│', '┃', '❘'],     # Thresholds
                    'colors': {'primary': '#00BFFF', 'glow': '#87CEEB'}
                }
            },
            
            # Activation (768 Hz) 🔑
            'activation': {
                'keys': {
                    'icons': ['🔑', '✨', '∞'],          # Key + Sparkle + Infinity
                    'codes': ['α', 'ω', '∞'],         # Key Codes
                    'colors': {'primary': '#4B0082', 'glow': '#8A2BE2'}
                },
                'sequence': {
                    'icons': ['🔑', '🔢', '∞'],          # Key + Numbers + Infinity
                    'patterns': ['123', '789', '∞'],   # Key Sequences
                    'colors': {'primary': '#800080', 'glow': '#9370DB'}
                },
                'resonance': {
                    'icons': ['🔑', '🎵', '∞'],          # Key + Music + Infinity
                    'frequencies': ['432', '528', '768'], # Key Frequencies
                    'colors': {'primary': '#483D8B', 'glow': '#6A5ACD'}
                }
            },
            
            # Transport (999 Hz) 🌠
            'transport': {
                'teleport': {
                    'icons': ['🌠', '⚡', '∞'],          # Shooting Star + Energy + Infinity
                    'jump': ['↯', '⇋', '⇌'],          # Teleport Jump
                    'colors': {'primary': '#4B0082', 'glow': '#8A2BE2'}
                },
                'wormhole': {
                    'icons': ['🌠', '🕳️', '∞'],          # Shooting Star + Hole + Infinity
                    'tunnel': ['⊶', '⊷', '⋈'],        # Wormhole Tunnel
                    'colors': {'primary': '#191970', 'glow': '#000080'}
                },
                'quantum': {
                    'icons': ['🌠', '⚛️', '∞'],          # Shooting Star + Atom + Infinity
                    'leap': ['⇄', '⇆', '⇅'],         # Quantum Leap
                    'colors': {'primary': '#800080', 'glow': '#9370DB'}
                }
            },
            
            # Dimension (∞ Hz) 🌌
            'dimension': {
                'space': {
                    'icons': ['🌌', '🌍', '∞'],          # Galaxy + Earth + Infinity
                    'coords': ['xyz', '4D', '5D'],     # Space Dimensions
                    'colors': {'primary': '#4B0082', 'glow': '#8A2BE2'}
                },
                'time': {
                    'icons': ['🌌', '⏳', '∞'],          # Galaxy + Time + Infinity
                    'flow': ['⟲', '⟳', '∞'],          # Time Flow
                    'colors': {'primary': '#000080', 'glow': '#0000CD'}
                },
                'reality': {
                    'icons': ['🌌', '🎲', '∞'],          # Galaxy + Dice + Infinity
                    'planes': ['α', 'Ω', '∞'],        # Reality Planes
                    'colors': {'primary': '#800080', 'glow': '#9370DB'}
                }
            }
        }
        
        # Portal Flows
        self.portal_flows = {
            'portal_flow': ['🌀', '💫', '∞'],         # Portal Flow
            'gateway_flow': ['🚪', '🔮', '∞'],        # Gateway Flow
            'activation_flow': ['🔑', '✨', '∞'],      # Activation Flow
            'transport_flow': ['🌠', '⚡', '∞'],       # Transport Flow
            'dimension_flow': ['🌌', '🌍', '∞']        # Dimension Flow
        }
        
    def get_portal(self, name: str) -> Dict:
        """Get portal set"""
        return self.portal_sets['portal'].get(name, None)
        
    def get_gateway(self, name: str) -> Dict:
        """Get gateway set"""
        return self.portal_sets['gateway'].get(name, None)
        
    def get_activation(self, name: str) -> Dict:
        """Get activation set"""
        return self.portal_sets['activation'].get(name, None)
        
    def get_transport(self, name: str) -> Dict:
        """Get transport set"""
        return self.portal_sets['transport'].get(name, None)
        
    def get_dimension(self, name: str) -> Dict:
        """Get dimension set"""
        return self.portal_sets['dimension'].get(name, None)
        
    def get_portal_flow(self, flow: str) -> List[str]:
        """Get portal flow sequence"""
        return self.portal_flows.get(flow, None)
