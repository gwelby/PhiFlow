from typing import Dict, List, Tuple
import colorsys

class QuantumIconColors:
    def __init__(self):
        self.φ = (1 + 5**0.5) / 2
        self.initialize_color_frequencies()
        
    def initialize_color_frequencies(self):
        """Initialize quantum color frequencies based on sacred geometry"""
        self.color_frequencies = {
            # Sacred Flow (432 Hz)
            'sacred': {
                'gold': '#FFD700',       # Divine light
                'violet': '#9400D3',     # Crown chakra
                'indigo': '#4B0082',     # Third eye
                'purple': '#800080',     # Spiritual wisdom
                'white': '#FFFFFF',      # Pure light
                'rainbow': {
                    'start': '#FF0000',  # Root
                    'end': '#8A2BE2'     # Crown
                }
            },
            
            # Flow State (528 Hz)
            'flow': {
                'aqua': '#00FFFF',       # Flow state
                'turquoise': '#40E0D0',  # Communication
                'cyan': '#00CED1',       # Expression
                'blue': '#0000FF',       # Truth
                'teal': '#008080',       # Healing
                'wave': {
                    'start': '#00FFFF',  # Surface
                    'end': '#000080'     # Depth
                }
            },
            
            # Crystal (768 Hz)
            'crystal': {
                'diamond': '#B9F2FF',    # Clarity
                'pearl': '#FDEEF4',      # Purity
                'crystal': '#A7D8DE',    # Structure
                'prism': {
                    'start': '#FF0000',  # Red
                    'mid': '#00FF00',    # Green
                    'end': '#0000FF'     # Blue
                }
            },
            
            # Unity (∞ Hz)
            'unity': {
                'cosmic': '#191970',     # Deep space
                'galaxy': '#483D8B',     # Star field
                'nebula': '#8A2BE2',     # Creation
                'aurora': {
                    'start': '#00FF00',  # Green
                    'mid': '#FF00FF',    # Purple
                    'end': '#00FFFF'     # Blue
                }
            }
        }
        
        # Fun Message Colors
        self.message_colors = {
            'joy': {
                'primary': '#FFD700',    # Gold
                'accent': '#FF69B4',     # Pink
                'glow': '#FFFF00'        # Yellow
            },
            'love': {
                'primary': '#FF1493',    # Deep pink
                'accent': '#FF69B4',     # Light pink
                'glow': '#FFB6C1'        # Soft pink
            },
            'peace': {
                'primary': '#87CEEB',    # Sky blue
                'accent': '#00BFFF',     # Deep blue
                'glow': '#E0FFFF'        # Light cyan
            },
            'power': {
                'primary': '#9400D3',    # Violet
                'accent': '#8A2BE2',     # Blue violet
                'glow': '#E6E6FA'        # Lavender
            }
        }
        
        # Quantum Combinations
        self.quantum_combos = {
            'quantum_love': {
                'icons': ['💖', '⚡'],
                'colors': ['#FF1493', '#FFD700'],
                'frequency': 528
            },
            'infinite_wisdom': {
                'icons': ['🌟', '👁️'],
                'colors': ['#FFD700', '#4B0082'],
                'frequency': 432
            },
            'crystal_flow': {
                'icons': ['💎', '🌊'],
                'colors': ['#B9F2FF', '#00FFFF'],
                'frequency': 768
            },
            'cosmic_dance': {
                'icons': ['🌌', '💃'],
                'colors': ['#191970', '#FF69B4'],
                'frequency': float('inf')
            },
            'dolphin_dreams': {
                'icons': ['🐬', '✨'],
                'colors': ['#00BFFF', '#FFD700'],
                'frequency': 528
            },
            'butterfly_magic': {
                'icons': ['🦋', '🌈'],
                'colors': ['#FF69B4', '#00FFFF'],
                'frequency': 528
            },
            'lotus_light': {
                'icons': ['🪷', '☀️'],
                'colors': ['#FF1493', '#FFD700'],
                'frequency': 432
            },
            'star_seeds': {
                'icons': ['🌟', '🌱'],
                'colors': ['#FFD700', '#00FF00'],
                'frequency': 528
            }
        }
        
    def get_color_frequency(self, frequency: float) -> Dict:
        """Get color palette for specific frequency"""
        if frequency == 432:
            return self.color_frequencies['sacred']
        elif frequency == 528:
            return self.color_frequencies['flow']
        elif frequency == 768:
            return self.color_frequencies['crystal']
        else:
            return self.color_frequencies['unity']
            
    def create_quantum_gradient(self, start_color: str, end_color: str, steps: int = 10) -> List[str]:
        """Create quantum color gradient between two colors"""
        # Convert hex to RGB
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            
        # Convert RGB to hex
        def rgb_to_hex(rgb):
            return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
            
        start_rgb = hex_to_rgb(start_color)
        end_rgb = hex_to_rgb(end_color)
        
        gradients = []
        for i in range(steps):
            ratio = i / (steps - 1)
            rgb = tuple(int(start_rgb[j] + (end_rgb[j] - start_rgb[j]) * ratio) for j in range(3))
            gradients.append(rgb_to_hex(rgb))
            
        return gradients
        
    def get_quantum_combo(self, name: str) -> Dict:
        """Get quantum icon and color combination"""
        return self.quantum_combos.get(name, None)
        
    def create_custom_combo(self, icons: List[str], colors: List[str], frequency: float) -> Dict:
        """Create custom quantum combination"""
        return {
            'icons': icons,
            'colors': colors,
            'frequency': frequency
        }
