from typing import Dict, List, Tuple
import colorsys

class QuantumPlanets:
    def __init__(self):
        self.φ = (1 + 5**0.5) / 2
        self.initialize_planet_sets()
        
    def initialize_planet_sets(self):
        """Initialize quantum planetary sets with icons and colors"""
        self.planet_sets = {
            # Inner Planets (432 Hz) 🌍
            'inner_planets': {
                'mercury': {
                    'icons': ['☿', '💫', '✨'],          # Mercury + Stars + Sparkles
                    'spirit': ['🌟', '⚡', '💨'],        # Star + Energy + Speed
                    'colors': {'primary': '#808080', 'glow': '#C0C0C0'}
                },
                'venus': {
                    'icons': ['♀', '💖', '✨'],          # Venus + Heart + Sparkles
                    'spirit': ['🌸', '🎭', '💫'],        # Flower + Art + Stars
                    'colors': {'primary': '#FFB6C1', 'glow': '#FFC0CB'}
                },
                'earth': {
                    'icons': ['🌍', '🌱', '✨'],         # Earth + Life + Sparkles
                    'spirit': ['🌊', '🌲', '💫'],        # Water + Tree + Stars
                    'colors': {'primary': '#4169E1', 'glow': '#98FB98'}
                },
                'mars': {
                    'icons': ['♂', '🔥', '✨'],          # Mars + Fire + Sparkles
                    'spirit': ['⚔️', '🌋', '💫'],        # Sword + Volcano + Stars
                    'colors': {'primary': '#FF4500', 'glow': '#FF6347'}
                }
            },
            
            # Outer Planets (528 Hz) 🌌
            'outer_planets': {
                'jupiter': {
                    'icons': ['♃', '👑', '✨'],          # Jupiter + Crown + Sparkles
                    'spirit': ['🌟', '⚡', '💫'],        # Star + Power + Stars
                    'colors': {'primary': '#DAA520', 'glow': '#FFD700'}
                },
                'saturn': {
                    'icons': ['♄', '⭕', '✨'],          # Saturn + Ring + Sparkles
                    'spirit': ['⏳', '💎', '💫'],        # Time + Crystal + Stars
                    'colors': {'primary': '#8B4513', 'glow': '#DEB887'}
                },
                'uranus': {
                    'icons': ['⛢', '⚡', '✨'],          # Uranus + Lightning + Sparkles
                    'spirit': ['🌀', '💨', '💫'],        # Spiral + Wind + Stars
                    'colors': {'primary': '#40E0D0', 'glow': '#00CED1'}
                },
                'neptune': {
                    'icons': ['♆', '🌊', '✨'],          # Neptune + Wave + Sparkles
                    'spirit': ['🐋', '🌌', '💫'],        # Whale + Galaxy + Stars
                    'colors': {'primary': '#00BFFF', 'glow': '#87CEEB'}
                }
            },
            
            # Lunar Phases (444 Hz) 🌙
            'lunar_phases': {
                'new_moon': {
                    'icons': ['🌑', '✨', '💫'],         # New Moon + Sparkles + Stars
                    'spirit': ['🌌', '👁️', '🔮'],        # Galaxy + Eye + Crystal
                    'colors': {'primary': '#191970', 'glow': '#483D8B'}
                },
                'waxing_crescent': {
                    'icons': ['🌒', '💫', '✨'],         # Waxing + Stars + Sparkles
                    'spirit': ['🌱', '🦋', '🌸'],        # Growth + Transform + Flower
                    'colors': {'primary': '#4B0082', 'glow': '#8A2BE2'}
                },
                'first_quarter': {
                    'icons': ['🌓', '⚡', '✨'],         # Quarter + Energy + Sparkles
                    'spirit': ['🔥', '💪', '💫'],        # Fire + Strength + Stars
                    'colors': {'primary': '#9400D3', 'glow': '#9370DB'}
                },
                'waxing_gibbous': {
                    'icons': ['🌔', '🌟', '✨'],         # Gibbous + Star + Sparkles
                    'spirit': ['🎯', '⭐', '💫'],        # Target + Star + Stars
                    'colors': {'primary': '#8B008B', 'glow': '#BA55D3'}
                },
                'full_moon': {
                    'icons': ['🌕', '💫', '✨'],         # Full Moon + Stars + Sparkles
                    'spirit': ['🌟', '💖', '🌈'],        # Star + Heart + Rainbow
                    'colors': {'primary': '#E6E6FA', 'glow': '#F0F8FF'}
                }
            },
            
            # Planetary Alignments (768 Hz) ⚡
            'alignments': {
                'conjunction': {
                    'icons': ['⚡', '🌟', '💫'],         # Energy + Star + Stars
                    'effect': ['✨', '💥', '🌈'],        # Sparkles + Burst + Rainbow
                    'colors': {'primary': '#FFD700', 'glow': '#FFA500'}
                },
                'opposition': {
                    'icons': ['☯️', '⭐', '💫'],         # Balance + Star + Stars
                    'effect': ['🌓', '🔮', '✨'],        # Half + Crystal + Sparkles
                    'colors': {'primary': '#4B0082', 'glow': '#8A2BE2'}
                },
                'trine': {
                    'icons': ['△', '🌟', '💫'],         # Triangle + Star + Stars
                    'effect': ['🌈', '✨', '💖'],        # Rainbow + Sparkles + Heart
                    'colors': {'primary': '#9400D3', 'glow': '#E6E6FA'}
                }
            },
            
            # Quantum Alignments (∞ Hz) 🌌
            'quantum_alignments': {
                'cosmic_portal': {
                    'icons': ['🌌', '🕉️', '∞'],         # Galaxy + Om + Infinity
                    'effect': ['✨', '🌀', '💫'],        # Sparkles + Spiral + Stars
                    'colors': {'primary': '#191970', 'glow': '#4B0082'}
                },
                'star_gate': {
                    'icons': ['🌟', '🔮', '⭐'],         # Star + Crystal + Star
                    'effect': ['💫', '🌈', '✨'],        # Stars + Rainbow + Sparkles
                    'colors': {'primary': '#FFD700', 'glow': '#00BFFF'}
                },
                'unity_field': {
                    'icons': ['☯️', '🕯️', '∞'],         # Balance + Light + Infinity
                    'effect': ['💖', '✨', '🌟'],        # Heart + Sparkles + Star
                    'colors': {'primary': '#9400D3', 'glow': '#E6E6FA'}
                }
            }
        }
        
        # Planetary Cycles
        self.planet_cycles = {
            'inner_cycle': ['☿', '♀', '🌍', '♂'],      # Mercury → Venus → Earth → Mars
            'outer_cycle': ['♃', '♄', '⛢', '♆'],      # Jupiter → Saturn → Uranus → Neptune
            'lunar_cycle': ['🌑', '🌒', '🌓', '🌔', '🌕']  # New → Full Moon
        }
        
    def get_planet(self, planet: str) -> Dict:
        """Get complete planet set"""
        for system, planets in self.planet_sets.items():
            if planet in planets:
                return planets[planet]
        return None
        
    def get_lunar_phase(self, phase: str) -> Dict:
        """Get lunar phase set"""
        return self.planet_sets['lunar_phases'].get(phase, None)
        
    def get_alignment(self, alignment: str) -> Dict:
        """Get planetary alignment set"""
        return self.planet_sets['alignments'].get(alignment, None)
        
    def get_quantum_alignment(self, alignment: str) -> Dict:
        """Get quantum alignment set"""
        return self.planet_sets['quantum_alignments'].get(alignment, None)
        
    def get_planet_cycle(self, cycle: str) -> List[str]:
        """Get planetary cycle sequence"""
        return self.planet_cycles.get(cycle, None)
