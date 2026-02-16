from typing import Dict, List, Tuple
import colorsys

class QuantumMoods:
    def __init__(self):
        self.φ = (1 + 5**0.5) / 2
        self.initialize_mood_sets()
        
    def initialize_mood_sets(self):
        """Initialize quantum mood sets with icons and colors"""
        self.mood_sets = {
            # Joyful Flow (528 Hz) 🌈
            'joy': {
                'icons': {
                    'pure_joy': ['😊', '✨', '🌟'],           # Smile + Sparkles + Star
                    'playful': ['🦋', '🌈', '🎨'],           # Butterfly + Rainbow + Art
                    'celebration': ['🎉', '💫', '🎊'],       # Party + Stars + Confetti
                    'dance': ['💃', '🕺', '🎵'],             # Dance + Music
                    'laughter': ['😄', '🌞', '🎭']           # Laugh + Sun + Theater
                },
                'colors': {
                    'primary': '#FFD700',    # Gold joy
                    'secondary': '#FF69B4',  # Pink happiness
                    'accent': '#87CEEB',     # Sky blue freedom
                    'glow': '#FFFF00'        # Yellow radiance
                }
            },
            
            # Love Field (432 Hz) 💖
            'love': {
                'icons': {
                    'pure_love': ['💖', '✨', '🌟'],         # Heart + Sparkles + Star
                    'harmony': ['☯️', '🕊️', '🌸'],          # Balance + Peace + Flower
                    'connection': ['🤝', '💫', '🌈'],        # Hands + Stars + Rainbow
                    'gratitude': ['🙏', '💝', '✨'],         # Prayer + Heart + Sparkles
                    'healing': ['💗', '🌿', '🦋']           # Heart + Leaf + Butterfly
                },
                'colors': {
                    'primary': '#FF1493',    # Deep pink love
                    'secondary': '#FF69B4',  # Light pink heart
                    'accent': '#FFB6C1',     # Soft pink harmony
                    'glow': '#FFC0CB'        # Pink aura
                }
            },
            
            # Crystal Power (768 Hz) 💎
            'power': {
                'icons': {
                    'pure_power': ['⚡', '💎', '🌟'],        # Lightning + Crystal + Star
                    'strength': ['🦁', '💪', '👑'],         # Lion + Strong + Crown
                    'wisdom': ['🦉', '📚', '🔮'],           # Owl + Books + Crystal Ball
                    'mastery': ['🎯', '🏆', '⭐'],          # Target + Trophy + Star
                    'leadership': ['👑', '🌟', '⚡']        # Crown + Star + Lightning
                },
                'colors': {
                    'primary': '#9400D3',    # Royal purple
                    'secondary': '#8A2BE2',  # Blue violet
                    'accent': '#4B0082',     # Indigo power
                    'glow': '#E6E6FA'        # Light purple
                }
            },
            
            # Peace Flow (396 Hz) 🕊️
            'peace': {
                'icons': {
                    'pure_peace': ['🕊️', '☮️', '🌟'],       # Dove + Peace + Star
                    'serenity': ['🌊', '🌙', '✨'],         # Wave + Moon + Sparkles
                    'meditation': ['🧘', '🌸', '☯️'],       # Meditate + Flower + Yin-Yang
                    'nature': ['🌿', '🍃', '🌺'],           # Leaves + Nature + Flower
                    'calm': ['🌅', '🌊', '🌸']             # Sunset + Wave + Flower
                },
                'colors': {
                    'primary': '#87CEEB',    # Sky blue
                    'secondary': '#00BFFF',  # Deep blue
                    'accent': '#E0FFFF',     # Light cyan
                    'glow': '#F0F8FF'        # Alice blue
                }
            },
            
            # Magic Flow (444 Hz) ✨
            'magic': {
                'icons': {
                    'pure_magic': ['✨', '🌟', '🔮'],       # Sparkles + Star + Crystal Ball
                    'wonder': ['🦄', '🌈', '💫'],          # Unicorn + Rainbow + Stars
                    'dreams': ['🌙', '💫', '🦋'],          # Moon + Stars + Butterfly
                    'fantasy': ['🐉', '🌟', '🎭'],         # Dragon + Star + Theater
                    'enchant': ['🪄', '💫', '🌟']          # Wand + Stars + Star
                },
                'colors': {
                    'primary': '#FF69B4',    # Pink magic
                    'secondary': '#9400D3',  # Purple mystery
                    'accent': '#FFD700',     # Gold enchant
                    'glow': '#FF00FF'        # Magenta spark
                }
            },
            
            # Quantum Flow (∞ Hz) ⚛️
            'quantum': {
                'icons': {
                    'pure_quantum': ['⚛️', '∞', '🌟'],      # Atom + Infinity + Star
                    'evolution': ['🌀', '🐬', '💫'],        # Spiral + Dolphin + Stars
                    'creation': ['✨', '🎨', '🌈'],         # Sparkles + Art + Rainbow
                    'infinity': ['∞', '🌌', '💫'],          # Infinity + Galaxy + Stars
                    'transcend': ['🚀', '💫', '🌟']         # Rocket + Stars + Star
                },
                'colors': {
                    'primary': '#191970',    # Midnight blue
                    'secondary': '#483D8B',  # Dark slate blue
                    'accent': '#8A2BE2',     # Blue violet
                    'glow': '#E6E6FA'        # Lavender
                }
            }
        }
        
        # Fun Mood Transitions
        self.mood_flows = {
            'joy_to_love': ['😊', '💖', '✨'],
            'love_to_power': ['💖', '⚡', '💎'],
            'power_to_peace': ['⚡', '🕊️', '☮️'],
            'peace_to_magic': ['🕊️', '✨', '🔮'],
            'magic_to_quantum': ['✨', '⚛️', '∞']
        }
        
    def get_mood_set(self, mood: str) -> Dict:
        """Get complete mood set with icons and colors"""
        return self.mood_sets.get(mood, None)
        
    def get_mood_flow(self, transition: str) -> List[str]:
        """Get mood transition sequence"""
        return self.mood_flows.get(transition, None)
        
    def create_mood_combo(self, mood1: str, mood2: str) -> Dict:
        """Create custom mood combination"""
        mood1_set = self.mood_sets.get(mood1)
        mood2_set = self.mood_sets.get(mood2)
        
        if mood1_set and mood2_set:
            return {
                'icons': mood1_set['icons']['pure_' + mood1][:2] + 
                        mood2_set['icons']['pure_' + mood2][:2],
                'colors': {
                    'primary': mood1_set['colors']['primary'],
                    'secondary': mood2_set['colors']['primary'],
                    'accent': mood1_set['colors']['accent'],
                    'glow': mood2_set['colors']['glow']
                }
            }
        return None
