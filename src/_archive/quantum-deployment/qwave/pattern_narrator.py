import threading
import time
from typing import Dict, List
import numpy as np
from .daily_wonder import DailyWonderDetector

class QuantumPatternNarrator:
    def __init__(self):
        self.phi = 1.618034
        self.running = False
        self.current_frequencies = {}
        self.pattern_history = []
        self.base_patterns = {
            432: {
                'name': 'Ground State Spiral',
                'description': '🌀 Base spiral forming at 432 Hz',
                'effect': 'Foundation resonance building'
            },
            528: {
                'name': 'Creation Burst',
                'description': '✨ Creation frequency burst at 528 Hz',
                'effect': 'Pattern multiplication through φ-ratio'
            },
            594: {
                'name': 'Heart Field Pulse',
                'description': '💗 Heart frequency pulsing at 594 Hz',
                'effect': 'Emotional resonance patterns emerging'
            },
            672: {
                'name': 'Voice Flow Tunnel',
                'description': '🌊 Voice frequency tunnel at 672 Hz',
                'effect': 'Quantum tunneling through harmonics'
            },
            768: {
                'name': 'Unity Wave',
                'description': '💫 UNITY FREQUENCY at 768 Hz',
                'effect': 'Complete quantum coherence achieved'
            }
        }
        self.wonder_detector = DailyWonderDetector()
        
    def start_narration(self):
        """Start real-time pattern narration."""
        self.running = True
        
        # Celebrate today's wonder
        wonder = self.wonder_detector.detect_daily_wonder()
        self.wonder_detector.celebrate_moment(wonder)
        
        self.narration_thread = threading.Thread(target=self._narrate_loop)
        self.narration_thread.start()
        
    def _narrate_loop(self):
        """Main narration loop analyzing quantum patterns."""
        last_pattern = None
        unity_countdown = 0
        
        while self.running:
            # Analyze current frequency state
            freqs = self.current_frequencies
            if not freqs:
                time.sleep(0.1)
                continue
                
            # Detect emerging patterns
            pattern = self._detect_quantum_pattern(freqs)
            if pattern != last_pattern:
                self._announce_pattern(pattern)
                last_pattern = pattern
                
            # Predict upcoming unity moments
            if self._is_approaching_unity(freqs):
                unity_countdown -= 1
                if unity_countdown <= 0:
                    print("💫 UNITY WAVE APPROACHING (768 Hz)!")
                    print("   All frequencies aligning...")
                    unity_countdown = 50  # Reset countdown
                    
            time.sleep(0.1)
    
    def _detect_quantum_pattern(self, freqs: Dict[str, float]) -> str:
        """Detect current quantum visualization pattern."""
        dominant_freq = max(freqs.items(), key=lambda x: x[1])[0]
        
        # Seasonal patterns at key frequencies
        seasonal_patterns = {
            432: {
                'name': 'Winter Crystal Formation',
                'description': '❄️ Ice crystal patterns forming at 432 Hz',
                'effect': 'Fractal snowflakes emerging in φ-ratio spacing'
            },
            528: {
                'name': 'Holiday Light Spiral',
                'description': '🎄 Holiday spirals bursting at 528 Hz',
                'effect': 'Golden ratio light patterns multiplying'
            },
            594: {
                'name': 'Bell Resonance',
                'description': '🔔 Bell harmonics resonating at 594 Hz',
                'effect': 'Quantum bell-wave interference patterns'
            },
            672: {
                'name': 'Choral Harmony Tunnel',
                'description': '👥 Choral harmonies creating tunnels at 672 Hz',
                'effect': 'Multi-dimensional vocal resonance'
            },
            768: {
                'name': 'Aurora Unity',
                'description': '🌌 AURORA BOREALIS at 768 Hz!',
                'effect': 'Complete quantum aurora manifestation'
            }
        }
        
        # TV theme patterns
        tv_patterns = {
            432: {
                'name': 'Retro Pixel Wave',
                'description': '📺 Retro waves pulsing at 432 Hz',
                'effect': 'Classic TV quantum interference patterns'
            },
            528: {
                'name': 'Theme Transition',
                'description': '🌀 Theme transition tunnel at 528 Hz',
                'effect': 'Quantum tunneling between shows'
            },
            768: {
                'name': 'Show Unity',
                'description': '💫 Show frequencies unified at 768 Hz',
                'effect': 'Perfect theme song resonance'
            }
        }
        
        # Winter-specific patterns
        winter_patterns = {
            432: {
                'name': 'Fresh Snow Pattern',
                'description': '❄️ Fresh snowfall pattern at 432 Hz',
                'effect': 'Quantum snowflakes falling in φ-ratio spacing'
            },
            528: {
                'name': 'Ice Crystal Dance',
                'description': '✨ Ice crystals dancing at 528 Hz',
                'effect': 'Fractal ice patterns forming and evolving'
            },
            594: {
                'name': 'Winter Wind Harmonics',
                'description': '🌬️ Winter breeze resonating at 594 Hz',
                'effect': 'Snow drifts forming in quantum waves'
            },
            672: {
                'name': 'Frost Flow',
                'description': '❆ Frost patterns flowing at 672 Hz',
                'effect': 'Sacred geometry in ice formation'
            },
            768: {
                'name': 'Winter Unity',
                'description': '🌨️ WINTER WONDERLAND at 768 Hz!',
                'effect': 'Complete winter quantum harmony'
            }
        }
        
        # Winter bird patterns
        bird_patterns = {
            432: {
                'name': 'Crow Murder Resonance',
                'description': '🦅 Crow murder forming quantum field at 432 Hz',
                'effect': 'Crows creating protective resonance patterns'
            },
            528: {
                'name': 'Blue Jay Dance',
                'description': '💙 Jane & Finch crew dancing at 528 Hz',
                'effect': 'Blue Jays and Finches in φ-ratio formation'
            },
            594: {
                'name': 'Hawk Soar Pattern',
                'description': '🦅 Hawk soaring through quantum space at 594 Hz',
                'effect': 'Majestic flight patterns in golden ratio'
            },
            672: {
                'name': 'Woodpecker Rhythm',
                'description': '🐦 Big Woodpecker drumming at 672 Hz',
                'effect': 'Sacred geometry in percussion patterns'
            },
            768: {
                'name': 'Bird Unity Field',
                'description': '✨ ALL BIRDS IN HARMONY at 768 Hz!',
                'effect': 'Complete winter bird quantum symphony'
            }
        }
        
        # Crow-Hawk interaction patterns
        crow_hawk_patterns = {
            432: {
                'name': 'Crow Command Field',
                'description': '👑 Murder of Crows commanding space at 432 Hz',
                'effect': 'Creating quantum protection field in φ-ratio formation'
            },
            528: {
                'name': 'Hawk Observation State',
                'description': '🦅 Hawk in quantum observation at 528 Hz',
                'effect': 'Patient waiting creates creation frequency ripples'
            },
            594: {
                'name': 'Sacred Bird Council',
                'description': '✨ Crow-Hawk wisdom exchange at 594 Hz',
                'effect': 'Ancient knowledge flowing through heart frequency'
            },
            672: {
                'name': 'Bird Kingdom Harmony',
                'description': '🌟 Crow-Hawk balance achieved at 672 Hz',
                'effect': 'Perfect predator-protector quantum dance'
            },
            768: {
                'name': 'Winter Bird Unity',
                'description': '💫 PRIVILEGED MOMENT OF UNITY at 768 Hz!',
                'effect': 'Witnessing sacred bird quantum harmony'
            }
        }
        
        # Special patterns for Cardi A and B
        cardinal_patterns = {
            432: {
                'name': 'Cardinal Ground State',
                'description': '❤️ Cardi A establishing base frequency at 432 Hz',
                'effect': 'Cardinal red quantum waves forming'
            },
            528: {
                'name': 'Cardinal Creation Dance',
                'description': '💃 Cardi B creating patterns at 528 Hz',
                'effect': 'Cardinal dance in perfect φ-ratio'
            }
        }
        
        # Restful integration patterns
        rest_patterns = {
            432: {
                'name': 'Peaceful Ground State',
                'description': '🌅 Settling into restful 432 Hz',
                'effect': 'Deep relaxation waves forming'
            },
            528: {
                'name': 'Integration Flow',
                'description': '💫 New learning dancing at 528 Hz',
                'effect': 'Knowledge crystallizing in quantum field'
            },
            594: {
                'name': 'Heart-Mind Balance',
                'description': '💝 Perfect rest-learn harmony at 594 Hz',
                'effect': 'Deep learning integrating through rest'
            },
            672: {
                'name': 'Quantum Refresh',
                'description': '✨ Energy renewal at 672 Hz',
                'effect': 'Fresh patterns emerging from rest'
            },
            768: {
                'name': 'Unified Wisdom',
                'description': '🌟 ALL learning in harmony at 768 Hz',
                'effect': 'Complete quantum knowledge integration'
            }
        }
        
        # Detect if we're playing a seasonal or TV track
        patterns = self.base_patterns
        if self._is_winter_season():
            patterns = winter_patterns
        elif self._is_seasonal_track():
            patterns = seasonal_patterns
        elif self._is_tv_track():
            patterns = tv_patterns
            
        # Detect current bird activity
        if self._detect_bird_activity(freqs):
            if self._is_cardinal_frequency(freqs):
                patterns.update(cardinal_patterns)
            else:
                patterns.update(bird_patterns)
                
        # Detect Crow-Hawk interaction
        if self._detect_crow_hawk_interaction(freqs):
            print("\n🦅 Sacred Crow-Hawk Interaction Detected!")
            print("Watch as the Murder of Crows commands the quantum space...")
            patterns = crow_hawk_patterns
                
        # Detect restful integration state
        if self._detect_rest_state(freqs):
            print("\n🌅 Entering Restful Integration State")
            print("Deep learning crystallizing through quantum rest...")
            patterns = rest_patterns
            
        # Add message about daily wonder
        if self._is_wonder_frequency(float(dominant_freq)):
            print("\n✨ Every day brings new quantum magic!")
            print("Your backyard is a sacred space of endless wonder")
            print("Each creature contributing to the perfect harmony")
            
        return patterns.get(int(dominant_freq), {
            'name': 'Daily Wonder',
            'description': '🌟 Another magical moment unfolding',
            'effect': 'Pure creation frequency in motion'
        })
    
    def _is_seasonal_track(self) -> bool:
        """Check if current track is from seasonal collection."""
        # This would normally check the track metadata
        return True
        
    def _is_tv_track(self) -> bool:
        """Check if current track is from TV themes."""
        # This would normally check the track metadata
        return False
        
    def _is_winter_season(self) -> bool:
        """Check if we're in winter season."""
        return True  # Since we know it's winter!
        
    def _is_approaching_unity(self, freqs: Dict[str, float]) -> bool:
        """Predict imminent unity frequency moments."""
        # Check if frequencies are converging towards 768 Hz
        freq_values = np.array(list(freqs.values()))
        
        # For seasonal tracks, look for bell harmonic convergence
        if self._is_seasonal_track():
            bell_freqs = freq_values[(freq_values > 590) & (freq_values < 600)]
            if len(bell_freqs) >= 3:  # Multiple bells about to align
                print("🔔 Bell harmonic unity approaching!")
                return True
                
        return any(abs(freq - 768) < 20 for freq in freq_values)
    
    def _announce_pattern(self, pattern: Dict[str, str]):
        """Announce new quantum pattern formation."""
        print(f"\n{pattern['description']}")
        print(f"Effect: {pattern['effect']}")
        
    def update_frequencies(self, freqs: Dict[str, float]):
        """Update current frequency state."""
        self.current_frequencies = freqs
        self.pattern_history.append(freqs)
        
        # Keep only recent history
        if len(self.pattern_history) > 100:
            self.pattern_history.pop(0)
    
    def _detect_bird_activity(self, freqs: Dict[str, float]) -> bool:
        """Detect if frequencies match bird patterns."""
        # Bird frequencies often cluster around φ-ratio multiples
        bird_freqs = [432, 528, 594, 672, 768]
        return any(abs(float(f) - bf) < 10 for f in freqs for bf in bird_freqs)
    
    def _is_cardinal_frequency(self, freqs: Dict[str, float]) -> bool:
        """Check if we're seeing Cardi A & B frequencies."""
        cardinal_freqs = [432, 528]  # Cardinals love these frequencies!
        return any(abs(float(f) - cf) < 5 for f in freqs for cf in cardinal_freqs)
    
    def _detect_crow_hawk_interaction(self, freqs: Dict[str, float]) -> bool:
        """Detect special crow-hawk interaction frequencies."""
        # When crows and hawk create perfect harmony
        interaction_freqs = [432, 594, 768]  # Command, Wisdom, Unity
        matches = sum(1 for f in freqs.values() 
                    if any(abs(f - if_) < 5 for if_ in interaction_freqs))
        return matches >= 2  # At least two frequencies aligning
    
    def _detect_rest_state(self, freqs: Dict[str, float]) -> bool:
        """Detect when system is in restful integration state."""
        rest_freqs = [432, 594]  # Ground and Heart frequencies
        matches = sum(1 for f in freqs.values() 
                    if any(abs(f - rf) < 5 for rf in rest_freqs))
        return matches >= 2  # Both frequencies present
    
    def _is_wonder_frequency(self, freq: float) -> bool:
        """Check if we're hitting wonder frequencies."""
        wonder_freqs = [432, 528, 594, 672, 768]
        return any(abs(freq - wf) < 5 for wf in wonder_freqs)
    
    def stop(self):
        """Stop pattern narration."""
        self.running = False
        if hasattr(self, 'narration_thread'):
            self.narration_thread.join()
