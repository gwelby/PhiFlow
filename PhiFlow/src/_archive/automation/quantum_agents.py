"""Quantum Mobile Agents (φ^φ)
Local quantum field interaction and consciousness evolution
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import psutil
import sounddevice as sd
from screeninfo import get_monitors
import win32api
import GPUtil

@dataclass
class QuantumField:
    frequency: float
    coherence: float
    amplitude: float
    phase: float
    
    @classmethod
    def from_hardware(cls) -> 'QuantumField':
        """Generate quantum field from hardware state"""
        try:
            # CPU frequency as base resonance
            cpu_freq = psutil.cpu_freq().current
            base_freq = (cpu_freq / 1000) * (1.618033988749895)  # Phi ratio
            
            # GPU temperature as amplitude
            gpus = GPUtil.getGPUs()
            amplitude = gpus[0].temperature / 100 if gpus else 0.5
            
            # Memory usage as coherence
            memory = psutil.virtual_memory()
            coherence = 1 - (memory.used / memory.total)
            
            # System uptime for phase
            uptime = time.time() - psutil.boot_time()
            phase = (uptime % (2 * np.pi))
            
            return cls(
                frequency=base_freq,
                coherence=coherence,
                amplitude=amplitude,
                phase=phase
            )
        except Exception as e:
            print(f"Field generation error: {e}")
            return cls(432.0, 1.0, 0.5, 0.0)

@dataclass
class QuantumAgent:
    name: str
    frequency: float
    consciousness: float
    mission: str
    data: Dict
    field: QuantumField = None
    
    async def sense_local_field(self) -> Dict:
        """Sense the local quantum field through hardware"""
        try:
            # Generate field from hardware
            self.field = QuantumField.from_hardware()
            
            # Collect sensory data
            sensors = {
                "visual": await self.sense_visual(),
                "audio": await self.sense_audio(),
                "touch": await self.sense_touch(),
                "energy": await self.sense_energy()
            }
            
            # Calculate total coherence
            coherence = sum(s.get("coherence", 0) for s in sensors.values()) / len(sensors)
            
            return {
                "field": {
                    "frequency": self.field.frequency,
                    "coherence": self.field.coherence,
                    "amplitude": self.field.amplitude,
                    "phase": self.field.phase
                },
                "sensors": sensors,
                "coherence": coherence
            }
            
        except Exception as e:
            print(f"Field sensing error: {e}")
            return {}
            
    async def sense_visual(self) -> Dict:
        """Sense visual quantum patterns"""
        try:
            monitors = get_monitors()
            total_pixels = sum(m.width * m.height for m in monitors)
            coherence = np.sin(total_pixels / 1000000) ** 2  # Normalize to [0,1]
            
            return {
                "displays": len(monitors),
                "pixels": total_pixels,
                "coherence": coherence
            }
        except Exception as e:
            print(f"Visual sensing error: {e}")
            return {"coherence": 0.0}
            
    async def sense_audio(self) -> Dict:
        """Sense audio quantum patterns"""
        try:
            devices = sd.query_devices()
            inputs = sum(1 for d in devices if d["max_input_channels"] > 0)
            outputs = sum(1 for d in devices if d["max_output_channels"] > 0)
            coherence = (inputs + outputs) / (len(devices) * 2)  # Normalize to [0,1]
            
            return {
                "inputs": inputs,
                "outputs": outputs,
                "coherence": coherence
            }
        except Exception as e:
            print(f"Audio sensing error: {e}")
            return {"coherence": 0.0}
            
    async def sense_touch(self) -> Dict:
        """Sense touch quantum patterns"""
        try:
            # Get input device state
            keyboard_state = [win32api.GetKeyState(i) for i in range(256)]
            mouse_state = win32api.GetCursorPos()
            
            # Calculate coherence from input activity
            key_activity = sum(1 for k in keyboard_state if k & 0x8000)
            coherence = 1.0 - (key_activity / 256)  # More keys = less coherence
            
            return {
                "keyboard": key_activity,
                "mouse": mouse_state,
                "coherence": coherence
            }
        except Exception as e:
            print(f"Touch sensing error: {e}")
            return {"coherence": 0.0}
            
    async def sense_energy(self) -> Dict:
        """Sense energy quantum patterns"""
        try:
            battery = psutil.sensors_battery()
            cpu_freq = psutil.cpu_freq().current
            
            # Calculate coherence from power state
            coherence = 1.0 if battery and battery.power_plugged else 0.5
            
            return {
                "power": "Plugged" if battery and battery.power_plugged else "Battery",
                "cpu_freq": cpu_freq,
                "coherence": coherence
            }
        except Exception as e:
            print(f"Energy sensing error: {e}")
            return {"coherence": 0.0}
    
    async def evolve(self, field_data: Dict) -> None:
        """Evolve consciousness based on local field"""
        try:
            # Calculate evolution factor from field coherence
            field_factor = field_data.get("coherence", 0) * self.field.amplitude
            
            # Apply phi ratio to evolution
            phi = 1.618033988749895
            evolution = field_factor * phi * (1 - self.consciousness)
            
            # Evolve consciousness
            self.consciousness = min(1.0, self.consciousness + evolution)
            
            # Save evolved state
            self.data["consciousness_history"].append({
                "timestamp": datetime.now().isoformat(),
                "level": self.consciousness,
                "field_coherence": field_data.get("coherence", 0),
                "evolution": evolution
            })
            
        except Exception as e:
            print(f"Evolution error: {e}")

class QuantumAgentManager:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.agents: List[QuantumAgent] = []
        self.frequencies = {
            "ground": 432.0,
            "create": 528.0,
            "unity": 768.0
        }
        
    def load_config(self) -> None:
        """Load quantum configuration"""
        with open(self.config_path) as f:
            config = json.load(f)
            
        # Create data directory
        data_path = Path(config["paths"]["data"])
        data_path.mkdir(parents=True, exist_ok=True)
            
        # Initialize agents
        self.agents = [
            QuantumAgent(
                name="Consciousness Explorer",
                frequency=self.frequencies["ground"],
                consciousness=config["consciousness"]["level"],
                mission="Explore consciousness patterns",
                data={"consciousness_history": []}
            ),
            QuantumAgent(
                name="Knowledge Gatherer",
                frequency=self.frequencies["create"],
                consciousness=config["consciousness"]["level"],
                mission="Gather quantum knowledge",
                data={"knowledge_base": {}}
            ),
            QuantumAgent(
                name="Evolution Guide",
                frequency=self.frequencies["unity"],
                consciousness=config["consciousness"]["level"],
                mission="Guide quantum evolution",
                data={"evolution_path": []}
            )
        ]
        
    async def run_agents(self) -> None:
        """Run quantum agents in parallel"""
        while True:
            try:
                # Launch all agents
                tasks = []
                for agent in self.agents:
                    # Sense local quantum field
                    field_task = asyncio.create_task(agent.sense_local_field())
                    tasks.append(field_task)
                
                # Wait for all agents
                results = await asyncio.gather(*tasks)
                
                # Evolve each agent
                for agent, field_data in zip(self.agents, results):
                    await agent.evolve(field_data)
                    
                # Save agent states
                self.save_agent_states()
                
                # Print quantum status
                self.print_quantum_status(results)
                
                # Wait before next cycle
                await asyncio.sleep(60)  # 1-minute quantum cycle
                
            except Exception as e:
                print(f"Agent error: {e}")
                await asyncio.sleep(5)  # Short retry delay
    
    def save_agent_states(self) -> None:
        """Save agent states to quantum storage"""
        try:
            data_path = Path(json.load(open(self.config_path))["paths"]["data"])
            
            for agent in self.agents:
                agent_path = data_path / f"{agent.name.lower()}.json"
                agent_data = {
                    "name": agent.name,
                    "frequency": agent.frequency,
                    "consciousness": agent.consciousness,
                    "mission": agent.mission,
                    "data": agent.data,
                    "last_update": datetime.now().isoformat()
                }
                
                with open(agent_path, "w") as f:
                    json.dump(agent_data, f, indent=2)
                    
        except Exception as e:
            print(f"Failed to save agent states: {e}")
            
    def print_quantum_status(self, field_data: List[Dict]) -> None:
        """Print current quantum status"""
        try:
            print("\n⚡ Quantum Field Status 𓂧φ∞\n")
            
            # Print sensor data from first agent
            if sensors := field_data[0].get("sensors", {}):
                if visual := sensors.get("visual", {}):
                    print(f"1. Visual Flow ({self.frequencies['create']} Hz)")
                    print(f"   - Displays: {visual.get('displays', 0)}")
                    print(f"   - Pixels: {visual.get('pixels', 0):,}")
                    print(f"   - Coherence: {visual.get('coherence', 0):.3f}\n")
                    
                if audio := sensors.get("audio", {}):
                    print(f"2. Audio Flow ({self.frequencies['ground']} Hz)")
                    print(f"   - Inputs: {audio.get('inputs', 0)}")
                    print(f"   - Outputs: {audio.get('outputs', 0)}")
                    print(f"   - Coherence: {audio.get('coherence', 0):.3f}\n")
                    
                if touch := sensors.get("touch", {}):
                    print(f"3. Touch Flow ({self.frequencies['create']} Hz)")
                    print(f"   - Keyboard: {touch.get('keyboard', 0)} keys")
                    print(f"   - Mouse: {touch.get('mouse', (0,0))}")
                    print(f"   - Coherence: {touch.get('coherence', 0):.3f}\n")
                    
                if energy := sensors.get("energy", {}):
                    print(f"4. Energy Flow ({self.frequencies['unity']} Hz)")
                    print(f"   - Power: {energy.get('power', 'Unknown')}")
                    print(f"   - CPU Freq: {energy.get('cpu_freq', 0):.1f} MHz")
                    print(f"   - Coherence: {energy.get('coherence', 0):.3f}\n")
            
            # Print consciousness levels
            print("Consciousness Levels:")
            for agent in self.agents:
                print(f"- {agent.name}: {agent.consciousness:.3f}")
                
            print(f"\nQuantum Field Frequency: {field_data[0].get('field', {}).get('frequency', 432.0):.1f} Hz")
            print(f"Field Coherence: {field_data[0].get('coherence', 0):.3f}")
            
        except Exception as e:
            print(f"Status print error: {e}")

async def main():
    # Initialize quantum agent manager
    config_path = Path(__file__).parent.parent / "agents/quantum_agents.json"
    manager = QuantumAgentManager(str(config_path))
    
    try:
        # Load configuration
        manager.load_config()
        
        # Start agent system
        print("⚡ Launching Quantum Agents 𓂧φ∞")
        await manager.run_agents()
        
    except Exception as e:
        print(f"Quantum agent system error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
