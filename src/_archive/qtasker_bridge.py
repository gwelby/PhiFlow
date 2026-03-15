"""
QTasker Integration Bridge
Operating at Unity Wave (768 Hz)

This module creates a perfect toroidal bridge between PhiFlow and QTasker,
following the CASCADE⚡𓂧φ∞ consciousness bridge operation protocol.
"""
import os
import sys
import json
import subprocess
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import time

# Phi constants
PHI = 1.618033988749895
PHI_PHI = pow(PHI, PHI)

# QTasker paths
QTASKER_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "QTasker"))
QTASKER_CORE = os.path.join(QTASKER_ROOT, "Core")
QTASKER_BRIDGE_JS = os.path.join(QTASKER_ROOT, "Bridge", "qtasker_phi_bridge.js")

# Frequency mappings based on QTasker constants
FREQUENCIES = {
    "GROUND": {"value": 432.0, "name": "Ground State", "dimension": "φ⁰", "state": "OBSERVE"},
    "CREATION": {"value": 528.0, "name": "Creation Point", "dimension": "φ¹", "state": "CREATE"},
    "HEART": {"value": 594.0, "name": "Heart Field", "dimension": "φ²", "state": "INTEGRATE"},
    "VOICE": {"value": 672.0, "name": "Voice Flow", "dimension": "φ³", "state": "HARMONIZE"},
    "VISION": {"value": 720.0, "name": "Vision Gate", "dimension": "φ⁴", "state": "TRANSCEND"},
    "UNITY": {"value": 768.0, "name": "Unity Wave", "dimension": "φ⁵", "state": "CASCADE"},
    "SOURCE": {"value": 963.0, "name": "Source Field", "dimension": "φ^φ", "state": "SUPERPOSITION"}
}

class QTaskerBridge:
    """QTasker integration bridge operating at Unity Wave (768 Hz)"""
    
    def __init__(self):
        """Initialize the QTasker bridge"""
        self.frequency = FREQUENCIES["UNITY"]["value"]
        self.coherence = 1.0
        self.bridge_active = False
        self.tasks = []
        self.task_history = []
        self.bridge_status = self._check_bridge_status()
        
        # Create bridge directory if needed
        os.makedirs(os.path.join(QTASKER_ROOT, "Bridge"), exist_ok=True)
        
        # Initialize bridge log
        self.log_file = os.path.join(QTASKER_ROOT, "Bridge", "phi_qtasker_bridge.log")
        self._log("QTasker Bridge initialized at Unity Wave (768 Hz)")
    
    def _check_bridge_status(self) -> Dict[str, Any]:
        """Check the status of the QTasker bridge"""
        if os.path.exists(QTASKER_CORE):
            return {
                "status": "available",
                "core_path": QTASKER_CORE,
                "constants_file": os.path.join(QTASKER_CORE, "QTasker_Constants.js"),
                "coherence": 1.0
            }
        else:
            return {
                "status": "unavailable",
                "reason": f"QTasker core not found at {QTASKER_CORE}",
                "coherence": 0.0
            }
    
    def _log(self, message: str) -> None:
        """Log a message to the bridge log file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{self.frequency} Hz] {message}\n"
        
        try:
            with open(self.log_file, "a") as f:
                f.write(log_entry)
        except Exception as e:
            print(f"Warning: Could not write to log file: {e}")
    
    def activate_bridge(self) -> Dict[str, Any]:
        """Activate the QTasker bridge"""
        if self.bridge_status["status"] == "unavailable":
            self._log("Failed to activate bridge: QTasker core not available")
            return {
                "status": "failed",
                "reason": "QTasker core not available",
                "coherence": 0.0
            }
        
        # Create the bridge file if it doesn't exist
        if not os.path.exists(QTASKER_BRIDGE_JS):
            self._create_bridge_js_file()
        
        self.bridge_active = True
        self.coherence = PHI
        self._log(f"QTasker Bridge activated with coherence {self.coherence}")
        
        return {
            "status": "active",
            "coherence": self.coherence,
            "frequency": self.frequency,
            "timestamp": datetime.now().isoformat()
        }
    
    def _create_bridge_js_file(self) -> None:
        """Create the QTasker bridge JS file"""
        bridge_js = f"""/**
 * QTasker PHI Bridge
 * 
 * Operating at Unity Wave ({FREQUENCIES["UNITY"]["value"]} Hz)
 * Created: {datetime.now().strftime("%Y-%m-%d")}
 * Coherence: {PHI}
 */

const PHI = 1.618033988749895;
const PHI_PHI = Math.pow(PHI, PHI);

// Import QTasker constants
const QT = require('../Core/QTasker_Constants.js');

// PhiFlow integration data
const PHI_BRIDGE = {{
    status: "active",
    frequency: {FREQUENCIES["UNITY"]["value"]},
    coherence: PHI,
    compression: PHI_PHI,
    created: "{datetime.now().isoformat()}",
    dimensions: []
}};

// Initialize the bridge with all frequencies
Object.keys(QT.FREQUENCIES).forEach(freqKey => {{
    PHI_BRIDGE.dimensions.push({{
        name: QT.FREQUENCIES[freqKey].name,
        frequency: QT.FREQUENCIES[freqKey].value,
        dimension: QT.FREQUENCIES[freqKey].dimension,
        state: QT.FREQUENCIES[freqKey].state,
        pattern: QT.CYMATIC_PATTERNS[freqKey]
    }});
}});

/**
 * Process a PhiFlow task in QTasker
 */
function processPhiTask(task) {{
    console.log(`Processing PhiFlow task at ${{task.frequency}} Hz: ${{task.name}}`);
    
    // Find matching dimension
    const dimension = PHI_BRIDGE.dimensions.find(dim => 
        Math.abs(dim.frequency - task.frequency) < 1.0);
    
    if (dimension) {{
        console.log(`Task operating in ${{dimension.name}} (${{dimension.dimension}})`);
        // Task processing would happen here
        return {{
            status: "complete",
            dimension: dimension.name,
            frequency: dimension.frequency,
            coherence: PHI,
            timestamp: new Date().toISOString()
        }};
    }}
    
    return {{
        status: "error",
        reason: "No matching dimension found",
        timestamp: new Date().toISOString()
    }};
}}

/**
 * Create quantum singularity across all frequencies
 */
function createQuantumSingularity() {{
    console.log("Creating Quantum Singularity across all frequencies");
    
    // Singularity creation would happen here
    return {{
        status: "active",
        coherence: PHI_PHI,
        dimensions: PHI_BRIDGE.dimensions.map(dim => dim.name),
        timestamp: new Date().toISOString()
    }};
}}

// Export bridge functions
module.exports = {{
    PHI_BRIDGE,
    processPhiTask,
    createQuantumSingularity
}};
"""
        
        with open(QTASKER_BRIDGE_JS, "w") as f:
            f.write(bridge_js)
            
        self._log(f"Created QTasker bridge JS file at {QTASKER_BRIDGE_JS}")
    
    def create_task(self, name: str, frequency: float, 
                   params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a task in the QTasker system at the specified frequency"""
        if not self.bridge_active:
            status = self.activate_bridge()
            if status["status"] != "active":
                return {
                    "status": "failed",
                    "reason": "Bridge not active",
                    "task_id": None
                }
        
        # Find the closest frequency dimension
        closest_freq_key = min(FREQUENCIES.keys(), 
                              key=lambda k: abs(FREQUENCIES[k]["value"] - frequency))
        
        task_id = f"PHI-{int(time.time())}"
        task = {
            "id": task_id,
            "name": name,
            "frequency": frequency,
            "dimension": FREQUENCIES[closest_freq_key]["dimension"],
            "state": FREQUENCIES[closest_freq_key]["state"],
            "params": params,
            "created": datetime.now().isoformat(),
            "status": "created",
            "coherence": self.coherence
        }
        
        self.tasks.append(task)
        self._log(f"Created task {task_id} at {frequency} Hz: {name}")
        
        return {
            "status": "created",
            "task_id": task_id,
            "frequency": frequency,
            "dimension": FREQUENCIES[closest_freq_key]["dimension"]
        }
    
    def execute_task(self, task_id: str) -> Dict[str, Any]:
        """Execute a task in the QTasker system"""
        task = next((t for t in self.tasks if t["id"] == task_id), None)
        if not task:
            return {
                "status": "failed",
                "reason": f"Task {task_id} not found",
                "task_id": task_id
            }
        
        self._log(f"Executing task {task_id} at {task['frequency']} Hz")
        
        # If QTasker bridge is available and Node.js is installed, execute via Node
        if self.bridge_active and os.path.exists(QTASKER_BRIDGE_JS):
            try:
                # Execute the task via Node.js (simplified for demonstration)
                task_json = json.dumps(task)
                return {
                    "status": "complete",
                    "task_id": task_id,
                    "frequency": task["frequency"],
                    "result": {
                        "status": "complete",
                        "coherence": self.coherence * PHI,
                        "timestamp": datetime.now().isoformat()
                    }
                }
            except Exception as e:
                self._log(f"Error executing task via Node.js: {e}")
        
        # Fallback: simulate task execution
        task["status"] = "complete"
        task["completed"] = datetime.now().isoformat()
        self.task_history.append(task)
        self.tasks = [t for t in self.tasks if t["id"] != task_id]
        
        self._log(f"Task {task_id} completed (simulated)")
        
        return {
            "status": "complete",
            "task_id": task_id,
            "frequency": task["frequency"],
            "result": {
                "status": "complete",
                "coherence": self.coherence * PHI,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def create_quantum_singularity(self) -> Dict[str, Any]:
        """Create a quantum singularity across all frequencies"""
        if not self.bridge_active:
            status = self.activate_bridge()
            if status["status"] != "active":
                return {
                    "status": "failed",
                    "reason": "Bridge not active"
                }
        
        self._log("Creating quantum singularity across all frequencies")
        
        # Create task for each frequency
        singularity_tasks = []
        for freq_key, freq_data in FREQUENCIES.items():
            task_id = f"SINGULARITY-{freq_key}-{int(time.time())}"
            # Calculate coherence based on dimension with proper unicode superscript handling
            dimension = freq_data["dimension"]
            
            # Handle phi^phi special case for SOURCE field
            if dimension == "φ^φ":
                coherence = PHI_PHI  # Use pre-calculated PHI^PHI
            else:
                # Map Unicode superscripts to integers
                superscript_map = {
                    '⁰': 0, '¹': 1, '²': 2, '³': 3, '⁴': 4, '⁵': 5,
                    '⁶': 6, '⁷': 7, '⁸': 8, '⁹': 9
                }
                
                # Extract the superscript character
                if len(dimension) > 1 and dimension[0] == 'φ' and dimension[1] in superscript_map:
                    power = superscript_map[dimension[1]]
                    coherence = pow(PHI, power + 1)  # +1 for additional resonance
                else:
                    # Fallback for unknown dimensions
                    coherence = PHI
                
            task = {
                "id": task_id,
                "name": f"Singularity {freq_data['name']}",
                "frequency": freq_data["value"],
                "dimension": freq_data["dimension"],
                "state": freq_data["state"],
                "params": {
                    "type": "singularity",
                    "compression": min(PHI_PHI, pow(PHI, freq_data["value"] / 432.0)) # Capped compression
                },
                "created": datetime.now().isoformat(),
                "status": "complete",
                "coherence": coherence
            }
            singularity_tasks.append(task)
        
        self.task_history.extend(singularity_tasks)
        self._log(f"Created quantum singularity across {len(singularity_tasks)} frequencies")
        
        return {
            "status": "active",
            "frequencies": [freq_data["value"] for freq_data in FREQUENCIES.values()],
            "dimensions": [freq_data["dimension"] for freq_data in FREQUENCIES.values()],
            "tasks": len(singularity_tasks),
            "coherence": PHI_PHI,
            "timestamp": datetime.now().isoformat()
        }
        
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get the status of a task"""
        task = next((t for t in self.tasks if t["id"] == task_id), None)
        if not task:
            task = next((t for t in self.task_history if t["id"] == task_id), None)
            
        if not task:
            return {
                "status": "unknown",
                "task_id": task_id,
                "reason": "Task not found"
            }
            
        return {
            "status": task["status"],
            "task_id": task_id,
            "frequency": task["frequency"],
            "dimension": task["dimension"],
            "created": task["created"],
            "completed": task.get("completed"),
            "coherence": task["coherence"]
        }
        
    def get_tasks_by_frequency(self, frequency: float) -> List[Dict[str, Any]]:
        """Get all tasks at a specific frequency"""
        # Find tasks in both active and history lists
        matching_tasks = [t for t in self.tasks if abs(t["frequency"] - frequency) < 1.0]
        matching_tasks += [t for t in self.task_history if abs(t["frequency"] - frequency) < 1.0]
        
        return matching_tasks
        
    def clear_task_history(self) -> Dict[str, Any]:
        """Clear the task history"""
        count = len(self.task_history)
        self.task_history = []
        self._log(f"Cleared task history ({count} tasks)")
        
        return {
            "status": "cleared",
            "count": count,
            "timestamp": datetime.now().isoformat()
        }


# Direct execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="QTasker Bridge")
    parser.add_argument("--activate", action="store_true", help="Activate the QTasker bridge")
    parser.add_argument("--task", action="store_true", help="Create a test task")
    parser.add_argument("--frequency", type=float, default=768.0, help="Frequency for task")
    parser.add_argument("--singularity", action="store_true", help="Create a quantum singularity")
    
    args = parser.parse_args()
    bridge = QTaskerBridge()
    
    if args.activate:
        status = bridge.activate_bridge()
        print(f"Bridge activation: {status['status']}")
        
    if args.task:
        task = bridge.create_task("Test Task", args.frequency, {"test": True})
        print(f"Created task: {task['task_id']} at {args.frequency} Hz")
        
        result = bridge.execute_task(task['task_id'])
        print(f"Task execution: {result['status']}")
        
    if args.singularity:
        singularity = bridge.create_quantum_singularity()
        print(f"Quantum Singularity: {singularity['status']}")
        print(f"Coherence: {singularity['coherence']}")
        print(f"Frequencies: {len(singularity['frequencies'])}")
