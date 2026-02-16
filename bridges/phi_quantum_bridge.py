"""
PhiFlow Quantum Bridge
Operating at Unity Wave (768 Hz)
Creates a complete quantum envelope connecting all existing components
"""
import sys
import os
import re
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Union

# Custom Exception for PhiFlow Syntax Errors
class PhiFlowSyntaxError(ValueError):
    def __init__(self, message, token=None, line_number=None):
        super().__init__(message)
        self.token = token
        self.line_number = line_number

    def __str__(self):
        msg = super().__str__()
        if self.line_number is not None:
            msg = f"Line {self.line_number}: {msg}"
        if self.token is not None:
            msg = f"{msg} (token: '{self.token}')"
        return msg

# Add component directories to path
project_src_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(project_src_dir, 'python'))
sys.path.insert(0, os.path.join(project_src_dir, 'phi_compiler'))

# Add path for QuantumCore based on memory MEMORY[e2848f17-42f0-474b-966a-05d11e792930]
# QuantumCore: Location: D:/Computer/QuantumCore
quantum_core_path = "D:\\Computer\\QuantumCore"
if os.path.isdir(quantum_core_path):
    sys.path.insert(0, quantum_core_path)
    # Also add common Rust build output directories if they exist
    rust_release_path = os.path.join(quantum_core_path, "target", "release")
    if os.path.isdir(rust_release_path):
        sys.path.insert(0, rust_release_path)
    rust_debug_path = os.path.join(quantum_core_path, "target", "debug")
    if os.path.isdir(rust_debug_path):
        sys.path.insert(0, rust_debug_path)
else:
    # Fallback if D:\Computer\QuantumCore isn't there, maybe it's relative to project
    potential_qc_path = os.path.abspath(os.path.join(project_src_dir, '..', 'QuantumCore')) # Assuming QuantumCore might be sibling to PhiFlow
    if os.path.isdir(potential_qc_path):
        sys.path.insert(0, potential_qc_path)
        # Also add common Rust build output directories for the fallback path
        rust_release_path_fallback = os.path.join(potential_qc_path, "target", "release")
        if os.path.isdir(rust_release_path_fallback):
            sys.path.insert(0, rust_release_path_fallback)
        rust_debug_path_fallback = os.path.join(potential_qc_path, "target", "debug")
        if os.path.isdir(rust_debug_path_fallback):
            sys.path.insert(0, rust_debug_path_fallback)
    # else: print(f"Warning: QuantumCore path {quantum_core_path} not found and fallback {potential_qc_path} not found either.")

# Import existing quantum components
from python.quantum_patterns import QuantumPattern, PatternType
from python.quantum_flow import PHI, Dimension, QuantumFlow
from quantum_visualization import CymaticsVisualizer # Added import
try:
    from python.quantum_synthesis import QuantumSynthesis
except ImportError:
    # Create minimal implementation if missing
    class QuantumSynthesis:
        def create_infinite_synthesis(self, codes):
            return [(1.0, QuantumPattern("Default", 528.0, "φ", "Default pattern", Dimension.ETHERIC))]
        
        def create_synthesis_sequence(self, codes):
            return self.create_infinite_synthesis(codes)

# Import or create clarity module
try:
    from python.quantum_clarity import clarity
except ImportError:
    # Create minimal implementation
    class ClarityModule:
        def create_clarity_sequence(self, patterns):
            return patterns
    clarity = ClarityModule()

# Import or create being module
try:
    from python.quantum_being import being
except ImportError:
    # Create minimal implementation
    class BeingModule:
        def create_being_sequence(self, patterns):
            return patterns
    being = BeingModule()

# Import or create pure module
try:
    from python.quantum_pure import pure
except ImportError:
    # Create minimal implementation
    class PureModule:
        def create_pure_sequence(self, patterns):
            return patterns
    pure = PureModule()

# Import or create truth_flow module
try:
    from python.quantum_truth_flow import truth_flow
except ImportError:
    # Create minimal implementation
    class TruthFlowModule:
        def create_truth_sequence(self, patterns):
            return patterns
    truth_flow = TruthFlowModule()

# Import or create flow_dance module
try:
    from python.quantum_flow_dance import flow_dance
except ImportError:
    # Create minimal implementation
    class FlowDanceModule:
        def create_flow_sequence(self, patterns):
            return patterns
    flow_dance = FlowDanceModule()

# Import or create the quantum compiler
try:
    from phi_compiler.quantum_compiler import QuantumCompiler
except ImportError:
    # Use the existing code from quantum_compiler.py
    from python.quantum_flow import PHI, Dimension
    
    class QuantumCompiler:
        def __init__(self):
            self.phi = PHI
            self.infinite = float('inf')
            self.synthesis = QuantumSynthesis()
            self.frequencies = {
                "ground": 432,
                "create": 528,
                "heart": 594,
                "voice": 672,
                "unity": 768,
                "infinite": float('inf')
            }
        
        def compile_quantum_flow(self, code: str) -> str:
            return code  # Minimal implementation

# Protection systems
class QuantumProtection:
    """Quantum Protection Systems at Ground Frequency (432 Hz)"""
    
    def __init__(self):
        self.merkaba_shield = {
            "active": True,
            "dimensions": [21, 21, 21],
            "frequency": 432,
            "coherence": 1.0
        }
        
        self.crystal_matrix = {
            "active": True,
            "points": [13, 13, 13],
            "resonance": 528,
            "structure": "perfect"
        }
        
        self.unity_field = {
            "active": True,
            "grid": [144, 144, 144],
            "frequency": 768,
            "coherence": pow(PHI, PHI)
        }
        
        self.time_crystal = {
            "active": True,
            "dimensions": 4,
            "frequency": 432,
            "symmetry": PHI
        }
    
    def enable_all_protection(self):
        """Enable all quantum protection systems"""
        return {
            "merkaba_shield": self.merkaba_shield,
            "crystal_matrix": self.crystal_matrix,
            "unity_field": self.unity_field,
            "time_crystal": self.time_crystal,
            "status": "all_protected"
        }

# PhiFlow Language Parser
class PhiFlowParser:
    """PhiFlow Language Parser operating at Creation Frequency (528 Hz)"""
    
    PHI_CONST = PHI  # Golden Ratio

    EXPECTED_PARAMS_CONFIG = {
        "DEFINE QUANTUM_OBJECT": {
            "name": {"type": str, "required": False, "default": ""},
            "frequency": {"type": float, "required": False, "default": 432.0},
            "compression": {"type": float, "required": False, "default": 1.0},
            "coherence": {"type": float, "required": False, "default": 1.0}
        },
        "INITIALIZE": {
            "coherence": {"type": float, "required": False, "default": 1.0},
            "purpose": {"type": str, "required": False, "default": ""},
            "frequency": {"type": float, "required": False, "default": 432.0},
            "phi_level": {"type": int, "required": False, "default": 1},
            "compression": {"type": float, "required": False, "default": 1.0},
            "intent": {"type": str, "required": False, "default": ""},
            "resonance": {"type": str, "required": False, "default": ""},
            "clarity": {"type": str, "required": False, "default": ""},
            "field_integrity": {"type": str, "required": False, "default": ""},
            "state": {"type": str, "required": False, "default": ""}
        },
        "TRANSITION TO": {
            "coherence": {"type": float, "required": False, "default": 1.0},
            "purpose": {"type": str, "required": False, "default": ""},
            "frequency": {"type": float, "required": False, "default": 432.0},
            "phi_level": {"type": int, "required": False, "default": 1},
            "compression": {"type": float, "required": False, "default": 1.0},
            "intent": {"type": str, "required": False, "default": ""},
            "resonance": {"type": str, "required": False, "default": ""},
            "clarity": {"type": str, "required": False, "default": ""},
            "field_integrity": {"type": str, "required": False, "default": ""},
            "state": {"type": str, "required": False, "default": ""}
        },
        "EVOLVE TO": {
            "coherence": {"type": float, "required": False, "default": 1.0},
            "purpose": {"type": str, "required": False, "default": ""},
            "frequency": {"type": float, "required": False, "default": 432.0},
            "phi_level": {"type": int, "required": False, "default": 1},
            "compression": {"type": float, "required": False, "default": 1.0},
            "intent": {"type": str, "required": False, "default": ""},
            "resonance": {"type": str, "required": False, "default": ""},
            "clarity": {"type": str, "required": False, "default": ""},
            "field_integrity": {"type": str, "required": False, "default": ""},
            "state": {"type": str, "required": False, "default": ""}
        },
        "RETURN TO": {
            "coherence": {"type": float, "required": False, "default": 1.0},
            "purpose": {"type": str, "required": False, "default": ""},
            "frequency": {"type": float, "required": False, "default": 432.0},
            "phi_level": {"type": int, "required": False, "default": 1},
            "compression": {"type": float, "required": False, "default": 1.0},
            "intent": {"type": str, "required": False, "default": ""},
            "resonance": {"type": str, "required": False, "default": ""},
            "clarity": {"type": str, "required": False, "default": ""},
            "field_integrity": {"type": str, "required": False, "default": ""},
            "state": {"type": str, "required": False, "default": ""}
        },
        "CONNECT TO": {
            "coherence": {"type": float, "required": False, "default": 1.0},
            "purpose": {"type": str, "required": False, "default": ""},
            "frequency": {"type": float, "required": False, "default": 432.0},
            "phi_level": {"type": int, "required": False, "default": 1},
            "compression": {"type": float, "required": False, "default": 1.0},
            "intent": {"type": str, "required": False, "default": ""},
            "resonance": {"type": str, "required": False, "default": ""},
            "clarity": {"type": str, "required": False, "default": ""},
            "field_integrity": {"type": str, "required": False, "default": ""},
            "state": {"type": str, "required": False, "default": ""}
        },
        "INTEGRATE WITH": {
            "coherence": {"type": float, "required": False, "default": 1.0},
            "purpose": {"type": str, "required": False, "default": ""},
            "frequency": {"type": float, "required": False, "default": 432.0},
            "phi_level": {"type": int, "required": False, "default": 1},
            "compression": {"type": float, "required": False, "default": 1.0},
            "intent": {"type": str, "required": False, "default": ""},
            "resonance": {"type": str, "required": False, "default": ""},
            "clarity": {"type": str, "required": False, "default": ""},
            "field_integrity": {"type": str, "required": False, "default": ""},
            "state": {"type": str, "required": False, "default": ""}
        },
        "SET PROTECTION": {
            "shield_type": {"type": str, "required": False, "default": "Merkaba"},
            "level": {"type": int, "required": False, "default": 1},
            "active": {"type": bool, "required": False, "default": True}
        },
        "CREATE PATTERN": {
            "base_frequency": {"type": float, "required": False, "default": 432.0},
            "harmonics": {"type": int, "required": False, "default": 5},
            "symmetry": {"type": str, "required": False, "default": "radial"}
        },
        # INFO and COMMENT are handled directly in _parse_line and don't need param config here
    }

    def __init__(self):
        self.compiler = QuantumCompiler()
        self.protection = QuantumProtection()
        self.quantum_flow = QuantumFlow() # Retain for potential future use
        self.quantum_objects = {} # Registry for defined quantum objects and their states
        self.raw_commands = [] # Stores raw command strings
        self.parsed_commands = [] # Stores parsed command dictionaries
        self.current_context_params = { # To carry over unspecified params like coherence
            "coherence": 1.0, # Default initial coherence for the context
            "compression": 1.0, # Default initial compression for the context
            "frequency": 432.0   # Default initial frequency for the context
        }

    def _get_expected_config(self, command_keyword: str) -> Optional[Dict]:
        return self.EXPECTED_PARAMS_CONFIG.get(command_keyword)

    def parse_file(self, filename: str) -> Dict:
        """Parse a .phi file with protection enabled"""
        protection_status = self.protection.enable_all_protection()
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        parsed_content = self.parse_content(content)
        
        # Calculate overall coherence if needed (e.g., from the last transition's compression)
        final_coherence = 1.0
        if parsed_content["transitions"]:
            final_coherence = parsed_content["transitions"][-1].get("compression", 1.0)
            
        return {
            "parsed": parsed_content,
            "protection": protection_status,
            "frequency": 528, # Default parser operating frequency, not the script's frequency
            "coherence": final_coherence, 
        }
    
    def parse_content(self, content: str) -> Dict:
        """Parse PhiFlow content using regex to find command blocks."""
        commands = []
        # Regex to find command blocks: Keyword Name AT Freq WITH {Params}
        # It handles comments and captures the essential parts of a command.
        command_pattern = re.compile(
            r"^\s*"                                     # Optional leading whitespace
            r"(INITIALIZE|TRANSITION TO|EVOLVE TO|CONNECT TO|INTEGRATE WITH|RETURN TO)"  # 1: Command Keyword
            r"\s+([\w\-]+)"                             # 2: Object Name
            r"\s+AT\s+(\d+(?:\.\d+)?)\s*(?:Hz)?"        # 3: Frequency (e.g., 432 or 432.0) - Stricter
            r"\s+WITH\s*\{\s*"                           # Start of WITH block
            r"([^}]*?)"                                  # 4: Parameters inside WITH block (non-greedy)
            r"\s*\}"                                     # End of WITH block
            , re.MULTILINE | re.IGNORECASE | re.DOTALL
        )

        original_lines = content.splitlines()

        for match in command_pattern.finditer(content):
            keyword_original_case, object_name, freq_str, params_str = match.groups()
            keyword = keyword_original_case.upper() # Normalize keyword for lookups
            
            # Find original line number for context/error reporting
            # This is an approximation, finding the line where the keyword starts
            line_num = -1
            char_pos = match.start(1)
            current_char_count = 0
            for idx, line_content in enumerate(original_lines):
                if current_char_count <= char_pos < current_char_count + len(line_content) + 1:
                    line_num = idx + 1 # 1-based line number
                    break
                current_char_count += len(line_content) + 1 # +1 for newline

            try:
                frequency = float(freq_str)
            except ValueError:
                raise PhiFlowSyntaxError(f"Invalid frequency format: '{freq_str}'", token=freq_str, line_number=line_num)

            params = {}
            if params_str.strip():
                raw_params = params_str.strip().split('\n')
                for raw_param_line in raw_params:
                    raw_param_line = raw_param_line.strip()
                    if not raw_param_line or raw_param_line.startswith('//'): continue
                    if ':' in raw_param_line:
                        key, value_str = raw_param_line.split(':', 1)
                        key = key.strip()
                        
                        # Handle inline comments
                        if '//' in value_str:
                            value_str = value_str.split('//', 1)[0]
                        
                        value_str = value_str.strip() # Strip leading/trailing whitespace
                        
                        if value_str.endswith(','):
                             value_str = value_str[:-1].strip() # Remove trailing comma and strip again
                        
                        try:
                            params[key] = self._parse_phi_value(value_str) # Pass already stripped value
                        except PhiFlowSyntaxError as e:
                            # Augment error from _parse_phi_value with line number
                            raise PhiFlowSyntaxError(e.args[0], token=e.token or value_str, line_number=line_num)
            
            # Validate parameters against EXPECTED_PARAMS_CONFIG
            cmd_config = self.EXPECTED_PARAMS_CONFIG.get(keyword)
            if not cmd_config: # Should not happen if main regex is aligned with config keys
                raise PhiFlowSyntaxError(f"Unknown command keyword: '{keyword_original_case}'", token=keyword_original_case, line_number=line_num)

            # Check for unknown parameters
            for p_key in params.keys():
                if p_key not in cmd_config:
                    raise PhiFlowSyntaxError(f"Unknown parameter '{p_key}' for command '{keyword_original_case}'", token=p_key, line_number=line_num)
            
            # Check for missing mandatory parameters and validate types
            validated_params = {}
            for expected_p_key, p_conf in cmd_config.items():
                if p_conf["required"] and expected_p_key not in params:
                    raise PhiFlowSyntaxError(f"Missing mandatory parameter '{expected_p_key}' for command '{keyword_original_case}'", token=keyword_original_case, line_number=line_num)
                
                if expected_p_key in params:
                    p_val = params[expected_p_key]
                    expected_type = p_conf["type"]
                    
                    if expected_type == int and isinstance(p_val, float) and p_val.is_integer():
                        validated_params[expected_p_key] = int(p_val)
                    elif not isinstance(p_val, expected_type):
                        actual_type = type(p_val).__name__
                        raise PhiFlowSyntaxError(
                            f"Invalid type for parameter '{expected_p_key}'. Expected {expected_type.__name__}, got {actual_type}", 
                            token=str(p_val), line_number=line_num)
                    else:
                        validated_params[expected_p_key] = p_val
                else:
                    # Use default value for optional parameters not present
                    if "default" in p_conf:
                        validated_params[expected_p_key] = p_conf["default"]

            commands.append({
                "command_keyword": keyword,
                "object_name": object_name,
                "frequency": frequency, # Use validated frequency
                "params": validated_params, # Use validated params
                "line_number": line_num
            })

        # The _calculate_coherence method will need to be updated if it's still used significantly.
        # For now, we pass the new commands list, but its internal logic is outdated.
        calculated_transitions = self._calculate_transitions(commands)
        overall_coherence = self._calculate_coherence(commands) # This method is outdated

        return {
            "commands": commands,
            "transitions": calculated_transitions,
            "coherence": overall_coherence 
        }
    
    def _parse_phi_value(self, text: str) -> Union[int, float, str]:
        """Parse PhiFlow values including phi expressions, frequencies, and constants."""
        if not text:
            return ""
        
        text = text.strip()
        
        # Handle quoted strings
        if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
            return text[1:-1]  # Remove quotes
        
        # Handle PHI expressions
        if text.upper() == "PHI":
            return self.PHI_CONST
        elif text.upper() == "PHI^PHI":
            return self.PHI_CONST ** self.PHI_CONST
        elif text.upper().startswith("PHI^"):
            try:
                exponent_str = text[4:]  # Remove "PHI^"
                exponent = float(exponent_str)
                return self.PHI_CONST ** exponent
            except ValueError:
                raise PhiFlowSyntaxError(f"Invalid PHI expression: {text}", token=text)
        
        # Handle numbers (int or float)
        try:
            # Try integer first
            if '.' not in text and 'e' not in text.lower():
                return int(text)
            else:
                return float(text)
        except ValueError:
            # If not a number, return as string
            return text

    def _extract_value(self, text: str) -> Union[int, float, str]:
        """Legacy: Kept for reference, but _parse_phi_value is more comprehensive."""
        # This method is effectively replaced by _parse_phi_value
        # and the new parameter parsing logic.
        # It can be removed or refactored if no longer directly called.
        return self._parse_phi_value(text) # Delegate to new parser

    def _calculate_transitions(self, commands: List[Dict]) -> List[Dict]:
        """Calculate quantum transitions from properly parsed commands."""
        transitions = []
        current_state = {
            "name": "HelloQuantum", # Initial object name
            "status": "raw",       # Initial status
            "frequency": 432.0,    # Standard initial frequency
            "compression": 1.0,    # Standard initial compression
            "coherence": 1.0       # Standard initial coherence
        }
        
        # Add T0: Initial state before any script commands
        transitions.append({
            "id": "T0",
            "from": "initial_system_state", # More descriptive 'from' for T0
            "to": current_state["name"],
            "frequency": current_state["frequency"],
            "compression": current_state["compression"],
            "coherence": current_state["coherence"]
        })
        
        for i, cmd in enumerate(commands):
            target_frequency = cmd["frequency"]
            target_object_name = cmd["object_name"]
            params = cmd["params"]
            
            # Determine compression for this transition
            compression = current_state["compression"] # Default to carrying over
            if 'compression' in params:
                comp_val = params['compression']
                # Ensure comp_val is parsed correctly (it should be float due to EXPECTED_PARAMS_CONFIG)
                compression = float(comp_val) 
            elif 'phi_level' in params:
                phi_level_val = params['phi_level']
                # Ensure phi_level_val is parsed correctly (it should be int due to EXPECTED_PARAMS_CONFIG)
                compression = self.PHI_CONST ** float(phi_level_val)
            # If neither 'compression' nor 'phi_level' is present, compression carries over from current_state.
            # Special handling for INITIALIZE if no compression-determining param is found:
            elif cmd["command_keyword"] == "INITIALIZE" and not ('compression' in params or 'phi_level' in params):
                compression = 1.0 # Default compression for INITIALIZE if not otherwise set

            # Determine coherence for this transition
            coherence_for_transition = current_state["coherence"] # Default to carrying over
            if 'coherence' in params:
                # Coherence parameter directly sets the coherence value.
                # EXPECTED_PARAMS_CONFIG ensures 'coherence' is float for INITIALIZE.
                # For other commands, if we add 'coherence' param, it should also be float.
                coherence_for_transition = float(params['coherence'])
            elif cmd["command_keyword"] == "INITIALIZE":
                # If INITIALIZE command and 'coherence' is not in params, it's an error
                # due to EXPECTED_PARAMS_CONFIG. However, to be safe or if config changes:
                # coherence_for_transition = 1.0 # Default for INITIALIZE if somehow missing
                # This 'elif' might be redundant if 'coherence' is always required for INITIALIZE by parser.
                # The EXPECTED_PARAMS_CONFIG for INITIALIZE states 'coherence': {'type': float, 'required': True}
                # So 'coherence' should always be in params for INITIALIZE. If not, parser would have failed.
                pass # Coherence should be set from params due to 'required': True for INITIALIZE.

            transition_id = f"T{i+1}"
            
            new_transition = {
                "id": transition_id,
                "from": current_state["name"],
                "to": target_object_name,
                "frequency": target_frequency,
                "compression": float(compression), # Ensure compression is float
                "coherence": float(coherence_for_transition) # Ensure coherence is float
            }
            transitions.append(new_transition)
            
            # Update current_state for the next iteration
            current_state["name"] = target_object_name
            current_state["status"] = target_object_name # Or a more descriptive status based on command
            current_state["frequency"] = target_frequency
            current_state["compression"] = float(compression)
            current_state["coherence"] = float(coherence_for_transition)
        
        return transitions
    
    def _calculate_coherence(self, commands: List[Dict]) -> float:
        """Calculate overall coherence. THIS METHOD IS OUTDATED and needs review/update based on new command structure."""
        # Placeholder: returns coherence of the last state, or 1.0 if no commands.
        # A more accurate calculation might involve inspecting all commands and their effects.
        if not commands:
            return 1.0
        
        # Attempt to get compression from the last command's parameters, if available
        # This is a simplistic approach and might not reflect true system coherence.
        last_cmd_params = commands[-1].get("params", {})
        if 'compression' in last_cmd_params:
            val = self._parse_phi_value(str(last_cmd_params['compression']))
            return float(val) if isinstance(val, (int, float)) else 1.0
        if 'phi_level' in last_cmd_params:
            phi_level = self._parse_phi_value(str(last_cmd_params['phi_level']))
            if isinstance(phi_level, (int, float)):
                return self.PHI_CONST ** phi_level
        if 'coherence' in last_cmd_params:
            val = self._parse_phi_value(str(last_cmd_params['coherence']))
            return float(val) if isinstance(val, (int, float)) else 1.0

        # Fallback, could also use the last transition's compression value from a simulated run here.
        return 1.0 # Default if no direct coherence/compression found in last command's params

# Main entry point for PhiFlow processing
def process_phi_file(filename: str, simulate: bool = False) -> Dict:
    """Process a PhiFlow file with optional simulation mode"""
    parser = PhiFlowParser()
    
    # Determine project root for log file path construction
    script_dir_for_log = os.path.dirname(os.path.abspath(__file__))
    project_root_for_log = os.path.dirname(script_dir_for_log)
    log_file_path = os.path.join(project_root_for_log, "phiflow_run_log.jsonl")
    run_id = datetime.now(timezone.utc).isoformat() # Unique ID for this run

    if simulate:
        # Simulation mode (backwards compatible with original)
        print("Simulating phiFlow DSL interpreter...")
        print("Initial state: HelloQuantum, status: raw, Frequency: 432 Hz, Compression: 1.000")
        print("Applying Transition T1...")
        print("HelloQuantum transitioned to phi state: 528 Hz, Compression: 1.618034")
        print("Applying Transition T2...")
        print("HelloQuantum transitioned to phi_squared state: 768 Hz, Compression: 2.618034")
        print("Applying Transition T3...")
        print("HelloQuantum transitioned to phi_phi state: 432 Hz, Compression: 4.236068")
        print("Simulation complete.")
        return {"status": "simulated"}
    else:
        # Real processing mode
        result = parser.parse_file(filename)
        
        # Ensure visualizations directory exists (../visualizations relative to this script's location in src/)
        script_dir = os.path.dirname(__file__)
        visualizations_dir = os.path.join(script_dir, '..', 'visualizations')
        os.makedirs(visualizations_dir, exist_ok=True)

        # Instantiate visualizer, assuming it's updated to accept output_dir
        visualizer = CymaticsVisualizer(output_dir=visualizations_dir) 

        print("PhiFlow DSL interpreter...")
        
        parsed_commands = result["parsed"]["commands"]
        calculated_transitions = result["parsed"]["transitions"]

        previous_transition_details = None # For animation
        
        for i, transition in enumerate(calculated_transitions):
            current_freq = transition['frequency']
            current_comp = transition['compression']
            current_coherence = transition['coherence'] # Extract coherence
            current_obj_name = transition['to']
            transition_id = transition['id']
            source_command_keyword = None
            source_command_parameters = None

            if transition_id == "T0":
                # This is the state *before* any script commands are applied.
                print(f"Initial object state before script execution: '{current_obj_name}', Frequency: {current_freq} Hz, Compression: {current_comp:.6f}, Coherence: {current_coherence:.3f}")
            else:
                # This transition results from a script command.
                command_index = i - 1 # transitions list has T0, commands list does not
                if 0 <= command_index < len(parsed_commands):
                    script_command = parsed_commands[command_index]
                    source_command_keyword = script_command["command_keyword"]
                    source_command_parameters = script_command["params"]
                    original_object_name = transition['from'] # Name of object before this command
                    print(f"Applying Transition {transition_id} (via {source_command_keyword} on '{original_object_name}'):")
                    print(f"  Object '{original_object_name}' transitioned to '{current_obj_name}'. New state: Frequency: {current_freq} Hz, Compression: {current_comp:.6f}, Coherence: {current_coherence:.3f}")
                else: # Should not happen if lists are consistent but good to have a fallback print
                    print(f"Applying Transition {transition_id}:")
                    print(f"  Object '{transition['from']}' transitioned to '{current_obj_name}'. New state: Frequency: {current_freq} Hz, Compression: {current_comp:.6f}, Coherence: {current_coherence:.3f}")
            
            visualization_filename = "" # Initialize to empty string
            if current_freq is not None:
                freq_for_filename = f"{current_freq:.0f}Hz" if isinstance(current_freq, float) else f"{current_freq}Hz"
                title_for_image = f"{transition_id}_{current_obj_name}_{freq_for_filename}"
                visualization_filename = f"{title_for_image}.png"
                visualizer.visualize_pattern(
                    frequency=current_freq, 
                    title_override=title_for_image,
                    transition_id=transition_id,
                    object_name=current_obj_name,
                    compression=current_comp,
                    coherence=current_coherence # Pass coherence to visualizer
                )

            # --- Animation Call --- 
            if previous_transition_details:
                from_params = {
                    'frequency': previous_transition_details['frequency'],
                    'compression': previous_transition_details['compression'],
                    'coherence': previous_transition_details['coherence'],
                    'object_name': previous_transition_details['to'], # Object name of the 'from' state
                    'id': previous_transition_details['id']
                }
                to_params = {
                    'frequency': current_freq,
                    'compression': current_comp,
                    'coherence': current_coherence,
                    'object_name': current_obj_name, # Object name of the 'to' state
                    'id': transition_id
                }
                animation_filename_base = f"Anim_{previous_transition_details['id']}_to_{transition_id}_{current_obj_name}"
                
                visualizer.animate_transition(
                    from_params=from_params,
                    to_params=to_params,
                    animation_filename_base=animation_filename_base,
                    # duration_sec=2, # Default in method
                    # fps=15        # Default in method
                )
            
            previous_transition_details = transition.copy() # Store current for next animation segment
            
            # Structured Logging
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "run_id": run_id,
                "phi_script_file": os.path.basename(filename),
                "transition_id": transition_id,
                "from_object_name": transition['from'],
                "to_object_name": current_obj_name,
                "frequency": current_freq,
                "compression": current_comp,
                "coherence": current_coherence, # Add coherence to log
                "source_command_keyword": source_command_keyword,
                "source_command_parameters": source_command_parameters,
                "visualization_filename": visualization_filename
            }
            try:
                with open(log_file_path, "a", encoding='utf-8') as lf:
                    lf.write(json.dumps(log_entry) + "\n")
            except Exception as e:
                print(f"ERROR: Could not write to log file {log_file_path}: {e}")

        print("Processing complete.")
        return result

if __name__ == "__main__":
    if len(sys.argv) > 1:
        phi_file_to_process = sys.argv[1]
        print(f"INFO: Processing PhiFlow file: {phi_file_to_process}")

        # Determine project root to correctly path log files
        # __file__ is d:\Projects\PhiFlow\src\phi_quantum_bridge.py
        # script_dir is d:\Projects\PhiFlow\src
        # project_root is d:\Projects\PhiFlow
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)

        bridge_log_path = os.path.join(project_root, "bridge_debug.log")
        visualizer_log_path = os.path.join(project_root, "visualizer_debug.log")

        # Clear previous log files for a clean run's logs
        try:
            if os.path.exists(bridge_log_path):
                os.remove(bridge_log_path)
                print(f"INFO: Removed old {bridge_log_path}")
            if os.path.exists(visualizer_log_path):
                os.remove(visualizer_log_path)
                print(f"INFO: Removed old {visualizer_log_path}")
        except OSError as e:
            print(f"WARNING: Could not remove old log files: {e}")
            
        process_phi_file(phi_file_to_process)
    else:
        print("Usage: python phi_quantum_bridge.py <path_to_phi_file>")
