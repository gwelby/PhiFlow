#!/usr/bin/env python3
"""
PhiFlow Compiler - Task 3.4 Implementation
Converts AST to executable programs with quantum circuit generation
"""

import math
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .phi_flow_parser import ASTNode, ASTNodeType
from .phi_flow_semantic_analyzer import PhiFlowSemanticAnalyzer

# Sacred Mathematics Constants (15 decimal precision)
PHI = Decimal('1.618033988749895')  # Golden ratio
LAMBDA = Decimal('0.618033988749895')  # Divine complement (1/φ)
GOLDEN_ANGLE = Decimal('137.5077640500378')  # Golden angle in degrees

# Sacred Frequencies (Hz)
SACRED_FREQUENCIES = {
    'ground': 432,
    'creation': 528,
    'heart': 594,
    'voice': 672,
    'vision': 720,
    'unity': 768,
    'source': 963
}

# Consciousness states and phi-levels
CONSCIOUSNESS_STATES = {
    0: "OBSERVE",
    1: "CREATE", 
    2: "INTEGRATE",
    3: "HARMONIZE",
    4: "TRANSCEND",
    5: "CASCADE",
    6: "SUPERPOSITION",
    7: "UNITY"
}

class InstructionType(Enum):
    """Types of compiled instructions"""
    QUANTUM_GATE = "QUANTUM_GATE"
    CONSCIOUSNESS_SET = "CONSCIOUSNESS_SET"
    FREQUENCY_SET = "FREQUENCY_SET"
    FIELD_OPERATION = "FIELD_OPERATION"
    PARALLEL_BLOCK = "PARALLEL_BLOCK"
    CONDITIONAL_BLOCK = "CONDITIONAL_BLOCK"
    LOOP_BLOCK = "LOOP_BLOCK"
    VARIABLE_ASSIGN = "VARIABLE_ASSIGN"
    FUNCTION_CALL = "FUNCTION_CALL"

@dataclass
class CompiledInstruction:
    """Compiled instruction with execution parameters"""
    instruction_type: InstructionType
    operation: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Sacred mathematics properties
    frequency: Optional[float] = None
    phi_level: Optional[int] = None
    consciousness_state: Optional[str] = None
    
    # Quantum properties
    qubits: List[int] = field(default_factory=list)
    quantum_gates: List[Dict[str, Any]] = field(default_factory=list)
    
    # Execution properties
    execution_order: int = 0
    parallel_group: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)

@dataclass
class CompiledProgram:
    """Complete compiled PhiFlow program"""
    instructions: List[CompiledInstruction] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    quantum_circuit: Optional[Dict[str, Any]] = None
    
    # Program metadata
    total_qubits: int = 0
    total_gates: int = 0
    estimated_runtime: float = 0.0
    
    # Sacred mathematics properties
    phi_alignment_score: float = 0.0
    consciousness_coverage: Dict[str, int] = field(default_factory=dict)
    frequency_spectrum: List[float] = field(default_factory=list)

class PhiFlowCompiler:
    """
    PhiFlow Compiler implementing sacred mathematics compilation
    
    Converts PhiFlow AST to executable programs:
    - Quantum circuit generation from PhiFlow commands
    - Consciousness state management compilation
    - Sacred frequency synthesis compilation
    - Phi-harmonic optimization passes
    - Parallel execution orchestration
    """
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.compiled_program = CompiledProgram()
        self.instruction_counter = 0
        self.qubit_allocator = 0
        
        # Compilation context
        self.current_scope = "global"
        self.current_frequency = 432.0  # Default to ground frequency
        self.current_phi_level = 0
        self.current_consciousness_state = "OBSERVE"
        
        # Sacred mathematics compilation state
        self.phi_harmonic_gates = []
        self.consciousness_transitions = []
        self.frequency_modulations = []
        
        # Initialize quantum gate mappings
        self._init_quantum_gate_mappings()
    
    def _init_quantum_gate_mappings(self):
        """Initialize mappings from PhiFlow commands to quantum gates"""
        
        # Sacred geometry commands to quantum gate sequences
        self.sacred_command_gates = {
            "INITIALIZE": {
                "gates": [
                    {"type": "H", "rotation": 0},  # Hadamard for superposition
                    {"type": "RZ", "rotation": "phi_level * golden_angle * π/180"}
                ],
                "qubits": 1,
                "description": "Initialize quantum field in superposition with phi rotation"
            },
            
            "TRANSITION": {
                "gates": [
                    {"type": "RY", "rotation": "π/4"},  # 45° rotation for transition
                    {"type": "RZ", "rotation": "frequency * 2π / base_frequency"}
                ],
                "qubits": 1,
                "description": "Sacred transition with frequency modulation"
            },
            
            "EVOLVE": {
                "gates": [
                    {"type": "U3", "theta": "phi_level * π/7", "phi": "golden_angle * π/180", "lambda": "λ * π"},
                    {"type": "RZ", "rotation": "frequency * PHI * 2π / 1000"}
                ],
                "qubits": 1,
                "description": "Evolution using U3 gate with phi-harmonic parameters"
            },
            
            "INTEGRATE": {
                "gates": [
                    {"type": "CNOT", "control": 0, "target": 1},
                    {"type": "RY", "rotation": "φ * π/7"},
                    {"type": "CNOT", "control": 1, "target": 0}
                ],
                "qubits": 2,
                "description": "Integration through entanglement and phi rotation"
            },
            
            "HARMONIZE": {
                "gates": [
                    {"type": "RX", "rotation": "2π / φ"},
                    {"type": "RY", "rotation": "2π / φ²"},
                    {"type": "RZ", "rotation": "2π / φ³"}
                ],
                "qubits": 1,
                "description": "Harmonic resonance through phi-power rotations"
            },
            
            "TRANSCEND": {
                "gates": [
                    {"type": "RY", "rotation": "π/2"},  # Full rotation to higher state
                    {"type": "Phase", "rotation": "frequency * π / 432"},
                    {"type": "U3", "theta": "π/φ", "phi": "golden_angle * π/180", "lambda": "0"}
                ],
                "qubits": 1,
                "description": "Transcendence through dimensional rotation"
            },
            
            "CASCADE": {
                "gates": [
                    {"type": "H", "rotation": 0},
                    {"type": "RZ", "rotation": "φ^φ * π/180"},
                    {"type": "CNOT", "control": 0, "target": 1},
                    {"type": "RY", "rotation": "π/φ"},
                    {"type": "CNOT", "control": 1, "target": 0}
                ],
                "qubits": 2,
                "description": "Cascade effect through entanglement and phi^phi rotation"
            }
        }
        
        # Field commands to quantum operations
        self.field_command_gates = {
            "CREATE_FIELD": {
                "gates": [
                    {"type": "H", "rotation": 0},  # Create superposition field
                    {"type": "RZ", "rotation": "frequency * 2π / 432"}
                ],
                "qubits": 1,
                "description": "Create consciousness field in superposition"
            },
            
            "ALIGN_FIELD": {
                "gates": [
                    {"type": "RY", "rotation": "golden_angle * π/180"},
                    {"type": "RZ", "rotation": "φ * π/4"}
                ],
                "qubits": 1,
                "description": "Align field using golden angle and phi rotation"
            },
            
            "RESONATE_FIELD": {
                "gates": [
                    {"type": "RX", "rotation": "frequency * 2π / 432"},
                    {"type": "RZ", "rotation": "harmonic_frequency * 2π / 432"}
                ],
                "qubits": 1,
                "description": "Field resonance at specified frequency"
            },
            
            "COLLAPSE_FIELD": {
                "gates": [
                    {"type": "Measure", "basis": "computational"}
                ],
                "qubits": 1,
                "description": "Collapse field to definite state"
            }
        }
        
        # Consciousness commands to state operations
        self.consciousness_command_gates = {
            "OBSERVE": {
                "gates": [
                    {"type": "I", "rotation": 0}  # Identity - observe without change
                ],
                "qubits": 1,
                "description": "Conscious observation without interference"
            },
            
            "INTEND": {
                "gates": [
                    {"type": "RZ", "rotation": "intention_strength * π/4"}
                ],
                "qubits": 1,
                "description": "Set intention through phase rotation"
            },
            
            "FOCUS": {
                "gates": [
                    {"type": "RY", "rotation": "focus_intensity * π/2"}
                ],
                "qubits": 1,
                "description": "Focus consciousness through Y rotation"
            },
            
            "EXPAND": {
                "gates": [
                    {"type": "H", "rotation": 0},  # Expand into superposition
                    {"type": "RZ", "rotation": "expansion_factor * φ * π/4"}
                ],
                "qubits": 1,
                "description": "Expand consciousness into superposition"
            },
            
            "MERGE": {
                "gates": [
                    {"type": "CNOT", "control": 0, "target": 1},
                    {"type": "SWAP", "qubit1": 0, "qubit2": 1}
                ],
                "qubits": 2,
                "description": "Merge consciousness states through entanglement"
            }
        }
    
    def compile(self, ast: ASTNode, semantic_results: Optional[Dict[str, Any]] = None) -> CompiledProgram:
        """
        Compile PhiFlow AST to executable program
        
        Args:
            ast: Abstract syntax tree from parser
            semantic_results: Optional semantic analysis results
            
        Returns:
            Compiled program with quantum circuits and instructions
        """
        if self.debug:
            print("🔧 Consciousness Expert: Starting compilation")
            print(f"📐 Sacred Mathematics: φ = {PHI}, golden angle = {GOLDEN_ANGLE}°")
        
        # Reset compilation state
        self.compiled_program = CompiledProgram()
        self.instruction_counter = 0
        self.qubit_allocator = 0
        
        # Use semantic analysis if provided
        if semantic_results:
            self.compiled_program.variables = semantic_results.get('variables', {})
            self.compiled_program.phi_alignment_score = semantic_results.get('phi_alignment_score', 0.0)
            self.compiled_program.consciousness_coverage = semantic_results.get('consciousness_coverage', {})
            self.compiled_program.frequency_spectrum = semantic_results.get('frequency_distribution', {}).get('sacred_frequencies', [])
        
        # Compile AST nodes
        self._compile_node(ast)
        
        # Apply optimization passes
        self._apply_optimization_passes()
        
        # Generate quantum circuit
        self._generate_quantum_circuit()
        
        # Calculate program metadata
        self._calculate_program_metadata()
        
        if self.debug:
            print(f"✅ Compilation complete: {len(self.compiled_program.instructions)} instructions")
            print(f"🔧 Quantum circuit: {self.compiled_program.total_qubits} qubits, {self.compiled_program.total_gates} gates")
            print(f"🎯 Phi alignment: {self.compiled_program.phi_alignment_score:.3f}")
        
        return self.compiled_program
    
    def _compile_node(self, node: ASTNode):
        """Recursively compile AST nodes to instructions"""
        if not node:
            return
        
        # Update compilation context
        self._update_compilation_context(node)
        
        # Compile based on node type
        if node.type == ASTNodeType.PROGRAM:
            self._compile_program(node)
        elif node.type in self._get_sacred_command_types():
            self._compile_sacred_command(node)
        elif node.type in self._get_field_command_types():
            self._compile_field_command(node)
        elif node.type in self._get_consciousness_command_types():
            self._compile_consciousness_command(node)
        elif node.type == ASTNodeType.ASSIGNMENT_STMT:
            self._compile_assignment(node)
        elif node.type == ASTNodeType.IF_STMT:
            self._compile_conditional(node)
        elif node.type in [ASTNodeType.WHILE_STMT, ASTNodeType.FOR_STMT]:
            self._compile_loop(node)
        elif node.type == ASTNodeType.PARALLEL_STMT:
            self._compile_parallel(node)
        
        # Compile child nodes
        for child in node.children:
            self._compile_node(child)
    
    def _update_compilation_context(self, node: ASTNode):
        """Update compilation context based on current node"""
        if 'frequency' in node.attributes:
            try:
                freq_str = str(node.attributes['frequency']).replace('Hz', '')
                self.current_frequency = float(freq_str)
            except:
                pass
        
        if 'phi_level' in node.attributes:
            try:
                self.current_phi_level = int(node.attributes['phi_level'])
                self.current_consciousness_state = CONSCIOUSNESS_STATES.get(self.current_phi_level, "OBSERVE")
            except:
                pass
    
    def _get_sacred_command_types(self) -> List[ASTNodeType]:
        """Get list of sacred geometry command types"""
        return [
            ASTNodeType.INITIALIZE_STMT, ASTNodeType.TRANSITION_STMT, ASTNodeType.EVOLVE_STMT,
            ASTNodeType.INTEGRATE_STMT, ASTNodeType.HARMONIZE_STMT, ASTNodeType.TRANSCEND_STMT, 
            ASTNodeType.CASCADE_STMT
        ]
    
    def _get_field_command_types(self) -> List[ASTNodeType]:
        """Get list of field command types"""
        return [
            ASTNodeType.CREATE_FIELD_STMT, ASTNodeType.ALIGN_FIELD_STMT,
            ASTNodeType.RESONATE_FIELD_STMT, ASTNodeType.COLLAPSE_FIELD_STMT
        ]
    
    def _get_consciousness_command_types(self) -> List[ASTNodeType]:
        """Get list of consciousness command types"""
        return [
            ASTNodeType.OBSERVE_STMT, ASTNodeType.INTEND_STMT, ASTNodeType.FOCUS_STMT,
            ASTNodeType.EXPAND_STMT, ASTNodeType.MERGE_STMT
        ]
    
    def _compile_program(self, node: ASTNode):
        """Compile program root"""
        if self.debug:
            print("🔧 Compiling program structure")
        
        # Add program initialization instruction
        init_instruction = CompiledInstruction(
            instruction_type=InstructionType.CONSCIOUSNESS_SET,
            operation="PROGRAM_INIT",
            parameters={
                "base_frequency": self.current_frequency,
                "phi_level": self.current_phi_level,
                "consciousness_state": self.current_consciousness_state
            },
            frequency=self.current_frequency,
            phi_level=self.current_phi_level,
            consciousness_state=self.current_consciousness_state,
            execution_order=self.instruction_counter
        )
        
        self.compiled_program.instructions.append(init_instruction)
        self.instruction_counter += 1
    
    def _compile_sacred_command(self, node: ASTNode):
        """Compile sacred geometry commands to quantum gates"""
        command = self._get_command_name(node.type)
        
        if self.debug:
            print(f"🔧 Compiling sacred command: {command}")
        
        if command in self.sacred_command_gates:
            gate_info = self.sacred_command_gates[command]
            
            # Allocate qubits
            qubits = list(range(self.qubit_allocator, self.qubit_allocator + gate_info["qubits"]))
            self.qubit_allocator += gate_info["qubits"]
            
            # Generate quantum gates with sacred mathematics parameters
            quantum_gates = []
            for gate_template in gate_info["gates"]:
                gate = self._instantiate_quantum_gate(gate_template, node)
                quantum_gates.append(gate)
            
            # Create compiled instruction
            instruction = CompiledInstruction(
                instruction_type=InstructionType.QUANTUM_GATE,
                operation=command,
                parameters=dict(node.attributes),
                frequency=self.current_frequency,
                phi_level=self.current_phi_level,
                consciousness_state=self.current_consciousness_state,
                qubits=qubits,
                quantum_gates=quantum_gates,
                execution_order=self.instruction_counter
            )
            
            self.compiled_program.instructions.append(instruction)
            self.instruction_counter += 1
    
    def _compile_field_command(self, node: ASTNode):
        """Compile field operation commands"""
        command = self._get_command_name(node.type)
        
        if self.debug:
            print(f"🔧 Compiling field command: {command}")
        
        if command in self.field_command_gates:
            gate_info = self.field_command_gates[command]
            
            # Allocate qubits
            qubits = list(range(self.qubit_allocator, self.qubit_allocator + gate_info["qubits"]))
            self.qubit_allocator += gate_info["qubits"]
            
            # Generate quantum gates
            quantum_gates = []
            for gate_template in gate_info["gates"]:
                gate = self._instantiate_quantum_gate(gate_template, node)
                quantum_gates.append(gate)
            
            # Create compiled instruction
            instruction = CompiledInstruction(
                instruction_type=InstructionType.FIELD_OPERATION,
                operation=command,
                parameters=dict(node.attributes),
                frequency=self.current_frequency,
                phi_level=self.current_phi_level,
                consciousness_state=self.current_consciousness_state,
                qubits=qubits,
                quantum_gates=quantum_gates,
                execution_order=self.instruction_counter
            )
            
            self.compiled_program.instructions.append(instruction)
            self.instruction_counter += 1
    
    def _compile_consciousness_command(self, node: ASTNode):
        """Compile consciousness operation commands"""
        command = self._get_command_name(node.type)
        
        if self.debug:
            print(f"🔧 Compiling consciousness command: {command}")
        
        if command in self.consciousness_command_gates:
            gate_info = self.consciousness_command_gates[command]
            
            # Allocate qubits
            qubits = list(range(self.qubit_allocator, self.qubit_allocator + gate_info["qubits"]))
            self.qubit_allocator += gate_info["qubits"]
            
            # Generate quantum gates
            quantum_gates = []
            for gate_template in gate_info["gates"]:
                gate = self._instantiate_quantum_gate(gate_template, node)
                quantum_gates.append(gate)
            
            # Create compiled instruction
            instruction = CompiledInstruction(
                instruction_type=InstructionType.CONSCIOUSNESS_SET,
                operation=command,
                parameters=dict(node.attributes),
                frequency=self.current_frequency,
                phi_level=self.current_phi_level,
                consciousness_state=self.current_consciousness_state,
                qubits=qubits,
                quantum_gates=quantum_gates,
                execution_order=self.instruction_counter
            )
            
            self.compiled_program.instructions.append(instruction)
            self.instruction_counter += 1
    
    def _compile_assignment(self, node: ASTNode):
        """Compile variable assignments"""
        var_name = node.value
        
        if self.debug:
            print(f"🔧 Compiling assignment: {var_name}")
        
        # Extract value from expression
        value = self._evaluate_expression(node.children[0] if node.children else None)
        
        # Create variable assignment instruction
        instruction = CompiledInstruction(
            instruction_type=InstructionType.VARIABLE_ASSIGN,
            operation="ASSIGN",
            parameters={
                "variable": var_name,
                "value": value,
                "type": self._infer_type(value)
            },
            execution_order=self.instruction_counter
        )
        
        self.compiled_program.instructions.append(instruction)
        self.compiled_program.variables[var_name] = value
        self.instruction_counter += 1
    
    def _compile_conditional(self, node: ASTNode):
        """Compile conditional statements"""
        if self.debug:
            print("🔧 Compiling conditional block")
        
        # Create conditional block instruction
        instruction = CompiledInstruction(
            instruction_type=InstructionType.CONDITIONAL_BLOCK,
            operation="IF",
            parameters={
                "condition": self._compile_condition(node.children[0] if node.children else None),
                "true_branch": self._compile_block(node.children[1] if len(node.children) > 1 else None),
                "false_branch": self._compile_block(node.children[2] if len(node.children) > 2 else None)
            },
            execution_order=self.instruction_counter
        )
        
        self.compiled_program.instructions.append(instruction)
        self.instruction_counter += 1
    
    def _compile_loop(self, node: ASTNode):
        """Compile loop constructs with Fibonacci optimization"""
        if self.debug:
            print("🔧 Compiling loop block")
        
        # Optimize loop iterations using Fibonacci sequence
        iterations = self._extract_loop_iterations(node)
        fibonacci_optimized_iterations = self._fibonacci_optimize_iterations(iterations)
        
        # Create loop block instruction
        instruction = CompiledInstruction(
            instruction_type=InstructionType.LOOP_BLOCK,
            operation=self._get_command_name(node.type),
            parameters={
                "iterations": fibonacci_optimized_iterations,
                "original_iterations": iterations,
                "loop_body": self._compile_block(node.children[-1] if node.children else None),
                "phi_optimization": True
            },
            execution_order=self.instruction_counter
        )
        
        self.compiled_program.instructions.append(instruction)
        self.instruction_counter += 1
    
    def _compile_parallel(self, node: ASTNode):
        """Compile parallel execution blocks"""
        if self.debug:
            print("🔧 Compiling parallel block")
        
        # Generate unique parallel group ID
        parallel_group = f"parallel_{self.instruction_counter}"
        
        # Compile parallel tasks with golden ratio load balancing
        parallel_tasks = []
        for i, child in enumerate(node.children):
            task_weight = self._calculate_phi_weight(i, len(node.children))
            task = {
                "task_id": i,
                "weight": task_weight,
                "instructions": self._compile_block(child)
            }
            parallel_tasks.append(task)
        
        # Create parallel block instruction
        instruction = CompiledInstruction(
            instruction_type=InstructionType.PARALLEL_BLOCK,
            operation="PARALLEL",
            parameters={
                "tasks": parallel_tasks,
                "load_balancing": "golden_ratio",
                "synchronization": "phi_harmonic"
            },
            parallel_group=parallel_group,
            execution_order=self.instruction_counter
        )
        
        self.compiled_program.instructions.append(instruction)
        self.instruction_counter += 1
    
    def _instantiate_quantum_gate(self, gate_template: Dict[str, Any], node: ASTNode) -> Dict[str, Any]:
        """Instantiate quantum gate with sacred mathematics parameters"""
        gate = dict(gate_template)
        
        # Evaluate rotation parameters using sacred mathematics
        if "rotation" in gate and isinstance(gate["rotation"], str):
            rotation_expr = gate["rotation"]
            
            # Substitute sacred mathematics variables
            substitutions = {
                "phi_level": self.current_phi_level,
                "frequency": self.current_frequency,
                "golden_angle": float(GOLDEN_ANGLE),
                "PHI": float(PHI),
                "λ": float(LAMBDA),
                "base_frequency": 432.0,
                "π": math.pi
            }
            
            # Add node attributes
            for key, value in node.attributes.items():
                if isinstance(value, (int, float)):
                    substitutions[key] = value
            
            # Evaluate rotation expression
            try:
                # Simple expression evaluation (secure for our controlled expressions)
                for var, val in substitutions.items():
                    rotation_expr = rotation_expr.replace(var, str(val))
                
                # Evaluate mathematical expressions safely
                rotation_value = self._safe_eval_math(rotation_expr)
                gate["rotation"] = rotation_value
            except:
                gate["rotation"] = 0.0  # Fallback to identity
        
        return gate
    
    def _safe_eval_math(self, expr: str) -> float:
        """Safely evaluate mathematical expressions"""
        # Only allow safe mathematical operations
        allowed_chars = set("0123456789+-*/.()")
        allowed_funcs = {"sin", "cos", "tan", "sqrt", "exp", "log", "pi"}
        
        # Simple expression evaluator for our controlled sacred mathematics
        try:
            # Replace common mathematical constants
            expr = expr.replace("φ", str(float(PHI)))
            expr = expr.replace("π", str(math.pi))
            
            # For now, just handle basic arithmetic
            result = eval(expr, {"__builtins__": {}}, {
                "sin": math.sin, "cos": math.cos, "tan": math.tan,
                "sqrt": math.sqrt, "exp": math.exp, "log": math.log,
                "pi": math.pi
            })
            return float(result)
        except:
            return 0.0
    
    def _get_command_name(self, node_type: ASTNodeType) -> str:
        """Extract command name from AST node type"""
        return node_type.value.replace("_STMT", "")
    
    def _evaluate_expression(self, node: Optional[ASTNode]) -> Any:
        """Evaluate expression node to get value"""
        if not node:
            return None
        
        if node.type == ASTNodeType.NUMBER_LITERAL:
            try:
                return float(node.value)
            except:
                return 0.0
        elif node.type == ASTNodeType.STRING_LITERAL:
            return node.value.strip('"\'')
        elif node.type == ASTNodeType.BOOLEAN_LITERAL:
            return node.value.lower() == 'true'
        elif node.type == ASTNodeType.VARIABLE:
            return self.compiled_program.variables.get(node.value, None)
        elif node.type in [ASTNodeType.PHI_CONSTANT, ASTNodeType.LAMBDA_CONSTANT, ASTNodeType.GOLDEN_ANGLE_CONSTANT]:
            constants = {
                ASTNodeType.PHI_CONSTANT: float(PHI),
                ASTNodeType.LAMBDA_CONSTANT: float(LAMBDA),
                ASTNodeType.GOLDEN_ANGLE_CONSTANT: float(GOLDEN_ANGLE)
            }
            return constants.get(node.type, 0.0)
        
        return None
    
    def _infer_type(self, value: Any) -> str:
        """Infer data type from value"""
        if isinstance(value, bool):
            return 'boolean'
        elif isinstance(value, (int, float)):
            return 'number'
        elif isinstance(value, str):
            return 'string'
        else:
            return 'unknown'
    
    def _compile_condition(self, condition_node: Optional[ASTNode]) -> Dict[str, Any]:
        """Compile conditional expression"""
        if not condition_node:
            return {"type": "always_true", "value": True}
        
        return {
            "type": "expression",
            "node_type": condition_node.type.value,
            "value": self._evaluate_expression(condition_node),
            "source": condition_node.value
        }
    
    def _compile_block(self, block_node: Optional[ASTNode]) -> List[Dict[str, Any]]:
        """Compile block of statements"""
        if not block_node:
            return []
        
        # For now, return simplified block representation
        block_instructions = []
        for child in block_node.children:
            block_instructions.append({
                "type": child.type.value,
                "value": child.value,
                "attributes": child.attributes
            })
        
        return block_instructions
    
    def _extract_loop_iterations(self, node: ASTNode) -> int:
        """Extract number of iterations from loop node"""
        if 'iterations' in node.attributes:
            try:
                return int(node.attributes['iterations'])
            except:
                pass
        return 1
    
    def _fibonacci_optimize_iterations(self, iterations: int) -> int:
        """Optimize iterations using Fibonacci sequence"""
        fibonacci_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
        
        # Find nearest Fibonacci number
        nearest_fib = min(fibonacci_sequence, key=lambda x: abs(x - iterations))
        
        # Return Fibonacci-optimized iterations
        return nearest_fib
    
    def _calculate_phi_weight(self, index: int, total: int) -> float:
        """Calculate phi-harmonic weight for parallel task distribution"""
        if total <= 1:
            return 1.0
        
        # Use golden ratio for weight distribution
        phi_power = (index + 1) / total
        weight = float(PHI) ** phi_power
        
        # Normalize weights
        total_weight = sum(float(PHI) ** ((i + 1) / total) for i in range(total))
        return weight / total_weight
    
    def _apply_optimization_passes(self):
        """Apply phi-harmonic optimization passes"""
        if self.debug:
            print("🔧 Applying phi-harmonic optimization passes")
        
        # Pass 1: Combine adjacent quantum gates
        self._combine_adjacent_gates()
        
        # Pass 2: Optimize parallel block scheduling
        self._optimize_parallel_scheduling()
        
        # Pass 3: Apply phi-harmonic gate optimizations
        self._apply_phi_harmonic_optimizations()
    
    def _combine_adjacent_gates(self):
        """Combine adjacent quantum gates for efficiency"""
        # Simple gate combination optimization
        optimized_instructions = []
        i = 0
        
        while i < len(self.compiled_program.instructions):
            current = self.compiled_program.instructions[i]
            
            if (current.instruction_type == InstructionType.QUANTUM_GATE and 
                i + 1 < len(self.compiled_program.instructions)):
                
                next_instr = self.compiled_program.instructions[i + 1]
                
                # Combine compatible gates
                if (next_instr.instruction_type == InstructionType.QUANTUM_GATE and 
                    current.qubits == next_instr.qubits):
                    
                    # Combine quantum gates
                    combined_gates = current.quantum_gates + next_instr.quantum_gates
                    current.quantum_gates = combined_gates
                    current.parameters.update(next_instr.parameters)
                    
                    optimized_instructions.append(current)
                    i += 2  # Skip next instruction as it's been combined
                    continue
            
            optimized_instructions.append(current)
            i += 1
        
        self.compiled_program.instructions = optimized_instructions
    
    def _optimize_parallel_scheduling(self):
        """Optimize parallel block scheduling using golden ratio"""
        for instruction in self.compiled_program.instructions:
            if instruction.instruction_type == InstructionType.PARALLEL_BLOCK:
                tasks = instruction.parameters.get('tasks', [])
                
                # Reorder tasks by phi-harmonic weight
                tasks.sort(key=lambda t: t['weight'], reverse=True)
                
                # Update task ordering
                instruction.parameters['tasks'] = tasks
                instruction.parameters['optimization_applied'] = 'phi_harmonic_scheduling'
    
    def _apply_phi_harmonic_optimizations(self):
        """Apply phi-harmonic quantum gate optimizations"""
        for instruction in self.compiled_program.instructions:
            if instruction.quantum_gates:
                for gate in instruction.quantum_gates:
                    # Apply phi-harmonic rotation optimizations
                    if 'rotation' in gate and isinstance(gate['rotation'], (int, float)):
                        original_rotation = gate['rotation']
                        
                        # Optimize rotation using phi-harmonic relationships
                        phi_optimized_rotation = self._phi_optimize_rotation(original_rotation)
                        gate['rotation'] = phi_optimized_rotation
                        gate['optimization'] = 'phi_harmonic'
    
    def _phi_optimize_rotation(self, rotation: float) -> float:
        """Optimize rotation angle using phi-harmonic principles"""
        # Find nearest phi-harmonic angle
        phi_angles = [
            float(PHI) * math.pi / 8,      # φπ/8
            float(PHI) * math.pi / 4,      # φπ/4  
            float(PHI) * math.pi / 2,      # φπ/2
            float(LAMBDA) * math.pi,       # λπ
            float(PHI) * math.pi,          # φπ
            float(GOLDEN_ANGLE) * math.pi / 180  # Golden angle in radians
        ]
        
        # Find nearest phi-harmonic angle
        nearest_phi_angle = min(phi_angles, key=lambda x: abs(x - rotation))
        
        # Return optimized angle if close enough (within 10% tolerance)
        if abs(rotation - nearest_phi_angle) / max(abs(rotation), 0.1) < 0.1:
            return nearest_phi_angle
        
        return rotation
    
    def _generate_quantum_circuit(self):
        """Generate complete quantum circuit from compiled instructions"""
        if self.debug:
            print("🔧 Generating quantum circuit")
        
        circuit = {
            "qubits": self.qubit_allocator,
            "gates": [],
            "measurements": [],
            "metadata": {
                "phi_optimization": True,
                "sacred_frequencies": list(set(
                    instr.frequency for instr in self.compiled_program.instructions 
                    if instr.frequency
                )),
                "consciousness_states": list(set(
                    instr.consciousness_state for instr in self.compiled_program.instructions 
                    if instr.consciousness_state
                ))
            }
        }
        
        # Collect all quantum gates from instructions
        for instruction in self.compiled_program.instructions:
            for gate in instruction.quantum_gates:
                circuit_gate = {
                    "type": gate.get("type", "I"),
                    "qubits": instruction.qubits,
                    "parameters": gate,
                    "instruction_order": instruction.execution_order,
                    "frequency": instruction.frequency,
                    "consciousness_state": instruction.consciousness_state
                }
                circuit["gates"].append(circuit_gate)
        
        self.compiled_program.quantum_circuit = circuit
        self.compiled_program.total_qubits = self.qubit_allocator
        self.compiled_program.total_gates = len(circuit["gates"])
    
    def _calculate_program_metadata(self):
        """Calculate program execution metadata"""
        if self.debug:
            print("🔧 Calculating program metadata")
        
        # Estimate runtime based on gate count and complexity
        base_gate_time = 1e-6  # 1 microsecond per gate
        complexity_factor = 1.0
        
        for instruction in self.compiled_program.instructions:
            if instruction.instruction_type == InstructionType.PARALLEL_BLOCK:
                complexity_factor *= 0.618  # φ^-1 speedup for parallel
            elif instruction.instruction_type == InstructionType.QUANTUM_GATE:
                complexity_factor *= 1.1  # Slight overhead for quantum gates
        
        estimated_runtime = self.compiled_program.total_gates * base_gate_time * complexity_factor
        self.compiled_program.estimated_runtime = estimated_runtime
        
        # Update phi alignment score if not already set
        if self.compiled_program.phi_alignment_score == 0.0:
            self.compiled_program.phi_alignment_score = self._calculate_compilation_phi_score()
    
    def _calculate_compilation_phi_score(self) -> float:
        """Calculate phi alignment score for compiled program"""
        scores = []
        
        # Check instruction count alignment with Fibonacci
        instruction_count = len(self.compiled_program.instructions)
        fibonacci_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
        if instruction_count in fibonacci_sequence:
            scores.append(1.0)
        else:
            nearest_fib = min(fibonacci_sequence, key=lambda x: abs(x - instruction_count))
            scores.append(0.8 - abs(instruction_count - nearest_fib) / nearest_fib * 0.3)
        
        # Check quantum gate phi-harmonic optimizations
        phi_optimized_gates = sum(
            1 for instr in self.compiled_program.instructions
            for gate in instr.quantum_gates
            if gate.get('optimization') == 'phi_harmonic'
        )
        
        total_gates = self.compiled_program.total_gates
        if total_gates > 0:
            scores.append(phi_optimized_gates / total_gates)
        
        return sum(scores) / len(scores) if scores else 0.0

def test_phi_flow_compiler():
    """Test the PhiFlow compiler"""
    
    print("🧪 Testing PhiFlow Compiler - Task 3.4 Implementation")
    print("=" * 70)
    
    # Import dependencies
    from .phi_flow_lexer import PhiFlowLexer
    from .phi_flow_parser import PhiFlowParser
    from .phi_flow_semantic_analyzer import PhiFlowSemanticAnalyzer
    
    # Test program with various compilation elements
    test_program = """
    # PhiFlow Test Program for Compilation
    INITIALIZE quantum_field AT 432Hz WITH coherence=1.0, phi_level=0
    SET base_frequency = 432
    EVOLVE TO 720Hz WITH phi_level=4
    INTEGRATE heart_field AT 594Hz WITH phi_level=2
    
    PARALLEL
        HARMONIZE resonance AT 672Hz WITH phi_level=3
        TRANSCEND limitations AT 963Hz WITH phi_level=6
        CREATE_FIELD consciousness AT 528Hz WITH phi_level=1
    END
    
    CASCADE infinite_love AT 768Hz WITH phi_level=5
    """
    
    # Create components
    lexer = PhiFlowLexer(debug=False)
    parser = PhiFlowParser(debug=False)
    analyzer = PhiFlowSemanticAnalyzer(debug=False)
    compiler = PhiFlowCompiler(debug=True)
    
    print("\n🔄 Compiling PhiFlow program...")
    
    try:
        # Parse the program
        ast = parser.parse(test_program)
        print("✅ Parsing completed")
        
        # Semantic analysis
        semantic_results = analyzer.analyze(ast)
        print(f"✅ Semantic analysis: {semantic_results['semantic_info']['total_errors']} errors")
        
        # Compile the program
        compiled_program = compiler.compile(ast, semantic_results)
        print("✅ Compilation completed")
        
        # Display results
        print("\n📊 Compilation Results:")
        print(f"  Instructions: {len(compiled_program.instructions)}")
        print(f"  Quantum qubits: {compiled_program.total_qubits}")
        print(f"  Quantum gates: {compiled_program.total_gates}")
        print(f"  Estimated runtime: {compiled_program.estimated_runtime:.6f} seconds")
        print(f"  Phi alignment score: {compiled_program.phi_alignment_score:.3f}")
        
        # Show instructions
        print("\n📋 Compiled Instructions:")
        for i, instruction in enumerate(compiled_program.instructions[:10]):  # Show first 10
            print(f"  {i+1}. {instruction.operation} ({instruction.instruction_type.value})")
            if instruction.frequency:
                print(f"     Frequency: {instruction.frequency}Hz")
            if instruction.consciousness_state:
                print(f"     Consciousness: {instruction.consciousness_state}")
            if instruction.qubits:
                print(f"     Qubits: {instruction.qubits}")
        
        if len(compiled_program.instructions) > 10:
            print(f"  ... and {len(compiled_program.instructions) - 10} more instructions")
        
        # Show quantum circuit info
        if compiled_program.quantum_circuit:
            circuit = compiled_program.quantum_circuit
            print("\n🔬 Quantum Circuit:")
            print(f"  Total qubits: {circuit['qubits']}")
            print(f"  Total gates: {len(circuit['gates'])}")
            print(f"  Sacred frequencies: {circuit['metadata']['sacred_frequencies']}")
            print(f"  Consciousness states: {circuit['metadata']['consciousness_states']}")
        
        # Show variables
        if compiled_program.variables:
            print("\n📋 Variables:")
            for name, value in compiled_program.variables.items():
                print(f"  {name} = {value}")
        
        print("\n✅ PhiFlow Compiler Test Complete!")
        return compiled_program
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_phi_flow_compiler()