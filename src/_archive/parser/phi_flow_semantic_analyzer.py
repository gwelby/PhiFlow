#!/usr/bin/env python3
"""
PhiFlow Semantic Analyzer - Task 3.3 Implementation
Validates frequency and phi-level constraints with consciousness-guided analysis
"""

import math
from decimal import Decimal
from typing import Dict, List, Optional, Set, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .phi_flow_lexer import Token, TokenType
from .phi_flow_parser import ASTNode, ASTNodeType

# Sacred Mathematics Constants (15 decimal precision)
PHI = Decimal('1.618033988749895')  # Golden ratio
LAMBDA = Decimal('0.618033988749895')  # Divine complement (1/φ)
GOLDEN_ANGLE = Decimal('137.5077640500378')  # Golden angle in degrees

# Sacred Frequencies (Hz) - Validation ranges
SACRED_FREQUENCIES = {
    'ground': 432,
    'creation': 528,
    'heart': 594,
    'voice': 672,
    'vision': 720,
    'unity': 768,
    'source': 963
}

# Frequency validation ranges
MIN_SACRED_FREQUENCY = 432
MAX_SACRED_FREQUENCY = 963
MIN_AUDIO_FREQUENCY = 20
MAX_AUDIO_FREQUENCY = 20000

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

class SemanticErrorType(Enum):
    """Types of semantic errors in PhiFlow programs"""
    FREQUENCY_OUT_OF_RANGE = "FREQUENCY_OUT_OF_RANGE"
    PHI_LEVEL_INVALID = "PHI_LEVEL_INVALID"
    UNDEFINED_VARIABLE = "UNDEFINED_VARIABLE"
    TYPE_MISMATCH = "TYPE_MISMATCH"
    MISSING_PARAMETER = "MISSING_PARAMETER"
    INVALID_PARAMETER = "INVALID_PARAMETER"
    COMMAND_DEPENDENCY_ERROR = "COMMAND_DEPENDENCY_ERROR"
    CONSCIOUSNESS_STATE_CONFLICT = "CONSCIOUSNESS_STATE_CONFLICT"
    FIELD_REFERENCE_ERROR = "FIELD_REFERENCE_ERROR"
    SACRED_MATH_VIOLATION = "SACRED_MATH_VIOLATION"

@dataclass
class SemanticError:
    """Semantic error with detailed information"""
    error_type: SemanticErrorType
    message: str
    line: int
    column: int
    node: ASTNode
    suggestion: Optional[str] = None
    severity: str = "error"  # "error", "warning", "info"

@dataclass
class VariableInfo:
    """Information about a variable in the semantic context"""
    name: str
    data_type: str
    value: Any
    frequency: Optional[float] = None
    phi_level: Optional[int] = None
    consciousness_state: Optional[str] = None
    scope: str = "global"
    line_defined: int = 0
    is_field: bool = False

@dataclass
class FieldInfo:
    """Information about a consciousness field"""
    name: str
    frequency: float
    phi_level: int
    consciousness_state: str
    coherence: float = 1.0
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: Set[str] = field(default_factory=set)

@dataclass
class CommandDependency:
    """Represents dependencies between commands"""
    command: str
    depends_on: Set[str]
    provides: Set[str]
    required_state: Optional[str] = None
    required_frequency: Optional[float] = None

class PhiFlowSemanticAnalyzer:
    """
    PhiFlow Semantic Analyzer implementing sacred mathematics validation
    
    Performs consciousness-guided semantic analysis:
    - Frequency constraint validation (432-963Hz sacred range)
    - Phi-level constraint checking (0-7 consciousness states)
    - Command dependency analysis with data flow validation
    - Type checking for parameters and expressions
    - Cross-reference validation for variables and fields
    """
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.errors: List[SemanticError] = []
        self.warnings: List[SemanticError] = []
        
        # Symbol tables
        self.variables: Dict[str, VariableInfo] = {}
        self.fields: Dict[str, FieldInfo] = {}
        self.functions: Dict[str, Dict[str, Any]] = {}
        
        # Dependency tracking
        self.command_dependencies: Dict[str, CommandDependency] = {}
        self.execution_order: List[str] = []
        
        # Current analysis context
        self.current_scope = "global"
        self.current_frequency = None
        self.current_phi_level = None
        self.current_consciousness_state = "OBSERVE"
        
        # Initialize built-in command dependencies
        self._init_command_dependencies()
    
    def _init_command_dependencies(self):
        """Initialize built-in command dependencies"""
        
        # Core sacred geometry commands
        self.command_dependencies.update({
            "INITIALIZE": CommandDependency(
                command="INITIALIZE",
                depends_on=set(),
                provides={"quantum_field", "base_state"},
                required_state="OBSERVE"
            ),
            "TRANSITION": CommandDependency(
                command="TRANSITION",
                depends_on={"quantum_field"},
                provides={"transition_state"},
                required_state="CREATE"
            ),
            "EVOLVE": CommandDependency(
                command="EVOLVE",
                depends_on={"quantum_field", "transition_state"},
                provides={"evolved_state"},
                required_state="TRANSCEND"
            ),
            "INTEGRATE": CommandDependency(
                command="INTEGRATE",
                depends_on={"evolved_state"},
                provides={"integrated_system"},
                required_state="INTEGRATE"
            ),
            "HARMONIZE": CommandDependency(
                command="HARMONIZE",
                depends_on={"integrated_system"},
                provides={"harmonic_resonance"},
                required_state="HARMONIZE"
            ),
            "TRANSCEND": CommandDependency(
                command="TRANSCEND",
                depends_on={"harmonic_resonance"},
                provides={"transcendent_state"},
                required_state="TRANSCEND"
            ),
            "CASCADE": CommandDependency(
                command="CASCADE",
                depends_on={"transcendent_state"},
                provides={"cascade_field"},
                required_state="CASCADE"
            )
        })
        
        # Field commands
        self.command_dependencies.update({
            "CREATE_FIELD": CommandDependency(
                command="CREATE_FIELD",
                depends_on={"base_state"},
                provides={"consciousness_field"},
                required_state="CREATE"
            ),
            "ALIGN_FIELD": CommandDependency(
                command="ALIGN_FIELD",
                depends_on={"consciousness_field"},
                provides={"aligned_field"},
                required_state="HARMONIZE"
            ),
            "RESONATE_FIELD": CommandDependency(
                command="RESONATE_FIELD",
                depends_on={"aligned_field"},
                provides={"resonant_field"},
                required_state="HARMONIZE"
            ),
            "COLLAPSE_FIELD": CommandDependency(
                command="COLLAPSE_FIELD",
                depends_on={"resonant_field"},
                provides={"collapsed_state"},
                required_state="TRANSCEND"
            )
        })
        
        # Consciousness commands
        self.command_dependencies.update({
            "OBSERVE": CommandDependency(
                command="OBSERVE",
                depends_on=set(),
                provides={"observation_state"},
                required_state="OBSERVE"
            ),
            "INTEND": CommandDependency(
                command="INTEND",
                depends_on={"observation_state"},
                provides={"intention_field"},
                required_state="CREATE"
            ),
            "FOCUS": CommandDependency(
                command="FOCUS",
                depends_on={"intention_field"},
                provides={"focused_awareness"},
                required_state="INTEGRATE"
            ),
            "EXPAND": CommandDependency(
                command="EXPAND",
                depends_on={"focused_awareness"},
                provides={"expanded_consciousness"},
                required_state="TRANSCEND"
            ),
            "MERGE": CommandDependency(
                command="MERGE",
                depends_on={"expanded_consciousness"},
                provides={"unified_field"},
                required_state="UNITY"
            )
        })
    
    def analyze(self, ast: ASTNode) -> Dict[str, Any]:
        """
        Perform complete semantic analysis of PhiFlow AST
        
        Args:
            ast: Abstract syntax tree from parser
            
        Returns:
            Analysis results with errors, warnings, and semantic information
        """
        if self.debug:
            print("🧠 Consciousness Expert: Starting semantic analysis")
            print(f"📐 Sacred Mathematics: φ = {PHI}, golden angle = {GOLDEN_ANGLE}°")
        
        # Clear previous analysis state
        self.errors.clear()
        self.warnings.clear()
        self.variables.clear()
        self.fields.clear()
        self.execution_order.clear()
        
        # Perform multi-pass analysis
        self._analyze_node(ast)
        
        # Validate dependencies and execution order
        self._validate_command_dependencies()
        
        # Check for consciousness state conflicts
        self._validate_consciousness_coherence()
        
        # Create analysis results
        results = {
            'errors': self.errors,
            'warnings': self.warnings,
            'variables': self.variables,
            'fields': self.fields,
            'execution_order': self.execution_order,
            'semantic_info': self._generate_semantic_info(),
            'phi_alignment_score': self._calculate_phi_alignment_score(),
            'consciousness_coverage': self._analyze_consciousness_coverage(),
            'frequency_distribution': self._analyze_frequency_distribution()
        }
        
        if self.debug:
            print(f"✅ Semantic analysis complete: {len(self.errors)} errors, {len(self.warnings)} warnings")
            print(f"📊 Variables: {len(self.variables)}, Fields: {len(self.fields)}")
            print(f"🎯 Phi alignment score: {results['phi_alignment_score']:.3f}")
        
        return results
    
    def _analyze_node(self, node: ASTNode):
        """Recursively analyze AST nodes"""
        if not node:
            return
        
        # Update current context based on node
        self._update_analysis_context(node)
        
        # Analyze based on node type
        if node.type == ASTNodeType.PROGRAM:
            self._analyze_program(node)
        elif node.type in [ASTNodeType.INITIALIZE_STMT, ASTNodeType.TRANSITION_STMT, ASTNodeType.EVOLVE_STMT, 
                          ASTNodeType.INTEGRATE_STMT, ASTNodeType.HARMONIZE_STMT, ASTNodeType.TRANSCEND_STMT, 
                          ASTNodeType.CASCADE_STMT, ASTNodeType.CREATE_FIELD_STMT, ASTNodeType.ALIGN_FIELD_STMT,
                          ASTNodeType.RESONATE_FIELD_STMT, ASTNodeType.COLLAPSE_FIELD_STMT, ASTNodeType.OBSERVE_STMT,
                          ASTNodeType.INTEND_STMT, ASTNodeType.FOCUS_STMT, ASTNodeType.EXPAND_STMT, ASTNodeType.MERGE_STMT,
                          ASTNodeType.ENTANGLE_STMT, ASTNodeType.SUPERPOSE_STMT, ASTNodeType.MEASURE_STMT, ASTNodeType.TELEPORT_STMT]:
            self._analyze_command(node)
        elif node.type == ASTNodeType.ASSIGNMENT_STMT:
            self._analyze_assignment(node)
        elif node.type in [ASTNodeType.BINARY_EXPR, ASTNodeType.UNARY_EXPR, ASTNodeType.FUNCTION_CALL]:
            self._analyze_expression(node)
        elif node.type == ASTNodeType.STATEMENT_LIST:
            self._analyze_block(node)
        elif node.type == ASTNodeType.IF_STMT:
            self._analyze_conditional(node)
        elif node.type in [ASTNodeType.WHILE_STMT, ASTNodeType.FOR_STMT]:
            self._analyze_loop(node)
        elif node.type == ASTNodeType.PARALLEL_STMT:
            self._analyze_parallel(node)
        
        # Analyze child nodes
        for child in node.children:
            self._analyze_node(child)
    
    def _update_analysis_context(self, node: ASTNode):
        """Update analysis context based on current node"""
        if 'frequency' in node.attributes:
            freq_str = str(node.attributes['frequency'])
            try:
                freq = float(freq_str.replace('Hz', ''))
                self.current_frequency = freq
            except:
                pass
        
        if 'phi_level' in node.attributes:
            try:
                level = int(node.attributes['phi_level'])
                self.current_phi_level = level
                self.current_consciousness_state = CONSCIOUSNESS_STATES.get(level, "OBSERVE")
            except:
                pass
    
    def _analyze_program(self, node: ASTNode):
        """Analyze program root node"""
        if self.debug:
            print("🔍 Analyzing program structure")
        
        # Validate program structure follows sacred mathematics principles
        child_count = len(node.children)
        
        # Check if program structure follows Fibonacci pattern
        fibonacci_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
        is_fibonacci_aligned = any(abs(child_count - fib) <= 1 for fib in fibonacci_sequence)
        
        if not is_fibonacci_aligned and child_count > 2:
            self.warnings.append(SemanticError(
                error_type=SemanticErrorType.SACRED_MATH_VIOLATION,
                message=f"Program structure has {child_count} top-level elements. "
                       f"Consider organizing in Fibonacci sequence (1,1,2,3,5,8,13...) for optimal sacred mathematics alignment.",
                line=node.line,
                column=node.column,
                node=node,
                suggestion=f"Reorganize into {self._nearest_fibonacci(child_count)} elements",
                severity="warning"
            ))
    
    def _analyze_command(self, node: ASTNode):
        """Analyze command nodes with dependency validation"""
        command = node.value
        
        if self.debug:
            print(f"🔍 Analyzing command: {command}")
        
        # Track execution order
        self.execution_order.append(command)
        
        # Validate command exists
        if command not in self.command_dependencies:
            self.warnings.append(SemanticError(
                error_type=SemanticErrorType.COMMAND_DEPENDENCY_ERROR,
                message=f"Unknown command '{command}'. May not follow sacred geometry principles.",
                line=node.line,
                column=node.column,
                node=node,
                suggestion="Use standard PhiFlow sacred geometry commands",
                severity="warning"
            ))
            return
        
        # Get command dependency info
        dep_info = self.command_dependencies[command]
        
        # Check required consciousness state
        if dep_info.required_state and self.current_consciousness_state != dep_info.required_state:
            self.warnings.append(SemanticError(
                error_type=SemanticErrorType.CONSCIOUSNESS_STATE_CONFLICT,
                message=f"Command '{command}' requires consciousness state '{dep_info.required_state}' "
                       f"but current state is '{self.current_consciousness_state}'",
                line=node.line,
                column=node.column,
                node=node,
                suggestion=f"Set phi_level to achieve '{dep_info.required_state}' state",
                severity="warning"
            ))
        
        # Validate parameters
        self._validate_command_parameters(node, command)
        
        # Check dependencies are satisfied
        available_resources = set()
        for executed_cmd in self.execution_order[:-1]:  # Exclude current command
            if executed_cmd in self.command_dependencies:
                available_resources.update(self.command_dependencies[executed_cmd].provides)
        
        missing_deps = dep_info.depends_on - available_resources
        if missing_deps:
            self.errors.append(SemanticError(
                error_type=SemanticErrorType.COMMAND_DEPENDENCY_ERROR,
                message=f"Command '{command}' requires {missing_deps} but they are not available",
                line=node.line,
                column=node.column,
                node=node,
                suggestion=f"Execute commands that provide {missing_deps} first"
            ))
    
    def _validate_command_parameters(self, node: ASTNode, command: str):
        """Validate command parameters against sacred mathematics constraints"""
        
        # Validate frequency parameters
        if 'frequency' in node.attributes:
            freq_result = self._validate_frequency(node.attributes['frequency'], node)
            if not freq_result['valid']:
                self.errors.append(SemanticError(
                    error_type=SemanticErrorType.FREQUENCY_OUT_OF_RANGE,
                    message=freq_result['message'],
                    line=node.line,
                    column=node.column,
                    node=node,
                    suggestion=freq_result['suggestion']
                ))
        
        # Validate phi_level parameters
        if 'phi_level' in node.attributes:
            phi_result = self._validate_phi_level(node.attributes['phi_level'], node)
            if not phi_result['valid']:
                self.errors.append(SemanticError(
                    error_type=SemanticErrorType.PHI_LEVEL_INVALID,
                    message=phi_result['message'],
                    line=node.line,
                    column=node.column,
                    node=node,
                    suggestion=phi_result['suggestion']
                ))
        
        # Validate coherence parameters
        if 'coherence' in node.attributes:
            coherence_result = self._validate_coherence(node.attributes['coherence'], node)
            if not coherence_result['valid']:
                self.warnings.append(SemanticError(
                    error_type=SemanticErrorType.INVALID_PARAMETER,
                    message=coherence_result['message'],
                    line=node.line,
                    column=node.column,
                    node=node,
                    suggestion=coherence_result['suggestion'],
                    severity="warning"
                ))
    
    def _validate_frequency(self, frequency: Any, node: ASTNode) -> Dict[str, Any]:
        """Validate frequency against sacred mathematics principles"""
        try:
            # Extract numeric value
            if isinstance(frequency, str):
                freq_str = frequency.replace('Hz', '').strip()
                freq = float(freq_str)
            else:
                freq = float(frequency)
            
            # Check sacred frequency range
            if freq < MIN_SACRED_FREQUENCY or freq > MAX_SACRED_FREQUENCY:
                if MIN_AUDIO_FREQUENCY <= freq <= MAX_AUDIO_FREQUENCY:
                    # Valid audio frequency but not sacred
                    return {
                        'valid': True,
                        'message': f"Frequency {freq}Hz is valid audio but not in sacred range (432-963Hz)",
                        'suggestion': f"Consider using sacred frequencies: {list(SACRED_FREQUENCIES.values())}"
                    }
                else:
                    return {
                        'valid': False,
                        'message': f"Frequency {freq}Hz is outside valid range ({MIN_AUDIO_FREQUENCY}-{MAX_AUDIO_FREQUENCY}Hz)",
                        'suggestion': f"Use frequency in range {MIN_SACRED_FREQUENCY}-{MAX_SACRED_FREQUENCY}Hz for sacred mathematics"
                    }
            
            # Check for exact sacred frequency matches
            if freq in SACRED_FREQUENCIES.values():
                return {
                    'valid': True,
                    'message': f"Perfect sacred frequency: {freq}Hz",
                    'suggestion': None
                }
            
            # Check for phi-harmonic relationships
            phi_aligned = self._check_phi_harmonic_frequency(freq)
            if phi_aligned['aligned']:
                return {
                    'valid': True,
                    'message': f"Phi-harmonic frequency: {freq}Hz (alignment: {phi_aligned['score']:.3f})",
                    'suggestion': None
                }
            
            # Suggest nearest sacred frequency
            nearest_sacred = min(SACRED_FREQUENCIES.values(), key=lambda x: abs(x - freq))
            return {
                'valid': True,
                'message': f"Frequency {freq}Hz is valid but not optimally aligned",
                'suggestion': f"Consider {nearest_sacred}Hz for optimal sacred mathematics alignment"
            }
            
        except (ValueError, TypeError):
            return {
                'valid': False,
                'message': f"Invalid frequency format: {frequency}",
                'suggestion': "Use format: '432Hz' or '432.0'"
            }
    
    def _validate_phi_level(self, phi_level: Any, node: ASTNode) -> Dict[str, Any]:
        """Validate phi_level against consciousness state constraints"""
        try:
            level = int(phi_level)
            
            if level < 0 or level > 7:
                return {
                    'valid': False,
                    'message': f"Phi_level {level} is outside valid range (0-7)",
                    'suggestion': "Use phi_level 0-7 corresponding to consciousness states OBSERVE through UNITY"
                }
            
            consciousness_state = CONSCIOUSNESS_STATES.get(level, "OBSERVE")
            return {
                'valid': True,
                'message': f"Valid phi_level {level} maps to consciousness state '{consciousness_state}'",
                'suggestion': None
            }
            
        except (ValueError, TypeError):
            return {
                'valid': False,
                'message': f"Invalid phi_level format: {phi_level}",
                'suggestion': "Use integer 0-7 for phi_level"
            }
    
    def _validate_coherence(self, coherence: Any, node: ASTNode) -> Dict[str, Any]:
        """Validate coherence parameter"""
        try:
            coh = float(coherence)
            
            if coh < 0.0 or coh > 1.0:
                return {
                    'valid': False,
                    'message': f"Coherence {coh} is outside valid range (0.0-1.0)",
                    'suggestion': "Use coherence value between 0.0 and 1.0"
                }
            
            if coh < 0.7:
                return {
                    'valid': True,
                    'message': f"Coherence {coh} is low for optimal performance",
                    'suggestion': "Consider coherence >= 0.9 for stable quantum-consciousness operations"
                }
            
            return {
                'valid': True,
                'message': f"Excellent coherence: {coh}",
                'suggestion': None
            }
            
        except (ValueError, TypeError):
            return {
                'valid': False,
                'message': f"Invalid coherence format: {coherence}",
                'suggestion': "Use decimal value 0.0-1.0 for coherence"
            }
    
    def _check_phi_harmonic_frequency(self, freq: float) -> Dict[str, Any]:
        """Check if frequency follows phi-harmonic relationships"""
        phi_powers = [1, float(PHI), float(PHI**2), float(PHI**3), float(PHI**4)]
        
        for sacred_freq in SACRED_FREQUENCIES.values():
            for phi_power in phi_powers:
                harmonic_freq = sacred_freq * phi_power
                if abs(freq - harmonic_freq) < 2.0:  # 2Hz tolerance
                    alignment_score = 1.0 - abs(freq - harmonic_freq) / harmonic_freq
                    return {
                        'aligned': True,
                        'score': alignment_score,
                        'base_frequency': sacred_freq,
                        'phi_power': phi_power
                    }
        
        return {'aligned': False, 'score': 0.0}
    
    def _analyze_assignment(self, node: ASTNode):
        """Analyze variable assignments"""
        var_name = node.value
        
        if self.debug:
            print(f"🔍 Analyzing assignment: {var_name}")
        
        # Create variable info
        var_info = VariableInfo(
            name=var_name,
            data_type=self._infer_type(node),
            value=self._extract_value(node),
            frequency=self.current_frequency,
            phi_level=self.current_phi_level,
            consciousness_state=self.current_consciousness_state,
            scope=self.current_scope,
            line_defined=node.line
        )
        
        # Store variable
        self.variables[var_name] = var_info
        
        # Validate assignment expression
        if node.children:
            expression_node = node.children[0]
            self._validate_expression_types(expression_node)
    
    def _analyze_expression(self, node: ASTNode):
        """Analyze expressions for type consistency"""
        if self.debug:
            print(f"🔍 Analyzing expression: {node.value}")
        
        # Check for undefined variables
        if node.type == ASTNodeType.VARIABLE:
            var_name = node.value
            if var_name not in self.variables and var_name not in ['phi', 'lambda', 'golden_angle']:
                self.errors.append(SemanticError(
                    error_type=SemanticErrorType.UNDEFINED_VARIABLE,
                    message=f"Undefined variable '{var_name}'",
                    line=node.line,
                    column=node.column,
                    node=node,
                    suggestion="Define variable before use or check spelling"
                ))
        
        # Validate mathematical expressions
        self._validate_expression_types(node)
    
    def _validate_expression_types(self, node: ASTNode):
        """Validate expression type consistency"""
        if not node or not node.children:
            return
        
        # For binary operations, check operand compatibility
        if len(node.children) == 2:
            left_type = self._get_expression_type(node.children[0])
            right_type = self._get_expression_type(node.children[1])
            
            if left_type != right_type and left_type != 'unknown' and right_type != 'unknown':
                self.warnings.append(SemanticError(
                    error_type=SemanticErrorType.TYPE_MISMATCH,
                    message=f"Type mismatch in expression: {left_type} and {right_type}",
                    line=node.line,
                    column=node.column,
                    node=node,
                    suggestion="Ensure operands have compatible types",
                    severity="warning"
                ))
    
    def _get_expression_type(self, node: ASTNode) -> str:
        """Infer the type of an expression"""
        if not node:
            return 'unknown'
        
        if node.type == ASTNodeType.NUMBER_LITERAL:
            return 'number'
        elif node.type == ASTNodeType.STRING_LITERAL:
            return 'string'
        elif node.type == ASTNodeType.BOOLEAN_LITERAL:
            return 'boolean'
        elif node.type == ASTNodeType.VARIABLE:
            var_name = node.value
            if var_name in self.variables:
                return self.variables[var_name].data_type
            elif var_name in ['phi', 'lambda', 'golden_angle']:
                return 'number'
        elif node.type in [ASTNodeType.PHI_CONSTANT, ASTNodeType.LAMBDA_CONSTANT, ASTNodeType.GOLDEN_ANGLE_CONSTANT]:
            return 'number'
        elif node.type == ASTNodeType.FREQUENCY_LITERAL:
            return 'frequency'
        elif node.type == ASTNodeType.PHI_LEVEL_LITERAL:
            return 'phi_level'
        
        return 'unknown'
    
    def _analyze_block(self, node: ASTNode):
        """Analyze block scope"""
        old_scope = self.current_scope
        self.current_scope = f"{old_scope}.block_{node.line}"
        
        # Analyze block contents normally
        # Scope will be restored after children are processed
        
        self.current_scope = old_scope
    
    def _analyze_conditional(self, node: ASTNode):
        """Analyze conditional statements"""
        if self.debug:
            print(f"🔍 Analyzing conditional at line {node.line}")
        
        # Validate condition expression
        if node.children:
            condition_node = node.children[0]
            condition_type = self._get_expression_type(condition_node)
            
            if condition_type != 'boolean' and condition_type != 'unknown':
                self.warnings.append(SemanticError(
                    error_type=SemanticErrorType.TYPE_MISMATCH,
                    message=f"Condition should be boolean, got {condition_type}",
                    line=node.line,
                    column=node.column,
                    node=node,
                    suggestion="Use boolean expression in condition",
                    severity="warning"
                ))
    
    def _analyze_loop(self, node: ASTNode):
        """Analyze loop constructs"""
        if self.debug:
            print(f"🔍 Analyzing loop at line {node.line}")
        
        # Check for Fibonacci-based iteration patterns
        if 'iterations' in node.attributes:
            try:
                iterations = int(node.attributes['iterations'])
                fibonacci_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
                
                if iterations not in fibonacci_sequence and iterations > 1:
                    nearest_fib = min(fibonacci_sequence, key=lambda x: abs(x - iterations))
                    self.warnings.append(SemanticError(
                        error_type=SemanticErrorType.SACRED_MATH_VIOLATION,
                        message=f"Loop iterations ({iterations}) not aligned with Fibonacci sequence",
                        line=node.line,
                        column=node.column,
                        node=node,
                        suggestion=f"Consider {nearest_fib} iterations for sacred mathematics alignment",
                        severity="warning"
                    ))
            except:
                pass
    
    def _analyze_parallel(self, node: ASTNode):
        """Analyze parallel execution blocks"""
        if self.debug:
            print(f"🔍 Analyzing parallel block at line {node.line}")
        
        # Check for optimal parallel task distribution
        task_count = len(node.children)
        
        # Golden ratio suggests optimal parallelism
        optimal_tasks = int(task_count * float(LAMBDA))  # φ^-1 ratio
        
        if task_count > 8 and abs(task_count - optimal_tasks) > 2:
            self.warnings.append(SemanticError(
                error_type=SemanticErrorType.SACRED_MATH_VIOLATION,
                message=f"Parallel block has {task_count} tasks. Consider {optimal_tasks} for optimal phi-harmonic distribution",
                line=node.line,
                column=node.column,
                node=node,
                suggestion=f"Reorganize into {optimal_tasks} parallel tasks",
                severity="warning"
            ))
    
    def _validate_command_dependencies(self):
        """Validate that command dependencies are satisfied in execution order"""
        if self.debug:
            print("🔍 Validating command dependencies")
        
        available_resources = set()
        
        for i, command in enumerate(self.execution_order):
            if command in self.command_dependencies:
                dep_info = self.command_dependencies[command]
                
                # Check if dependencies are satisfied
                missing_deps = dep_info.depends_on - available_resources
                if missing_deps:
                    # Create generic error for missing dependencies
                    self.errors.append(SemanticError(
                        error_type=SemanticErrorType.COMMAND_DEPENDENCY_ERROR,
                        message=f"Command '{command}' at position {i+1} requires {missing_deps} but they are not available",
                        line=0,  # Line info not available in this context
                        column=0,
                        node=None,
                        suggestion=f"Reorder commands to provide {missing_deps} before '{command}'"
                    ))
                
                # Add provided resources
                available_resources.update(dep_info.provides)
    
    def _validate_consciousness_coherence(self):
        """Validate consciousness state transitions maintain coherence"""
        if self.debug:
            print("🔍 Validating consciousness coherence")
        
        # Check for abrupt consciousness state transitions
        current_state_level = 0
        
        for i, command in enumerate(self.execution_order):
            if command in self.command_dependencies:
                required_state = self.command_dependencies[command].required_state
                if required_state:
                    required_level = self._get_consciousness_level(required_state)
                    
                    # Check for jarring transitions (> 2 levels)
                    if abs(required_level - current_state_level) > 2:
                        self.warnings.append(SemanticError(
                            error_type=SemanticErrorType.CONSCIOUSNESS_STATE_CONFLICT,
                            message=f"Abrupt consciousness transition from level {current_state_level} to {required_level} at command '{command}'",
                            line=0,
                            column=0,
                            node=None,
                            suggestion="Add intermediate consciousness transition commands for smoother evolution",
                            severity="warning"
                        ))
                    
                    current_state_level = required_level
    
    def _get_consciousness_level(self, state: str) -> int:
        """Get numeric level for consciousness state"""
        for level, name in CONSCIOUSNESS_STATES.items():
            if name == state:
                return level
        return 0
    
    def _infer_type(self, node: ASTNode) -> str:
        """Infer the data type of a node"""
        if node.children and node.children[0]:
            return self._get_expression_type(node.children[0])
        return 'unknown'
    
    def _extract_value(self, node: ASTNode) -> Any:
        """Extract the value from an assignment node"""
        if node.children and node.children[0]:
            child = node.children[0]
            if child.node_type == NodeType.NUMBER:
                try:
                    return float(child.value)
                except:
                    return child.value
            return child.value
        return None
    
    def _nearest_fibonacci(self, n: int) -> int:
        """Find nearest Fibonacci number to n"""
        fibonacci_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
        return min(fibonacci_sequence, key=lambda x: abs(x - n))
    
    def _generate_semantic_info(self) -> Dict[str, Any]:
        """Generate semantic analysis summary"""
        return {
            'total_errors': len(self.errors),
            'total_warnings': len(self.warnings),
            'variables_defined': len(self.variables),
            'fields_created': len(self.fields),
            'commands_analyzed': len(self.execution_order),
            'consciousness_states_used': len(set(var.consciousness_state for var in self.variables.values() if var.consciousness_state)),
            'frequencies_used': len(set(var.frequency for var in self.variables.values() if var.frequency)),
            'dependency_satisfied': len([e for e in self.errors if e.error_type == SemanticErrorType.COMMAND_DEPENDENCY_ERROR]) == 0
        }
    
    def _calculate_phi_alignment_score(self) -> float:
        """Calculate overall phi-alignment score for the program"""
        scores = []
        
        # Check variable count alignment
        var_count = len(self.variables)
        if var_count > 0:
            var_fib_alignment = 1.0 if var_count in [1, 1, 2, 3, 5, 8, 13, 21] else 0.5
            scores.append(var_fib_alignment)
        
        # Check command count alignment  
        cmd_count = len(self.execution_order)
        if cmd_count > 0:
            cmd_fib_alignment = 1.0 if cmd_count in [1, 1, 2, 3, 5, 8, 13, 21] else 0.5
            scores.append(cmd_fib_alignment)
        
        # Check frequency alignments
        for var in self.variables.values():
            if var.frequency:
                phi_check = self._check_phi_harmonic_frequency(var.frequency)
                scores.append(phi_check['score'])
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _analyze_consciousness_coverage(self) -> Dict[str, int]:
        """Analyze coverage of consciousness states"""
        coverage = {}
        
        # Count consciousness states in commands
        for command in self.execution_order:
            if command in self.command_dependencies:
                state = self.command_dependencies[command].required_state
                if state:
                    coverage[state] = coverage.get(state, 0) + 1
        
        # Count consciousness states in variables
        for var in self.variables.values():
            if var.consciousness_state:
                coverage[var.consciousness_state] = coverage.get(var.consciousness_state, 0) + 1
        
        return coverage
    
    def _analyze_frequency_distribution(self) -> Dict[str, List[float]]:
        """Analyze frequency distribution in the program"""
        distribution = {
            'sacred_frequencies': [],
            'phi_harmonic_frequencies': [],
            'other_frequencies': []
        }
        
        for var in self.variables.values():
            if var.frequency:
                if var.frequency in SACRED_FREQUENCIES.values():
                    distribution['sacred_frequencies'].append(var.frequency)
                elif self._check_phi_harmonic_frequency(var.frequency)['aligned']:
                    distribution['phi_harmonic_frequencies'].append(var.frequency)
                else:
                    distribution['other_frequencies'].append(var.frequency)
        
        return distribution

def test_phi_flow_semantic_analyzer():
    """Test the PhiFlow semantic analyzer"""
    
    print("🧪 Testing PhiFlow Semantic Analyzer - Task 3.3 Implementation")
    print("=" * 70)
    
    # Import parser for testing
    from .phi_flow_parser import PhiFlowParser
    from .phi_flow_lexer import PhiFlowLexer
    
    # Test program with various semantic elements
    test_program = """
    # PhiFlow Test Program with Semantic Analysis
    INITIALIZE quantum_field AT 432Hz WITH coherence=1.0, phi_level=0
    SET base_frequency = 432 * phi
    SET harmonic_frequency = base_frequency * 1.618
    
    IF consciousness_state == "TRANSCEND" THEN
        EVOLVE TO 963Hz WITH phi_level=7
        HARMONIZE all_systems AT 720Hz WITH resonance=golden_angle
    ELSE
        OBSERVE state AT ground WITH stability=true, phi_level=0
    ENDIF
    
    CREATE_FIELD consciousness_field AT 594Hz WITH phi_level=2
    
    PARALLEL
        EVOLVE consciousness AT creation WITH phi_level=1
        INTEGRATE heart_field AT 594Hz WITH phi_level=2
        TRANSCEND limitations AT unity WITH phi_level=7
        HARMONIZE resonance AT 720Hz WITH phi_level=4
        CASCADE infinite_love AT 963Hz WITH phi_level=6
    END
    
    # Test invalid elements
    SET invalid_frequency = 50000  # Outside valid range
    UNKNOWN_COMMAND test WITH phi_level=10  # Invalid phi_level and command
    """
    
    # Create components
    lexer = PhiFlowLexer(debug=False)
    parser = PhiFlowParser(debug=False)
    analyzer = PhiFlowSemanticAnalyzer(debug=True)
    
    # Parse the test program
    print("\n🔄 Parsing PhiFlow program for semantic analysis...")
    tokens = lexer.tokenize(test_program)
    ast = parser.parse(tokens)
    
    # Perform semantic analysis
    print("\n🧠 Performing semantic analysis...")
    results = analyzer.analyze(ast)
    
    # Display results
    print("\n📊 Semantic Analysis Results:")
    print(f"  Total errors: {results['semantic_info']['total_errors']}")
    print(f"  Total warnings: {results['semantic_info']['total_warnings']}")
    print(f"  Variables defined: {results['semantic_info']['variables_defined']}")
    print(f"  Commands analyzed: {results['semantic_info']['commands_analyzed']}")
    print(f"  Phi alignment score: {results['phi_alignment_score']:.3f}")
    
    # Show errors
    if results['errors']:
        print("\n❌ Semantic Errors:")
        for error in results['errors']:
            print(f"  Line {error.line}: {error.message}")
            if error.suggestion:
                print(f"    Suggestion: {error.suggestion}")
    
    # Show warnings
    if results['warnings']:
        print("\n⚠️ Semantic Warnings:")
        for warning in results['warnings']:
            print(f"  Line {warning.line}: {warning.message}")
            if warning.suggestion:
                print(f"    Suggestion: {warning.suggestion}")
    
    # Show consciousness coverage
    print("\n🧠 Consciousness State Coverage:")
    for state, count in results['consciousness_coverage'].items():
        print(f"  {state}: {count} occurrences")
    
    # Show frequency distribution
    print("\n📐 Frequency Distribution:")
    freq_dist = results['frequency_distribution']
    if freq_dist['sacred_frequencies']:
        print(f"  Sacred frequencies: {freq_dist['sacred_frequencies']}")
    if freq_dist['phi_harmonic_frequencies']:
        print(f"  Phi-harmonic frequencies: {freq_dist['phi_harmonic_frequencies']}")
    if freq_dist['other_frequencies']:
        print(f"  Other frequencies: {freq_dist['other_frequencies']}")
    
    # Show variables
    print("\n📋 Variables Analyzed:")
    for name, var_info in results['variables'].items():
        print(f"  {name}: {var_info.data_type} = {var_info.value}")
        if var_info.frequency:
            print(f"    Frequency: {var_info.frequency}Hz")
        if var_info.consciousness_state:
            print(f"    Consciousness: {var_info.consciousness_state}")
    
    print("\n✅ PhiFlow Semantic Analyzer Test Complete!")
    return results

if __name__ == "__main__":
    test_phi_flow_semantic_analyzer()