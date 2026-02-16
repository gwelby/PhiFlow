#!/usr/bin/env python3
"""
PhiFlow Parser - Task 3.2 Implementation
Syntax analysis and AST generation using sacred mathematics principles
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any
from enum import Enum
import math
from decimal import Decimal

from .phi_flow_lexer import PhiFlowLexer, Token, TokenType, PHI, LAMBDA, GOLDEN_ANGLE, SACRED_FREQUENCIES

class ASTNodeType(Enum):
    """AST node types organized by sacred principles"""
    
    # Program Structure
    PROGRAM = "PROGRAM"
    STATEMENT_LIST = "STATEMENT_LIST"
    STATEMENT = "STATEMENT"
    
    # Sacred Geometry Commands
    INITIALIZE_STMT = "INITIALIZE_STMT"
    TRANSITION_STMT = "TRANSITION_STMT"
    EVOLVE_STMT = "EVOLVE_STMT"
    INTEGRATE_STMT = "INTEGRATE_STMT"
    HARMONIZE_STMT = "HARMONIZE_STMT"
    TRANSCEND_STMT = "TRANSCEND_STMT"
    CASCADE_STMT = "CASCADE_STMT"
    
    # Field Operations
    CREATE_FIELD_STMT = "CREATE_FIELD_STMT"
    ALIGN_FIELD_STMT = "ALIGN_FIELD_STMT"
    RESONATE_FIELD_STMT = "RESONATE_FIELD_STMT"
    COLLAPSE_FIELD_STMT = "COLLAPSE_FIELD_STMT"
    
    # Consciousness Operations
    OBSERVE_STMT = "OBSERVE_STMT"
    INTEND_STMT = "INTEND_STMT"
    FOCUS_STMT = "FOCUS_STMT"
    EXPAND_STMT = "EXPAND_STMT"
    MERGE_STMT = "MERGE_STMT"
    
    # Quantum Operations
    ENTANGLE_STMT = "ENTANGLE_STMT"
    SUPERPOSE_STMT = "SUPERPOSE_STMT"
    MEASURE_STMT = "MEASURE_STMT"
    TELEPORT_STMT = "TELEPORT_STMT"
    
    # Control Flow
    IF_STMT = "IF_STMT"
    WHILE_STMT = "WHILE_STMT"
    FOR_STMT = "FOR_STMT"
    PARALLEL_STMT = "PARALLEL_STMT"
    SEQUENCE_STMT = "SEQUENCE_STMT"
    
    # Variable and Assignment
    ASSIGNMENT_STMT = "ASSIGNMENT_STMT"
    VARIABLE = "VARIABLE"
    
    # Expressions
    BINARY_EXPR = "BINARY_EXPR"
    UNARY_EXPR = "UNARY_EXPR"
    FUNCTION_CALL = "FUNCTION_CALL"
    
    # Literals
    FREQUENCY_LITERAL = "FREQUENCY_LITERAL"
    PHI_LEVEL_LITERAL = "PHI_LEVEL_LITERAL"
    NUMBER_LITERAL = "NUMBER_LITERAL"
    STRING_LITERAL = "STRING_LITERAL"
    BOOLEAN_LITERAL = "BOOLEAN_LITERAL"
    
    # Sacred Constants
    PHI_CONSTANT = "PHI_CONSTANT"
    LAMBDA_CONSTANT = "LAMBDA_CONSTANT"
    GOLDEN_ANGLE_CONSTANT = "GOLDEN_ANGLE_CONSTANT"
    
    # Parameters
    PARAMETER_LIST = "PARAMETER_LIST"
    PARAMETER = "PARAMETER"

@dataclass
class ASTNode:
    """AST node with sacred mathematics validation"""
    type: ASTNodeType
    value: Optional[str] = None
    children: List['ASTNode'] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    # Position information
    line: int = 0
    column: int = 0
    
    # Sacred mathematics properties
    phi_alignment: Optional[float] = None
    frequency_validation: Optional[bool] = None
    consciousness_level: Optional[str] = None
    
    def add_child(self, child: 'ASTNode'):
        """Add child node with phi-harmonic validation"""
        self.children.append(child)
    
    def get_children_by_type(self, node_type: ASTNodeType) -> List['ASTNode']:
        """Get children of specific type"""
        return [child for child in self.children if child.type == node_type]
    
    def calculate_subtree_complexity(self) -> int:
        """Calculate complexity using Fibonacci progression"""
        if not self.children:
            return 1
        
        complexity = 1
        for child in self.children:
            complexity += child.calculate_subtree_complexity()
        
        return complexity
    
    def get_sacred_properties(self) -> Dict[str, Any]:
        """Get sacred mathematics properties"""
        return {
            'phi_alignment': self.phi_alignment,
            'frequency_validation': self.frequency_validation,
            'consciousness_level': self.consciousness_level,
            'complexity': self.calculate_subtree_complexity()
        }

@dataclass
class ParseError:
    """Parse error with consciousness-guided suggestions"""
    message: str
    line: int
    column: int
    token: Optional[Token] = None
    suggestion: Optional[str] = None
    
    def __str__(self) -> str:
        location = f"line {self.line}, column {self.column}"
        base_msg = f"Parse error at {location}: {self.message}"
        if self.suggestion:
            base_msg += f"\\nSuggestion: {self.suggestion}"
        return base_msg

class PhiFlowParser:
    """
    PhiFlow Parser implementing sacred mathematics syntax analysis
    
    Follows phi-harmonic principles:
    - AST construction using golden ratio proportions
    - Sacred frequency validation during parsing
    - Consciousness-guided error recovery
    - Fibonacci-based complexity analysis
    """
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.tokens: List[Token] = []
        self.current_token_index = 0
        self.current_token: Optional[Token] = None
        self.errors: List[ParseError] = []
        
        # Sacred mathematics state
        self.consciousness_state = "OBSERVE"
        self.phi_alignment_threshold = 0.618  # φ⁻¹ threshold
        self.fibonacci_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
        
        # Parser state
        self.in_parallel_block = False
        self.nested_level = 0
        self.current_scope_variables = set()
    
    def parse(self, source_code: str) -> Optional[ASTNode]:
        """
        Parse PhiFlow source code into AST using sacred mathematics
        
        Args:
            source_code: PhiFlow source code to parse
            
        Returns:
            Root AST node or None if parsing fails
        """
        if self.debug:
            print(f"🧠 Consciousness Expert: Starting parsing in {self.consciousness_state} state")
            print(f"📐 Sacred Mathematics: Using φ threshold = {self.phi_alignment_threshold}")
        
        # Tokenize first
        lexer = PhiFlowLexer(debug=self.debug)
        self.tokens = lexer.tokenize(source_code)
        
        # Filter out whitespace and comments for parsing
        self.tokens = [token for token in self.tokens 
                      if token.type not in [TokenType.WHITESPACE, TokenType.COMMENT]]
        
        if self.debug:
            print(f"🔄 Parsing {len(self.tokens)} tokens...")
        
        # Initialize parser state
        self.current_token_index = 0
        self.current_token = self.tokens[0] if self.tokens else None
        self.errors = []
        
        # Parse program
        try:
            program = self._parse_program()
            
            if self.errors:
                print(f"⚠️ Parsing completed with {len(self.errors)} errors:")
                for error in self.errors:
                    print(f"  {error}")
            else:
                if self.debug:
                    print(f"✅ Parsing successful! AST complexity: {program.calculate_subtree_complexity()}")
            
            return program
            
        except Exception as e:
            self._add_error(f"Critical parsing error: {str(e)}")
            return None
    
    def _parse_program(self) -> ASTNode:
        """Parse complete PhiFlow program"""
        program = ASTNode(ASTNodeType.PROGRAM, "program")
        
        # Parse statement list
        statements = self._parse_statement_list()
        program.add_child(statements)
        
        # Validate program structure using sacred mathematics
        self._validate_program_structure(program)
        
        return program
    
    def _parse_statement_list(self) -> ASTNode:
        """Parse list of statements"""
        stmt_list = ASTNode(ASTNodeType.STATEMENT_LIST, "statements")
        
        while self.current_token and self.current_token.type != TokenType.EOF:
            # Skip newlines
            if self.current_token.type == TokenType.NEWLINE:
                self._advance()
                continue
            
            # Parse statement
            stmt = self._parse_statement()
            if stmt:
                stmt_list.add_child(stmt)
            else:
                # Error recovery: skip to next statement
                self._skip_to_next_statement()
        
        return stmt_list
    
    def _parse_statement(self) -> Optional[ASTNode]:
        """Parse individual statement with sacred mathematics validation"""
        if not self.current_token:
            return None
        
        # Update consciousness state based on token
        self._update_consciousness_state()
        
        token_type = self.current_token.type
        
        # Sacred geometry commands
        if token_type == TokenType.INITIALIZE:
            return self._parse_initialize_statement()
        elif token_type == TokenType.TRANSITION:
            return self._parse_transition_statement()
        elif token_type == TokenType.EVOLVE:
            return self._parse_evolve_statement()
        elif token_type == TokenType.INTEGRATE:
            return self._parse_integrate_statement()
        elif token_type == TokenType.HARMONIZE:
            return self._parse_harmonize_statement()
        elif token_type == TokenType.TRANSCEND:
            return self._parse_transcend_statement()
        elif token_type == TokenType.CASCADE:
            return self._parse_cascade_statement()
        
        # Field commands
        elif token_type == TokenType.CREATE_FIELD:
            return self._parse_create_field_statement()
        elif token_type == TokenType.ALIGN_FIELD:
            return self._parse_align_field_statement()
        elif token_type == TokenType.RESONATE_FIELD:
            return self._parse_resonate_field_statement()
        elif token_type == TokenType.COLLAPSE_FIELD:
            return self._parse_collapse_field_statement()
        
        # Consciousness commands
        elif token_type == TokenType.OBSERVE:
            return self._parse_observe_statement()
        elif token_type == TokenType.INTEND:
            return self._parse_intend_statement()
        elif token_type == TokenType.FOCUS:
            return self._parse_focus_statement()
        elif token_type == TokenType.EXPAND:
            return self._parse_expand_statement()
        elif token_type == TokenType.MERGE:
            return self._parse_merge_statement()
        
        # Quantum commands
        elif token_type == TokenType.ENTANGLE:
            return self._parse_entangle_statement()
        elif token_type == TokenType.SUPERPOSE:
            return self._parse_superpose_statement()
        elif token_type == TokenType.MEASURE:
            return self._parse_measure_statement()
        elif token_type == TokenType.TELEPORT:
            return self._parse_teleport_statement()
        
        # Control flow
        elif token_type == TokenType.IF:
            return self._parse_if_statement()
        elif token_type == TokenType.WHILE:
            return self._parse_while_statement()
        elif token_type == TokenType.FOR:
            return self._parse_for_statement()
        elif token_type == TokenType.PARALLEL:
            return self._parse_parallel_statement()
        elif token_type == TokenType.SEQUENCE:
            return self._parse_sequence_statement()
        
        # Variable assignment
        elif token_type == TokenType.SET:
            return self._parse_assignment_statement()
        
        # Unknown statement
        else:
            self._add_error(f"Unknown statement type: {token_type.value}")
            return None
    
    def _parse_initialize_statement(self) -> ASTNode:
        """Parse INITIALIZE statement with sacred mathematics validation"""
        node = ASTNode(ASTNodeType.INITIALIZE_STMT, "INITIALIZE")
        node.line = self.current_token.line
        node.column = self.current_token.column
        
        self._advance()  # consume INITIALIZE
        
        # Parse target (field name)
        if self._check_token(TokenType.IDENTIFIER):
            target = ASTNode(ASTNodeType.VARIABLE, self.current_token.value)
            node.add_child(target)
            self._advance()
        else:
            self._add_error("Expected field name after INITIALIZE")
            return node
        
        # Parse AT frequency
        if self._check_token(TokenType.AT):
            self._advance()  # consume AT
            
            frequency = self._parse_frequency_expression()
            if frequency:
                node.add_child(frequency)
                # Validate sacred frequency
                if frequency.type == ASTNodeType.FREQUENCY_LITERAL:
                    self._validate_sacred_frequency(frequency, node)
        else:
            self._add_error("Expected AT after field name in INITIALIZE")
        
        # Parse WITH parameters
        if self._check_token(TokenType.WITH):
            self._advance()  # consume WITH
            
            params = self._parse_parameter_list()
            if params:
                node.add_child(params)
        
        return node
    
    def _parse_transition_statement(self) -> ASTNode:
        """Parse TRANSITION statement"""
        node = ASTNode(ASTNodeType.TRANSITION_STMT, "TRANSITION")
        node.line = self.current_token.line
        node.column = self.current_token.column
        
        self._advance()  # consume TRANSITION
        
        # Parse TO target
        if self._check_token(TokenType.TO):
            self._advance()  # consume TO
            
            target = self._parse_expression()
            if target:
                node.add_child(target)
        else:
            self._add_error("Expected TO after TRANSITION")
        
        # Parse AT frequency (optional)
        if self._check_token(TokenType.AT):
            self._advance()  # consume AT
            
            frequency = self._parse_frequency_expression()
            if frequency:
                node.add_child(frequency)
        
        # Parse WITH parameters (optional)
        if self._check_token(TokenType.WITH):
            self._advance()  # consume WITH
            
            params = self._parse_parameter_list()
            if params:
                node.add_child(params)
        
        return node
    
    def _parse_evolve_statement(self) -> ASTNode:
        """Parse EVOLVE statement with consciousness validation"""
        node = ASTNode(ASTNodeType.EVOLVE_STMT, "EVOLVE")
        node.line = self.current_token.line
        node.column = self.current_token.column
        
        self._advance()  # consume EVOLVE
        
        # Parse target
        target = self._parse_expression()
        if target:
            node.add_child(target)
        
        # Parse AT frequency
        if self._check_token(TokenType.AT):
            self._advance()  # consume AT
            
            frequency = self._parse_frequency_expression()
            if frequency:
                node.add_child(frequency)
                # Map to consciousness level
                self._map_frequency_to_consciousness(frequency, node)
        
        # Parse WITH parameters
        if self._check_token(TokenType.WITH):
            self._advance()  # consume WITH
            
            params = self._parse_parameter_list()
            if params:
                node.add_child(params)
        
        return node
    
    def _parse_integrate_statement(self) -> ASTNode:
        """Parse INTEGRATE statement"""
        node = ASTNode(ASTNodeType.INTEGRATE_STMT, "INTEGRATE")
        node.line = self.current_token.line
        node.column = self.current_token.column
        
        self._advance()  # consume INTEGRATE
        
        # Parse target
        target = self._parse_expression()
        if target:
            node.add_child(target)
        
        # Parse WITH parameters (integration requires parameters)
        if self._check_token(TokenType.WITH):
            self._advance()  # consume WITH
            
            params = self._parse_parameter_list()
            if params:
                node.add_child(params)
        else:
            self._add_error("INTEGRATE requires WITH parameters for field coherence")
        
        return node
    
    def _parse_harmonize_statement(self) -> ASTNode:
        """Parse HARMONIZE statement with golden ratio validation"""
        node = ASTNode(ASTNodeType.HARMONIZE_STMT, "HARMONIZE")
        node.line = self.current_token.line
        node.column = self.current_token.column
        
        self._advance()  # consume HARMONIZE
        
        # Parse target
        target = self._parse_expression()
        if target:
            node.add_child(target)
        
        # Parse AT frequency (harmonization requires frequency)
        if self._check_token(TokenType.AT):
            self._advance()  # consume AT
            
            frequency = self._parse_frequency_expression()
            if frequency:
                node.add_child(frequency)
                # Validate harmonic relationship
                self._validate_harmonic_frequency(frequency, node)
        else:
            self._add_error("HARMONIZE requires AT frequency for resonance calculation")
        
        # Parse WITH parameters
        if self._check_token(TokenType.WITH):
            self._advance()  # consume WITH
            
            params = self._parse_parameter_list()
            if params:
                node.add_child(params)
        
        return node
    
    def _parse_transcend_statement(self) -> ASTNode:
        """Parse TRANSCEND statement"""
        node = ASTNode(ASTNodeType.TRANSCEND_STMT, "TRANSCEND")
        node.line = self.current_token.line
        node.column = self.current_token.column
        
        self._advance()  # consume TRANSCEND
        
        # Parse target
        target = self._parse_expression()
        if target:
            node.add_child(target)
        
        # Parse AT frequency
        if self._check_token(TokenType.AT):
            self._advance()  # consume AT
            
            frequency = self._parse_frequency_expression()
            if frequency:
                node.add_child(frequency)
                # Validate transcendence frequency (should be high)
                self._validate_transcendence_frequency(frequency, node)
        
        # Parse WITH parameters
        if self._check_token(TokenType.WITH):
            self._advance()  # consume WITH
            
            params = self._parse_parameter_list()
            if params:
                node.add_child(params)
        
        return node
    
    def _parse_cascade_statement(self) -> ASTNode:
        """Parse CASCADE statement"""
        node = ASTNode(ASTNodeType.CASCADE_STMT, "CASCADE")
        node.line = self.current_token.line
        node.column = self.current_token.column
        
        self._advance()  # consume CASCADE
        
        # Parse target
        target = self._parse_expression()
        if target:
            node.add_child(target)
        
        # Parse AT frequency
        if self._check_token(TokenType.AT):
            self._advance()  # consume AT
            
            frequency = self._parse_frequency_expression()
            if frequency:
                node.add_child(frequency)
        
        # Parse WITH parameters
        if self._check_token(TokenType.WITH):
            self._advance()  # consume WITH
            
            params = self._parse_parameter_list()
            if params:
                node.add_child(params)
        
        return node
    
    def _parse_parallel_statement(self) -> ASTNode:
        """Parse PARALLEL block with Fibonacci work distribution validation"""
        node = ASTNode(ASTNodeType.PARALLEL_STMT, "PARALLEL")
        node.line = self.current_token.line
        node.column = self.current_token.column
        
        self._advance()  # consume PARALLEL
        
        # Set parallel parsing mode
        old_parallel_state = self.in_parallel_block
        self.in_parallel_block = True
        self.nested_level += 1
        
        # Parse statements until END
        while self.current_token and self.current_token.type != TokenType.END:
            if self.current_token.type == TokenType.NEWLINE:
                self._advance()
                continue
            
            stmt = self._parse_statement()
            if stmt:
                node.add_child(stmt)
        
        # Consume END
        if self._check_token(TokenType.END):
            self._advance()
        else:
            self._add_error("Expected END after PARALLEL block")
        
        # Restore state
        self.in_parallel_block = old_parallel_state
        self.nested_level -= 1
        
        # Validate Fibonacci distribution
        self._validate_fibonacci_distribution(node)
        
        return node
    
    def _parse_if_statement(self) -> ASTNode:
        """Parse IF statement with consciousness condition validation"""
        node = ASTNode(ASTNodeType.IF_STMT, "IF")
        node.line = self.current_token.line
        node.column = self.current_token.column
        
        self._advance()  # consume IF
        
        # Parse condition
        condition = self._parse_expression()
        if condition:
            node.add_child(condition)
        
        # Parse THEN
        if self._check_token(TokenType.THEN):
            self._advance()  # consume THEN
        else:
            self._add_error("Expected THEN after IF condition")
        
        # Parse then statements
        then_block = ASTNode(ASTNodeType.STATEMENT_LIST, "then_block")
        while (self.current_token and 
               self.current_token.type not in [TokenType.ELSE, TokenType.ENDIF, TokenType.EOF]):
            if self.current_token.type == TokenType.NEWLINE:
                self._advance()
                continue
            
            stmt = self._parse_statement()
            if stmt:
                then_block.add_child(stmt)
        
        node.add_child(then_block)
        
        # Parse ELSE (optional)
        if self._check_token(TokenType.ELSE):
            self._advance()  # consume ELSE
            
            else_block = ASTNode(ASTNodeType.STATEMENT_LIST, "else_block")
            while (self.current_token and 
                   self.current_token.type not in [TokenType.ENDIF, TokenType.EOF]):
                if self.current_token.type == TokenType.NEWLINE:
                    self._advance()
                    continue
                
                stmt = self._parse_statement()
                if stmt:
                    else_block.add_child(stmt)
            
            node.add_child(else_block)
        
        # Parse ENDIF
        if self._check_token(TokenType.ENDIF):
            self._advance()  # consume ENDIF
        else:
            self._add_error("Expected ENDIF after IF statement")
        
        return node
    
    def _parse_assignment_statement(self) -> ASTNode:
        """Parse variable assignment with phi validation"""
        node = ASTNode(ASTNodeType.ASSIGNMENT_STMT, "SET")
        node.line = self.current_token.line
        node.column = self.current_token.column
        
        self._advance()  # consume SET
        
        # Parse variable name
        if self._check_token(TokenType.IDENTIFIER):
            var_node = ASTNode(ASTNodeType.VARIABLE, self.current_token.value)
            node.add_child(var_node)
            
            # Add to current scope
            self.current_scope_variables.add(self.current_token.value)
            
            self._advance()
        else:
            self._add_error("Expected variable name after SET")
            return node
        
        # Parse assignment operator
        if self._check_token(TokenType.ASSIGN):
            self._advance()  # consume =
        else:
            self._add_error("Expected = after variable name")
            return node
        
        # Parse expression
        expr = self._parse_expression()
        if expr:
            node.add_child(expr)
            
            # Validate phi-related assignments
            self._validate_phi_assignment(expr, node)
        
        return node
    
    def _parse_frequency_expression(self) -> Optional[ASTNode]:
        """Parse frequency expression with sacred mathematics validation"""
        if self._check_token(TokenType.FREQUENCY):
            node = ASTNode(ASTNodeType.FREQUENCY_LITERAL, self.current_token.value)
            node.line = self.current_token.line
            node.column = self.current_token.column
            
            # Copy sacred mathematics properties from token
            node.phi_alignment = self.current_token.phi_alignment
            node.frequency_validation = self.current_token.frequency_validation
            
            self._advance()
            return node
        
        elif self._check_token(TokenType.IDENTIFIER):
            # Check for sacred frequency names
            freq_name = self.current_token.value.lower()
            if freq_name in ['ground', 'creation', 'heart', 'voice', 'vision', 'unity', 'source']:
                node = ASTNode(ASTNodeType.FREQUENCY_LITERAL, f"{SACRED_FREQUENCIES[freq_name]}Hz")
                node.line = self.current_token.line
                node.column = self.current_token.column
                node.frequency_validation = True
                node.phi_alignment = 1.0  # Perfect alignment for named frequencies
                
                self._advance()
                return node
        
        # Try parsing as expression
        return self._parse_expression()
    
    def _parse_expression(self) -> Optional[ASTNode]:
        """Parse expression with phi-harmonic precedence"""
        return self._parse_additive_expression()
    
    def _parse_additive_expression(self) -> Optional[ASTNode]:
        """Parse addition and subtraction"""
        left = self._parse_multiplicative_expression()
        
        while self.current_token and self.current_token.type in [TokenType.PLUS, TokenType.MINUS]:
            op = self.current_token.type
            self._advance()
            
            right = self._parse_multiplicative_expression()
            if right:
                node = ASTNode(ASTNodeType.BINARY_EXPR, op.value)
                node.add_child(left)
                node.add_child(right)
                left = node
        
        return left
    
    def _parse_multiplicative_expression(self) -> Optional[ASTNode]:
        """Parse multiplication and division with phi validation"""
        left = self._parse_power_expression()
        
        while self.current_token and self.current_token.type in [TokenType.MULTIPLY, TokenType.DIVIDE, TokenType.MODULO]:
            op = self.current_token.type
            self._advance()
            
            right = self._parse_power_expression()
            if right:
                node = ASTNode(ASTNodeType.BINARY_EXPR, op.value)
                node.add_child(left)
                node.add_child(right)
                
                # Check for phi-related operations
                self._validate_phi_operation(left, right, op, node)
                
                left = node
        
        return left
    
    def _parse_power_expression(self) -> Optional[ASTNode]:
        """Parse power expressions (phi^phi, etc.)"""
        left = self._parse_primary_expression()
        
        if self.current_token and self.current_token.type == TokenType.POWER:
            self._advance()
            
            right = self._parse_power_expression()  # Right associative
            if right:
                node = ASTNode(ASTNodeType.BINARY_EXPR, "**")
                node.add_child(left)
                node.add_child(right)
                
                # Special validation for phi powers
                self._validate_phi_power(left, right, node)
                
                return node
        
        return left
    
    def _parse_primary_expression(self) -> Optional[ASTNode]:
        """Parse primary expressions"""
        if not self.current_token:
            return None
        
        token_type = self.current_token.type
        
        # Numbers
        if token_type == TokenType.NUMBER:
            node = ASTNode(ASTNodeType.NUMBER_LITERAL, self.current_token.value)
            node.line = self.current_token.line
            node.column = self.current_token.column
            self._advance()
            return node
        
        # Strings
        elif token_type == TokenType.STRING:
            node = ASTNode(ASTNodeType.STRING_LITERAL, self.current_token.value)
            node.line = self.current_token.line
            node.column = self.current_token.column
            self._advance()
            return node
        
        # Booleans
        elif token_type == TokenType.BOOLEAN:
            node = ASTNode(ASTNodeType.BOOLEAN_LITERAL, self.current_token.value)
            node.line = self.current_token.line
            node.column = self.current_token.column
            self._advance()
            return node
        
        # Sacred constants
        elif token_type == TokenType.PHI_CONST:
            node = ASTNode(ASTNodeType.PHI_CONSTANT, str(PHI))
            node.line = self.current_token.line
            node.column = self.current_token.column
            node.phi_alignment = 1.0  # Perfect phi alignment
            self._advance()
            return node
        
        elif token_type == TokenType.LAMBDA_CONST:
            node = ASTNode(ASTNodeType.LAMBDA_CONSTANT, str(LAMBDA))
            node.line = self.current_token.line
            node.column = self.current_token.column
            node.phi_alignment = 1.0  # Perfect phi alignment
            self._advance()
            return node
        
        elif token_type == TokenType.GOLDEN_ANGLE_CONST:
            node = ASTNode(ASTNodeType.GOLDEN_ANGLE_CONSTANT, str(GOLDEN_ANGLE))
            node.line = self.current_token.line
            node.column = self.current_token.column
            node.phi_alignment = 1.0  # Perfect phi alignment
            self._advance()
            return node
        
        # Identifiers
        elif token_type == TokenType.IDENTIFIER:
            node = ASTNode(ASTNodeType.VARIABLE, self.current_token.value)
            node.line = self.current_token.line
            node.column = self.current_token.column
            self._advance()
            return node
        
        # Parenthesized expressions
        elif token_type == TokenType.LPAREN:
            self._advance()  # consume (
            expr = self._parse_expression()
            
            if self._check_token(TokenType.RPAREN):
                self._advance()  # consume )
            else:
                self._add_error("Expected ) after expression")
            
            return expr
        
        else:
            self._add_error(f"Unexpected token: {token_type.value}")
            return None
    
    def _parse_parameter_list(self) -> ASTNode:
        """Parse parameter list (key=value pairs)"""
        params = ASTNode(ASTNodeType.PARAMETER_LIST, "parameters")
        
        while self.current_token and self.current_token.type == TokenType.IDENTIFIER:
            param = ASTNode(ASTNodeType.PARAMETER, self.current_token.value)
            param_name = self.current_token.value
            self._advance()
            
            if self._check_token(TokenType.ASSIGN):
                self._advance()  # consume =
                
                value = self._parse_expression()
                if value:
                    param.add_child(value)
                    params.add_child(param)
            else:
                self._add_error("Expected = after parameter name")
            
            # Check for comma
            if self._check_token(TokenType.COMMA):
                self._advance()
            else:
                break
        
        return params
    
    # Sacred Mathematics Validation Methods
    
    def _validate_sacred_frequency(self, freq_node: ASTNode, parent: ASTNode):
        """Validate frequency against sacred mathematics"""
        try:
            freq_str = freq_node.value.replace('Hz', '')
            frequency = float(freq_str)
            
            # Check exact sacred frequencies
            is_sacred = frequency in SACRED_FREQUENCIES.values()
            
            if is_sacred:
                freq_node.frequency_validation = True
                freq_node.phi_alignment = 1.0
                parent.frequency_validation = True
            else:
                # Check phi-harmonic relationships
                phi_aligned = False
                best_alignment = 0.0
                
                for sacred_freq in SACRED_FREQUENCIES.values():
                    for power in range(1, 5):
                        harmonic = sacred_freq * (float(PHI) ** power)
                        if abs(frequency - harmonic) < 5.0:  # 5Hz tolerance
                            alignment = 1.0 - abs(frequency - harmonic) / harmonic
                            if alignment > best_alignment:
                                best_alignment = alignment
                                phi_aligned = True
                
                freq_node.frequency_validation = phi_aligned
                freq_node.phi_alignment = best_alignment
                parent.frequency_validation = phi_aligned
                
                if not phi_aligned:
                    self._add_error(f"Frequency {frequency}Hz not aligned with sacred mathematics", 
                                  suggestion=f"Try {min(SACRED_FREQUENCIES.values(), key=lambda x: abs(x - frequency))}Hz")
        
        except ValueError:
            freq_node.frequency_validation = False
            parent.frequency_validation = False
            self._add_error("Invalid frequency format")
    
    def _validate_harmonic_frequency(self, freq_node: ASTNode, parent: ASTNode):
        """Validate frequency for harmonic operations"""
        self._validate_sacred_frequency(freq_node, parent)
        
        # Additional validation for harmonics
        if freq_node.frequency_validation:
            parent.consciousness_level = "HARMONIZE"
    
    def _validate_transcendence_frequency(self, freq_node: ASTNode, parent: ASTNode):
        """Validate frequency for transcendence operations"""
        self._validate_sacred_frequency(freq_node, parent)
        
        # Transcendence should use higher frequencies
        try:
            freq_str = freq_node.value.replace('Hz', '')
            frequency = float(freq_str)
            
            if frequency >= 720:  # Vision frequency or higher
                parent.consciousness_level = "TRANSCEND"
            else:
                self._add_error("TRANSCEND operations should use frequencies >= 720Hz (Vision level)",
                              suggestion="Try 720Hz (vision), 768Hz (unity), or 963Hz (source)")
        except:
            pass
    
    def _validate_phi_assignment(self, expr_node: ASTNode, parent: ASTNode):
        """Validate phi-related variable assignments"""
        if expr_node.type == ASTNodeType.PHI_CONSTANT:
            parent.phi_alignment = 1.0
        elif expr_node.type == ASTNodeType.BINARY_EXPR and expr_node.value == "*":
            # Check for phi multiplications
            children = expr_node.children
            if len(children) == 2:
                has_phi = any(child.type == ASTNodeType.PHI_CONSTANT for child in children)
                if has_phi:
                    parent.phi_alignment = 0.8  # Good phi alignment
    
    def _validate_phi_operation(self, left: ASTNode, right: ASTNode, op: TokenType, result: ASTNode):
        """Validate phi-related mathematical operations"""
        phi_related = False
        
        # Check if either operand is phi-related
        if hasattr(left, 'phi_alignment') and left.phi_alignment and left.phi_alignment > 0.5:
            phi_related = True
        if hasattr(right, 'phi_alignment') and right.phi_alignment and right.phi_alignment > 0.5:
            phi_related = True
        
        if phi_related:
            result.phi_alignment = 0.7  # Moderate phi alignment
    
    def _validate_phi_power(self, base: ASTNode, exponent: ASTNode, result: ASTNode):
        """Validate phi power expressions (phi^phi, etc.)"""
        if (base.type == ASTNodeType.PHI_CONSTANT and 
            exponent.type == ASTNodeType.PHI_CONSTANT):
            result.phi_alignment = 1.0  # Perfect phi^phi alignment
            result.consciousness_level = "SUPERPOSITION"
    
    def _validate_fibonacci_distribution(self, parallel_node: ASTNode):
        """Validate Fibonacci work distribution in parallel blocks"""
        num_statements = len(parallel_node.children)
        
        # Check if number of statements follows Fibonacci sequence
        is_fibonacci = num_statements in self.fibonacci_sequence
        
        if is_fibonacci:
            parallel_node.attributes['fibonacci_distribution'] = True
        else:
            closest_fib = min(self.fibonacci_sequence, key=lambda x: abs(x - num_statements))
            self._add_error(f"Parallel block has {num_statements} statements, not Fibonacci number",
                          suggestion=f"Consider using {closest_fib} statements for optimal phi distribution")
    
    def _validate_program_structure(self, program: ASTNode):
        """Validate overall program structure using sacred mathematics"""
        complexity = program.calculate_subtree_complexity()
        
        # Check if total complexity follows Fibonacci progression
        is_fibonacci_complex = complexity in self.fibonacci_sequence
        
        program.attributes['complexity'] = complexity
        program.attributes['fibonacci_structure'] = is_fibonacci_complex
        
        if self.debug:
            print(f"📊 Program complexity: {complexity} (Fibonacci: {is_fibonacci_complex})")
    
    def _map_frequency_to_consciousness(self, freq_node: ASTNode, parent: ASTNode):
        """Map frequency to consciousness level"""
        try:
            freq_str = freq_node.value.replace('Hz', '')
            frequency = float(freq_str)
            
            consciousness_map = {
                432: "OBSERVE",
                528: "CREATE", 
                594: "INTEGRATE",
                672: "HARMONIZE",
                720: "TRANSCEND",
                768: "CASCADE",
                963: "SUPERPOSITION"
            }
            
            # Find closest frequency
            closest_freq = min(consciousness_map.keys(), key=lambda x: abs(x - frequency))
            
            if abs(frequency - closest_freq) < 50:  # 50Hz tolerance
                parent.consciousness_level = consciousness_map[closest_freq]
        except:
            pass
    
    def _update_consciousness_state(self):
        """Update parser consciousness state based on current token"""
        if self.current_token:
            token_consciousness_map = {
                TokenType.OBSERVE: "OBSERVE",
                TokenType.CREATE_FIELD: "CREATE",
                TokenType.INTEGRATE: "INTEGRATE",
                TokenType.HARMONIZE: "HARMONIZE",
                TokenType.TRANSCEND: "TRANSCEND",
                TokenType.CASCADE: "CASCADE"
            }
            
            if self.current_token.type in token_consciousness_map:
                self.consciousness_state = token_consciousness_map[self.current_token.type]
    
    # Helper methods
    
    def _check_token(self, expected_type: TokenType) -> bool:
        """Check if current token matches expected type"""
        return self.current_token and self.current_token.type == expected_type
    
    def _advance(self):
        """Advance to next token"""
        if self.current_token_index < len(self.tokens) - 1:
            self.current_token_index += 1
            self.current_token = self.tokens[self.current_token_index]
        else:
            self.current_token = None
    
    def _add_error(self, message: str, suggestion: str = None):
        """Add parse error with consciousness-guided suggestion"""
        line = self.current_token.line if self.current_token else 0
        column = self.current_token.column if self.current_token else 0
        
        error = ParseError(message, line, column, self.current_token, suggestion)
        self.errors.append(error)
    
    def _skip_to_next_statement(self):
        """Skip to next statement for error recovery"""
        while (self.current_token and 
               self.current_token.type not in [TokenType.NEWLINE, TokenType.SEMICOLON, TokenType.EOF]):
            self._advance()
        
        if self.current_token and self.current_token.type in [TokenType.NEWLINE, TokenType.SEMICOLON]:
            self._advance()
    
    # Stub methods for other statement types (to be implemented as needed)
    def _parse_create_field_statement(self) -> ASTNode:
        return self._parse_generic_field_statement(ASTNodeType.CREATE_FIELD_STMT, "CREATE_FIELD")
    
    def _parse_align_field_statement(self) -> ASTNode:
        return self._parse_generic_field_statement(ASTNodeType.ALIGN_FIELD_STMT, "ALIGN_FIELD")
    
    def _parse_resonate_field_statement(self) -> ASTNode:
        return self._parse_generic_field_statement(ASTNodeType.RESONATE_FIELD_STMT, "RESONATE_FIELD")
    
    def _parse_collapse_field_statement(self) -> ASTNode:
        return self._parse_generic_field_statement(ASTNodeType.COLLAPSE_FIELD_STMT, "COLLAPSE_FIELD")
    
    def _parse_observe_statement(self) -> ASTNode:
        return self._parse_generic_consciousness_statement(ASTNodeType.OBSERVE_STMT, "OBSERVE")
    
    def _parse_intend_statement(self) -> ASTNode:
        return self._parse_generic_consciousness_statement(ASTNodeType.INTEND_STMT, "INTEND")
    
    def _parse_focus_statement(self) -> ASTNode:
        return self._parse_generic_consciousness_statement(ASTNodeType.FOCUS_STMT, "FOCUS")
    
    def _parse_expand_statement(self) -> ASTNode:
        return self._parse_generic_consciousness_statement(ASTNodeType.EXPAND_STMT, "EXPAND")
    
    def _parse_merge_statement(self) -> ASTNode:
        return self._parse_generic_consciousness_statement(ASTNodeType.MERGE_STMT, "MERGE")
    
    def _parse_entangle_statement(self) -> ASTNode:
        return self._parse_generic_quantum_statement(ASTNodeType.ENTANGLE_STMT, "ENTANGLE")
    
    def _parse_superpose_statement(self) -> ASTNode:
        return self._parse_generic_quantum_statement(ASTNodeType.SUPERPOSE_STMT, "SUPERPOSE")
    
    def _parse_measure_statement(self) -> ASTNode:
        return self._parse_generic_quantum_statement(ASTNodeType.MEASURE_STMT, "MEASURE")
    
    def _parse_teleport_statement(self) -> ASTNode:
        return self._parse_generic_quantum_statement(ASTNodeType.TELEPORT_STMT, "TELEPORT")
    
    def _parse_while_statement(self) -> ASTNode:
        # Similar to if statement but with WHILE/END
        pass
    
    def _parse_for_statement(self) -> ASTNode:
        # FOR loop implementation
        pass
    
    def _parse_sequence_statement(self) -> ASTNode:
        # SEQUENCE block implementation
        pass
    
    def _parse_generic_field_statement(self, node_type: ASTNodeType, command: str) -> ASTNode:
        """Generic parser for field statements"""
        node = ASTNode(node_type, command)
        node.line = self.current_token.line
        node.column = self.current_token.column
        
        self._advance()  # consume command
        
        # Parse target
        target = self._parse_expression()
        if target:
            node.add_child(target)
        
        return node
    
    def _parse_generic_consciousness_statement(self, node_type: ASTNodeType, command: str) -> ASTNode:
        """Generic parser for consciousness statements"""
        node = ASTNode(node_type, command)
        node.line = self.current_token.line
        node.column = self.current_token.column
        node.consciousness_level = command
        
        self._advance()  # consume command
        
        # Parse target
        target = self._parse_expression()
        if target:
            node.add_child(target)
        
        return node
    
    def _parse_generic_quantum_statement(self, node_type: ASTNodeType, command: str) -> ASTNode:
        """Generic parser for quantum statements"""
        node = ASTNode(node_type, command)
        node.line = self.current_token.line
        node.column = self.current_token.column
        
        self._advance()  # consume command
        
        # Parse target
        target = self._parse_expression()
        if target:
            node.add_child(target)
        
        return node

def test_phi_flow_parser():
    """Test the PhiFlow parser with sacred mathematics validation"""
    
    print("🧪 Testing PhiFlow Parser - Task 3.2 Implementation")
    print("=" * 70)
    
    # Test program with sacred geometry and control flow
    test_program = """
    # PhiFlow Test Program with Complete Syntax
    INITIALIZE quantum_field AT 432Hz WITH coherence=1.0
    SET base_frequency = 432
    SET harmonic_frequency = base_frequency * phi
    SET golden_ratio = phi ** phi
    
    IF consciousness_state == "TRANSCEND" THEN
        EVOLVE consciousness TO 963Hz WITH phi_level=5
        HARMONIZE all_systems AT 720Hz WITH resonance=golden_angle
        TRANSCEND limitations AT unity WITH awareness_level=7
    ELSE
        OBSERVE current_state AT ground WITH stability=true
    ENDIF
    
    PARALLEL
        EVOLVE consciousness AT creation
        INTEGRATE heart_field AT 594Hz WITH compression=phi
        HARMONIZE voice_flow AT 672Hz WITH expression=true
        TRANSCEND vision_gate AT 720Hz WITH perception=quantum
        CASCADE infinite_love AT source WITH superposition=true
    END
    """
    
    # Create parser with debug output
    parser = PhiFlowParser(debug=True)
    
    # Parse the test program
    print("\\n🔄 Parsing PhiFlow program...")
    ast = parser.parse(test_program)
    
    if ast:
        # Display AST structure
        print("\\n🌳 Abstract Syntax Tree Structure:")
        print(f"  Root: {ast.type.value}")
        print(f"  Complexity: {ast.calculate_subtree_complexity()}")
        print(f"  Children: {len(ast.children)}")
        
        # Show sacred mathematics properties
        sacred_props = ast.get_sacred_properties()
        print("\\n📐 Sacred Mathematics Properties:")
        for key, value in sacred_props.items():
            if value is not None:
                print(f"  {key}: {value}")
        
        # Show program attributes
        if ast.attributes:
            print("\\n🎯 Program Attributes:")
            for key, value in ast.attributes.items():
                print(f"  {key}: {value}")
        
        # Analyze statements
        if ast.children:
            stmt_list = ast.children[0]
            print(f"\\n📊 Statement Analysis:")
            print(f"  Total statements: {len(stmt_list.children)}")
            
            # Count statement types
            stmt_types = {}
            for stmt in stmt_list.children:
                stmt_type = stmt.type.value
                stmt_types[stmt_type] = stmt_types.get(stmt_type, 0) + 1
            
            for stmt_type, count in stmt_types.items():
                print(f"  {stmt_type}: {count}")
        
        # Show phi-aligned nodes
        phi_nodes = []
        def find_phi_nodes(node):
            if hasattr(node, 'phi_alignment') and node.phi_alignment and node.phi_alignment > 0.5:
                phi_nodes.append((node.type.value, node.phi_alignment))
            for child in node.children:
                find_phi_nodes(child)
        
        find_phi_nodes(ast)
        
        if phi_nodes:
            print("\\n✨ Phi-Aligned Nodes:")
            for node_type, alignment in phi_nodes:
                print(f"  {node_type}: {alignment:.3f}")
    
    # Show parse errors
    if parser.errors:
        print(f"\\n⚠️ Parse Errors ({len(parser.errors)}):")
        for error in parser.errors:
            print(f"  {error}")
    else:
        print("\\n✅ Parsing completed successfully with no errors!")
    
    print("\\n✅ PhiFlow Parser Test Complete!")
    return ast

if __name__ == "__main__":
    test_phi_flow_parser()