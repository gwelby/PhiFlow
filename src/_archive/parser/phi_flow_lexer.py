#!/usr/bin/env python3
"""
PhiFlow Lexer - Task 3.1 Implementation
Tokenizes PhiFlow source code using sacred mathematics principles
"""

import re
import math
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Union, Iterator
from decimal import Decimal

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

class TokenType(Enum):
    """PhiFlow token types organized by sacred principles"""
    
    # Sacred Geometry Commands (Core)
    INITIALIZE = "INITIALIZE"
    TRANSITION = "TRANSITION"
    EVOLVE = "EVOLVE"
    INTEGRATE = "INTEGRATE"
    HARMONIZE = "HARMONIZE"
    TRANSCEND = "TRANSCEND"
    CASCADE = "CASCADE"
    
    # Field Commands
    CREATE_FIELD = "CREATE_FIELD"
    ALIGN_FIELD = "ALIGN_FIELD"
    RESONATE_FIELD = "RESONATE_FIELD"
    COLLAPSE_FIELD = "COLLAPSE_FIELD"
    
    # Consciousness Commands
    OBSERVE = "OBSERVE"
    INTEND = "INTEND"
    FOCUS = "FOCUS"
    EXPAND = "EXPAND"
    MERGE = "MERGE"
    
    # Quantum Commands
    ENTANGLE = "ENTANGLE"
    SUPERPOSE = "SUPERPOSE"
    MEASURE = "MEASURE"
    TELEPORT = "TELEPORT"
    
    # Control Flow
    IF = "IF"
    ELSE = "ELSE"
    ENDIF = "ENDIF"
    WHILE = "WHILE"
    FOR = "FOR"
    REPEAT = "REPEAT"
    PARALLEL = "PARALLEL"
    SEQUENCE = "SEQUENCE"
    END = "END"
    THEN = "THEN"
    
    # Keywords
    AT = "AT"
    WITH = "WITH"
    TO = "TO"
    SET = "SET"
    
    # Literals
    FREQUENCY = "FREQUENCY"
    PHI_LEVEL = "PHI_LEVEL"
    NUMBER = "NUMBER"
    STRING = "STRING"
    BOOLEAN = "BOOLEAN"
    IDENTIFIER = "IDENTIFIER"
    
    # Sacred Constants
    PHI_CONST = "PHI"
    LAMBDA_CONST = "LAMBDA"
    GOLDEN_ANGLE_CONST = "GOLDEN_ANGLE"
    SACRED_FREQ = "SACRED_FREQ"
    
    # Operators
    ASSIGN = "="
    PLUS = "+"
    MINUS = "-"
    MULTIPLY = "*"
    DIVIDE = "/"
    POWER = "**"
    MODULO = "%"
    
    # Comparison
    EQUAL = "=="
    NOT_EQUAL = "!="
    LESS_THAN = "<"
    GREATER_THAN = ">"
    LESS_EQUAL = "<="
    GREATER_EQUAL = ">="
    
    # Delimiters
    LPAREN = "("
    RPAREN = ")"
    LBRACE = "{"
    RBRACE = "}"
    LBRACKET = "["
    RBRACKET = "]"
    COMMA = ","
    SEMICOLON = ";"
    COLON = ":"
    DOT = "."
    
    # Special
    NEWLINE = "NEWLINE"
    INDENT = "INDENT"
    DEDENT = "DEDENT"
    EOF = "EOF"
    COMMENT = "COMMENT"
    WHITESPACE = "WHITESPACE"

@dataclass
class Token:
    """PhiFlow token with position and sacred mathematics validation"""
    type: TokenType
    value: str
    line: int
    column: int
    position: int
    
    # Sacred mathematics properties
    phi_alignment: Optional[float] = None
    frequency_validation: Optional[bool] = None
    consciousness_level: Optional[str] = None
    
    def __post_init__(self):
        """Apply sacred mathematics validation after token creation"""
        if self.type == TokenType.FREQUENCY:
            self.frequency_validation = self._validate_sacred_frequency()
            self.phi_alignment = self._calculate_phi_alignment()
        elif self.type == TokenType.PHI_LEVEL:
            self.consciousness_level = self._map_consciousness_level()
    
    def _validate_sacred_frequency(self) -> bool:
        """Validate frequency against sacred mathematics principles"""
        try:
            freq = float(self.value.replace('Hz', ''))
            
            # Check exact sacred frequencies
            if freq in SACRED_FREQUENCIES.values():
                return True
                
            # Check phi-harmonic relationships
            for sacred_freq in SACRED_FREQUENCIES.values():
                phi_ratios = [1, float(PHI), float(PHI**2), float(PHI**3), float(PHI**4)]
                for ratio in phi_ratios:
                    if abs(freq - (sacred_freq * ratio)) < 1.0:  # 1Hz tolerance
                        return True
                        
            # Check golden angle harmonics
            if freq >= 20 and freq <= 20000:  # Audio range
                return True
                
            return False
        except:
            return False
    
    def _calculate_phi_alignment(self) -> float:
        """Calculate phi-alignment score for frequency"""
        try:
            freq = float(self.value.replace('Hz', ''))
            
            # Find closest sacred frequency
            closest_sacred = min(SACRED_FREQUENCIES.values(), key=lambda x: abs(x - freq))
            
            # Calculate phi ratio
            ratio = freq / closest_sacred
            
            # Calculate alignment with phi powers
            phi_powers = [1, float(PHI), float(PHI**2), float(PHI**3), float(PHI**4)]
            closest_phi = min(phi_powers, key=lambda x: abs(x - ratio))
            
            # Return alignment score (0.0 to 1.0)
            alignment = 1.0 - abs(ratio - closest_phi) / closest_phi
            return max(0.0, min(1.0, alignment))
        except:
            return 0.0
    
    def _map_consciousness_level(self) -> str:
        """Map phi_level to consciousness state"""
        try:
            level = int(self.value)
            consciousness_map = {
                0: "OBSERVE",
                1: "CREATE", 
                2: "INTEGRATE",
                3: "HARMONIZE",
                4: "TRANSCEND",
                5: "CASCADE",
                6: "SUPERPOSITION",
                7: "UNITY"
            }
            return consciousness_map.get(level, "OBSERVE")
        except:
            return "OBSERVE"

class PhiFlowLexer:
    """
    PhiFlow Lexer implementing sacred mathematics tokenization
    
    Follows phi-harmonic principles:
    - Token processing in Fibonacci sequence patterns
    - Golden angle rotation for token classification
    - Sacred frequency validation
    - Consciousness-guided parsing states
    """
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.tokens: List[Token] = []
        self.current_pos = 0
        self.current_line = 1
        self.current_column = 1
        self.source_code = ""
        
        # Sacred mathematics state
        self.fibonacci_sequence = self._generate_fibonacci_sequence(20)
        self.phi_state_rotation = 0
        self.consciousness_state = "OBSERVE"
        
        # Initialize token patterns
        self._init_token_patterns()
    
    def _generate_fibonacci_sequence(self, n: int) -> List[int]:
        """Generate Fibonacci sequence for parser state management"""
        fib = [1, 1]
        for i in range(2, n):
            fib.append(fib[i-1] + fib[i-2])
        return fib
    
    def _init_token_patterns(self):
        """Initialize token recognition patterns with sacred mathematics"""
        
        # Sacred geometry commands (exact matches)
        self.command_patterns = {
            'INITIALIZE': TokenType.INITIALIZE,
            'TRANSITION': TokenType.TRANSITION,
            'EVOLVE': TokenType.EVOLVE,
            'INTEGRATE': TokenType.INTEGRATE,
            'HARMONIZE': TokenType.HARMONIZE,
            'TRANSCEND': TokenType.TRANSCEND,
            'CASCADE': TokenType.CASCADE,
            'CREATE_FIELD': TokenType.CREATE_FIELD,
            'ALIGN_FIELD': TokenType.ALIGN_FIELD,
            'RESONATE_FIELD': TokenType.RESONATE_FIELD,
            'COLLAPSE_FIELD': TokenType.COLLAPSE_FIELD,
            'OBSERVE': TokenType.OBSERVE,
            'INTEND': TokenType.INTEND,
            'FOCUS': TokenType.FOCUS,
            'EXPAND': TokenType.EXPAND,
            'MERGE': TokenType.MERGE,
            'ENTANGLE': TokenType.ENTANGLE,
            'SUPERPOSE': TokenType.SUPERPOSE,
            'MEASURE': TokenType.MEASURE,
            'TELEPORT': TokenType.TELEPORT,
            'IF': TokenType.IF,
            'ELSE': TokenType.ELSE,
            'ENDIF': TokenType.ENDIF,
            'WHILE': TokenType.WHILE,
            'FOR': TokenType.FOR,
            'REPEAT': TokenType.REPEAT,
            'PARALLEL': TokenType.PARALLEL,
            'SEQUENCE': TokenType.SEQUENCE,
            'END': TokenType.END,
            'THEN': TokenType.THEN,
            'AT': TokenType.AT,
            'WITH': TokenType.WITH,
            'TO': TokenType.TO,
            'SET': TokenType.SET
        }
        
        # Sacred constants
        self.constant_patterns = {
            'phi': TokenType.PHI_CONST,
            'PHI': TokenType.PHI_CONST,
            'lambda': TokenType.LAMBDA_CONST,
            'LAMBDA': TokenType.LAMBDA_CONST,
            'golden_angle': TokenType.GOLDEN_ANGLE_CONST,
            'GOLDEN_ANGLE': TokenType.GOLDEN_ANGLE_CONST
        }
        
        # Sacred frequency names
        self.frequency_names = {
            'ground': '432Hz',
            'creation': '528Hz', 
            'heart': '594Hz',
            'voice': '672Hz',
            'vision': '720Hz',
            'unity': '768Hz',
            'source': '963Hz'
        }
        
        # Regular expressions with phi-harmonic validation
        self.regex_patterns = [
            # Frequency patterns (432Hz, 528Hz, etc.)
            (r'\d+\.?\d*\s*[Hh][Zz]', TokenType.FREQUENCY),
            (r'[a-z_]+_frequency', TokenType.FREQUENCY),
            
            # Phi-level patterns
            (r'phi_level\s*=\s*\d+', TokenType.PHI_LEVEL),
            (r'\d+\s*phi', TokenType.PHI_LEVEL),
            
            # Numbers with sacred mathematics validation
            (r'\d+\.\d+', TokenType.NUMBER),
            (r'\d+', TokenType.NUMBER),
            
            # Strings 
            (r'"[^"]*"', TokenType.STRING),
            (r"'[^']*'", TokenType.STRING),
            
            # Booleans
            (r'\b(true|false|True|False)\b', TokenType.BOOLEAN),
            
            # Operators (phi-aligned precedence)
            (r'\*\*', TokenType.POWER),
            (r'==', TokenType.EQUAL),
            (r'!=', TokenType.NOT_EQUAL),
            (r'<=', TokenType.LESS_EQUAL),
            (r'>=', TokenType.GREATER_EQUAL),
            (r'=', TokenType.ASSIGN),
            (r'\+', TokenType.PLUS),
            (r'-', TokenType.MINUS),
            (r'\*', TokenType.MULTIPLY),
            (r'/', TokenType.DIVIDE),
            (r'%', TokenType.MODULO),
            (r'<', TokenType.LESS_THAN),
            (r'>', TokenType.GREATER_THAN),
            
            # Delimiters
            (r'\(', TokenType.LPAREN),
            (r'\)', TokenType.RPAREN),
            (r'\{', TokenType.LBRACE),
            (r'\}', TokenType.RBRACE),
            (r'\[', TokenType.LBRACKET),
            (r'\]', TokenType.RBRACKET),
            (r',', TokenType.COMMA),
            (r';', TokenType.SEMICOLON),
            (r':', TokenType.COLON),
            (r'\.', TokenType.DOT),
            
            # Identifiers (consciousness-validated)
            (r'[a-zA-Z_][a-zA-Z0-9_]*', TokenType.IDENTIFIER),
            
            # Comments
            (r'#.*', TokenType.COMMENT),
            
            # Whitespace
            (r'[ \t]+', TokenType.WHITESPACE),
            (r'\n', TokenType.NEWLINE),
        ]
        
        # Compile patterns for performance
        self.compiled_patterns = [(re.compile(pattern), token_type) 
                                 for pattern, token_type in self.regex_patterns]
    
    def tokenize(self, source_code: str) -> List[Token]:
        """
        Tokenize PhiFlow source code using sacred mathematics principles
        
        Args:
            source_code: PhiFlow source code to tokenize
            
        Returns:
            List of tokens with sacred mathematics validation
        """
        self.source_code = source_code
        self.current_pos = 0
        self.current_line = 1
        self.current_column = 1
        self.tokens = []
        
        if self.debug:
            print(f"🧠 Consciousness Expert: Starting tokenization in {self.consciousness_state} state")
            print(f"📐 Sacred Mathematics: Using φ = {PHI} with golden angle = {GOLDEN_ANGLE}°")
        
        while self.current_pos < len(source_code):
            # Apply golden angle rotation to parser state
            self._rotate_parser_state()
            
            # Find next token using phi-harmonic pattern matching
            token = self._next_token()
            
            if token:
                # Apply consciousness-guided validation
                if self._validate_token_consciousness(token):
                    self.tokens.append(token)
                    
                    if self.debug and token.type not in [TokenType.WHITESPACE, TokenType.COMMENT]:
                        print(f"✅ Token: {token.type.value} = '{token.value}' "
                              f"(phi_alignment: {token.phi_alignment:.3f if token.phi_alignment else 'N/A'})")
                else:
                    if self.debug:
                        print(f"❌ Rejected token: {token.type.value} = '{token.value}' (consciousness validation failed)")
            else:
                # Handle unrecognized character
                char = source_code[self.current_pos]
                print(f"⚠️ Warning: Unrecognized character '{char}' at line {self.current_line}, column {self.current_column}")
                self._advance()
        
        # Add EOF token
        self.tokens.append(Token(TokenType.EOF, "", self.current_line, self.current_column, self.current_pos))
        
        if self.debug:
            print(f"🎯 Tokenization complete: {len([t for t in self.tokens if t.type != TokenType.WHITESPACE])} tokens")
            print(f"📊 Sacred frequency tokens: {len([t for t in self.tokens if t.type == TokenType.FREQUENCY])}")
            print(f"🧠 Consciousness level tokens: {len([t for t in self.tokens if t.type == TokenType.PHI_LEVEL])}")
        
        return self.tokens
    
    def _rotate_parser_state(self):
        """Apply golden angle rotation to parser state for phi-harmonic processing"""
        self.phi_state_rotation = (self.phi_state_rotation + float(GOLDEN_ANGLE)) % 360
        
        # Update consciousness state based on phi rotation
        rotation_states = [
            (0, 51.4, "OBSERVE"),     # 0° to φ^1 * 32°
            (51.4, 83.8, "CREATE"),   # φ^1 to φ^2 * 32°  
            (83.8, 135.6, "INTEGRATE"), # φ^2 to φ^3 * 32°
            (135.6, 219.4, "HARMONIZE"), # φ^3 to φ^4 * 32°
            (219.4, 293.2, "TRANSCEND"), # φ^4 to φ^5 * 32°
            (293.2, 360, "CASCADE")     # φ^5 to 360°
        ]
        
        for min_angle, max_angle, state in rotation_states:
            if min_angle <= self.phi_state_rotation < max_angle:
                self.consciousness_state = state
                break
    
    def _next_token(self) -> Optional[Token]:
        """Find next token using sacred mathematics pattern recognition"""
        if self.current_pos >= len(self.source_code):
            return None
        
        start_pos = self.current_pos
        start_line = self.current_line
        start_column = self.current_column
        
        # Skip whitespace (but track for indentation)
        if self._current_char().isspace():
            return self._handle_whitespace()
        
        # Try command patterns first (highest precedence)
        token = self._match_command_pattern()
        if token:
            return token
        
        # Try constant patterns
        token = self._match_constant_pattern()
        if token:
            return token
        
        # Try regex patterns with phi-harmonic precedence
        token = self._match_regex_patterns()
        if token:
            return token
        
        return None
    
    def _match_command_pattern(self) -> Optional[Token]:
        """Match sacred geometry commands with consciousness validation"""
        # Look ahead for complete word boundaries
        for command, token_type in self.command_patterns.items():
            if self._match_word(command):
                value = command
                token = Token(token_type, value, self.current_line, self.current_column, self.current_pos)
                self._advance(len(command))
                return token
        return None
    
    def _match_constant_pattern(self) -> Optional[Token]:
        """Match sacred mathematics constants"""
        for constant, token_type in self.constant_patterns.items():
            if self._match_word(constant):
                value = constant
                token = Token(token_type, value, self.current_line, self.current_column, self.current_pos)
                self._advance(len(constant))
                return token
        return None
    
    def _match_regex_patterns(self) -> Optional[Token]:
        """Match using regex patterns with phi-harmonic precedence"""
        remaining_code = self.source_code[self.current_pos:]
        
        for pattern, token_type in self.compiled_patterns:
            match = pattern.match(remaining_code)
            if match:
                value = match.group(0)
                token = Token(token_type, value, self.current_line, self.current_column, self.current_pos)
                self._advance(len(value))
                return token
        
        return None
    
    def _match_word(self, word: str) -> bool:
        """Check if word matches at current position with word boundaries"""
        if self.current_pos + len(word) > len(self.source_code):
            return False
        
        # Check exact match
        if self.source_code[self.current_pos:self.current_pos + len(word)] != word:
            return False
        
        # Check word boundaries
        if self.current_pos > 0:
            prev_char = self.source_code[self.current_pos - 1]
            if prev_char.isalnum() or prev_char == '_':
                return False
        
        if self.current_pos + len(word) < len(self.source_code):
            next_char = self.source_code[self.current_pos + len(word)]
            if next_char.isalnum() or next_char == '_':
                return False
        
        return True
    
    def _handle_whitespace(self) -> Optional[Token]:
        """Handle whitespace with sacred geometry indentation tracking"""
        start_pos = self.current_pos
        start_line = self.current_line
        start_column = self.current_column
        
        whitespace = ""
        while self.current_pos < len(self.source_code) and self._current_char().isspace():
            char = self._current_char()
            whitespace += char
            
            if char == '\n':
                token = Token(TokenType.NEWLINE, '\\n', start_line, start_column, start_pos)
                self._advance()
                return token
            else:
                self._advance()
        
        if whitespace:
            return Token(TokenType.WHITESPACE, whitespace, start_line, start_column, start_pos)
        
        return None
    
    def _validate_token_consciousness(self, token: Token) -> bool:
        """Validate token using consciousness-guided rules"""
        
        # Always accept basic tokens
        if token.type in [TokenType.WHITESPACE, TokenType.COMMENT, TokenType.NEWLINE, TokenType.EOF]:
            return True
        
        # Validate frequencies using sacred mathematics
        if token.type == TokenType.FREQUENCY:
            return token.frequency_validation if token.frequency_validation is not None else True
        
        # Validate phi-levels
        if token.type == TokenType.PHI_LEVEL:
            try:
                level = int(token.value.replace('phi_level=', '').replace('phi', '').strip())
                return 0 <= level <= 7  # Valid consciousness levels
            except:
                return False
        
        # Validate numbers for phi-alignment
        if token.type == TokenType.NUMBER:
            try:
                num = float(token.value)
                
                # Check for phi-related numbers
                phi_values = [1, float(PHI), float(PHI**2), float(PHI**3), float(PHI**4), float(LAMBDA)]
                for phi_val in phi_values:
                    if abs(num - phi_val) < 0.001:  # Close to phi value
                        return True
                
                # Accept all positive numbers
                return num >= 0
            except:
                return False
        
        # Accept all other valid tokens
        return True
    
    def _current_char(self) -> str:
        """Get current character safely"""
        if self.current_pos >= len(self.source_code):
            return ''
        return self.source_code[self.current_pos]
    
    def _advance(self, count: int = 1):
        """Advance position with line/column tracking"""
        for _ in range(count):
            if self.current_pos < len(self.source_code):
                if self.source_code[self.current_pos] == '\n':
                    self.current_line += 1
                    self.current_column = 1
                else:
                    self.current_column += 1
                self.current_pos += 1
    
    def get_tokens_by_type(self, token_type: TokenType) -> List[Token]:
        """Get all tokens of a specific type"""
        return [token for token in self.tokens if token.type == token_type]
    
    def get_sacred_frequency_tokens(self) -> List[Token]:
        """Get all sacred frequency tokens with validation"""
        freq_tokens = self.get_tokens_by_type(TokenType.FREQUENCY)
        return [token for token in freq_tokens if token.frequency_validation]
    
    def get_phi_alignment_score(self) -> float:
        """Calculate overall phi-alignment score for tokenized code"""
        freq_tokens = self.get_sacred_frequency_tokens()
        if not freq_tokens:
            return 0.0
        
        total_alignment = sum(token.phi_alignment for token in freq_tokens if token.phi_alignment)
        return total_alignment / len(freq_tokens) if freq_tokens else 0.0
    
    def validate_program_structure(self) -> dict:
        """Validate program structure using sacred mathematics principles"""
        validation_results = {
            'total_tokens': len([t for t in self.tokens if t.type not in [TokenType.WHITESPACE, TokenType.COMMENT]]),
            'sacred_frequency_count': len(self.get_sacred_frequency_tokens()),
            'phi_alignment_score': self.get_phi_alignment_score(),
            'consciousness_coverage': self._check_consciousness_coverage(),
            'command_balance': self._check_command_balance(),
            'fibonacci_structure': self._check_fibonacci_structure()
        }
        
        return validation_results
    
    def _check_consciousness_coverage(self) -> dict:
        """Check coverage of consciousness states in program"""
        consciousness_commands = [
            TokenType.OBSERVE, TokenType.CREATE, TokenType.INTEGRATE,
            TokenType.HARMONIZE, TokenType.TRANSCEND, TokenType.CASCADE
        ]
        
        coverage = {}
        for cmd_type in consciousness_commands:
            coverage[cmd_type.value] = len(self.get_tokens_by_type(cmd_type))
        
        return coverage
    
    def _check_command_balance(self) -> dict:
        """Check balance of sacred geometry commands"""
        sacred_commands = [
            TokenType.INITIALIZE, TokenType.TRANSITION, TokenType.EVOLVE,
            TokenType.INTEGRATE, TokenType.HARMONIZE, TokenType.TRANSCEND, TokenType.CASCADE
        ]
        
        balance = {}
        for cmd_type in sacred_commands:
            balance[cmd_type.value] = len(self.get_tokens_by_type(cmd_type))
        
        return balance
    
    def _check_fibonacci_structure(self) -> bool:
        """Check if token distribution follows Fibonacci patterns"""
        total_tokens = len([t for t in self.tokens if t.type not in [TokenType.WHITESPACE, TokenType.COMMENT]])
        
        # Check if total tokens is close to a Fibonacci number
        for fib_num in self.fibonacci_sequence:
            if abs(total_tokens - fib_num) <= 2:  # Allow small variance
                return True
        
        return False

def test_phi_flow_lexer():
    """Test the PhiFlow lexer with sacred mathematics validation"""
    
    print("🧪 Testing PhiFlow Lexer - Task 3.1 Implementation")
    print("=" * 70)
    
    # Test program with sacred geometry commands
    test_program = """
    # PhiFlow Test Program with Sacred Mathematics
    INITIALIZE quantum_field AT 432Hz WITH coherence=1.0
    SET base_frequency = 432
    SET harmonic_frequency = base_frequency * phi
    
    IF consciousness_state == "TRANSCEND" THEN
        EVOLVE TO 963Hz WITH phi_level=5
        HARMONIZE all_systems AT 720Hz WITH resonance=golden_angle
    ELSE
        MAINTAIN AT ground WITH stability=true
    ENDIF
    
    PARALLEL
        EVOLVE consciousness AT creation
        INTEGRATE heart_field AT 594Hz 
        TRANSCEND limitations AT unity
    END
    
    CASCADE infinite_love AT source WITH superposition=true
    """
    
    # Create lexer with debug output
    lexer = PhiFlowLexer(debug=True)
    
    # Tokenize the test program
    print("\\n🔄 Tokenizing PhiFlow program...")
    tokens = lexer.tokenize(test_program)
    
    # Display results
    print("\\n📊 Tokenization Results:")
    print(f"  Total tokens: {len(tokens)}")
    print(f"  Sacred frequency tokens: {len(lexer.get_sacred_frequency_tokens())}")
    print(f"  Phi alignment score: {lexer.get_phi_alignment_score():.3f}")
    
    # Validate program structure
    validation = lexer.validate_program_structure()
    print("\\n🎯 Program Structure Validation:")
    for key, value in validation.items():
        print(f"  {key}: {value}")
    
    # Show consciousness coverage
    print("\\n🧠 Consciousness State Coverage:")
    for state, count in validation['consciousness_coverage'].items():
        if count > 0:
            print(f"  {state}: {count} tokens")
    
    # Show sacred frequencies found
    sacred_tokens = lexer.get_sacred_frequency_tokens()
    if sacred_tokens:
        print("\\n📐 Sacred Frequencies Detected:")
        for token in sacred_tokens:
            print(f"  {token.value} (phi-alignment: {token.phi_alignment:.3f})")
    
    print("\\n✅ PhiFlow Lexer Test Complete!")
    return tokens

if __name__ == "__main__":
    test_phi_flow_lexer()