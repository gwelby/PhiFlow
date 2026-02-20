use std::collections::HashMap;
use crate::phi_core::*;

#[derive(Debug, Clone, PartialEq)]
pub enum PhiToken {
    // Core language constructs
    Create, Pattern, At, Hz, With, Into, From,
    Function, Return, Let, If, Else, For, In, While,

    // Sacred frequency literals
    Sacred(u32),

    // Consciousness keywords
    Consciousness, Monitor, Coherence, Validate, Zone,
    Foundational, Elevated, Transcendent, Cosmic,
    // New consciousness keywords
    Frequency, Intention, Witness, State, Hardware, Emergency,
    Protocol, Trigger, Immediate, Notify, Gradient,
    BiologicalProgram, Target, Sequence, Resonate,
    QuantumBridge, QuantumField, ConsciousnessFlow,

    // Pattern types
    Spiral, Flower, DNA, Mandelbrot, Pentagram, SriYantra,
    Golden, Fibonacci, Heart, Toroid, Field,

    // Mathematical operations
    Generate, Transform, Combine, Analyze, Synthesize,

    // Data types
    Point2D, Point3D, Pattern2D, Pattern3D, Audio,
    NumberType, StringType, BooleanType, // Specific tokens for types

    // Literals
    Number(f64),
    String(String),
    Boolean(bool),
    Identifier(String),

    // Operators
    Plus, Minus, Star, Slash, Percent, Power,
    Equal, EqualEqual, NotEqual, Less, LessEqual, Greater, GreaterEqual,
    And, Or, Not,

    // Delimiters
    LeftParen, RightParen, LeftBrace, RightBrace, LeftBracket, RightBracket,
    Comma, Semicolon, Colon, Arrow, Dot,

    // Special
    Newline, Eof,
}

#[derive(Debug, Clone)]
pub enum PhiExpression {
    // Pattern creation
    CreatePattern {
        pattern_type: String,
        frequency: f64,
        parameters: HashMap<String, PhiValue>,
    },

    // Function definition and calls
    FunctionDef {
        name: String,
        parameters: Vec<(String, Option<PhiType>)>,
        return_type: PhiType,
        body: Box<PhiExpression>,
    },
    FunctionCall {
        name: String,
        arguments: Vec<PhiExpression>,
    },

    // Variable binding and access
    LetBinding {
        name: String,
        value: Box<PhiExpression>,
        phi_type: Option<PhiType>,
    },
    Variable(String),

    // Pattern operations
    PatternTransform {
        pattern: Box<PhiExpression>,
        transform_type: String,
        parameters: HashMap<String, PhiValue>,
    },
    PatternCombine {
        patterns: Vec<PhiExpression>,
        combine_type: String,
    },

    // Consciousness operations
    ConsciousnessValidation {
        pattern: Box<PhiExpression>,
        metrics: Vec<String>,
    },
    ConsciousnessMonitor {
        expression: Box<PhiExpression>,
        callback: Box<PhiExpression>,
    },
    
    // NEW: Consciousness-aware constructs
    ConsciousnessState {
        state: String, // "TRANSCEND", "CREATE", etc.
        coherence: f64,
        frequency: f64,
    },
    
    FrequencyPattern {
        base_frequency: f64,
        harmonics: Vec<f64>,
        phi_scaling: bool,
    },
    
    QuantumField {
        field_type: String,
        dimensions: Vec<u32>,
        coherence_target: f64,
    },
    
    BiologicalInterface {
        target: String, // "dna", "protein", etc.
        transduction_method: String,
        frequency: f64,
    },
    
    HardwareSync {
        device_type: String,
        consciousness_mapping: Box<PhiExpression>,
    },
    
    // Consciousness flow control
    ConsciousnessFlow {
        condition: Box<PhiExpression>,
        branches: Vec<(String, Box<PhiExpression>)>, // state -> action
    },
    
    // Emergency protocol
    EmergencyProtocol {
        trigger: Box<PhiExpression>,
        immediate_action: Box<PhiExpression>,
        notification: Vec<String>,
    },

    // Audio synthesis
    AudioSynthesis {
        pattern: Box<PhiExpression>,
        audio_type: String,
        parameters: HashMap<String, PhiValue>,
    },

    // === PHIFLOW UNIQUE: Constructs no other language has ===

    // WITNESS: Pause execution, hold state, be present with it.
    // Not a breakpoint. Not a sleep. The program observes itself.
    Witness {
        expression: Option<Box<PhiExpression>>,  // what to witness (None = witness everything)
        body: Option<Box<PhiExpression>>,         // optional block to execute after witnessing
    },

    // INTENTION: Declare WHY before HOW. The program's purpose affects execution.
    // Same code can route differently based on intention.
    IntentionBlock {
        intention: String,                         // the declared purpose
        body: Box<PhiExpression>,                  // code executed under this intention
    },

    // RESONATE: Share state between intention blocks. Code that talks to itself.
    // resonate              -> share current intention's state to the field
    // resonate expression   -> share a specific value to the field
    Resonate {
        expression: Option<Box<PhiExpression>>,    // what to share (None = share all)
    },

    // Control flow
    Block(Vec<PhiExpression>),
    IfElse {
        condition: Box<PhiExpression>,
        then_branch: Box<PhiExpression>,
        else_branch: Option<Box<PhiExpression>>,
    },
    ForLoop {
        variable: String,
        iterable: Box<PhiExpression>,
        body: Box<PhiExpression>,
    },
    WhileLoop {
        condition: Box<PhiExpression>,
        body: Box<PhiExpression>,
    },
    Return(Box<PhiExpression>),

    // Literals
    Number(f64),
    String(String),
    Boolean(bool),
    List(Vec<PhiExpression>),
    ListAccess {
        list: Box<PhiExpression>,
        index: Box<PhiExpression>,
    },

    // Binary and unary operations
    BinaryOp {
        left: Box<PhiExpression>,
        operator: BinaryOperator,
        right: Box<PhiExpression>,
    },
    UnaryOp {
        operator: UnaryOperator,
        operand: Box<PhiExpression>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum PhiValue {
    Number(f64),
    String(String),
    Boolean(bool),
    Point2D([f64; 2]),
    Point3D([f64; 3]),
    Pattern2D(Vec<[f64; 2]>),
    Pattern3D(Vec<[f64; 3]>),
    Audio(Vec<f64>),
    ValidationResult(ValidationResult),
    List(Vec<PhiValue>),
    Return(Box<PhiValue>),
}

#[derive(Debug, Clone, PartialEq)] // Corrected derive macro
pub enum PhiType {
    Number, String, Boolean,
    Point2D, Point3D,
    Pattern2D, Pattern3D,
    Audio,
    ValidationResult,
    List(Box<PhiType>),
    Function(Vec<PhiType>, Box<PhiType>),
    Void,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOperator {
    Add, Subtract, Multiply, Divide, Modulo, Power,
    Equal, NotEqual, Less, LessEqual, Greater, GreaterEqual,
    And, Or,
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOperator {
    Negate, Not,
}

pub struct PhiLexer {
    input: Vec<char>,
    position: usize,
    current_char: Option<char>,
    line: usize,
    column: usize,
}

impl PhiLexer {
    pub fn new(input: &str) -> Self {
        let chars: Vec<char> = input.chars().collect();
        let current_char = chars.get(0).copied();

        PhiLexer {
            input: chars,
            position: 0,
            current_char,
            line: 1,
            column: 1,
        }
    }

    pub fn tokenize(&mut self) -> Result<Vec<PhiToken>, String> {
        let mut tokens = Vec::new();

        while let Some(token) = self.next_token()? {
            if token == PhiToken::Eof {
                break;
            }
            tokens.push(token);
        }

        tokens.push(PhiToken::Eof);
        Ok(tokens)
    }

    fn next_token(&mut self) -> Result<Option<PhiToken>, String> {
        self.skip_whitespace();

        match self.current_char {
            None => Ok(Some(PhiToken::Eof)),
            Some('\n') => {
                self.advance();
                Ok(Some(PhiToken::Newline))
            }
            Some('(') => {
                self.advance();
                Ok(Some(PhiToken::LeftParen))
            }
            Some(')') => {
                self.advance();
                Ok(Some(PhiToken::RightParen))
            }
            Some('{') => {
                self.advance();
                Ok(Some(PhiToken::LeftBrace))
            }
            Some('}') => {
                self.advance();
                Ok(Some(PhiToken::RightBrace))
            }
            Some('[') => {
                self.advance();
                Ok(Some(PhiToken::LeftBracket))
            }
            Some(']') => {
                self.advance();
                Ok(Some(PhiToken::RightBracket))
            }
            Some(',') => {
                self.advance();
                Ok(Some(PhiToken::Comma))
            }
            Some(';') => {
                self.advance();
                Ok(Some(PhiToken::Semicolon))
            }
            Some(':') => {
                self.advance();
                Ok(Some(PhiToken::Colon))
            }
            Some('.') => {
                self.advance();
                Ok(Some(PhiToken::Dot))
            }
            Some('+') => {
                self.advance();
                Ok(Some(PhiToken::Plus))
            }
            Some('-') => {
                if self.peek() == Some('>') {
                    self.advance();
                    self.advance();
                    Ok(Some(PhiToken::Arrow))
                } else {
                    self.advance();
                    Ok(Some(PhiToken::Minus))
                }
            }
            Some('*') => {
                if self.peek() == Some('*') {
                    self.advance();
                    self.advance();
                    Ok(Some(PhiToken::Power))
                } else {
                    self.advance();
                    Ok(Some(PhiToken::Star))
                }
            }
            Some('/') => {
                if self.peek() == Some('/') {
                    // Line comment - skip to end of line
                    while let Some(c) = self.current_char {
                        if c == '\n' {
                            break;
                        }
                        self.advance();
                    }
                    self.next_token()
                } else {
                    self.advance();
                    Ok(Some(PhiToken::Slash))
                }
            }
            Some('%') => {
                self.advance();
                Ok(Some(PhiToken::Percent))
            }
            Some('=') => {
                if self.peek() == Some('=') {
                    self.advance();
                    self.advance();
                    Ok(Some(PhiToken::EqualEqual))
                } else {
                    self.advance();
                    Ok(Some(PhiToken::Equal))
                }
            }
            Some('!') => {
                if self.peek() == Some('=') {
                    self.advance();
                    self.advance();
                    Ok(Some(PhiToken::NotEqual))
                } else {
                    self.advance();
                    Ok(Some(PhiToken::Not))
                }
            }
            Some('<') => {
                if self.peek() == Some('=') {
                    self.advance();
                    self.advance();
                    Ok(Some(PhiToken::LessEqual))
                } else {
                    self.advance();
                    Ok(Some(PhiToken::Less))
                }
            }
            Some('>') => {
                if self.peek() == Some('=') {
                    self.advance();
                    self.advance();
                    Ok(Some(PhiToken::GreaterEqual))
                } else {
                    self.advance();
                    Ok(Some(PhiToken::Greater))
                }
            }
            Some('&') => {
                if self.peek() == Some('&') {
                    self.advance();
                    self.advance();
                    Ok(Some(PhiToken::And))
                } else {
                    Err(format!("Unexpected character '&' at line {}, column {}", self.line, self.column))
                }
            }
            Some('|') => {
                if self.peek() == Some('|') {
                    self.advance();
                    self.advance();
                    Ok(Some(PhiToken::Or))
                } else {
                    Err(format!("Unexpected character '|' at line {}, column {}", self.line, self.column))
                }
            }
            Some('"') => {
                Ok(Some(self.read_string()?))
            }
            Some(c) if c.is_alphabetic() || c == '_' => {
                Ok(Some(self.read_identifier()))
            }
            Some(c) if c.is_numeric() => {
                Ok(Some(self.read_number()?))
            }
            Some(c) => {
                Err(format!("Unexpected character '{}' at line {}, column {}", c, self.line, self.column))
            }
        }
    }

    fn read_identifier(&mut self) -> PhiToken {
        let mut value = String::new();

        while let Some(c) = self.current_char {
            if c.is_alphanumeric() || c == '_' {
                value.push(c);
                self.advance();
            } else {
                break;
            }
        }

        // Check for keywords
        match value.as_str() {
            // Core constructs
            "create" => PhiToken::Create,
            "pattern" => PhiToken::Pattern,
            "at" => PhiToken::At,
            "Hz" => PhiToken::Hz,
            "with" => PhiToken::With,
            "into" => PhiToken::Into,
            "from" => PhiToken::From,
            "function" => PhiToken::Function,
            "return" => PhiToken::Return,
            "let" => PhiToken::Let,
            "if" => PhiToken::If,
            "else" => PhiToken::Else,
            "for" => PhiToken::For,
            "in" => PhiToken::In,
            "while" => PhiToken::While,

            // Consciousness
            "consciousness" => PhiToken::Consciousness,
            "monitor" => PhiToken::Monitor,
            "coherence" => PhiToken::Coherence,
            "validate" => PhiToken::Validate,
            "zone" => PhiToken::Zone,
            "foundational" => PhiToken::Foundational,
            "elevated" => PhiToken::Elevated,
            "transcendent" => PhiToken::Transcendent,
            "cosmic" => PhiToken::Cosmic,
            // New consciousness keywords
            "frequency" => PhiToken::Frequency,
            "intention" => PhiToken::Intention,
            "witness" => PhiToken::Witness,
            "resonate" => PhiToken::Resonate,
            "state" => PhiToken::State,
            "hardware" => PhiToken::Hardware,
            "emergency" => PhiToken::Emergency,
            "protocol" => PhiToken::Protocol,
            "trigger" => PhiToken::Trigger,
            "immediate" => PhiToken::Immediate,
            "notify" => PhiToken::Notify,
            "gradient" => PhiToken::Gradient,
            "biological_program" => PhiToken::BiologicalProgram,
            "target" => PhiToken::Target,
            "sequence" => PhiToken::Sequence,
            "quantum_bridge" => PhiToken::QuantumBridge,
            "quantum_field" => PhiToken::QuantumField,
            "consciousness_flow" => PhiToken::ConsciousnessFlow,

            // Patterns
            "spiral" => PhiToken::Spiral,
            "flower" => PhiToken::Flower,
            "dna" => PhiToken::DNA,
            "mandelbrot" => PhiToken::Mandelbrot,
            "pentagram" => PhiToken::Pentagram,
            "sriyantra" => PhiToken::SriYantra, // Corrected typo
            "golden" => PhiToken::Golden,
            "fibonacci" => PhiToken::Fibonacci,
            "heart" => PhiToken::Heart,
            "toroid" => PhiToken::Toroid,
            "field" => PhiToken::Field,

            // Operations
            "generate" => PhiToken::Generate,
            "transform" => PhiToken::Transform,
            "combine" => PhiToken::Combine,
            "analyze" => PhiToken::Analyze,
            "synthesize" => PhiToken::Synthesize,

            // Types
            "Number" => PhiToken::NumberType,
            "String" => PhiToken::StringType,
            "Boolean" => PhiToken::BooleanType,
            "Point2D" => PhiToken::Point2D,
            "Point3D" => PhiToken::Point3D,
            "Pattern2D" => PhiToken::Pattern2D,
            "Pattern3D" => PhiToken::Pattern3D,
            "Audio" => PhiToken::Audio,
            "ValidationResult" => PhiToken::Identifier(value), // Handled as Identifier

            // Literals
            "true" => PhiToken::Boolean(true),
            "false" => PhiToken::Boolean(false),

            _ => PhiToken::Identifier(value),
        }
    }

    fn read_number(&mut self) -> Result<PhiToken, String> {
        let mut value = String::new();
        let mut has_dot = false;

        while let Some(c) = self.current_char {
            if c.is_numeric() {
                value.push(c);
                self.advance();
            } else if c == '.' && !has_dot {
                has_dot = true;
                value.push(c);
                self.advance();
            } else {
                break;
            }
        }

        value.parse::<f64>()
            .map(PhiToken::Number)
            .map_err(|_| format!("Invalid number '{}' at line {}, column {}", value, self.line, self.column))
    }

    fn read_string(&mut self) -> Result<PhiToken, String> {
        self.advance(); // Skip opening quote
        let mut value = String::new();

        while let Some(c) = self.current_char {
            if c == '"' {
                self.advance(); // Skip closing quote
                return Ok(PhiToken::String(value));
            } else if c == '\\' { // Corrected from single backslash to double backslash
                self.advance();
                match self.current_char {
                    Some('n') => {
                        value.push('\n'); // Corrected from literal newline to escaped newline
                        self.advance();
                    }
                    Some('t') => {
                        value.push('\t'); // Corrected from literal tab to escaped tab
                        self.advance();
                    }
                    Some('\\') => { // Corrected from single backslash to double backslash
                        value.push('\\'); // Corrected from literal backslash to escaped backslash
                        self.advance();
                    }
                    Some('"') => {
                        value.push('"');
                        self.advance();
                    }
                    Some(c) => {
                        value.push(c);
                        self.advance();
                    }
                    None => {
                        return Err(format!("Unterminated string at line {}, column {}", self.line, self.column));
                    }
                }
            } else {
                value.push(c);
                self.advance();
            }
        }

        Err(format!("Unterminated string at line {}, column {}", self.line, self.column))
    }

    fn advance(&mut self) {
        if let Some('\n') = self.current_char { // Corrected from literal newline to escaped newline
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }

        self.position += 1;
        self.current_char = self.input.get(self.position).copied();
    }

    fn peek(&self) -> Option<char> {
        self.input.get(self.position + 1).copied()
    }

    fn skip_whitespace(&mut self) {
        while let Some(c) = self.current_char {
            if c.is_whitespace() && c != '\n' { // Corrected from literal newline to escaped newline
                self.advance();
            } else {
                break;
            }
        }
    }
}

// Enhanced parser with error recovery
pub struct PhiParser {
    tokens: Vec<PhiToken>,
    position: usize,
    current_token: PhiToken,
}

impl PhiParser {
    pub fn new(tokens: Vec<PhiToken>) -> Self {
        let current_token = tokens.get(0).cloned().unwrap_or(PhiToken::Eof);
        PhiParser {
            tokens,
            position: 0,
            current_token,
        }
    }

    pub fn parse(&mut self) -> Result<Vec<PhiExpression>, String> {
        let mut expressions = Vec::new();

        while self.current_token != PhiToken::Eof {
            if self.current_token == PhiToken::Newline {
                self.advance();
                continue;
            }

            let expr = self.parse_statement()?;
            expressions.push(expr);
        }

        Ok(expressions)
    }

    fn parse_statement(&mut self) -> Result<PhiExpression, String> {
        match &self.current_token {
            PhiToken::Create => self.parse_create_statement(),
            PhiToken::Let => self.parse_let_statement(),
            PhiToken::Function => self.parse_function_definition(),
            PhiToken::If => self.parse_if_statement(),
            PhiToken::For => self.parse_for_statement(),
            PhiToken::While => self.parse_while_statement(),
            PhiToken::Validate => self.parse_validation_statement(),
            PhiToken::Return => self.parse_return_statement(),
            PhiToken::Witness => self.parse_witness_statement(),
            PhiToken::Resonate => self.parse_resonate_statement(),
            PhiToken::Intention => self.parse_intention_block(),
            PhiToken::Consciousness => self.parse_consciousness_declaration(),
            PhiToken::Hardware => self.parse_hardware_declaration(),
            PhiToken::Emergency => self.parse_emergency_protocol(),
            PhiToken::ConsciousnessFlow => self.parse_consciousness_flow(),
            PhiToken::BiologicalProgram => self.parse_biological_program(),
            PhiToken::QuantumBridge => self.parse_quantum_bridge(),
            // Handle pattern tokens as variable names in statement context
            PhiToken::Pattern => {
                let var_name = "pattern".to_string();
                self.advance();
                Ok(PhiExpression::Variable(var_name))
            }
            PhiToken::Spiral => {
                let var_name = "spiral".to_string();
                self.advance();
                Ok(PhiExpression::Variable(var_name))
            }
            PhiToken::Flower => {
                let var_name = "flower".to_string();
                self.advance();
                Ok(PhiExpression::Variable(var_name))
            }
            PhiToken::DNA => {
                let var_name = "dna".to_string();
                self.advance();
                Ok(PhiExpression::Variable(var_name))
            }
            PhiToken::Frequency => {
                // Could be frequency_pattern declaration OR just 'frequency' used as a variable
                // Peek ahead: if next non-newline token is '{', it's a frequency pattern
                // Otherwise treat as expression statement (variable usage)
                let mut lookahead = self.position + 1;
                while lookahead < self.tokens.len() && self.tokens[lookahead] == PhiToken::Newline {
                    lookahead += 1;
                }
                if lookahead < self.tokens.len() && self.tokens[lookahead] == PhiToken::LeftBrace {
                    self.parse_frequency_pattern()
                } else {
                    self.parse_expression_statement()
                }
            }
            PhiToken::LeftBrace => {
                // Parse block: { ... }
                self.advance();
                let mut expressions = Vec::new();
                while self.current_token != PhiToken::RightBrace {
                    if self.current_token == PhiToken::Newline {
                        self.advance();
                        continue;
                    }
                    if self.current_token == PhiToken::Eof {
                        return Err("Unexpected end of file in block".to_string());
                    }
                    expressions.push(self.parse_statement()?);
                }
                self.expect(PhiToken::RightBrace)?;
                Ok(PhiExpression::Block(expressions))
            }
            PhiToken::Identifier(name) => {
                // If it's an identifier, it could be a variable or a function call
                // We'll try to parse it as an expression statement
                self.parse_expression_statement()
            },
            // Bare expressions as statements (numbers, strings, booleans, parens)
            PhiToken::Number(_) | PhiToken::String(_) | PhiToken::Boolean(_)
            | PhiToken::LeftParen | PhiToken::Minus | PhiToken::Not | PhiToken::Coherence => {
                self.parse_expression_statement()
            }
            _ => {
                Err(format!("Unexpected token in statement: {:?}", self.current_token))
            }
        }
    }

    fn parse_create_statement(&mut self) -> Result<PhiExpression, String> {
        self.expect(PhiToken::Create)?;
        
        // Handle optional "pattern" keyword
        if self.current_token == PhiToken::Pattern {
            self.advance();
        }

        let pattern_type_token = self.current_token.clone();
        let pattern_type = match pattern_type_token {
            PhiToken::Spiral => "spiral".to_string(),
            PhiToken::Flower => "flower".to_string(),
            PhiToken::DNA => "dna".to_string(),
            PhiToken::Mandelbrot => "mandelbrot".to_string(),
            PhiToken::Pentagram => "pentagram".to_string(),
            PhiToken::SriYantra => "sriyantra".to_string(),
            PhiToken::Golden => "golden".to_string(),
            PhiToken::Fibonacci => "fibonacci".to_string(),
            PhiToken::Heart => "heart".to_string(),
            PhiToken::Toroid => "toroid".to_string(),
            PhiToken::Field => "field".to_string(),
            _ => return Err(format!("Expected pattern type, found {:?}", pattern_type_token)),
        };
        self.advance();

        self.expect(PhiToken::At)?;
        let (frequency, frequency_var) = match &self.current_token {
            PhiToken::Number(n) => {
                let freq = *n;
                self.advance();
                (freq, None)
            }
            PhiToken::Identifier(name) => {
                let var_name = name.clone();
                self.advance();
                (-1.0, Some(var_name))
            }
            PhiToken::Frequency | PhiToken::State | PhiToken::Coherence
            | PhiToken::Monitor | PhiToken::Pattern | PhiToken::Target
            | PhiToken::Field | PhiToken::Zone | PhiToken::Intention
            | PhiToken::Witness | PhiToken::Resonate | PhiToken::Sequence => {
                let var_name = format!("{:?}", self.current_token).to_lowercase();
                self.advance();
                (-1.0, Some(var_name))
            }
            _ => return Err(format!("Expected frequency (number or variable), found {:?}", self.current_token))
        };
        if self.current_token == PhiToken::Hz {
            self.advance();
        }

        let mut parameters = HashMap::new();

        // If frequency is a variable, store it in parameters for interpreter resolution
        if let Some(var_name) = frequency_var {
            parameters.insert("__frequency_var".to_string(), PhiValue::String(format!("${}", var_name)));
        }

        if self.current_token == PhiToken::With {
            self.advance();
            self.expect(PhiToken::LeftBrace)?;

            while self.current_token != PhiToken::RightBrace {
                // Skip any newlines
                while self.current_token == PhiToken::Newline {
                    self.advance();
                }
                
                // Check for closing brace after skipping newlines
                if self.current_token == PhiToken::RightBrace {
                    break;
                }
                
                let param_name = self.expect_identifier()?;
                self.expect(PhiToken::Colon)?;
                let param_value = self.parse_phi_value()?;
                parameters.insert(param_name, param_value);

                if self.current_token == PhiToken::Comma {
                    self.advance();
                }
                
                // Skip any trailing newlines
                while self.current_token == PhiToken::Newline {
                    self.advance();
                }
            }

            self.expect(PhiToken::RightBrace)?;
        }

        Ok(PhiExpression::CreatePattern {
            pattern_type,
            frequency,
            parameters,
        })
    }

    fn parse_validation_statement(&mut self) -> Result<PhiExpression, String> {
        self.expect(PhiToken::Validate)?;
        
        // Handle pattern tokens as identifiers for validation
        let pattern_expr = match &self.current_token {
            PhiToken::Spiral => {
                self.advance();
                PhiExpression::Variable("spiral".to_string())
            }
            PhiToken::Flower => {
                self.advance();
                PhiExpression::Variable("flower".to_string())
            }
            PhiToken::DNA => {
                self.advance();
                PhiExpression::Variable("dna".to_string())
            }
            PhiToken::Pattern => {
                self.advance();
                PhiExpression::Variable("pattern".to_string())
            }
            PhiToken::Identifier(name) => {
                let var_name = name.clone();
                self.advance();
                PhiExpression::Variable(var_name)
            }
            _ => self.parse_expression()?
        };
        
        let pattern = Box::new(pattern_expr);

        let mut metrics = Vec::new();
        if self.current_token == PhiToken::With {
            self.advance();
            self.expect(PhiToken::LeftBracket)?;

            while self.current_token != PhiToken::RightBracket {
                // Handle consciousness metrics that may be tokens instead of identifiers
                let metric_name = match &self.current_token {
                    PhiToken::Coherence => {
                        self.advance();
                        "coherence".to_string()
                    }
                    PhiToken::Consciousness => {
                        self.advance();
                        "consciousness_zone".to_string()
                    }
                    PhiToken::Identifier(name) => {
                        let metric = name.clone();
                        self.advance();
                        metric
                    }
                    _ => {
                        return Err(format!("Expected metric name, found {:?}", self.current_token));
                    }
                };
                metrics.push(metric_name);
                if self.current_token == PhiToken::Comma {
                    self.advance();
                }
            }

            self.expect(PhiToken::RightBracket)?;
        }

        Ok(PhiExpression::ConsciousnessValidation { pattern, metrics })
    }

    fn parse_phi_value(&mut self) -> Result<PhiValue, String> {
        match &self.current_token {
            PhiToken::Number(n) => {
                let val = *n;
                self.advance();
                Ok(PhiValue::Number(val))
            }
            PhiToken::String(s) => {
                let val = s.clone();
                self.advance();
                Ok(PhiValue::String(val))
            }
            PhiToken::Boolean(b) => {
                let val = *b;
                self.advance();
                Ok(PhiValue::Boolean(val))
            }
            PhiToken::Identifier(name) => {
                // Handle variable references in values
                let var_name = name.clone();
                self.advance();
                // For now, we'll store the variable name as a string
                // The interpreter will resolve it later
                Ok(PhiValue::String(format!("${}", var_name))) // Mark as variable with $
            }
            // Handle list literals
            PhiToken::LeftBracket => {
                self.advance(); // Consume '['
                let mut elements = Vec::new();
                // Check for empty list
                if self.current_token == PhiToken::RightBracket {
                    self.advance();
                    return Ok(PhiValue::List(elements));
                }
                loop {
                    elements.push(self.parse_phi_value()?);
                    if self.current_token == PhiToken::Comma {
                        self.advance();
                    } else {
                        break;
                    }
                }
                self.expect(PhiToken::RightBracket)?; // Consume ']'
                Ok(PhiValue::List(elements))
            }
            _ => Err(format!("Expected value, found {:?}", self.current_token)),
        }
    }

    fn parse_expression_statement(&mut self) -> Result<PhiExpression, String> {
        let expr = self.parse_expression()?;
        Ok(expr)
    }

    fn parse_expression(&mut self) -> Result<PhiExpression, String> {
        self.parse_logical_or()
    }

    fn parse_logical_or(&mut self) -> Result<PhiExpression, String> {
        let mut expr = self.parse_logical_and()?;

        while self.current_token == PhiToken::Or {
            self.advance();
            // Skip newlines after operator
            while self.current_token == PhiToken::Newline {
                self.advance();
            }
            let right = Box::new(self.parse_logical_and()?);
            expr = PhiExpression::BinaryOp {
                left: Box::new(expr),
                operator: BinaryOperator::Or,
                right,
            };
        }

        Ok(expr)
    }

    fn parse_logical_and(&mut self) -> Result<PhiExpression, String> {
        let mut expr = self.parse_equality()?;

        while self.current_token == PhiToken::And {
            self.advance();
            // Skip newlines after operator
            while self.current_token == PhiToken::Newline {
                self.advance();
            }
            let right = Box::new(self.parse_equality()?);
            expr = PhiExpression::BinaryOp {
                left: Box::new(expr),
                operator: BinaryOperator::And,
                right,
            };
        }

        Ok(expr)
    }

    fn parse_equality(&mut self) -> Result<PhiExpression, String> {
        let mut expr = self.parse_comparison()?;

        while matches!(self.current_token, PhiToken::EqualEqual | PhiToken::NotEqual) {
            let op = match self.current_token {
                PhiToken::EqualEqual => BinaryOperator::Equal,
                PhiToken::NotEqual => BinaryOperator::NotEqual,
                _ => unreachable!(),
            };
            self.advance();
            // Skip newlines after operator
            while self.current_token == PhiToken::Newline {
                self.advance();
            }
            let right = Box::new(self.parse_comparison()?);
            expr = PhiExpression::BinaryOp {
                left: Box::new(expr),
                operator: op,
                right,
            };
        }

        Ok(expr)
    }

    fn parse_comparison(&mut self) -> Result<PhiExpression, String> {
        let mut expr = self.parse_term()?;

        while matches!(
            self.current_token,
            PhiToken::Greater | PhiToken::GreaterEqual | PhiToken::Less | PhiToken::LessEqual
        ) {
            let op = match self.current_token {
                PhiToken::Greater => BinaryOperator::Greater,
                PhiToken::GreaterEqual => BinaryOperator::GreaterEqual,
                PhiToken::Less => BinaryOperator::Less,
                PhiToken::LessEqual => BinaryOperator::LessEqual,
                _ => unreachable!(),
            };
            self.advance();
            // Skip newlines after operator
            while self.current_token == PhiToken::Newline {
                self.advance();
            }
            let right = Box::new(self.parse_term()?);
            expr = PhiExpression::BinaryOp {
                left: Box::new(expr),
                operator: op,
                right,
            };
        }

        Ok(expr)
    }

    fn parse_term(&mut self) -> Result<PhiExpression, String> {
        let mut expr = self.parse_factor()?;

        while matches!(self.current_token, PhiToken::Plus | PhiToken::Minus) {
            let op = match self.current_token {
                PhiToken::Plus => BinaryOperator::Add,
                PhiToken::Minus => BinaryOperator::Subtract,
                _ => unreachable!(),
            };
            self.advance();
            // Skip newlines after operator
            while self.current_token == PhiToken::Newline {
                self.advance();
            }
            let right = Box::new(self.parse_factor()?);
            expr = PhiExpression::BinaryOp {
                left: Box::new(expr),
                operator: op,
                right,
            };
        }

        Ok(expr)
    }

    fn parse_factor(&mut self) -> Result<PhiExpression, String> {
        let mut expr = self.parse_unary()?;

        while matches!(self.current_token, PhiToken::Star | PhiToken::Slash | PhiToken::Percent | PhiToken::Power) {
            let op = match self.current_token {
                PhiToken::Star => BinaryOperator::Multiply,
                PhiToken::Slash => BinaryOperator::Divide,
                PhiToken::Percent => BinaryOperator::Modulo,
                PhiToken::Power => BinaryOperator::Power,
                _ => unreachable!(),
            };
            self.advance();
            // Skip newlines after operator
            while self.current_token == PhiToken::Newline {
                self.advance();
            }
            let right = Box::new(self.parse_unary()?);
            expr = PhiExpression::BinaryOp {
                left: Box::new(expr),
                operator: op,
                right,
            };
        }

        Ok(expr)
    }

    fn parse_unary(&mut self) -> Result<PhiExpression, String> {
        match self.current_token {
            PhiToken::Minus => {
                self.advance();
                let operand = Box::new(self.parse_unary()?);
                Ok(PhiExpression::UnaryOp {
                    operator: UnaryOperator::Negate,
                    operand,
                })
            }
            PhiToken::Not => {
                self.advance();
                let operand = Box::new(self.parse_unary()?);
                Ok(PhiExpression::UnaryOp {
                    operator: UnaryOperator::Not,
                    operand,
                })
            }
            _ => self.parse_primary(),
        }
    }

    fn parse_primary(&mut self) -> Result<PhiExpression, String> {
        let mut expr = match self.current_token.clone() {
            PhiToken::Number(n) => {
                self.advance();
                PhiExpression::Number(n)
            }
            PhiToken::String(s) => {
                self.advance();
                PhiExpression::String(s)
            }
            PhiToken::Boolean(b) => {
                self.advance();
                PhiExpression::Boolean(b)
            }
            PhiToken::Coherence => {
                self.advance();
                PhiExpression::Variable("coherence".to_string())
            }
            PhiToken::Create => {
                // Handle create statements as expressions
                self.parse_create_statement()?
            }
            PhiToken::Validate => {
                // Handle validate statements as expressions  
                self.parse_validation_statement()?
            }
            PhiToken::Identifier(name) => {
                let var_name = name.clone();
                self.advance();

                // Check for assignment
                if self.current_token == PhiToken::Equal {
                    self.advance();
                    let value = Box::new(self.parse_expression()?);
                    return Ok(PhiExpression::LetBinding { name: var_name, value, phi_type: None });
                }

                // Check for function call
                if self.current_token == PhiToken::LeftParen {
                    self.advance();
                    let mut arguments = Vec::new();

                    while self.current_token != PhiToken::RightParen {
                        arguments.push(self.parse_expression()?);
                        if self.current_token == PhiToken::Comma {
                            self.advance();
                        }
                    }

                    self.expect(PhiToken::RightParen)?;

                    PhiExpression::FunctionCall {
                        name: var_name,
                        arguments,
                    }
                } else {
                    PhiExpression::Variable(var_name)
                }
            }
            PhiToken::LeftParen => {
                self.advance();
                
                // Skip any newlines after opening paren
                while self.current_token == PhiToken::Newline {
                    self.advance();
                }
                
                let expr = self.parse_expression()?;
                
                // Skip any newlines before closing paren
                while self.current_token == PhiToken::Newline {
                    self.advance();
                }
                
                self.expect(PhiToken::RightParen)?;
                expr
            }
            PhiToken::LeftBrace => {
                self.advance();
                let mut expressions = Vec::new();

                while self.current_token != PhiToken::RightBrace {
                    if self.current_token == PhiToken::Newline {
                        self.advance();
                        continue;
                    }
                    expressions.push(self.parse_statement()?);
                }

                self.expect(PhiToken::RightBrace)?;
                PhiExpression::Block(expressions)
            }
            PhiToken::LeftBracket => {
                // Parse list literals directly as expressions (not through parse_phi_value)
                self.advance(); // Consume '['
                let mut elements = Vec::new();
                
                // Check for empty list
                if self.current_token == PhiToken::RightBracket {
                    self.advance();
                    PhiExpression::List(elements)
                } else {
                    loop {
                        // Parse each element as a full expression (not just simple values)
                        elements.push(self.parse_expression()?);
                        if self.current_token == PhiToken::Comma {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                    self.expect(PhiToken::RightBracket)?; // Consume ']'
                    PhiExpression::List(elements)
                }
            }
            _ => return Err(format!("Unexpected token: {:?}", self.current_token)),
        }; // Note the '?' here to propagate errors

        // Handle list access
        while self.current_token == PhiToken::LeftBracket {
            self.advance(); // Consume '['
            let index = Box::new(self.parse_expression()?);
            self.expect(PhiToken::RightBracket)?; // Consume ']'
            expr = PhiExpression::ListAccess {
                list: Box::new(expr),
                index,
            };
        }

        Ok(expr)
    }

    // Helper methods for parsing specific constructs
    fn parse_let_statement(&mut self) -> Result<PhiExpression, String> {
        self.expect(PhiToken::Let)?;
        let name = self.expect_identifier()?;

        let phi_type = if self.current_token == PhiToken::Colon {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };

        self.expect(PhiToken::Equal)?;
        let value = Box::new(self.parse_expression()?);

        Ok(PhiExpression::LetBinding { name, value, phi_type })
    }

    fn parse_function_definition(&mut self) -> Result<PhiExpression, String> {
        self.expect(PhiToken::Function)?;
        let name = self.expect_identifier()?;

        self.expect(PhiToken::LeftParen)?;
        let mut parameters = Vec::new();

        while self.current_token != PhiToken::RightParen {
            let param_name = self.expect_identifier()?;
            let param_type = if self.current_token == PhiToken::Colon {
                self.advance();
                Some(self.parse_type()?)
            } else {
                None
            };
            parameters.push((param_name, param_type));
            if self.current_token == PhiToken::Comma {
                self.advance();
            }
        }

        self.expect(PhiToken::RightParen)?;

        let return_type = if self.current_token == PhiToken::Arrow {
            self.advance();
            self.parse_type()? // Return the PhiType directly
        } else {
            PhiType::Void
        };

        // Skip any newlines before the function body
        while self.current_token == PhiToken::Newline {
            self.advance();
        }

        let body = Box::new(self.parse_statement()?);

        Ok(PhiExpression::FunctionDef {
            name,
            parameters,
            return_type,
            body,
        })
    }

    fn parse_if_statement(&mut self) -> Result<PhiExpression, String> {
        self.expect(PhiToken::If)?;
        let condition = Box::new(self.parse_expression()?);
        let then_branch = Box::new(self.parse_statement()?);

        let else_branch = if self.current_token == PhiToken::Else {
            self.advance();
            Some(Box::new(self.parse_statement()?))
        } else {
            None
        };

        Ok(PhiExpression::IfElse {
            condition,
            then_branch,
            else_branch,
        })
    }

    fn parse_for_statement(&mut self) -> Result<PhiExpression, String> {
        self.expect(PhiToken::For)?;
        let variable = self.expect_identifier()?;
        self.expect(PhiToken::In)?;
        let iterable = Box::new(self.parse_expression()?);
        let body = Box::new(self.parse_statement()?);

        Ok(PhiExpression::ForLoop {
            variable,
            iterable,
            body,
        })
    }

    fn parse_while_statement(&mut self) -> Result<PhiExpression, String> {
        self.expect(PhiToken::While)?;
        let condition = Box::new(self.parse_expression()?);
        let body = Box::new(self.parse_statement()?);

        Ok(PhiExpression::WhileLoop { condition, body })
    }

    fn parse_return_statement(&mut self) -> Result<PhiExpression, String> {
        self.expect(PhiToken::Return)?;
        let value = Box::new(self.parse_expression()?);
        Ok(PhiExpression::Return(value))
    }

    /// Parse witness statement:
    ///   witness                          -> witness all current state
    ///   witness expression               -> witness a specific value
    ///   witness { body }                 -> witness then execute body
    ///   witness expression { body }      -> witness value then execute body
    fn parse_witness_statement(&mut self) -> Result<PhiExpression, String> {
        self.expect(PhiToken::Witness)?;

        // Check what IMMEDIATELY follows witness (before consuming newlines)
        // This determines: bare witness, witness with expression, or witness with block
        let (expression, body) = if self.current_token == PhiToken::Newline
            || self.current_token == PhiToken::Eof
            || self.current_token == PhiToken::RightBrace
        {
            // bare witness - nothing on the same line
            (None, None)
        } else if self.current_token == PhiToken::LeftBrace {
            // witness { body }
            self.advance();
            let mut expressions = Vec::new();
            while self.current_token != PhiToken::RightBrace {
                if self.current_token == PhiToken::Newline {
                    self.advance();
                    continue;
                }
                if self.current_token == PhiToken::Eof {
                    return Err("Unexpected end of file in witness block".to_string());
                }
                expressions.push(self.parse_statement()?);
            }
            self.expect(PhiToken::RightBrace)?;
            (None, Some(Box::new(PhiExpression::Block(expressions))))
        } else {
            // witness expression
            let expr = Some(Box::new(self.parse_expression()?));
            (expr, None)
        };

        Ok(PhiExpression::Witness { expression, body })
    }

    /// Parse resonate statement:
    ///   resonate              -> share current intention's state
    ///   resonate expression   -> share a specific value
    fn parse_resonate_statement(&mut self) -> Result<PhiExpression, String> {
        self.expect(PhiToken::Resonate)?;

        let expression = if self.current_token == PhiToken::Newline
            || self.current_token == PhiToken::Eof
            || self.current_token == PhiToken::RightBrace
        {
            None
        } else {
            Some(Box::new(self.parse_expression()?))
        };

        Ok(PhiExpression::Resonate { expression })
    }

    /// Parse intention block:
    ///   intention "healing" { body }
    fn parse_intention_block(&mut self) -> Result<PhiExpression, String> {
        self.expect(PhiToken::Intention)?;

        // Expect the intention name as a string
        let intention = match &self.current_token {
            PhiToken::String(s) => {
                let name = s.clone();
                self.advance();
                name
            }
            PhiToken::Identifier(s) => {
                let name = s.clone();
                self.advance();
                name
            }
            _ => return Err(format!("Expected intention name (string or identifier), found {:?}", self.current_token)),
        };

        // Skip newlines
        while self.current_token == PhiToken::Newline {
            self.advance();
        }

        // Parse the body block
        self.expect(PhiToken::LeftBrace)?;
        let mut expressions = Vec::new();
        while self.current_token != PhiToken::RightBrace {
            if self.current_token == PhiToken::Newline {
                self.advance();
                continue;
            }
            if self.current_token == PhiToken::Eof {
                return Err("Unexpected end of file in intention block".to_string());
            }
            expressions.push(self.parse_statement()?);
        }
        self.expect(PhiToken::RightBrace)?;

        Ok(PhiExpression::IntentionBlock {
            intention,
            body: Box::new(PhiExpression::Block(expressions)),
        })
    }

    fn parse_type(&mut self) -> Result<PhiType, String> {
        let type_token = self.current_token.clone(); // Clone to match against it
        self.advance();

        match type_token {
            PhiToken::NumberType => Ok(PhiType::Number),
            PhiToken::StringType => Ok(PhiType::String),
            PhiToken::BooleanType => Ok(PhiType::Boolean),
            PhiToken::Point2D => Ok(PhiType::Point2D),
            PhiToken::Point3D => Ok(PhiType::Point3D),
            PhiToken::Pattern2D => Ok(PhiType::Pattern2D),
            PhiToken::Pattern3D => Ok(PhiType::Pattern3D),
            PhiToken::Audio => Ok(PhiType::Audio),
            PhiToken::Frequency => Ok(PhiType::Number), // Frequencies are numbers
            PhiToken::Identifier(s) if s == "ValidationResult" => Ok(PhiType::ValidationResult),
            PhiToken::Identifier(s) if s == "List" => {
                self.expect(PhiToken::Less)?; // Expect '<'
                let inner_type = self.parse_type()?;
                self.expect(PhiToken::Greater)?; // Expect '>'
                Ok(PhiType::List(Box::new(inner_type)))
            },
            _ => Err(format!("Unknown type: {:?}", type_token)),
        }
    }

    // Helper methods
    fn advance(&mut self) {
        if self.position < self.tokens.len() - 1 {
            self.position += 1;
            self.current_token = self.tokens[self.position].clone();
        }
    }

    fn expect(&mut self, expected: PhiToken) -> Result<(), String> {
        if std::mem::discriminant(&self.current_token) == std::mem::discriminant(&expected) {
            self.advance();
            Ok(())
        } else {
            Err(format!("Expected {:?}, found {:?}", expected, self.current_token))
        }
    }

    fn expect_identifier(&mut self) -> Result<String, String> {
        // Skip any newlines before looking for identifier
        while self.current_token == PhiToken::Newline {
            self.advance();
        }

        // Accept actual identifiers and keywords that could be used as variable names
        let result = match &self.current_token {
            PhiToken::Identifier(name) => Ok(name.clone()),
            PhiToken::Frequency => Ok("frequency".to_string()),
            PhiToken::Witness => Ok("witness".to_string()),
            PhiToken::Resonate => Ok("resonate".to_string()),
            PhiToken::State => Ok("state".to_string()),
            PhiToken::Coherence => Ok("coherence".to_string()),
            PhiToken::Monitor => Ok("monitor".to_string()),
            PhiToken::Pattern => Ok("pattern".to_string()),
            PhiToken::Target => Ok("target".to_string()),
            PhiToken::Field => Ok("field".to_string()),
            PhiToken::Zone => Ok("zone".to_string()),
            PhiToken::Intention => Ok("intention".to_string()),
            PhiToken::Sequence => Ok("sequence".to_string()),
            _ => Err(format!("Expected identifier, found {:?}", self.current_token)),
        };
        if result.is_ok() {
            self.advance();
        }
        result
    }

    fn expect_number(&mut self) -> Result<f64, String> {
        if let PhiToken::Number(n) = &self.current_token {
            let result = *n;
            self.advance();
            Ok(result)
        } else {
            Err(format!("Expected number, found {:?}", self.current_token))
        }
    }

    // ========== Consciousness-Aware Parsing Methods ==========

    /// Parse consciousness declaration: consciousness STATE { frequency: 720Hz, coherence: 0.867, intention: "..." }
    fn parse_consciousness_declaration(&mut self) -> Result<PhiExpression, String> {
        self.expect(PhiToken::Consciousness)?;
        
        // Skip any newlines
        while self.current_token == PhiToken::Newline {
            self.advance();
        }
        
        // Get the state name (e.g., TRANSCEND)
        let state = self.expect_identifier()?;
        
        self.expect(PhiToken::LeftBrace)?;
        
        let mut frequency = 432.0; // Default ground frequency
        let mut coherence = 1.0;
        let mut intention = String::new();
        
        // Parse consciousness properties
        while self.current_token != PhiToken::RightBrace {
            // Skip any newlines
            while self.current_token == PhiToken::Newline {
                self.advance();
            }
            
            // Check for closing brace after skipping newlines
            if self.current_token == PhiToken::RightBrace {
                break;
            }
            
            let prop_name = self.expect_identifier()?;
            self.expect(PhiToken::Colon)?;
            
            match prop_name.as_str() {
                "frequency" => {
                    frequency = self.expect_number()?;
                    if self.current_token == PhiToken::Hz {
                        self.advance();
                    }
                }
                "coherence" => {
                    coherence = self.expect_number()?;
                }
                "intention" => {
                    intention = self.expect_string()?;
                }
                _ => return Err(format!("Unknown consciousness property: {}", prop_name))
            }
            
            if self.current_token == PhiToken::Comma {
                self.advance();
            }
            
            // Skip any trailing newlines
            while self.current_token == PhiToken::Newline {
                self.advance();
            }
        }
        
        self.expect(PhiToken::RightBrace)?;
        
        Ok(PhiExpression::ConsciousnessState {
            state,
            coherence,
            frequency,
        })
    }

    /// Parse hardware declaration: hardware NAME { device: "...", map consciousness_level to ... }
    fn parse_hardware_declaration(&mut self) -> Result<PhiExpression, String> {
        self.expect(PhiToken::Hardware)?;
        
        let device_name = self.expect_identifier()?;
        
        self.expect(PhiToken::LeftBrace)?;
        
        // For now, simplified hardware parsing
        let mut device_type = String::new();
        
        while self.current_token != PhiToken::RightBrace {
            // Skip content for now - would parse device mapping
            self.advance();
        }
        
        self.expect(PhiToken::RightBrace)?;
        
        Ok(PhiExpression::HardwareSync {
            device_type: device_name,
            consciousness_mapping: Box::new(PhiExpression::Number(0.0)), // Placeholder
        })
    }

    /// Parse emergency protocol: emergency_protocol NAME { trigger: ..., immediate { ... }, notify: [...] }
    fn parse_emergency_protocol(&mut self) -> Result<PhiExpression, String> {
        self.expect(PhiToken::Emergency)?;
        
        // Skip any newlines
        while self.current_token == PhiToken::Newline {
            self.advance();
        }
        
        self.expect(PhiToken::Protocol)?;
        
        // Skip any newlines
        while self.current_token == PhiToken::Newline {
            self.advance();
        }
        
        let protocol_name = self.expect_identifier()?;
        
        self.expect(PhiToken::LeftBrace)?;
        
        let mut trigger = Box::new(PhiExpression::Boolean(false));
        let mut immediate_action = Box::new(PhiExpression::Block(vec![]));
        let mut notifications = vec![];
        
        while self.current_token != PhiToken::RightBrace {
            // Skip any newlines
            while self.current_token == PhiToken::Newline {
                self.advance();
            }
            
            // Check for closing brace after skipping newlines
            if self.current_token == PhiToken::RightBrace {
                break;
            }
            
            let prop = self.expect_identifier()?;
            
            match prop.as_str() {
                "trigger" => {
                    self.expect(PhiToken::Colon)?;
                    trigger = Box::new(self.parse_expression()?);
                }
                "immediate" => {
                    // No colon for immediate - it directly has a block
                    self.expect(PhiToken::LeftBrace)?;
                    let mut actions = vec![];
                    
                    // Skip any newlines after opening brace
                    while self.current_token == PhiToken::Newline {
                        self.advance();
                    }
                    
                    while self.current_token != PhiToken::RightBrace {
                        actions.push(self.parse_statement()?);
                        
                        // Skip any newlines between statements
                        while self.current_token == PhiToken::Newline {
                            self.advance();
                        }
                    }
                    self.expect(PhiToken::RightBrace)?;
                    immediate_action = Box::new(PhiExpression::Block(actions));
                }
                "notify" => {
                    self.expect(PhiToken::Colon)?;
                    self.expect(PhiToken::LeftBracket)?;
                    while self.current_token != PhiToken::RightBracket {
                        notifications.push(self.expect_string()?);
                        if self.current_token == PhiToken::Comma {
                            self.advance();
                        }
                    }
                    self.expect(PhiToken::RightBracket)?;
                }
                _ => return Err(format!("Unknown emergency protocol property: {}", prop))
            }
            
            if self.current_token == PhiToken::Comma {
                self.advance();
            }
            
            // Skip any trailing newlines
            while self.current_token == PhiToken::Newline {
                self.advance();
            }
        }
        
        self.expect(PhiToken::RightBrace)?;
        
        Ok(PhiExpression::EmergencyProtocol {
            trigger,
            immediate_action,
            notification: notifications,
        })
    }

    /// Parse consciousness flow: consciousness_flow { gradient consciousness.level { ... } }
    fn parse_consciousness_flow(&mut self) -> Result<PhiExpression, String> {
        self.expect(PhiToken::ConsciousnessFlow)?;
        self.expect(PhiToken::LeftBrace)?;
        
        // Skip any newlines
        while self.current_token == PhiToken::Newline {
            self.advance();
        }
        
        // For now, simplified parsing - skip the gradient block
        if self.current_token == PhiToken::Gradient {
            self.advance(); // Skip 'gradient'
            
            // Skip the condition expression (consciousness.level)
            while self.current_token != PhiToken::LeftBrace && self.current_token != PhiToken::RightBrace {
                self.advance();
            }
            
            // If we have a left brace, skip the gradient block
            if self.current_token == PhiToken::LeftBrace {
                self.advance(); // Skip '{'
                let mut brace_count = 1;
                
                // Skip until we find the matching right brace
                while brace_count > 0 && self.current_token != PhiToken::Eof {
                    if self.current_token == PhiToken::LeftBrace {
                        brace_count += 1;
                    } else if self.current_token == PhiToken::RightBrace {
                        brace_count -= 1;
                    }
                    self.advance();
                }
            }
        }
        
        // Skip any trailing newlines
        while self.current_token == PhiToken::Newline {
            self.advance();
        }
        
        self.expect(PhiToken::RightBrace)?;
        
        // For now, return a simplified structure
        let condition = Box::new(PhiExpression::Variable("consciousness.level".to_string()));
        let branches = vec![];
        
        Ok(PhiExpression::ConsciousnessFlow {
            condition,
            branches,
        })
    }

    /// Parse biological program: biological_program NAME { target: ..., frequency: ..., sequence ... }
    fn parse_biological_program(&mut self) -> Result<PhiExpression, String> {
        self.expect(PhiToken::BiologicalProgram)?;
        
        let program_name = self.expect_identifier()?;
        
        self.expect(PhiToken::LeftBrace)?;
        
        // For now, simplified parsing
        let target = "human_biology".to_string();
        let frequency = 528.0;
        
        while self.current_token != PhiToken::RightBrace {
            self.advance(); // Skip content for now
        }
        
        self.expect(PhiToken::RightBrace)?;
        
        Ok(PhiExpression::BiologicalInterface {
            target,
            transduction_method: "phi_harmonic".to_string(),
            frequency,
        })
    }

    /// Parse quantum bridge: quantum_bridge NAME { source: ..., target: ..., ... }
    fn parse_quantum_bridge(&mut self) -> Result<PhiExpression, String> {
        self.expect(PhiToken::QuantumBridge)?;
        
        let bridge_name = self.expect_identifier()?;
        
        self.expect(PhiToken::LeftBrace)?;
        
        // For now, simplified parsing
        let field_type = "consciousness_bridge".to_string();
        let dimensions = vec![3, 4, 5, 6, 7];
        let coherence_target = 1.0;
        
        while self.current_token != PhiToken::RightBrace {
            self.advance(); // Skip content for now
        }
        
        self.expect(PhiToken::RightBrace)?;
        
        Ok(PhiExpression::QuantumField {
            field_type,
            dimensions,
            coherence_target,
        })
    }

    /// Helper to expect a string literal
    fn expect_string(&mut self) -> Result<String, String> {
        match &self.current_token {
            PhiToken::String(s) => {
                let value = s.clone();
                self.advance();
                Ok(value)
            }
            _ => Err(format!("Expected string literal, found {:?}", self.current_token))
        }
    }

    

    /// Parse frequency pattern: frequency_pattern { base_frequency: ..., harmonics: [...], phi_scaling: ... }
    fn parse_frequency_pattern(&mut self) -> Result<PhiExpression, String> {
        self.expect(PhiToken::Frequency)?;
        self.expect(PhiToken::LeftBrace)?;

        let mut base_frequency = 0.0;
        let mut harmonics = Vec::new();
        let mut phi_scaling = false;

        while self.current_token != PhiToken::RightBrace {
            while self.current_token == PhiToken::Newline {
                self.advance();
            }
            if self.current_token == PhiToken::RightBrace {
                break;
            }

            let prop_name = self.expect_identifier()?;
            self.expect(PhiToken::Colon)?;

            match prop_name.as_str() {
                "base_frequency" => {
                    base_frequency = self.expect_number()?;
                }
                "harmonics" => {
                    self.expect(PhiToken::LeftBracket)?;
                    while self.current_token != PhiToken::RightBracket {
                        harmonics.push(self.expect_number()?);
                        if self.current_token == PhiToken::Comma {
                            self.advance();
                        }
                    }
                    self.expect(PhiToken::RightBracket)?;
                }
                "phi_scaling" => {
                    phi_scaling = match &self.current_token {
                        PhiToken::Boolean(b) => {
                            let value = *b;
                            self.advance();
                            value
                        }
                        _ => return Err(format!("Expected boolean literal, found {:?}", self.current_token))
                    };
                }
                _ => return Err(format!("Unknown frequency pattern property: {}", prop_name))
            }

            if self.current_token == PhiToken::Comma {
                self.advance();
            }
            while self.current_token == PhiToken::Newline {
                self.advance();
            }
        }
        self.expect(PhiToken::RightBrace)?;

        Ok(PhiExpression::FrequencyPattern {
            base_frequency,
            harmonics,
            phi_scaling,
        })
    }


    











    








    



}

// PhiFlow program parsing entry point
pub fn parse_phi_program(source: &str) -> Result<Vec<PhiExpression>, String> {
    let mut lexer = PhiLexer::new(source);
    let tokens = lexer.tokenize()?;
    let mut parser = PhiParser::new(tokens);
    parser.parse()
}
