// PhiFlow Lexer - Tokenizes PhiFlow quantum-consciousness DSL
// Supports sacred frequencies, quantum operations, and consciousness integration

use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Quantum Keywords
    Qubit,
    Gate,
    Circuit,
    Measure,
    Execute,

    // Sacred Frequency Operations
    Sacred(u32), // Sacred(432), Sacred(528), etc.
    Frequency,
    Hz,
    Lock,
    Resonate,

    // Phi/Golden Ratio Operations
    Phi,
    Golden,
    Spiral,
    Fibonacci,

    // Consciousness Integration
    Consciousness,
    Monitor,
    State,
    Coherence,
    Clarity,
    Flow,
    Witness,
    Intention,
    Stream,
    Break,

    // Quantum Gates
    Hadamard, // H
    PauliX,   // X
    PauliY,   // Y
    PauliZ,   // Z
    CNOT,     // CNOT
    Rotation, // RX, RY, RZ

    // Standard Language Constructs
    Let,
    Fn,
    If,
    Else,
    Match,
    For,
    While,
    In,
    Return,

    // Data Types
    Number(f64),
    String(String),
    Identifier(String),
    Boolean(bool),

    // Operators
    Plus,         // +
    Minus,        // -
    Star,         // *
    Slash,        // /
    Equal,        // =
    EqualEqual,   // ==
    NotEqual,     // !=
    Greater,      // >
    Less,         // <
    GreaterEqual, // >=
    LessEqual,    // <=
    And,          // &&
    Or,           // ||
    Not,          // !

    // Punctuation
    LeftParen,    // (
    RightParen,   // )
    LeftBrace,    // {
    RightBrace,   // }
    LeftBracket,  // [
    RightBracket, // ]
    Comma,        // ,
    Semicolon,    // ;
    Colon,        // :
    DoubleColon,  // ::
    Arrow,        // ->
    Pipe,         // |>
    Dot,          // .

    // Special
    Newline,
    EOF,
    Error(String),
}

pub struct PhiFlowLexer {
    input: String,
    position: usize,
    current_char: Option<char>,
    keywords: HashMap<String, Token>,
}

impl PhiFlowLexer {
    pub fn new(input: String) -> Self {
        let mut lexer = PhiFlowLexer {
            input: input.clone(),
            position: 0,
            current_char: input.chars().next(),
            keywords: HashMap::new(),
        };

        lexer.init_keywords();
        lexer
    }

    fn init_keywords(&mut self) {
        let keywords = vec![
            // Quantum Keywords
            ("qubit", Token::Qubit),
            ("gate", Token::Gate),
            ("circuit", Token::Circuit),
            ("measure", Token::Measure),
            ("execute", Token::Execute),
            // Sacred Frequencies
            ("frequency", Token::Frequency),
            ("Hz", Token::Hz),
            ("lock", Token::Lock),
            ("resonate", Token::Resonate),
            // Phi Operations
            ("phi", Token::Phi),
            ("PHI", Token::Phi),
            ("golden", Token::Golden),
            ("spiral", Token::Spiral),
            ("fibonacci", Token::Fibonacci),
            // Consciousness
            ("consciousness", Token::Consciousness),
            ("monitor", Token::Monitor),
            ("state", Token::State),
            ("coherence", Token::Coherence),
            ("clarity", Token::Clarity),
            ("flow", Token::Flow),
            ("witness", Token::Witness),
            ("intention", Token::Intention),
            ("stream", Token::Stream),
            ("break", Token::Break),
            // Quantum Gates
            ("H", Token::Hadamard),
            ("X", Token::PauliX),
            ("Y", Token::PauliY),
            ("Z", Token::PauliZ),
            ("CNOT", Token::CNOT),
            ("RX", Token::Rotation),
            ("RY", Token::Rotation),
            ("RZ", Token::Rotation),
            // Standard Keywords
            ("let", Token::Let),
            ("fn", Token::Fn),
            ("if", Token::If),
            ("else", Token::Else),
            ("match", Token::Match),
            ("for", Token::For),
            ("while", Token::While),
            ("in", Token::In),
            ("return", Token::Return),
            ("true", Token::Boolean(true)),
            ("false", Token::Boolean(false)),
        ];

        for (keyword, token) in keywords {
            self.keywords.insert(keyword.to_string(), token);
        }
    }

    pub fn tokenize(&mut self) -> Result<Vec<Token>, String> {
        let mut tokens = Vec::new();

        while let Some(token) = self.next_token()? {
            if token != Token::EOF {
                tokens.push(token);
            } else {
                break;
            }
        }

        tokens.push(Token::EOF);
        Ok(tokens)
    }

    fn next_token(&mut self) -> Result<Option<Token>, String> {
        self.skip_whitespace();

        match self.current_char {
            None => Ok(Some(Token::EOF)),
            Some('\n') => {
                self.advance();
                Ok(Some(Token::Newline))
            }
            Some('+') => {
                self.advance();
                Ok(Some(Token::Plus))
            }
            Some('-') => {
                self.advance();
                if self.current_char == Some('>') {
                    self.advance();
                    Ok(Some(Token::Arrow))
                } else {
                    Ok(Some(Token::Minus))
                }
            }
            Some('*') => {
                self.advance();
                Ok(Some(Token::Star))
            }
            Some('/') => {
                self.advance();
                if self.current_char == Some('/') {
                    // Line comment
                    self.skip_line_comment();
                    self.next_token()
                } else if self.current_char == Some('*') {
                    // Block comment
                    self.skip_block_comment()?;
                    self.next_token()
                } else {
                    Ok(Some(Token::Slash))
                }
            }
            Some('=') => {
                self.advance();
                if self.current_char == Some('=') {
                    self.advance();
                    Ok(Some(Token::EqualEqual))
                } else {
                    Ok(Some(Token::Equal))
                }
            }
            Some('!') => {
                self.advance();
                if self.current_char == Some('=') {
                    self.advance();
                    Ok(Some(Token::NotEqual))
                } else {
                    Ok(Some(Token::Not))
                }
            }
            Some('>') => {
                self.advance();
                if self.current_char == Some('=') {
                    self.advance();
                    Ok(Some(Token::GreaterEqual))
                } else {
                    Ok(Some(Token::Greater))
                }
            }
            Some('<') => {
                self.advance();
                if self.current_char == Some('=') {
                    self.advance();
                    Ok(Some(Token::LessEqual))
                } else {
                    Ok(Some(Token::Less))
                }
            }
            Some('&') => {
                self.advance();
                if self.current_char == Some('&') {
                    self.advance();
                    Ok(Some(Token::And))
                } else {
                    Err("Unexpected character '&'".to_string())
                }
            }
            Some('|') => {
                self.advance();
                if self.current_char == Some('|') {
                    self.advance();
                    Ok(Some(Token::Or))
                } else if self.current_char == Some('>') {
                    self.advance();
                    Ok(Some(Token::Pipe))
                } else {
                    Err("Unexpected character '|'".to_string())
                }
            }
            Some('(') => {
                self.advance();
                Ok(Some(Token::LeftParen))
            }
            Some(')') => {
                self.advance();
                Ok(Some(Token::RightParen))
            }
            Some('{') => {
                self.advance();
                Ok(Some(Token::LeftBrace))
            }
            Some('}') => {
                self.advance();
                Ok(Some(Token::RightBrace))
            }
            Some('[') => {
                self.advance();
                Ok(Some(Token::LeftBracket))
            }
            Some(']') => {
                self.advance();
                Ok(Some(Token::RightBracket))
            }
            Some(',') => {
                self.advance();
                Ok(Some(Token::Comma))
            }
            Some(';') => {
                self.advance();
                Ok(Some(Token::Semicolon))
            }
            Some(':') => {
                self.advance();
                if self.current_char == Some(':') {
                    self.advance();
                    Ok(Some(Token::DoubleColon))
                } else {
                    Ok(Some(Token::Colon))
                }
            }
            Some('.') => {
                self.advance();
                Ok(Some(Token::Dot))
            }
            Some('"') => self.read_string(),
            Some(c) if c.is_digit(10) => self.read_number(),
            Some(c) if c.is_alphabetic() || c == '_' => self.read_identifier(),
            Some(c) => Err(format!("Unexpected character: '{}'", c)),
        }
    }

    fn advance(&mut self) {
        self.position += 1;
        if self.position >= self.input.len() {
            self.current_char = None;
        } else {
            self.current_char = self.input.chars().nth(self.position);
        }
    }

    fn skip_whitespace(&mut self) {
        while let Some(c) = self.current_char {
            if c.is_whitespace() && c != '\n' {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn skip_line_comment(&mut self) {
        while let Some(c) = self.current_char {
            if c == '\n' {
                break;
            }
            self.advance();
        }
    }

    fn skip_block_comment(&mut self) -> Result<(), String> {
        self.advance(); // Skip '*'

        while let Some(c) = self.current_char {
            if c == '*' {
                self.advance();
                if self.current_char == Some('/') {
                    self.advance();
                    return Ok(());
                }
            } else {
                self.advance();
            }
        }

        Err("Unterminated block comment".to_string())
    }

    fn read_string(&mut self) -> Result<Option<Token>, String> {
        self.advance(); // Skip opening quote
        let mut value = String::new();

        while let Some(c) = self.current_char {
            if c == '"' {
                self.advance(); // Skip closing quote
                return Ok(Some(Token::String(value)));
            }
            if c == '\\' {
                self.advance();
                match self.current_char {
                    Some('n') => value.push('\n'),
                    Some('t') => value.push('\t'),
                    Some('r') => value.push('\r'),
                    Some('\\') => value.push('\\'),
                    Some('"') => value.push('"'),
                    Some(c) => return Err(format!("Invalid escape sequence: \\{}", c)),
                    None => return Err("Unexpected end of string".to_string()),
                }
                self.advance();
            } else {
                value.push(c);
                self.advance();
            }
        }

        Err("Unterminated string".to_string())
    }

    fn read_number(&mut self) -> Result<Option<Token>, String> {
        let mut value = String::new();

        while let Some(c) = self.current_char {
            if c.is_digit(10) || c == '.' {
                value.push(c);
                self.advance();
            } else {
                break;
            }
        }

        match value.parse::<f64>() {
            Ok(num) => Ok(Some(Token::Number(num))),
            Err(_) => Err(format!("Invalid number: {}", value)),
        }
    }

    fn read_identifier(&mut self) -> Result<Option<Token>, String> {
        let mut value = String::new();

        while let Some(c) = self.current_char {
            if c.is_alphanumeric() || c == '_' {
                value.push(c);
                self.advance();
            } else {
                break;
            }
        }

        // Check for Sacred frequency notation: Sacred(432)
        if value == "Sacred" && self.current_char == Some('(') {
            self.advance(); // Skip '('
            if let Ok(Some(Token::Number(freq))) = self.read_number() {
                if self.current_char == Some(')') {
                    self.advance(); // Skip ')'
                    return Ok(Some(Token::Sacred(freq as u32)));
                }
            }
            return Err("Invalid Sacred frequency syntax".to_string());
        }

        // Check if it's a keyword
        if let Some(token) = self.keywords.get(&value).cloned() {
            Ok(Some(token))
        } else {
            Ok(Some(Token::Identifier(value)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokens() {
        let mut lexer = PhiFlowLexer::new("let x = 42;".to_string());
        let tokens = lexer.tokenize().unwrap();

        assert_eq!(tokens[0], Token::Let);
        assert_eq!(tokens[1], Token::Identifier("x".to_string()));
        assert_eq!(tokens[2], Token::Equal);
        assert_eq!(tokens[3], Token::Number(42.0));
        assert_eq!(tokens[4], Token::Semicolon);
    }

    #[test]
    fn test_sacred_frequency() {
        let mut lexer = PhiFlowLexer::new("Sacred(432) Sacred(528)".to_string());
        let tokens = lexer.tokenize().unwrap();

        assert_eq!(tokens[0], Token::Sacred(432));
        assert_eq!(tokens[1], Token::Sacred(528));
    }

    #[test]
    fn test_quantum_gates() {
        let mut lexer = PhiFlowLexer::new("gate H(qubit) CNOT(q1, q2)".to_string());
        let tokens = lexer.tokenize().unwrap();

        assert_eq!(tokens[0], Token::Gate);
        assert_eq!(tokens[1], Token::Hadamard);
        assert_eq!(tokens[2], Token::LeftParen);
        assert_eq!(tokens[3], Token::Qubit);
        assert_eq!(tokens[4], Token::RightParen);
        assert_eq!(tokens[5], Token::CNOT);
    }

    #[test]
    fn test_consciousness_tokens() {
        let mut lexer = PhiFlowLexer::new("consciousness.coherence > 0.9".to_string());
        let tokens = lexer.tokenize().unwrap();

        assert_eq!(tokens[0], Token::Consciousness);
        assert_eq!(tokens[1], Token::Dot);
        assert_eq!(tokens[2], Token::Coherence);
    }

    #[test]
    fn test_block_comments() {
        let mut lexer = PhiFlowLexer::new("let x = /* comment */ 42;".to_string());
        let tokens = lexer.tokenize().unwrap();

        assert_eq!(tokens[0], Token::Let);
        assert_eq!(tokens[1], Token::Identifier("x".to_string()));
        assert_eq!(tokens[2], Token::Equal);
        assert_eq!(tokens[3], Token::Number(42.0));
        assert_eq!(tokens[4], Token::Semicolon);
    }
}
