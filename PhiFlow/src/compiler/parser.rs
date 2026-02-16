// PhiFlow Parser - Converts tokens to Abstract Syntax Tree
// Implements recursive descent parser for PhiFlow quantum-consciousness DSL

use crate::compiler::lexer::{Token, PhiFlowLexer};
use crate::compiler::ast::{
    PhiFlowExpression, PhiFlowProgram, QuantumGate, QuantumGateType,
    BinaryOperator, UnaryOperator, ComparisonOperator, LogicalOperator,
    ConsciousnessCondition, ConsciousnessMetric, BrainwaveType,
    PhiFlowType, Parameter, MatchArm, Pattern,
    ConsciousnessConfig, QuantumConfig
};
use std::collections::HashMap;

pub struct PhiFlowParser {
    tokens: Vec<Token>,
    current: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("Unexpected token: {token:?} at position {position}")]
    UnexpectedToken { token: Token, position: usize },
    
    #[error("Expected token {expected:?}, found {found:?} at position {position}")]
    ExpectedToken { expected: String, found: Token, position: usize },
    
    #[error("Unexpected end of input")]
    UnexpectedEOF,
    
    #[error("Invalid quantum gate: {gate}")]
    InvalidQuantumGate { gate: String },
    
    #[error("Invalid sacred frequency: {frequency}")]
    InvalidSacredFrequency { frequency: u32 },
    
    #[error("Invalid consciousness metric: {metric}")]
    InvalidConsciousnessMetric { metric: String },
    
    #[error("Syntax error: {message}")]
    SyntaxError { message: String },
}

type ParseResult<T> = Result<T, ParseError>;

impl PhiFlowParser {
    pub fn new(tokens: Vec<Token>) -> Self {
        PhiFlowParser {
            tokens,
            current: 0,
        }
    }
    
    pub fn parse(&mut self) -> ParseResult<PhiFlowProgram> {
        let mut functions = Vec::new();
        let mut main = None;
        let imports = Vec::new();
        let consciousness_config = None;
        let quantum_config = None;
        
        // Collect all top-level statements/expressions
        let mut main_statements = Vec::new();
        
        while !self.is_at_end() {
            // Skip newlines
            if self.check(&Token::Newline) {
                self.advance();
                continue;
            }
            
            match self.peek() {
                Token::Fn => {
                    let function = self.parse_function()?;
                    if let PhiFlowExpression::FunctionDefinition { name, .. } = &function {
                        if name == "main" {
                            main = Some(function);
                        } else {
                            functions.push(function);
                        }
                    }
                }
                _ => {
                    // Parse top-level statement/expression
                    let expr = self.parse_expression()?;
                    main_statements.push(expr);
                    
                    // Optional semicolon at top level
                    self.match_token(&Token::Semicolon);
                }
            }
        }
        
        // If we have multiple statements, wrap them in a block
        // If we have one statement, use it directly
        if !main_statements.is_empty() && main.is_none() {
            main = if main_statements.len() == 1 {
                Some(main_statements.into_iter().next().unwrap())
            } else {
                Some(PhiFlowExpression::Block(main_statements))
            };
        }
        
        Ok(PhiFlowProgram {
            functions,
            main,
            imports,
            consciousness_config,
            quantum_config,
        })
    }
    
    fn parse_expression(&mut self) -> ParseResult<PhiFlowExpression> {
        self.parse_assignment()
    }
    
    fn parse_assignment(&mut self) -> ParseResult<PhiFlowExpression> {
        if self.match_token(&Token::Let) {
            self.parse_let_binding()
        } else {
            self.parse_logical_or()
        }
    }
    
    fn parse_let_binding(&mut self) -> ParseResult<PhiFlowExpression> {
        let variable = self.consume_identifier("Expected variable name")?;
        
        let type_annotation = if self.match_token(&Token::Colon) {
            Some(self.parse_type()?)
        } else {
            None
        };
        
        self.consume(&Token::Equal, "Expected '=' after variable name")?;
        let value = Box::new(self.parse_expression()?);
        
        Ok(PhiFlowExpression::Let {
            variable,
            type_annotation,
            value,
        })
    }
    
    fn parse_logical_or(&mut self) -> ParseResult<PhiFlowExpression> {
        let mut expr = self.parse_logical_and()?;
        
        while self.match_token(&Token::Or) {
            let operator = BinaryOperator::Or;
            let right = Box::new(self.parse_logical_and()?);
            expr = PhiFlowExpression::BinaryOp {
                left: Box::new(expr),
                operator,
                right,
            };
        }
        
        Ok(expr)
    }
    
    fn parse_logical_and(&mut self) -> ParseResult<PhiFlowExpression> {
        let mut expr = self.parse_equality()?;
        
        while self.match_token(&Token::And) {
            let operator = BinaryOperator::And;
            let right = Box::new(self.parse_equality()?);
            expr = PhiFlowExpression::BinaryOp {
                left: Box::new(expr),
                operator,
                right,
            };
        }
        
        Ok(expr)
    }
    
    fn parse_equality(&mut self) -> ParseResult<PhiFlowExpression> {
        let mut expr = self.parse_comparison()?;
        
        while let Some(operator) = self.match_equality_operator() {
            let right = Box::new(self.parse_comparison()?);
            expr = PhiFlowExpression::BinaryOp {
                left: Box::new(expr),
                operator,
                right,
            };
        }
        
        Ok(expr)
    }
    
    fn match_equality_operator(&mut self) -> Option<BinaryOperator> {
        if self.match_token(&Token::EqualEqual) {
            Some(BinaryOperator::Equal)
        } else if self.match_token(&Token::NotEqual) {
            Some(BinaryOperator::NotEqual)
        } else {
            None
        }
    }
    
    fn parse_comparison(&mut self) -> ParseResult<PhiFlowExpression> {
        let mut expr = self.parse_term()?;
        
        while let Some(operator) = self.match_comparison_operator() {
            let right = Box::new(self.parse_term()?);
            expr = PhiFlowExpression::BinaryOp {
                left: Box::new(expr),
                operator,
                right,
            };
        }
        
        Ok(expr)
    }
    
    fn match_comparison_operator(&mut self) -> Option<BinaryOperator> {
        if self.match_token(&Token::Greater) {
            Some(BinaryOperator::Greater)
        } else if self.match_token(&Token::GreaterEqual) {
            Some(BinaryOperator::GreaterEqual)
        } else if self.match_token(&Token::Less) {
            Some(BinaryOperator::Less)
        } else if self.match_token(&Token::LessEqual) {
            Some(BinaryOperator::LessEqual)
        } else {
            None
        }
    }
    
    fn parse_term(&mut self) -> ParseResult<PhiFlowExpression> {
        let mut expr = self.parse_factor()?;
        
        while let Some(operator) = self.match_term_operator() {
            let right = Box::new(self.parse_factor()?);
            expr = PhiFlowExpression::BinaryOp {
                left: Box::new(expr),
                operator,
                right,
            };
        }
        
        Ok(expr)
    }
    
    fn match_term_operator(&mut self) -> Option<BinaryOperator> {
        if self.match_token(&Token::Plus) {
            Some(BinaryOperator::Add)
        } else if self.match_token(&Token::Minus) {
            Some(BinaryOperator::Subtract)
        } else {
            None
        }
    }
    
    fn parse_factor(&mut self) -> ParseResult<PhiFlowExpression> {
        let mut expr = self.parse_unary()?;
        
        while let Some(operator) = self.match_factor_operator() {
            let right = Box::new(self.parse_unary()?);
            expr = PhiFlowExpression::BinaryOp {
                left: Box::new(expr),
                operator,
                right,
            };
        }
        
        Ok(expr)
    }
    
    fn match_factor_operator(&mut self) -> Option<BinaryOperator> {
        if self.match_token(&Token::Star) {
            Some(BinaryOperator::Multiply)
        } else if self.match_token(&Token::Slash) {
            Some(BinaryOperator::Divide)
        } else {
            None
        }
    }
    
    fn parse_unary(&mut self) -> ParseResult<PhiFlowExpression> {
        if let Some(operator) = self.match_unary_operator() {
            let operand = Box::new(self.parse_unary()?);
            Ok(PhiFlowExpression::UnaryOp { operator, operand })
        } else {
            self.parse_call()
        }
    }
    
    fn match_unary_operator(&mut self) -> Option<UnaryOperator> {
        if self.match_token(&Token::Minus) {
            Some(UnaryOperator::Negate)
        } else if self.match_token(&Token::Not) {
            Some(UnaryOperator::Not)
        } else {
            None
        }
    }
    
    fn parse_call(&mut self) -> ParseResult<PhiFlowExpression> {
        let mut expr = self.parse_primary()?;
        
        loop {
            if self.check(&Token::LeftParen) {
                expr = self.finish_call(expr)?;
            } else if self.check(&Token::LeftBracket) {
                expr = self.finish_array_index(expr)?;
            } else {
                break;
            }
        }
        
        Ok(expr)
    }
    
    fn finish_call(&mut self, callee: PhiFlowExpression) -> ParseResult<PhiFlowExpression> {
        self.consume(&Token::LeftParen, "Expected '('")?;
        
        let mut args = Vec::new();
        if !self.check(&Token::RightParen) {
            loop {
                args.push(self.parse_expression()?);
                if !self.match_token(&Token::Comma) {
                    break;
                }
            }
        }
        
        self.consume(&Token::RightParen, "Expected ')' after arguments")?;
        
        if let PhiFlowExpression::Variable(name) = callee {
            Ok(PhiFlowExpression::FunctionCall { name, args })
        } else {
            Err(ParseError::SyntaxError {
                message: "Invalid function call".to_string(),
            })
        }
    }
    
    fn finish_array_index(&mut self, array: PhiFlowExpression) -> ParseResult<PhiFlowExpression> {
        self.consume(&Token::LeftBracket, "Expected '['")?;
        let index = Box::new(self.parse_expression()?);
        self.consume(&Token::RightBracket, "Expected ']' after array index")?;
        
        Ok(PhiFlowExpression::ArrayIndex {
            array: Box::new(array),
            index,
        })
    }
    
    fn parse_primary(&mut self) -> ParseResult<PhiFlowExpression> {
        match self.peek() {
            Token::Number(n) => {
                let value = *n;
                self.advance();
                Ok(PhiFlowExpression::Number(value))
            }
            Token::String(s) => {
                let value = s.clone();
                self.advance();
                Ok(PhiFlowExpression::String(value))
            }
            Token::Boolean(b) => {
                let value = *b;
                self.advance();
                Ok(PhiFlowExpression::Boolean(value))
            }
            Token::Sacred(freq) => {
                let frequency = *freq;
                self.advance();
                let operation = if self.check(&Token::LeftBrace) {
                    Box::new(self.parse_block()?)
                } else if self.check(&Token::LeftParen) {
                    self.advance();
                    let expr = Box::new(self.parse_expression()?);
                    self.consume(&Token::RightParen, "Expected ')' after sacred operation")?;
                    expr
                } else {
                    return Err(ParseError::ExpectedToken {
                        expected: "Expected '{' or '(' after sacred frequency".to_string(),
                        found: self.peek().clone(),
                        position: self.current,
                    });
                };
                Ok(PhiFlowExpression::SacredFrequency { frequency, operation })
            }
            Token::Resonate => {
                self.advance();
                Ok(PhiFlowExpression::Variable("resonate".to_string()))
            }
            Token::Witness => {
                self.advance();
                if self.check(&Token::LeftParen) {
                    self.parse_witness_expression()
                } else {
                    Ok(PhiFlowExpression::Variable("witness".to_string()))
                }
            }
            Token::Intention => {
                self.advance();
                if self.check(&Token::LeftParen) {
                    self.parse_intention_expression()
                } else {
                    Ok(PhiFlowExpression::Variable("intention".to_string()))
                }
            }
            Token::Consciousness => {
                self.advance();
                if self.check(&Token::Dot) {
                    self.parse_consciousness_expression()
                } else {
                    Ok(PhiFlowExpression::Variable("consciousness".to_string()))
                }
            }
            Token::Qubit => {
                self.advance();
                self.parse_quantum_expression()
            }
            Token::Gate => {
                self.advance();
                self.parse_quantum_gate()
            }
            Token::Circuit => {
                self.advance();
                self.parse_quantum_circuit()
            }
            Token::Identifier(name) => {
                let var_name = name.clone();
                self.advance();
                Ok(PhiFlowExpression::Variable(var_name))
            }
            Token::Phi => {
                self.advance();
                Ok(PhiFlowExpression::Variable("PHI".to_string()))
            }
            Token::LeftParen => {
                self.advance();
                let expr = self.parse_expression()?;
                self.consume(&Token::RightParen, "Expected ')' after expression")?;
                Ok(expr)
            }
            Token::LeftBrace => {
                self.parse_block()
            }
            Token::If => {
                self.parse_if_expression()
            }
            Token::For => {
                self.parse_for_expression()
            }
            Token::While => {
                self.parse_while_expression()
            }
            Token::LeftBracket => {
                self.parse_array_literal()
            }
            _ => Err(ParseError::UnexpectedToken {
                token: self.peek().clone(),
                position: self.current,
            }),
        }
    }
    
    fn parse_consciousness_expression(&mut self) -> ParseResult<PhiFlowExpression> {
        if self.match_token(&Token::Monitor) {
            self.parse_consciousness_monitor()
        } else {
            // Simple consciousness binding
            let state_name = self.consume_identifier("Expected consciousness state name")?;
            self.consume(&Token::LeftBrace, "Expected '{' after consciousness state")?;
            let expression = Box::new(self.parse_expression()?);
            self.consume(&Token::RightBrace, "Expected '}' after consciousness expression")?;
            
            Ok(PhiFlowExpression::ConsciousnessBinding {
                state_name,
                expression,
            })
        }
    }
    
    fn parse_consciousness_monitor(&mut self) -> ParseResult<PhiFlowExpression> {
        self.consume(&Token::LeftParen, "Expected '(' after monitor")?;
        
        let mut metrics = Vec::new();
        if !self.check(&Token::RightParen) {
            loop {
                let metric = self.parse_consciousness_metric()?;
                metrics.push(metric);
                if !self.match_token(&Token::Comma) {
                    break;
                }
            }
        }
        
        self.consume(&Token::RightParen, "Expected ')' after metrics")?;
        self.consume(&Token::LeftBrace, "Expected '{' after monitor parameters")?;
        let callback = Box::new(self.parse_expression()?);
        self.consume(&Token::RightBrace, "Expected '}' after monitor callback")?;
        
        Ok(PhiFlowExpression::ConsciousnessMonitor { metrics, callback })
    }

    fn parse_witness_expression(&mut self) -> ParseResult<PhiFlowExpression> {
        self.consume(&Token::LeftParen, "Expected '(' after witness")?;
        let expression = Box::new(self.parse_expression()?);
        self.consume(&Token::RightParen, "Expected ')' after witnessed expression")?;
        
        Ok(PhiFlowExpression::Witness(expression))
    }

    fn parse_intention_expression(&mut self) -> ParseResult<PhiFlowExpression> {
        self.consume(&Token::LeftParen, "Expected '(' after intention")?;
        
        let content = if let Token::String(s) = self.peek() {
            let s_val = s.clone();
            self.advance();
            s_val
        } else {
            return Err(ParseError::ExpectedToken {
                expected: "intention string".to_string(),
                found: self.peek().clone(),
                position: self.current,
            });
        };

        self.consume(&Token::Comma, "Expected ',' after intention content")?;
        let target = Box::new(self.parse_expression()?);
        self.consume(&Token::RightParen, "Expected ')' after intention target")?;
        
        Ok(PhiFlowExpression::Intention { content, target })
    }
    
    fn parse_consciousness_metric(&mut self) -> ParseResult<ConsciousnessMetric> {
        let identifier = self.consume_identifier("Expected consciousness metric")?;
        
        match identifier.as_str() {
            "coherence" => Ok(ConsciousnessMetric::Coherence),
            "clarity" => Ok(ConsciousnessMetric::Clarity),
            "flow" => Ok(ConsciousnessMetric::FlowState),
            "phi" => Ok(ConsciousnessMetric::PhiResonance),
            _ => Err(ParseError::InvalidConsciousnessMetric { metric: identifier }),
        }
    }
    
    fn parse_quantum_expression(&mut self) -> ParseResult<PhiFlowExpression> {
        let qubit_name = self.consume_identifier("Expected qubit name")?;
        Ok(PhiFlowExpression::Variable(qubit_name))
    }
    
    fn parse_quantum_gate(&mut self) -> ParseResult<PhiFlowExpression> {
        let gate_type = self.parse_gate_type()?;
        
        self.consume(&Token::LeftParen, "Expected '(' after gate")?;
        let mut qubits = Vec::new();
        let mut parameters = Vec::new();
        
        if !self.check(&Token::RightParen) {
            loop {
                if let Token::Identifier(qubit) = self.peek() {
                    qubits.push(qubit.clone());
                    self.advance();
                } else if let Token::Number(param) = self.peek() {
                    parameters.push(*param);
                    self.advance();
                } else {
                    return Err(ParseError::ExpectedToken {
                        expected: "qubit or parameter".to_string(),
                        found: self.peek().clone(),
                        position: self.current,
                    });
                }
                
                if !self.match_token(&Token::Comma) {
                    break;
                }
            }
        }
        
        self.consume(&Token::RightParen, "Expected ')' after gate parameters")?;
        
        Ok(PhiFlowExpression::QuantumGate {
            gate_type,
            qubits,
            parameters,
        })
    }
    
    fn parse_gate_type(&mut self) -> ParseResult<QuantumGateType> {
        match self.peek() {
            Token::Hadamard => {
                self.advance();
                Ok(QuantumGateType::Hadamard)
            }
            Token::PauliX => {
                self.advance();
                Ok(QuantumGateType::PauliX)
            }
            Token::PauliY => {
                self.advance();
                Ok(QuantumGateType::PauliY)
            }
            Token::PauliZ => {
                self.advance();
                Ok(QuantumGateType::PauliZ)
            }
            Token::CNOT => {
                self.advance();
                Ok(QuantumGateType::CNOT)
            }
            Token::Rotation => {
                self.advance();
                // Determine rotation type based on next token
                if let Token::Identifier(rot_type) = self.peek() {
                    match rot_type.as_str() {
                        "X" => {
                            self.advance();
                            Ok(QuantumGateType::RotationX(0.0)) // Angle will be filled in later
                        }
                        "Y" => {
                            self.advance();
                            Ok(QuantumGateType::RotationY(0.0))
                        }
                        "Z" => {
                            self.advance();
                            Ok(QuantumGateType::RotationZ(0.0))
                        }
                        _ => Err(ParseError::InvalidQuantumGate {
                            gate: format!("R{}", rot_type),
                        }),
                    }
                } else {
                    Err(ParseError::ExpectedToken {
                        expected: "rotation axis (X, Y, or Z)".to_string(),
                        found: self.peek().clone(),
                        position: self.current,
                    })
                }
            }
            _ => Err(ParseError::InvalidQuantumGate {
                gate: format!("{:?}", self.peek()),
            }),
        }
    }
    
    fn parse_quantum_circuit(&mut self) -> ParseResult<PhiFlowExpression> {
        self.consume(&Token::LeftBrace, "Expected '{' after circuit")?;
        
        let mut qubits = Vec::new();
        let mut gates = Vec::new();
        
        while !self.check(&Token::RightBrace) && !self.is_at_end() {
            if self.match_token(&Token::Newline) {
                continue;
            }
            
            if self.check(&Token::Qubit) {
                self.advance();
                let qubit_name = self.consume_identifier("Expected qubit name")?;
                qubits.push(qubit_name);
                self.consume(&Token::Semicolon, "Expected ';' after qubit declaration")?;
            } else if self.check(&Token::Gate) {
                if let PhiFlowExpression::QuantumGate { gate_type, qubits: gate_qubits, parameters } = self.parse_quantum_gate()? {
                    gates.push(QuantumGate {
                        gate_type,
                        qubits: gate_qubits,
                        parameters,
                        consciousness_controlled: false,
                    });
                    self.consume(&Token::Semicolon, "Expected ';' after gate")?;
                }
            } else {
                return Err(ParseError::ExpectedToken {
                    expected: "qubit or gate".to_string(),
                    found: self.peek().clone(),
                    position: self.current,
                });
            }
        }
        
        self.consume(&Token::RightBrace, "Expected '}' after circuit")?;
        
        Ok(PhiFlowExpression::QuantumCircuit { qubits, gates })
    }
    
    fn parse_block(&mut self) -> ParseResult<PhiFlowExpression> {
        self.consume(&Token::LeftBrace, "Expected '{'")?;
        
        let mut expressions = Vec::new();
        while !self.check(&Token::RightBrace) && !self.is_at_end() {
            if self.match_token(&Token::Newline) {
                continue;
            }
            expressions.push(self.parse_expression()?);
            // Optional semicolon
            self.match_token(&Token::Semicolon);
        }
        
        self.consume(&Token::RightBrace, "Expected '}'")?;
        Ok(PhiFlowExpression::Block(expressions))
    }
    
    fn parse_if_expression(&mut self) -> ParseResult<PhiFlowExpression> {
        self.consume(&Token::If, "Expected 'if'")?;
        let condition = Box::new(self.parse_expression()?);
        let then_branch = Box::new(self.parse_block()?);
        
        let else_branch = if self.match_token(&Token::Else) {
            if self.check(&Token::If) {
                Some(Box::new(self.parse_if_expression()?))
            } else {
                Some(Box::new(self.parse_block()?))
            }
        } else {
            None
        };
        
        Ok(PhiFlowExpression::If {
            condition,
            then_branch,
            else_branch,
        })
    }
    
    fn parse_function(&mut self) -> ParseResult<PhiFlowExpression> {
        self.consume(&Token::Fn, "Expected 'fn'")?;
        let name = self.consume_identifier("Expected function name")?;
        
        self.consume(&Token::LeftParen, "Expected '(' after function name")?;
        let mut parameters = Vec::new();
        
        if !self.check(&Token::RightParen) {
            loop {
                let param_name = self.consume_identifier("Expected parameter name")?;
                self.consume(&Token::Colon, "Expected ':' after parameter name")?;
                let param_type = self.parse_type()?;
                
                parameters.push(Parameter {
                    name: param_name,
                    param_type,
                    default_value: None,
                });
                
                if !self.match_token(&Token::Comma) {
                    break;
                }
            }
        }
        
        self.consume(&Token::RightParen, "Expected ')' after parameters")?;
        
        let return_type = if self.match_token(&Token::Arrow) {
            Some(self.parse_type()?)
        } else {
            None
        };
        
        let body = Box::new(self.parse_block()?);
        
        Ok(PhiFlowExpression::FunctionDefinition {
            name,
            parameters,
            return_type,
            body,
        })
    }
    
    fn parse_type(&mut self) -> ParseResult<PhiFlowType> {
        let identifier = self.consume_identifier("Expected type name")?;
        
        match identifier.as_str() {
            "f64" => Ok(PhiFlowType::Float64),
            "i32" => Ok(PhiFlowType::Integer),
            "bool" => Ok(PhiFlowType::Boolean),
            "string" => Ok(PhiFlowType::String),
            "qubit" => Ok(PhiFlowType::Qubit),
            "circuit" => Ok(PhiFlowType::QuantumCircuit),
            "consciousness" => Ok(PhiFlowType::ConsciousnessState),
            _ => Ok(PhiFlowType::Custom(identifier)),
        }
    }
    
    // Helper functions
    fn match_token(&mut self, token: &Token) -> bool {
        if self.check(token) {
            self.advance();
            true
        } else {
            false
        }
    }
    
    fn check(&self, token: &Token) -> bool {
        if self.is_at_end() {
            false
        } else {
            std::mem::discriminant(self.peek()) == std::mem::discriminant(token)
        }
    }
    
    fn parse_for_expression(&mut self) -> ParseResult<PhiFlowExpression> {
        self.advance(); // consume 'for'
        
        let variable = self.consume_identifier("Expected variable name after 'for'")?;
        self.consume(&Token::In, "Expected 'in' after for variable")?;
        let iterable = Box::new(self.parse_expression()?);
        
        self.consume(&Token::LeftBrace, "Expected '{' after for condition")?;
        let body = Box::new(self.parse_expression()?);
        self.consume(&Token::RightBrace, "Expected '}' after for body")?;
        
        Ok(PhiFlowExpression::For {
            variable,
            iterable,
            body,
        })
    }
    
    fn parse_while_expression(&mut self) -> ParseResult<PhiFlowExpression> {
        self.advance(); // consume 'while'
        
        let condition = Box::new(self.parse_expression()?);
        
        self.consume(&Token::LeftBrace, "Expected '{' after while condition")?;
        let body = Box::new(self.parse_expression()?);
        self.consume(&Token::RightBrace, "Expected '}' after while body")?;
        
        Ok(PhiFlowExpression::While {
            condition,
            body,
        })
    }
    
    fn parse_array_literal(&mut self) -> ParseResult<PhiFlowExpression> {
        self.advance(); // consume '['
        
        let mut elements = Vec::new();
        
        if !self.check(&Token::RightBracket) {
            loop {
                elements.push(self.parse_expression()?);
                if !self.match_token(&Token::Comma) {
                    break;
                }
            }
        }
        
        self.consume(&Token::RightBracket, "Expected ']' after array elements")?;
        
        Ok(PhiFlowExpression::Array(elements))
    }

    fn advance(&mut self) -> &Token {
        if !self.is_at_end() {
            self.current += 1;
        }
        self.previous()
    }
    
    fn is_at_end(&self) -> bool {
        self.current >= self.tokens.len() || matches!(self.peek(), Token::EOF)
    }
    
    fn peek(&self) -> &Token {
        self.tokens.get(self.current).unwrap_or(&Token::EOF)
    }
    
    fn previous(&self) -> &Token {
        &self.tokens[self.current - 1]
    }
    
    fn consume(&mut self, token: &Token, message: &str) -> ParseResult<()> {
        if self.check(token) {
            self.advance();
            Ok(())
        } else {
            Err(ParseError::ExpectedToken {
                expected: message.to_string(),
                found: self.peek().clone(),
                position: self.current,
            })
        }
    }
    
    fn consume_identifier(&mut self, message: &str) -> ParseResult<String> {
        if let Token::Identifier(name) = self.peek() {
            let result = name.clone();
            self.advance();
            Ok(result)
        } else {
            Err(ParseError::ExpectedToken {
                expected: message.to_string(),
                found: self.peek().clone(),
                position: self.current,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::lexer::PhiFlowLexer;
    
    fn parse_expression_from_string(input: &str) -> ParseResult<PhiFlowExpression> {
        let mut lexer = PhiFlowLexer::new(input.to_string());
        let tokens = lexer.tokenize().unwrap();
        let mut parser = PhiFlowParser::new(tokens);
        parser.parse_expression()
    }
    
    #[test]
    fn test_parse_number() {
        let expr = parse_expression_from_string("42.5").unwrap();
        assert_eq!(expr, PhiFlowExpression::Number(42.5));
    }
    
    #[test]
    fn test_parse_binary_op() {
        let expr = parse_expression_from_string("2 + 3").unwrap();
        match expr {
            PhiFlowExpression::BinaryOp { left, operator, right } => {
                assert_eq!(*left, PhiFlowExpression::Number(2.0));
                assert_eq!(operator, BinaryOperator::Add);
                assert_eq!(*right, PhiFlowExpression::Number(3.0));
            }
            _ => panic!("Expected binary operation"),
        }
    }
    
    #[test]
    fn test_parse_sacred_frequency() {
        let expr = parse_expression_from_string("Sacred(432)(resonate)").unwrap();
        match expr {
            PhiFlowExpression::SacredFrequency { frequency, operation } => {
                assert_eq!(frequency, 432);
                assert_eq!(*operation, PhiFlowExpression::Variable("resonate".to_string()));
            }
            _ => panic!("Expected sacred frequency expression"),
        }
    }
    
    #[test]
    fn test_parse_quantum_gate() {
        let expr = parse_expression_from_string("gate H(q0)").unwrap();
        match expr {
            PhiFlowExpression::QuantumGate { gate_type, qubits, .. } => {
                assert_eq!(gate_type, QuantumGateType::Hadamard);
                assert_eq!(qubits, vec!["q0".to_string()]);
            }
            _ => panic!("Expected quantum gate expression"),
        }
    }
    
    #[test]
    fn test_parse_let_binding() {
        let expr = parse_expression_from_string("let x = 42").unwrap();
        match expr {
            PhiFlowExpression::Let { variable, value, .. } => {
                assert_eq!(variable, "x");
                assert_eq!(*value, PhiFlowExpression::Number(42.0));
            }
            _ => panic!("Expected let binding"),
        }
    }

    #[test]
    fn test_parse_witness_intention() {
        let expr = parse_expression_from_string("witness(42)").unwrap();
        match expr {
            PhiFlowExpression::Witness(inner) => {
                assert_eq!(*inner, PhiFlowExpression::Number(42.0));
            }
            _ => panic!("Expected witness expression"),
        }

        let expr = parse_expression_from_string("intention(\"healing\", 528)").unwrap();
        match expr {
            PhiFlowExpression::Intention { content, target } => {
                assert_eq!(content, "healing");
                assert_eq!(*target, PhiFlowExpression::Number(528.0));
            }
            _ => panic!("Expected intention expression"),
        }
    }
}