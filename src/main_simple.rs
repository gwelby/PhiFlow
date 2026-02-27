// Simple version of main.rs to test basic compilation
use phiflow::{PhiFlowLexer, PhiFlowParser, PhiFlowInterpreter};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("PhiFlow Simple Test");
    
    // Simple test - parse and execute "5"
    let source = "5";
    
    // Tokenize
    let mut lexer = PhiFlowLexer::new(source.to_string());
    let tokens = lexer.tokenize()?;
    
    // Parse
    let mut parser = PhiFlowParser::new(tokens);
    let program = parser.parse()?;
    
    // Execute
    let mut interpreter = PhiFlowInterpreter::new();
    let result = interpreter.execute_program(program).await?;
    
    println!("Result: {:?}", result);
    
    Ok(())
}