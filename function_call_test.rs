#[cfg(test)]
mod function_call_tests {
    use super::*;
    use crate::compiler::ast::*;
    use crate::vm::interpreter::*;

    #[tokio::test]
    async fn test_user_function_call() {
        let mut interpreter = PhiFlowInterpreter::new();
        
        // Define a simple function: fn add(x: f64, y: f64) -> f64 { x + y }
        let function_def = PhiFlowExpression::FunctionDefinition {
            name: "add".to_string(),
            parameters: vec\![
                Parameter {
                    name: "x".to_string(),
                    param_type: PhiFlowType::Float64,
                    default_value: None,
                },
                Parameter {
                    name: "y".to_string(),
                    param_type: PhiFlowType::Float64,
                    default_value: None,
                },
            ],
            return_type: Some(PhiFlowType::Float64),
            body: Box::new(PhiFlowExpression::BinaryOp {
                left: Box::new(PhiFlowExpression::Variable("x".to_string())),
                operator: BinaryOperator::Add,
                right: Box::new(PhiFlowExpression::Variable("y".to_string())),
            }),
        };
        
        // Store the function
        if let PhiFlowExpression::FunctionDefinition { name, .. } = &function_def {
            interpreter.functions.insert(name.clone(), function_def);
        }
        
        // Call the function: add(2, 3)
        let function_call = PhiFlowExpression::FunctionCall {
            name: "add".to_string(),
            args: vec\![
                PhiFlowExpression::Number(2.0),
                PhiFlowExpression::Number(3.0),
            ],
        };
        
        let result = interpreter.evaluate_expression(&function_call).await.unwrap();
        assert_eq\!(result, PhiFlowValue::Number(5.0));
    }
}
EOF < /dev/null
