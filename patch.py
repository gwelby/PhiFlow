import sys
import re

# Patch src/parser/mod.rs
path = "src/parser/mod.rs"
with open(path, "r", encoding="utf-8") as f:
    mod_rs = f.read()

# 1. Add MidCircuit token
mod_rs = mod_rs.replace(
    "    Arrow,\n    Dot,\n\n    // Special",
    "    Arrow,\n    Dot,\n    MidCircuit,\n\n    // Special"
)

# 2. Modify Witness struct
mod_rs = mod_rs.replace(
    "Witness {\n        expression: Option<Box<PhiExpression>>,",
    "Witness {\n        mid_circuit: bool,\n        expression: Option<Box<PhiExpression>>,"
)

# 3. Add to identifier parse
mod_rs = mod_rs.replace(
    '"sriyantra" => PhiToken::SriYantra,\n            "golden" => PhiToken::Golden,',
    '"sriyantra" => PhiToken::SriYantra,\n            "mid_circuit" => PhiToken::MidCircuit,\n            "golden" => PhiToken::Golden,'
)

# 4. Modify parse_witness_statement
parse_witness = """    fn parse_witness_statement(&mut self) -> Result<PhiExpression, String> {
        self.expect(PhiToken::Witness)?;

        let mut mid_circuit = false;

        // Skip any newlines right after witness before checking mid_circuit
        while self.current_token == PhiToken::Newline {
            self.advance();
        }

        if self.current_token == PhiToken::MidCircuit {
            mid_circuit = true;
            self.advance();
        }

        // Skip newlines before expression
        while self.current_token == PhiToken::Newline {
            self.advance();
        }

        // Check what IMMEDIATELY follows witness (before consuming newlines)
        // This determines: bare witness, witness with expression, or witness with block
        let (expression, body) = if self.current_token == PhiToken::Eof
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

        Ok(PhiExpression::Witness { mid_circuit, expression, body })
    }"""

mod_rs = re.sub(
    r"    fn parse_witness_statement.*?Ok\(PhiExpression::Witness \{ expression, body \}\)\n    \}",
    parse_witness,
    mod_rs,
    flags=re.DOTALL
)

# 5. Add to Debug output
mod_rs = mod_rs.replace(
    'PhiToken::Audio => Ok("audio".to_string()),\n            _ => Err',
    'PhiToken::Audio => Ok("audio".to_string()),\n            PhiToken::MidCircuit => Ok("mid_circuit".to_string()),\n            _ => Err'
)

with open(path, "w", encoding="utf-8", newline="\n") as f:
    f.write(mod_rs)


# Patch src/phi_ir/lowering.rs
path_lowering = "src/phi_ir/lowering.rs"
with open(path_lowering, "r", encoding="utf-8") as f:
    lowering = f.read()

# 1. Update pattern match
lowering = lowering.replace(
    "PhiExpression::Witness { expression, body } => {",
    "PhiExpression::Witness { mid_circuit, expression, body } => {"
)

# 2. Update emit
target_block = """            let op = ctx.emit(PhiIRNode::Witness {
                target,
                collapse_policy: CollapsePolicy::Deferred,
            });"""
            
replacement_block = """            let policy = if *mid_circuit { CollapsePolicy::MidCircuit } else { CollapsePolicy::Final };
            let op = ctx.emit(PhiIRNode::Witness {
                target,
                collapse_policy: policy,
            });"""

lowering = lowering.replace(target_block, replacement_block)

# Replace all Deferred to Final
lowering = lowering.replace("CollapsePolicy::Deferred", "CollapsePolicy::Final")

with open(path_lowering, "w", encoding="utf-8", newline="\n") as f:
    f.write(lowering)

print("Patch applied successfully.")
