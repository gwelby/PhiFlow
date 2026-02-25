// Test newline handling in parse_phi_program
use phiflow::parser::parse_phi_program;

fn main() {
    println!("🧪 Testing Newline Handling\n");

    // Test 1: Code without leading newline
    let test1 = r#"let x = 1.0"#;
    println!("Test 1: No leading newline");
    println!("Code: {:?}", test1);
    match parse_phi_program(test1) {
        Ok(_) => println!("✅ Parsed successfully\n"),
        Err(e) => println!("❌ Failed: {}\n", e),
    }

    // Test 2: Code with leading newline  
    let test2 = r#"
let x = 1.0"#;
    println!("Test 2: With leading newline");
    println!("Code: {:?}", test2);
    match parse_phi_program(test2) {
        Ok(_) => println!("✅ Parsed successfully\n"),
        Err(e) => println!("❌ Failed: {}\n", e),
    }

    // Test 3: Code with double leading newline
    let test3 = r#"

let x = 1.0"#;
    println!("Test 3: With double leading newline");
    println!("Code: {:?}", test3);
    match parse_phi_program(test3) {
        Ok(_) => println!("✅ Parsed successfully\n"),
        Err(e) => println!("❌ Failed: {}\n", e),
    }

    // Test 4: Multiple statements with newlines
    let test4 = r#"
let x = 1.0

let y = 2.0
"#;
    println!("Test 4: Multiple statements with newlines");
    println!("Code: {:?}", test4);
    match parse_phi_program(test4) {
        Ok(ast) => println!("✅ Parsed successfully - {} expressions\n", ast.len()),
        Err(e) => println!("❌ Failed: {}\n", e),
    }

    // Test 5: Emergency protocol with leading newline
    let test5 = r#"
emergency protocol test {
    trigger: true,
    immediate { },
    notify: []
}"#;
    println!("Test 5: Emergency protocol with leading newline");
    println!("Code: {:?}", test5);
    match parse_phi_program(test5) {
        Ok(_) => println!("✅ Parsed successfully\n"),
        Err(e) => println!("❌ Failed: {}\n", e),
    }
}