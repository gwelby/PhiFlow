#!/bin/bash

# PhiFlow Test Runner - Comprehensive testing script for PhiFlow quantum consciousness programming language

echo "🧬 PhiFlow Comprehensive Test Suite"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[FAIL]${NC} $1"
}

# Check if cargo is available
if ! command -v cargo &> /dev/null; then
    print_error "Cargo is not installed. Please install Rust and Cargo."
    exit 1
fi

print_status "Starting PhiFlow test suite..."

# Build the project first
print_status "Building PhiFlow project..."
if cargo build; then
    print_success "Build successful"
else
    print_error "Build failed"
    exit 1
fi

# Function to run test category
run_test_category() {
    local test_name=$1
    local test_description=$2
    
    echo ""
    print_status "Running $test_description..."
    
    if cargo test --test $test_name; then
        print_success "$test_description completed successfully"
        return 0
    else
        print_error "$test_description failed"
        return 1
    fi
}

# Track test results
total_tests=0
passed_tests=0

# Run unit tests
total_tests=$((total_tests + 1))
if run_test_category "unit_tests" "Unit Tests (Individual Components)"; then
    passed_tests=$((passed_tests + 1))
fi

# Run quantum tests
total_tests=$((total_tests + 1))
if run_test_category "quantum_tests" "Quantum Tests (Quantum Computing Functionality)"; then
    passed_tests=$((passed_tests + 1))
fi

# Run integration tests
total_tests=$((total_tests + 1))
if run_test_category "integration_tests" "Integration Tests (Full System Pipeline)"; then
    passed_tests=$((passed_tests + 1))
fi

# Run performance tests
total_tests=$((total_tests + 1))
if run_test_category "performance_tests" "Performance Tests (Benchmarks & Scaling)"; then
    passed_tests=$((passed_tests + 1))
fi

# Run doctests
echo ""
print_status "Running documentation tests..."
total_tests=$((total_tests + 1))
if cargo test --doc; then
    print_success "Documentation tests completed successfully"
    passed_tests=$((passed_tests + 1))
else
    print_error "Documentation tests failed"
fi

# Run clippy linting
echo ""
print_status "Running Clippy linting..."
if cargo clippy -- -D warnings; then
    print_success "Clippy linting passed"
else
    print_warning "Clippy linting found issues (not blocking)"
fi

# Run formatting check
echo ""
print_status "Checking code formatting..."
if cargo fmt -- --check; then
    print_success "Code formatting is correct"
else
    print_warning "Code formatting issues found (run 'cargo fmt' to fix)"
fi

# Test CLI functionality
echo ""
print_status "Testing CLI functionality..."
if cargo run -- --help > /dev/null 2>&1; then
    print_success "CLI help command works"
else
    print_error "CLI help command failed"
fi

# Test specific CLI commands
print_status "Testing CLI quantum commands..."
if cargo run -- quantum list > /dev/null 2>&1; then
    print_success "CLI quantum list command works"
else
    print_warning "CLI quantum list command failed (may need backend setup)"
fi

# Test CLI info command
if cargo run -- info > /dev/null 2>&1; then
    print_success "CLI info command works"
else
    print_error "CLI info command failed"
fi

# Test example PhiFlow programs
echo ""
print_status "Testing example PhiFlow programs..."

example_files=(
    "examples/basic_test.phi"
    "examples/hello_quantum.phi"
    "examples/quantum_phi.phi"
)

for example in "${example_files[@]}"; do
    if [ -f "$example" ]; then
        print_status "Testing $example..."
        if cargo run -- "$example" > /dev/null 2>&1; then
            print_success "$example executed successfully"
        else
            print_warning "$example execution failed (may require specific setup)"
        fi
    else
        print_warning "$example not found"
    fi
done

# Summary
echo ""
echo "=========================================="
echo "🧬 PhiFlow Test Suite Summary"
echo "=========================================="
echo ""

if [ $passed_tests -eq $total_tests ]; then
    print_success "All test categories passed! ($passed_tests/$total_tests)"
    echo ""
    echo "🌟 PhiFlow is ready for quantum consciousness programming!"
    echo "✨ Key features tested:"
    echo "   • Sacred frequency quantum operations"
    echo "   • Phi-harmonic quantum gates"
    echo "   • Quantum simulator backend"
    echo "   • Consciousness-quantum coupling"
    echo "   • CLI quantum management"
    echo "   • Multi-dimensional quantum circuits"
    echo ""
    exit 0
else
    print_error "Some tests failed ($passed_tests/$total_tests passed)"
    echo ""
    echo "🔧 Issues found - please check the test output above"
    echo "💡 Common solutions:"
    echo "   • Run 'cargo build' to ensure compilation"
    echo "   • Check that all dependencies are installed"
    echo "   • Verify quantum backend configuration"
    echo "   • Run 'cargo fmt' to fix formatting issues"
    echo ""
    exit 1
fi