#!/usr/bin/env python3
"""
Test PhiFlow Compiler - Task 3.4 Testing
"""

from src._archive.parser.phi_flow_parser import PhiFlowParser
from src._archive.parser.phi_flow_semantic_analyzer import PhiFlowSemanticAnalyzer
from src._archive.parser.phi_flow_compiler import PhiFlowCompiler

def test_compiler():
    """Test compiler with simple PhiFlow program"""
    
    print("🧪 Testing PhiFlow Compiler - Task 3.4")
    print("=" * 50)
    
    # Simple test program
    test_program = """
    INITIALIZE quantum_field AT 432Hz WITH coherence=1.0, phi_level=0
    EVOLVE TO 720Hz WITH phi_level=4
    INTEGRATE heart_field AT 594Hz WITH phi_level=2
    """
    
    # Create components
    parser = PhiFlowParser(debug=False)
    analyzer = PhiFlowSemanticAnalyzer(debug=False)
    compiler = PhiFlowCompiler(debug=True)
    
    print("\n🔄 Processing test program...")
    
    try:
        # Parse
        ast = parser.parse(test_program)
        print(f"✅ Parsing completed")
        
        # Semantic analysis
        semantic_results = analyzer.analyze(ast)
        print(f"✅ Semantic analysis completed")
        
        # Compile
        compiled_program = compiler.compile(ast, semantic_results)
        print(f"✅ Compilation completed")
        
        # Show results
        print(f"\n📊 Compilation Summary:")
        print(f"  Instructions: {len(compiled_program.instructions)}")
        print(f"  Quantum qubits: {compiled_program.total_qubits}")
        print(f"  Quantum gates: {compiled_program.total_gates}")
        print(f"  Phi alignment: {compiled_program.phi_alignment_score:.3f}")
        
        # Show first few instructions
        print(f"\n📋 Sample Instructions:")
        for i, instr in enumerate(compiled_program.instructions[:3]):
            print(f"  {i+1}. {instr.operation} - {instr.instruction_type.value}")
            if instr.frequency:
                print(f"     Frequency: {instr.frequency}Hz")
        
        print("\n✅ Compiler Test Complete!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_compiler()