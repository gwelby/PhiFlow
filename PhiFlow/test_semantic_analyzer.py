#!/usr/bin/env python3
"""
Test PhiFlow Semantic Analyzer - Task 3.3 Testing
"""

from src.parser.phi_flow_lexer import PhiFlowLexer
from src.parser.phi_flow_parser import PhiFlowParser
from src.parser.phi_flow_semantic_analyzer import PhiFlowSemanticAnalyzer

def test_semantic_analyzer():
    """Test semantic analyzer with simple PhiFlow program"""
    
    print("🧪 Testing PhiFlow Semantic Analyzer - Task 3.3")
    print("=" * 50)
    
    # Simple test program
    test_program = """
    INITIALIZE quantum_field AT 432Hz WITH coherence=1.0
    SET base_frequency = 432
    EVOLVE TO 720Hz WITH phi_level=4
    """
    
    # Create components
    lexer = PhiFlowLexer(debug=False)
    parser = PhiFlowParser(debug=False)
    analyzer = PhiFlowSemanticAnalyzer(debug=True)
    
    print("\n🔄 Processing test program...")
    
    try:
        # Parse (parser does its own tokenization)
        ast = parser.parse(test_program)
        print(f"✅ Parsing: AST generated")
        
        # Semantic analysis
        results = analyzer.analyze(ast)
        print(f"✅ Semantic Analysis: {results['semantic_info']['total_errors']} errors, {results['semantic_info']['total_warnings']} warnings")
        
        # Show results
        if results['errors']:
            print("\n❌ Errors found:")
            for error in results['errors']:
                print(f"  {error.message}")
        
        if results['warnings']:
            print("\n⚠️ Warnings:")
            for warning in results['warnings']:
                print(f"  {warning.message}")
        
        print(f"\n📊 Analysis Summary:")
        print(f"  Phi alignment score: {results['phi_alignment_score']:.3f}")
        print(f"  Variables: {results['semantic_info']['variables_defined']}")
        print(f"  Commands: {results['semantic_info']['commands_analyzed']}")
        
        print("\n✅ Semantic Analyzer Test Complete!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_semantic_analyzer()