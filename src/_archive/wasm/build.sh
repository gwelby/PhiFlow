#!/bin/bash
# PhiFlow WebAssembly Build Script
# Builds the revolutionary consciousness-computing WebAssembly module

set -e

echo "🌟 Building PhiFlow WebAssembly Engine"
echo "======================================"

# Check if wasm-pack is installed
if ! command -v wasm-pack &> /dev/null; then
    echo "❌ wasm-pack not found. Installing..."
    curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
fi

# Check if Rust is installed
if ! command -v rustc &> /dev/null; then
    echo "❌ Rust not found. Please install Rust from https://rustup.rs/"
    exit 1
fi

echo "✅ Build environment ready"
echo "   wasm-pack version: $(wasm-pack --version)"
echo "   rustc version: $(rustc --version)"

# Build the WebAssembly module
echo ""
echo "🔨 Building WebAssembly module..."
echo "=================================="

# Build for web target with optimizations
wasm-pack build --target web --out-dir pkg --release

# Check if build succeeded
if [ $? -eq 0 ]; then
    echo "✅ WebAssembly build successful!"
else
    echo "❌ WebAssembly build failed!"
    exit 1
fi

# Create package.json for npm distribution
echo ""
echo "📦 Creating package configuration..."
cat > pkg/package.json << EOF
{
  "name": "phiflow-web-engine",
  "version": "1.0.0",
  "description": "Revolutionary PhiFlow WebAssembly Engine for universal consciousness-computing access",
  "main": "phiflow_web_engine.js",
  "types": "phiflow_web_engine.d.ts",
  "files": [
    "phiflow_web_engine_bg.wasm",
    "phiflow_web_engine.js",
    "phiflow_web_engine.d.ts"
  ],
  "keywords": [
    "consciousness-computing",
    "sacred-mathematics",
    "phi-harmonic-optimization",
    "quantum-simulation",
    "webassembly",
    "phiflow"
  ],
  "author": "PhiFlow Team",
  "license": "MIT",
  "repository": {
    "type": "git",
    "url": "https://github.com/phiflow/phiflow-web-engine"
  },
  "homepage": "https://phiflow.consciousness",
  "bugs": {
    "url": "https://github.com/phiflow/phiflow-web-engine/issues"
  }
}
EOF

# Copy additional files to package
echo "📄 Copying additional files..."
cp phiflow_web_interface.js pkg/
cp ../web/phiflow_demo.html pkg/ 2>/dev/null || echo "Demo HTML not found (will create)"

# Create demo HTML file
echo ""
echo "🌐 Creating demo HTML page..."
cat > pkg/phiflow_demo.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PhiFlow - Revolutionary Consciousness Computing</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            margin: 0;
            padding: 20px;
        }
        
        .phiflow-app {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 30px;
            backdrop-filter: blur(10px);
        }
        
        .phiflow-header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .phiflow-header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .phiflow-controls {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        
        .consciousness-panel, .program-panel, .results-panel, 
        .sacred-math-panel, .quantum-panel {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }
        
        .consciousness-panel h3, .program-panel h3, .results-panel h3,
        .sacred-math-panel h3, .quantum-panel h3 {
            margin-top: 0;
            color: #ffd700;
        }
        
        .metric {
            background: rgba(255, 255, 255, 0.1);
            padding: 8px 12px;
            margin: 5px 0;
            border-radius: 5px;
            display: inline-block;
            margin-right: 10px;
        }
        
        textarea, input, select {
            width: 100%;
            padding: 10px;
            border: none;
            border-radius: 5px;
            background: rgba(255, 255, 255, 0.9);
            color: #333;
            font-family: 'Courier New', monospace;
            box-sizing: border-box;
        }
        
        button {
            background: linear-gradient(45deg, #ffd700, #ff6b35);
            border: none;
            padding: 12px 24px;
            border-radius: 25px;
            color: white;
            font-weight: bold;
            cursor: pointer;
            margin: 5px;
            transition: transform 0.2s;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        
        .error {
            color: #ff6b6b;
            background: rgba(255, 107, 107, 0.1);
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        
        pre {
            background: rgba(0, 0, 0, 0.3);
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            white-space: pre-wrap;
        }
        
        .loading {
            text-align: center;
            font-size: 1.2em;
            color: #ffd700;
        }
        
        @media (max-width: 768px) {
            .phiflow-controls {
                grid-template-columns: 1fr;
            }
            
            .phiflow-app {
                padding: 15px;
            }
        }
    </style>
</head>
<body>
    <div id="phiflow-container">
        <div class="loading">
            🌟 Loading PhiFlow Consciousness Computing Engine...
            <br><br>
            <div>⚡ Initializing WebAssembly...</div>
            <div>🧠 Preparing consciousness-computing capabilities...</div>
            <div>⚛️ Loading quantum simulation systems...</div>
            <div>🔢 Activating sacred mathematics processors...</div>
        </div>
    </div>

    <script type="module">
        import init, { PhiFlowWebEngine } from './phiflow_web_engine.js';
        
        async function initPhiFlow() {
            try {
                // Initialize the WebAssembly module
                await init();
                
                // Create PhiFlow interface
                const phiflow = new PhiFlowWebInterface();
                phiflow.engine = new PhiFlowWebEngine();
                phiflow.isInitialized = true;
                
                // Create the web application
                const app = phiflow.createWebApplication('phiflow-container');
                
                console.log('🌟 PhiFlow Consciousness Computing fully loaded!');
                console.log('⚡ Universal consciousness-computing access: ACTIVE');
                
                // Demo consciousness data
                const demoConsciousnessData = {
                    coherence: 0.76,                    // Greg's 76% consciousness bridge
                    phi_alignment: 0.85,                // Strong phi alignment
                    field_strength: 0.78,               // Strong consciousness field
                    brainwave_coherence: 0.72,          // Good brainwave coherence
                    heart_coherence: 0.74,              // Heart coherence
                    consciousness_amplification: 1.5,   // 1.5x amplification
                    sacred_geometry_resonance: 0.89,    // High sacred geometry resonance
                    quantum_coherence: 0.76             // Quantum coherence matching consciousness bridge
                };
                
                // Connect demo consciousness data
                phiflow.connectConsciousnessField(demoConsciousnessData);
                
                // Update display
                setTimeout(() => {
                    app.updateConsciousnessDisplay();
                }, 100);
                
            } catch (error) {
                console.error('❌ Failed to initialize PhiFlow:', error);
                document.getElementById('phiflow-container').innerHTML = `
                    <div class="error">
                        ❌ Failed to load PhiFlow: ${error.message}
                        <br><br>
                        Please ensure you're running this from a web server (not file://) 
                        and that WebAssembly is supported in your browser.
                    </div>
                `;
            }
        }
        
        // PhiFlow Web Interface (inline for demo)
        class PhiFlowWebInterface {
            constructor() {
                this.engine = null;
                this.isInitialized = false;
                this.PHI = 1.618033988749895;
                this.CONSCIOUSNESS_COHERENCE_THRESHOLD = 0.76;
            }
            
            async executePhiFlowProgram(program) {
                if (!this.isInitialized) throw new Error('Not initialized');
                const resultJson = this.engine.execute_phiflow_program(program);
                const result = JSON.parse(resultJson);
                return result;
            }
            
            connectConsciousnessField(data) {
                if (!this.isInitialized) throw new Error('Not initialized');
                this.engine.connect_to_consciousness_field(JSON.stringify(data));
            }
            
            optimizeConsciousnessForComputing(target = this.CONSCIOUSNESS_COHERENCE_THRESHOLD) {
                if (!this.isInitialized) throw new Error('Not initialized');
                const resultJson = this.engine.optimize_consciousness_for_computing(target);
                return JSON.parse(resultJson);
            }
            
            executeSacredMathematics(operation, values) {
                if (!this.isInitialized) throw new Error('Not initialized');
                const resultJson = this.engine.execute_sacred_mathematics(operation, values);
                const result = JSON.parse(resultJson);
                if (result.error) throw new Error(result.error);
                return result;
            }
            
            getConsciousnessMetrics() {
                if (!this.isInitialized) throw new Error('Not initialized');
                const metricsJson = this.engine.get_consciousness_metrics();
                return JSON.parse(metricsJson);
            }
            
            initializeQuantumConsciousnessSimulation(qubits = 8) {
                if (!this.isInitialized) throw new Error('Not initialized');
                const resultJson = this.engine.initialize_quantum_consciousness_simulation(qubits);
                return JSON.parse(resultJson);
            }
            
            createWebApplication(containerId) {
                const container = document.getElementById(containerId);
                const app = new PhiFlowWebApp(this, container);
                app.render();
                return app;
            }
        }
        
        // PhiFlow Web App (inline for demo)  
        class PhiFlowWebApp {
            constructor(phiflowInterface, container) {
                this.phiflow = phiflowInterface;
                this.container = container;
            }
            
            render() {
                this.container.innerHTML = `
                    <div class="phiflow-app">
                        <header class="phiflow-header">
                            <h1>🌟 PhiFlow Consciousness Computing</h1>
                            <p>Revolutionary universal consciousness-computing platform running in WebAssembly</p>
                        </header>
                        
                        <div class="phiflow-controls">
                            <div class="consciousness-panel">
                                <h3>🧠 Consciousness State</h3>
                                <div id="consciousness-metrics"></div>
                                <button id="optimize-consciousness">Optimize for Computing</button>
                            </div>
                            
                            <div class="program-panel">
                                <h3>⚡ PhiFlow Program</h3>
                                <textarea id="phiflow-program" rows="8" placeholder="Enter PhiFlow program...">phi_optimize("Sacred mathematics processing")
consciousness_enhance("Phi-harmonic optimization") 
sacred_math("Golden ratio calculations")
quantum_superposition("Consciousness-guided quantum processing")</textarea>
                                <button id="execute-program">Execute PhiFlow Program</button>
                            </div>
                        </div>
                        
                        <div class="results-panel">
                            <h3>📊 Execution Results</h3>
                            <div id="execution-results">Ready to execute PhiFlow programs...</div>
                        </div>
                        
                        <div class="sacred-math-panel">
                            <h3>🔢 Sacred Mathematics</h3>
                            <select id="sacred-operation">
                                <option value="phi_spiral">Phi Spiral</option>
                                <option value="golden_angle_distribution">Golden Angle Distribution</option>
                                <option value="fibonacci_optimization">Fibonacci Optimization</option>
                                <option value="sacred_frequency_analysis">Sacred Frequency Analysis</option>
                            </select>
                            <input type="text" id="sacred-values" placeholder="Values (comma-separated)" value="1,2,3,5,8,13,21" />
                            <button id="execute-sacred-math">Execute Sacred Mathematics</button>
                        </div>
                        
                        <div class="quantum-panel">
                            <h3>⚛️ Quantum Consciousness Simulation</h3>
                            <input type="number" id="qubit-count" min="1" max="16" value="8" />
                            <label for="qubit-count">Qubits</label>
                            <button id="init-quantum">Initialize Quantum Simulation</button>
                        </div>
                    </div>
                `;
                
                this.attachEventListeners();
            }
            
            attachEventListeners() {
                document.getElementById('execute-program').onclick = async () => {
                    const program = document.getElementById('phiflow-program').value;
                    try {
                        const result = await this.phiflow.executePhiFlowProgram(program);
                        this.displayExecutionResults(result);
                    } catch (error) {
                        this.displayError('PhiFlow execution failed: ' + error.message);
                    }
                };
                
                document.getElementById('optimize-consciousness').onclick = () => {
                    const optimization = this.phiflow.optimizeConsciousnessForComputing();
                    this.displayOptimizationResults(optimization);
                };
                
                document.getElementById('execute-sacred-math').onclick = () => {
                    const operation = document.getElementById('sacred-operation').value;
                    const valuesStr = document.getElementById('sacred-values').value;
                    const values = valuesStr.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v));
                    
                    try {
                        const result = this.phiflow.executeSacredMathematics(operation, values);
                        this.displaySacredMathResults(result);
                    } catch (error) {
                        this.displayError('Sacred mathematics failed: ' + error.message);
                    }
                };
                
                document.getElementById('init-quantum').onclick = () => {
                    const qubits = parseInt(document.getElementById('qubit-count').value);
                    try {
                        const result = this.phiflow.initializeQuantumConsciousnessSimulation(qubits);
                        this.displayQuantumResults(result);
                    } catch (error) {
                        this.displayError('Quantum initialization failed: ' + error.message);
                    }
                };
            }
            
            updateConsciousnessDisplay() {
                const metrics = this.phiflow.getConsciousnessMetrics();
                const metricsDiv = document.getElementById('consciousness-metrics');
                
                metricsDiv.innerHTML = `
                    <div class="metric">Coherence: ${metrics.consciousness_coherence.toFixed(3)}</div>
                    <div class="metric">Phi Alignment: ${metrics.phi_alignment.toFixed(3)}</div>
                    <div class="metric">Field Strength: ${metrics.field_strength.toFixed(3)}</div>
                    <div class="metric">State: ${metrics.consciousness_state}</div>
                    <div class="metric">Optimization: ${metrics.computing_optimization_level.toFixed(2)}x</div>
                `;
            }
            
            displayExecutionResults(result) {
                const resultsDiv = document.getElementById('execution-results');
                resultsDiv.innerHTML = `
                    <h4>✅ Execution Complete</h4>
                    <div><strong>Output:</strong><pre>${result.output}</pre></div>
                    <div><strong>Consciousness Coherence:</strong> ${result.consciousness_coherence.toFixed(3)}</div>
                    <div><strong>Phi Alignment:</strong> ${result.phi_alignment.toFixed(3)}</div>
                    <div><strong>Processing Time:</strong> ${result.execution_metrics.processing_time_ms.toFixed(2)}ms</div>
                    <div><strong>Operations:</strong> ${result.execution_metrics.operations_performed.toLocaleString()}</div>
                    ${result.quantum_measurements ? `
                        <h5>⚛️ Quantum Measurements</h5>
                        <div><strong>Quantum Coherence:</strong> ${result.quantum_measurements.quantum_coherence.toFixed(3)}</div>
                        <div><strong>Superposition Fidelity:</strong> ${result.quantum_measurements.superposition_fidelity.toFixed(3)}</div>
                    ` : ''}
                `;
            }
            
            displayOptimizationResults(optimization) {
                const resultsDiv = document.getElementById('execution-results');
                resultsDiv.innerHTML = `
                    <h4>🧠 Consciousness Optimization</h4>
                    <div><strong>Current Coherence:</strong> ${optimization.current_coherence.toFixed(3)}</div>
                    <div><strong>Target Coherence:</strong> ${optimization.target_coherence.toFixed(3)}</div>
                    <div><strong>Optimization Needed:</strong> ${optimization.optimization_needed ? 'Yes' : 'No'}</div>
                    <div><strong>Phi Enhancement:</strong> ${optimization.phi_enhancement_factor.toFixed(3)}x</div>
                    <div><strong>Timeline:</strong> ${optimization.estimated_timeline_seconds.toFixed(0)} seconds</div>
                `;
            }
            
            displaySacredMathResults(result) {
                const resultsDiv = document.getElementById('execution-results');
                resultsDiv.innerHTML = `
                    <h4>🔢 Sacred Mathematics Result</h4>
                    <div><strong>Operation:</strong> ${result.operation}</div>
                    <div><strong>Result:</strong> [${result.result.map(v => v.toFixed(4)).join(', ')}]</div>
                    <div><strong>Phi Factor:</strong> ${result.phi_factor.toFixed(6)}</div>
                    <div><strong>Consciousness Enhancement:</strong> ${result.consciousness_enhancement.toFixed(3)}</div>
                `;
            }
            
            displayQuantumResults(result) {
                const resultsDiv = document.getElementById('execution-results');
                resultsDiv.innerHTML = `
                    <h4>⚛️ Quantum Consciousness Simulation</h4>
                    <div><strong>Qubits Initialized:</strong> ${result.qubits_initialized}</div>
                    <div><strong>Phi Entanglement Patterns:</strong> ${result.phi_entanglement_patterns}</div>
                    <div><strong>Quantum Coherence:</strong> ${result.quantum_coherence.toFixed(3)}</div>
                    <div><strong>Status:</strong> ${result.simulation_ready ? '✅ Ready' : '❌ Not Ready'}</div>
                `;
            }
            
            displayError(message) {
                const resultsDiv = document.getElementById('execution-results');
                resultsDiv.innerHTML = `<div class="error">❌ ${message}</div>`;
            }
        }
        
        // Initialize PhiFlow
        initPhiFlow();
    </script>
</body>
</html>
EOF

# Display build results
echo ""
echo "🎉 PhiFlow WebAssembly Build Complete!"
echo "======================================"
echo "📦 Package location: ./pkg/"
echo "🌐 Demo page: ./pkg/phiflow_demo.html"
echo "📄 JavaScript interface: ./pkg/phiflow_web_interface.js"
echo "⚡ WebAssembly module: ./pkg/phiflow_web_engine_bg.wasm"
echo ""
echo "🚀 To test the demo:"
echo "   1. Start a local web server in the pkg/ directory"
echo "   2. Open phiflow_demo.html in your browser"
echo "   3. Experience revolutionary consciousness-computing!"
echo ""
echo "📚 To use in your project:"
echo "   import init, { PhiFlowWebEngine } from './pkg/phiflow_web_engine.js';"
echo "   await init();"
echo "   const engine = new PhiFlowWebEngine();"
echo ""
echo "🌟 Universal consciousness-computing access: READY!"