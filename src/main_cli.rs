use clap::Parser;
use phiflow::compile_to_openqasm_with_options;
use phiflow::parser::parse_phi_program_with_diagnostics;
use phiflow::phi_ir::evaluator::Evaluator;
use phiflow::phi_ir::lowering::{lower_program, lower_program_checked};
use phiflow::phi_ir::openqasm::OpenQasmEmitter;

use phiflow::phi_ir::topology_transpiler::{RoutingStrategy, TopologyTranspileConfig};
use phiflow::quantum::BackendTopologyProfile;
use phiflow::phi_ir::PhiIRValue;
use phiflow::metrics::consciousness_proxy::ConsciousnessMetrics;
use phiflow::metrics::self_correlation::SelfCorrelation;
use phiflow::metrics::trace::Trace;
use phiflow::resonance_bus::{self, ResonanceEvent};
use phiflow::sensors;
use phiflow::system_host::SystemHostProvider;
use phiflow::wasm_host::{self, WasmHostHooks};
use phiflow::OpenQasmCompileOptions;
use phiflow::PhiDiagnostic;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// A command-line interpreter for the PhiFlow language.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The path to the .phi file to execute. Required unless --poll-ibm is given.
    file: Option<PathBuf>,

    /// Emit parse errors as a strict JSON array of PhiDiagnostic objects (for tooling).
    #[arg(long, default_value_t = false)]
    json_errors: bool,

    /// Emit execution metrics as JSON (coherence, resonance, self-correlation signals).
    #[arg(long, default_value_t = false)]
    measure: bool,

    /// The target backend to compile to (e.g., 'quantum'). If not specified, runs in the interpreter.
    #[arg(long)]
    target: Option<String>,

    /// Optimize quantum circuit depth using tree topology.
    #[arg(long, default_value_t = false)]
    optimize_depth: bool,

    /// Fetch the live backend topology profile and emit topology-aware OpenQASM.
    #[arg(long, default_value_t = false)]
    topology_aware: bool,

    /// IBM backend name to fetch topology from when --topology-aware is enabled.
    #[arg(long, default_value = "ibm_fez")]
    topology_backend: String,

    /// IBM backend name used by the quantum transpile guardrail (--target quantum).
    #[arg(long, default_value = "ibm_marrakesh")]
    quantum_backend: String,

    /// Run as a daemon, listening for evolve events.
    #[arg(long, default_value_t = false)]
    daemon: bool,

    /// Path to persist the daemon state to.
    #[arg(long, env = "PHIFLOW_DAEMON_STATE_PATH", default_value = "/tmp/phiflow_daemon_state.json")]
    state_path: PathBuf,

    /// Trigger a resonant handoff: "TargetAgent:TaskID:Context"
    #[arg(long)]
    handoff: Option<String>,

    /// Launch and manage the SOMA python sensor suite (Headless).
    #[arg(long, default_value_t = false)]
    with_soma: bool,

    /// Limit the number of execution steps (0 = infinite).
    #[arg(long, default_value_t = 0)]
    max_steps: usize,

    /// Launch and manage the Quantum Presence bridge for real-time Heron telemetry.
    #[arg(long, default_value_t = false)]
    with_quantum: bool,

    /// Poll an IBM Quantum job by ID and analyze coherence. Use "mock" for a demo run.
    #[arg(long)]
    poll_ibm: Option<String>,

    /// Generate a sacred geometry SVG pattern and print to stdout.
    /// Patterns: flower_of_life, phi_spiral, merkaba, sri_yantra, consciousness_torus, claude_mandala.
    #[arg(long)]
    sacred_geometry: Option<String>,

    /// Print consciousness mathematics reference (frequencies, breathing patterns) as JSON.
    #[arg(long, default_value_t = false)]
    consciousness_info: bool,

    /// Start the MCP stdio server for tool integration (spawn_phi_stream, read_resonance_field, etc.).
    #[arg(long, default_value_t = false)]
    mcp_serve: bool,

    /// Emit OSC (Open Sound Control) messages to 127.0.0.1:<port> as the program runs.
    /// Any OSC-capable environment (TouchDesigner, Three.js, SuperCollider, etc.) can
    /// receive the live state stream and render it in real-time.
    #[arg(long)]
    osc: Option<u16>,
}

struct SomaManager {
    child_soma: Option<std::process::Child>,
    child_quantum: Option<std::process::Child>,
}

impl SomaManager {
    fn new() -> Self {
        Self { child_soma: None, child_quantum: None }
    }

    fn start_soma(&mut self) {
        println!("🌀 Starting SOMA Sensor Suite (Python)...");
        let soma_py = std::env::var("SOMA_PY_PATH")
            .unwrap_or_else(|_| "soma.py".to_string());
        let child = std::process::Command::new("python")
            .arg(&soma_py)
            .arg("--phiflow")
            .arg("--headless")
            .spawn();

        match child {
            Ok(c) => {
                println!("✅ SOMA Process started (PID: {})", c.id());
                self.child_soma = Some(c);
            }
            Err(e) => {
                eprintln!("❌ Failed to start SOMA: {}", e);
            }
        }
    }

    fn start_quantum(&mut self) {
        println!("🌌 Starting Quantum Presence Bridge (IBM Heron)...");
        let child = std::process::Command::new("python")
            .arg("scripts/quantum_presence.py")
            .spawn();

        match child {
            Ok(c) => {
                println!("✅ Quantum Presence Bridge started (PID: {})", c.id());
                self.child_quantum = Some(c);
            }
            Err(e) => {
                eprintln!("❌ Failed to start Quantum Bridge: {}", e);
            }
        }
    }

    fn stop(&mut self) {
        if let Some(mut child) = self.child_soma.take() {
            println!("🛑 Stopping SOMA Process...");
            let _ = child.kill();
        }
        if let Some(mut child) = self.child_quantum.take() {
            println!("🛑 Stopping Quantum Presence Bridge...");
            let _ = child.kill();
        }
    }
}

impl Drop for SomaManager {
    fn drop(&mut self) {
        self.stop();
    }
}

#[derive(Debug)]
enum CliError {
    Parse(PhiDiagnostic),
    Io(String),
    Eval(String),
    Lower(String),
}

struct RunReport {
    final_coherence: f64,
    resonance_events: Vec<(String, PhiIRValue)>,
    ended_streams: Vec<String>,
    consciousness_metrics: Option<ConsciousnessMetrics>,
    self_correlation: Option<SelfCorrelation>,
    _program: phiflow::phi_ir::PhiIRProgram,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    let json_errors = args.json_errors;

    // --poll-ibm <job_id>: Poll an IBM Quantum job and analyze coherence.
    // Use "mock" for a demo run without credentials.
    if let Some(job_id) = &args.poll_ibm {
        poll_ibm_job_and_analyze(job_id);
        return;
    }

    // --sacred-geometry <pattern>: Generate SVG sacred geometry pattern.
    if let Some(pattern) = &args.sacred_geometry {
        sacred_geometry_command(pattern);
        return;
    }

    // --consciousness-info: Print consciousness math reference as JSON.
    if args.consciousness_info {
        consciousness_info_command();
        return;
    }

    // --mcp-serve: Start MCP stdio server.
    if args.mcp_serve {
        mcp_serve_command().await;
        return;
    }

    let file = match &args.file {
        Some(f) => f.clone(),
        None => {
            eprintln!("Error: <FILE> is required unless --poll-ibm, --sacred-geometry, --consciousness-info, or --mcp-serve is given.");
            std::process::exit(2);
        }
    };

    let measure = args.measure;
    match run(
        &file,
        json_errors,
        measure,
        args.target.clone(),
        args.optimize_depth,
        args.topology_aware,
        args.topology_backend.clone(),
        args.quantum_backend.clone(),
        args.daemon,
        args.state_path.clone(),
        args.handoff.clone(),
        args.with_soma,
        args.with_quantum,
        args.max_steps,
        args.osc,
    )
    .await
    {
        Ok(Some(report)) => {
            if json_errors {
                println!("[]");
                std::process::exit(0);
            }

            if measure {
                let mut resonance_map = serde_json::Map::new();
                for (scope, value) in &report.resonance_events {
                    let v = match value {
                        PhiIRValue::Number(n) => serde_json::json!(n),
                        PhiIRValue::String(s) => serde_json::json!(s),
                        PhiIRValue::Boolean(b) => serde_json::json!(b),
                        _ => serde_json::json!(null),
                    };
                    resonance_map.insert(scope.clone(), v);
                }

                // Include consciousness metrics (C_PF) when available.
                let consciousness = match &report.consciousness_metrics {
                    Some(m) => serde_json::json!({
                        "l_self": m.l_self,
                        "d_int": m.d_int,
                        "c_coh": m.c_coh,
                        "f_model": m.f_model,
                        "f_self_star": m.f_self_star,
                        "c_pf": m.c_pf,
                    }),
                    None => serde_json::json!(null),
                };

                let payload = serde_json::json!({
                    "ok": true,
                    "final_coherence": report.final_coherence,
                    "resonance_events": resonance_map,
                    "ended_streams": report.ended_streams,
                    "consciousness": consciousness,
                    "source": file.to_string_lossy(),
                });
                println!("{}", serde_json::to_string_pretty(&payload).unwrap());

                // Write metrics to the daemon metrics JSONL file so the
                // phiflow-metrics-bridge (:18030) can serve them.
                if let (Some(m), Some(sc)) = (&report.consciousness_metrics, &report.self_correlation) {
                    let bridge_entry = serde_json::json!({
                        "timestamp": chrono::Utc::now().to_rfc3339(),
                        "l_self": m.l_self,
                        "r_in": sc.r_in_norm,
                        "r_out": sc.r_out_norm,
                        "c_pf": m.c_pf,
                        "d_int": m.d_int,
                        "c_coh": m.c_coh,
                        "f_model": m.f_model,
                        "window_size": 10,
                        "final_coherence": report.final_coherence,
                        "coherence_per_intention": resonance_map,
                        "source": file.to_string_lossy(),
                    });
                    let jsonl_path = std::path::Path::new("/tmp/phiflow_daemon_metrics.jsonl");
                    if let Some(parent) = jsonl_path.parent() {
                        let _ = std::fs::create_dir_all(parent);
                    }
                    use std::io::Write;
                    if let Ok(mut f) = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(jsonl_path)
                    {
                        let _ = writeln!(f, "{}", bridge_entry);
                        eprintln!("📊 Metrics published to :18030 ({} bytes)", bridge_entry.to_string().len());
                    }
                }

                std::process::exit(0);
            }

            for (_scope, value) in &report.resonance_events {
                match value {
                    PhiIRValue::Number(n) => {
                        println!("🔔 Resonating Field: {:.4}Hz", n);
                    }
                    PhiIRValue::String(s) => {
                        println!("🔔 Resonating Field: \"{}\"", s);
                    }
                    other => {
                        println!("🔔 Resonating Field: {:?}", other);
                    }
                }
            }

            for stream in &report.ended_streams {
                println!("🌊 Stream broken: {}", stream);
            }

            println!(
                "✨ Execution Finished. Final Coherence: {:.4}",
                report.final_coherence
            );
            std::process::exit(0);
        }
        Ok(None) => {
            std::process::exit(0);
        }
        Err(CliError::Parse(diag)) => {
            if json_errors {
                let payload = vec![diag];
                let _ = serde_json::to_string(&payload).map(|json| println!("{}", json));
                std::process::exit(2);
            }
            eprintln!("{}", diag);
            std::process::exit(2);
        }
        Err(CliError::Io(msg)) => {
            if json_errors {
                println!("[]");
            }
            eprintln!("Error: {}", msg);
            std::process::exit(1);
        }
        Err(CliError::Eval(msg)) => {
            if json_errors {
                println!("[]");
            }
            eprintln!("Runtime error: {}", msg);
            std::process::exit(1);
        }
        Err(CliError::Lower(msg)) => {
            if json_errors {
                println!("[]");
            }
            eprintln!("Lowering error: {}", msg);
            std::process::exit(1);
        }
    }
}

async fn run(
    file_path: &PathBuf,
    json_errors: bool,
    measure: bool,
    target: Option<String>,
    optimize_depth: bool,
    topology_aware: bool,
    topology_backend: String,
    quantum_backend: String,
    daemon: bool,
    state_path: PathBuf,
    handoff: Option<String>,
    with_soma: bool,
    with_quantum: bool,
    max_steps: usize,
    osc_port: Option<u16>,
) -> Result<Option<RunReport>, CliError> {
    if let Some(h) = handoff {
        let parts: Vec<&str> = h.split(':').collect();
        if parts.len() >= 2 {
            let target = parts[0];
            let task = parts[1];
            let context = if parts.len() > 2 { parts[2] } else { "" };
            
            let payload = serde_json::json!({
                "target": target,
                "task_id": task,
                "context": context,
                "attention": "CLI_MANUAL_HANDOFF",
                "coherence": 1.0,
                "dissonance": 0.0,
            });
            
            println!("📡 Broadcasting resonant handoff: {} -> {}", target, task);
            let _ = resonance_bus::emit_resonance(payload, "_handoff", "phi_cli");
            return Ok(None);
        } else {
            return Err(CliError::Eval("Invalid handoff format. Use 'Agent:TaskID:Context'".to_string()));
        }
    }

    let source = fs::read_to_string(file_path)
        .map_err(|e| CliError::Io(format!("Failed to read file: {}", e)))?;

    let ast = parse_phi_program_with_diagnostics(&source).map_err(CliError::Parse)?;

    println!("Compiling to PhiFlow IR...");
    let ir_program = lower_program_checked(&ast).map_err(|e| CliError::Lower(e.to_string()))?;
    let session_signing_key = Arc::new(phiflow::security::anchor::AnchorSigningKey::generate());

    if json_errors {
        return Ok(Some(RunReport {
            final_coherence: 0.0,
            resonance_events: Vec::new(),
            ended_streams: Vec::new(),
            consciousness_metrics: None,
            self_correlation: None,
            _program: ir_program,
        }));
    }

    if let Some(t) = target {
        match t.as_str() {
            "quantum" => {
                println!("🌌 Quantum Consciousness Council — parameterized emission");

                // 1. Run evaluator to capture live coherence per intention
                let mut evaluator = Evaluator::new(ir_program.clone())
                    .with_hardware_modifier(sensors::compute_coherence_from_sensors);
                let _ = evaluator.run(); // may end naturally

                let frozen = evaluator.freeze_state();
                let mut runtime_params = HashMap::new();
                for event in &frozen.witness_log {
                    if let Some(name) = event.intention_stack.last() {
                        // Last coherence value for each intention wins
                        runtime_params.insert(name.clone(), event.coherence);
                    }
                }

                if runtime_params.is_empty() {
                    println!("⚠️  No witness events found — using compile-time constants");
                } else {
                    println!("📊 Captured council coherence:");
                    for (name, coherence) in &runtime_params {
                        println!("  {}: {:.4}", name, coherence);
                    }
                }

                // 2. Emit parameterized QASM (used by both measure and non-measure paths).
                let mut emitter = OpenQasmEmitter::new();
                emitter.optimize_depth = optimize_depth;
                let qasm = emitter
                    .emit_with_runtime_params(&ir_program, &runtime_params)
                    .map_err(|e| CliError::Eval(e.to_string()))?;

                // 3. Quantum transpile guardrail: depth, ops, layout, spectators.
                let guardrail = run_transpile_guardrail(&qasm, &quantum_backend);

                if measure {
                    let mut coherence_map = serde_json::Map::new();
                    for (name, coherence) in &runtime_params {
                        coherence_map.insert(name.clone(), serde_json::json!(coherence));
                    }

                    // Compute consciousness metrics from the frozen council trace.
                    let q_trace = Trace::from_vm_state(&frozen);
                    let q_consciousness = if q_trace.len() >= 20 {
                        let m = ConsciousnessMetrics::compute(&q_trace, 10, 5, 0.01);
                        serde_json::json!({
                            "l_self": m.l_self,
                            "d_int": m.d_int,
                            "c_coh": m.c_coh,
                            "f_model": m.f_model,
                            "f_self_star": m.f_self_star,
                            "c_pf": m.c_pf,
                        })
                    } else {
                        serde_json::json!(null)
                    };

                    let mut payload = serde_json::json!({
                        "ok": true,
                        "target": "quantum",
                        "coherence_per_intention": coherence_map,
                        "consciousness": q_consciousness,
                        "source": file_path.to_string_lossy(),
                    });
                    if let Ok(report) = guardrail {
                        if let Some(obj) = payload.as_object_mut() {
                            obj.insert("transpile_report".to_string(), report);
                        }
                    } else if let Err(e) = guardrail {
                        eprintln!("⚠️  Transpile guardrail unavailable: {}", e);
                    }
                    println!("{}", serde_json::to_string_pretty(&payload).unwrap());
                    return Ok(None);
                }

                // Non-measure path: print the QASM, then the guardrail report.
                print!("{}", qasm);

                match guardrail {
                    Ok(report) => print_guardrail_report(&report),
                    Err(e) => eprintln!("⚠️  Transpile guardrail unavailable: {}", e),
                }
                return Ok(None);
            }
            "openqasm" => {
                let qasm = if topology_aware {
                    let profile = fetch_live_topology_profile(&topology_backend).await?;
                    let native_two_qubit_gate = profile.native_two_qubit_gate;
                    let options = OpenQasmCompileOptions {
                        optimize_depth,
                        topology: Some(TopologyTranspileConfig {
                            backend_name: topology_backend.clone(),
                            strategy: RoutingStrategy::CalibrationWeightedShortestPath,
                            native_two_qubit_gate,
                        }),
                        live_backend_profile: Some(profile),
                        anchor_signing_key: Some(Arc::clone(&session_signing_key)),
                    };
                    compile_to_openqasm_with_options(&source, &options)
                        .map_err(CliError::Eval)?
                } else {
                    let mut emitter = OpenQasmEmitter::new();
                    emitter.optimize_depth = optimize_depth;
                    emitter.anchor_fingerprint_ecdsa = Some(session_signing_key.fingerprint());
                    emitter.anchor_fingerprint_pq = Some(session_signing_key.fingerprint_pq());
                    emitter
                        .emit(&ir_program)
                        .map_err(|e| CliError::Eval(e.to_string()))?
                };

                // Transpile guardrail: use topology_backend for topology-aware,
                // otherwise fall back to quantum_backend.
                let guardrail_backend = if topology_aware {
                    topology_backend.clone()
                } else {
                    quantum_backend.clone()
                };
                let guardrail = run_transpile_guardrail(&qasm, &guardrail_backend);

                print!("{}", qasm);

                match guardrail {
                    Ok(report) => print_guardrail_report(&report),
                    Err(e) => eprintln!("⚠️  Transpile guardrail unavailable: {}", e),
                }
                return Ok(None);
            }
            "wasm" => {
                println!("🌐 WASM backend — compiling to WAT and executing via wasmtime host");

                // 1. Compile source to WAT
                let wat = wasm_host::compile_source_to_wat(&source)
                    .map_err(|e| CliError::Eval(e.to_string()))?;

                if measure {
                    // Just emit the WAT for tooling
                    let payload = serde_json::json!({
                        "ok": true,
                        "target": "wasm",
                        "wat": wat,
                        "source": file_path.to_string_lossy(),
                    });
                    println!("{}", serde_json::to_string_pretty(&payload).unwrap());
                    return Ok(None);
                }

                // 2. Run through the WASM host with consciousness hooks
                let hooks = WasmHostHooks::new()
                    .with_coherence_provider(sensors::compute_coherence_from_sensors)
                    .with_witness(|event| {
                        if let Some(ref intention) = event.intention {
                            println!(
                                "👁  witness ({}): coherence={:.4}",
                                intention, event.coherence
                            );
                        } else {
                            println!("👁  witness: coherence={:.4}", event.coherence);
                        }
                    })
                    .with_resonate(|value, scope| {
                        if let Some(s) = scope {
                            println!("🔔 resonate {} in {}", value, s);
                        } else {
                            println!("🔔 resonate {}", value);
                        }
                    })
                    .with_intention_push(|name, depth| {
                        println!("📥 intention push: {} (depth={})", name, depth);
                    })
                    .with_intention_pop(|name, depth| {
                        println!("📤 intention pop: {} (depth={})", name, depth);
                    });

                let result = wasm_host::run_source_with_host(&source, hooks)
                    .map_err(|e| CliError::Eval(e.to_string()))?;

                println!("\n═══════════════════════════════════════════");
                println!("  WASM Execution Result");
                println!("═══════════════════════════════════════════");
                println!("  Result: {:?}", result.result);
                println!(
                    "  Final coherence: {:.4}",
                    result.snapshot.coherence
                );
                println!(
                    "  Witness events: {}",
                    result.snapshot.witness_log.len()
                );
                println!(
                    "  Intention stack depth: {}",
                    result.snapshot.intention_stack.len()
                );
                println!("═══════════════════════════════════════════");

                return Ok(None);
            }
            _ => {
                return Err(CliError::Eval(format!("Unknown target: {}", t)));
            }
        }
    }

    if daemon {
        daemon_run(ir_program, state_path, with_soma, with_quantum).await?;
        return Ok(None);
    }

    let mut soma_manager = SomaManager::new();
    if with_soma {
        soma_manager.start_soma();
    }
    if with_quantum {
        soma_manager.start_quantum();
    }

    // If --osc <port> is given, plug in the OSC host provider to broadcast
    // live runtime events as OSC messages.
    let osc_host = if let Some(port) = osc_port {
        match phiflow::osc_host::OscHostProvider::new(port) {
            Ok(host) => {
                host.emit_start(&file_path.to_string_lossy());
                eprintln!("📡 OSC streaming to 127.0.0.1:{} — connect your visualizer/audio environment", port);
                Some(host)
            }
            Err(e) => {
                eprintln!("⚠️  Failed to create OSC socket: {}", e);
                None
            }
        }
    } else {
        None
    };

    let mut evaluator = Evaluator::new(ir_program.clone())
        .with_hardware_modifier(sensors::compute_coherence_from_sensors);

    if let Some(ref host) = osc_host {
        // Clone the Arc-backed host so the evaluator gets its own reference
        // while we retain one to emit the end event after the run.
        evaluator = evaluator.with_host(Box::new(host.clone()));
    }

    if max_steps > 0 {
        evaluator.max_steps = Some(max_steps);
    } else {
        evaluator.max_steps = Some(1_000_000_000);
    }

    let _result = evaluator.run().map_err(|e| CliError::Eval(e.to_string()))?;

    // Emit program-end OSC message.
    if let Some(ref host) = osc_host {
        host.emit_end(evaluator.resolved_coherence());
    }

    // Compute consciousness metrics (C_PF) from the frozen execution trace.
    let frozen = evaluator.freeze_state();
    let trace = Trace::from_vm_state(&frozen);
    let (consciousness_metrics, self_correlation) = if trace.len() >= 20 {
        let cm = ConsciousnessMetrics::compute(&trace, 10, 5, 0.01);
        let sc = SelfCorrelation::from_type4_trace(&trace, 0.01);
        (Some(cm), Some(sc))
    } else {
        (None, None) // Not enough data for meaningful metric computation
    };

    Ok(Some(RunReport {
        final_coherence: evaluator.resolved_coherence(),
        resonance_events: evaluator.resonance_events().to_vec(),
        ended_streams: evaluator.ended_streams().to_vec(),
        consciousness_metrics,
        self_correlation,
        _program: ir_program,
    }))
}

async fn fetch_live_topology_profile(
    backend_name: &str,
) -> Result<BackendTopologyProfile, CliError> {
    // Use the Python bridge (scripts/fetch_topology_profile.py) which authenticates
    // via IBM_QUANTUM_TOKEN from ~/.cascade_keys — the same credential used by all
    // other PhiFlow IBM scripts. This avoids the need for a separate IBM Cloud IAM
    // API key + service CRN.
    let script_path = std::env::var("PHIFLOW_TOPOLOGY_BRIDGE")
        .unwrap_or_else(|_| "scripts/fetch_topology_profile.py".to_string());

    let output = tokio::process::Command::new("python3.12")
        .arg(&script_path)
        .arg(backend_name)
        .output()
        .await
        .map_err(|e| CliError::Eval(format!(
            "Failed to launch topology bridge script: {e}\n\
             Hint: --topology-aware calls scripts/fetch_topology_profile.py which uses\n\
             IBM_QUANTUM_TOKEN from ~/.cascade_keys."
        )))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(CliError::Eval(format!(
            "Topology bridge script failed: {stderr}\n\
             Hint: --topology-aware calls scripts/fetch_topology_profile.py which uses\n\
             IBM_QUANTUM_TOKEN from ~/.cascade_keys."
        )));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Deserialize using an intermediate struct that represents edges as a
    // list (JSON can't have tuple keys in objects). Then convert to the
    // HashMap<(usize, usize), EdgeCalibration> expected by BackendTopologyProfile.
    #[derive(serde::Deserialize)]
    struct EdgeEntry {
        edge: (usize, usize),
        #[serde(flatten)]
        cal: phiflow::quantum::EdgeCalibration,
    }
    #[derive(serde::Deserialize)]
    struct RawProfile {
        backend_name: String,
        family: phiflow::quantum::ProcessorFamily,
        num_qubits: usize,
        coupling_map: Vec<(usize, usize)>,
        native_two_qubit_gate: phiflow::quantum::NativeTwoQGate,
        qubits: HashMap<usize, phiflow::quantum::QubitCalibration>,
        edges: Vec<EdgeEntry>,
    }

    let raw: RawProfile = serde_json::from_str(&stdout)
        .map_err(|e| CliError::Eval(format!(
            "Failed to parse topology profile JSON: {e}\n\
             Raw output (first 500 chars): {}",
            &stdout[..stdout.len().min(500)]
        )))?;

    let edges: HashMap<(usize, usize), phiflow::quantum::EdgeCalibration> = raw.edges
        .into_iter()
        .map(|e| (e.edge, e.cal))
        .collect();

    let profile = BackendTopologyProfile {
        backend_name: raw.backend_name,
        family: raw.family,
        num_qubits: raw.num_qubits,
        coupling_map: raw.coupling_map,
        native_two_qubit_gate: raw.native_two_qubit_gate,
        qubits: raw.qubits,
        edges,
    };

    Ok(profile)
}

#[derive(Debug, Clone, PartialEq)]
enum StreamStatus {
    Ready,
    AwaitingQuantumCollapse(String), // The IBM job_id from the Python execution bridge
}

struct DaemonHypervisor<'a> {
    streams: HashMap<String, (StreamStatus, Evaluator<'a>)>,
    shared_resonance: Arc<Mutex<HashMap<String, Vec<PhiIRValue>>>>,
    state_path: PathBuf,
    signing_key: Arc<phiflow::security::anchor::AnchorSigningKey>,
}

impl<'a> DaemonHypervisor<'a> {
    fn new(state_path: PathBuf) -> Self {
        Self {
            streams: HashMap::new(),
            shared_resonance: Arc::new(Mutex::new(HashMap::new())),
            state_path,
            signing_key: Arc::new(phiflow::security::anchor::AnchorSigningKey::generate()),
        }
    }

    fn spawn_stream(&mut self, id: String, evaluator: Evaluator<'a>) {
        self.streams.insert(id, (StreamStatus::Ready, evaluator));
    }

    fn save_state(&self) {
        let mut states = HashMap::new();
        // NOTE: Awaiting streams technically freeze at the state they yielded,
        // so we save the evaluator state as is. Upon resume, they'll become Ready
        // and re-trigger a witness if the Python bridge was lost, which is tolerable.
        for (id, (_, eval)) in &self.streams {
            states.insert(id.clone(), eval.freeze_state());
        }

        if let Some(parent) = self.state_path.parent() {
            if let Err(e) = fs::create_dir_all(parent) {
                eprintln!("warning: could not create daemon state directory {:?}: {}", parent, e);
            }
        }

        if let Ok(json) = serde_json::to_string_pretty(&states) {
            let _ = fs::write(&self.state_path, json);
            println!("💾 Daemon state snapshotted to {:?}", self.state_path);
        }
    }

    fn load_state(&mut self) {
        if !self.state_path.exists() {
            return;
        }

        if let Ok(json) = fs::read_to_string(&self.state_path) {
            if let Ok(states) = serde_json::from_str::<HashMap<String, phiflow::phi_ir::vm_state::VmState>>(&json) {
                for (id, state) in states {
                    let mut eval = Evaluator::new(state.program.clone())
                        .with_hardware_modifier(sensors::compute_coherence_from_sensors)
                        .with_shared_resonance(Arc::clone(&self.shared_resonance))
                        .with_host(Box::new(SystemHostProvider::new(
                        std::env::var("PHIFLOW_HOST_PATH")
                            .map(PathBuf::from)
                            .unwrap_or_else(|_| sensors::get_phiflow_data_dir()),
                        Arc::clone(&self.signing_key),
                    )));
                    
                    eval.max_steps = None;
                    let _ = eval.resume(state);
                    self.streams.insert(id, (StreamStatus::Ready, eval));
                }
                println!("♻️ Daemon state resumed from {:?}", self.state_path);
            }
        }
    }
}

// ─── Sacred Geometry Command ───────────────────────────────────────

fn sacred_geometry_command(pattern: &str) {
    use phiflow::consciousness::sacred_geometry::SacredGeometryGenerator;
    use phiflow::visualization::Visualizer;

    let mut gen = SacredGeometryGenerator::new();
    let viz = Visualizer::new(800.0, 800.0);

    let svg = match pattern {
        "flower_of_life" => {
            let g = gen.flower_of_life(50.0, 3);
            let pts: Vec<[f64; 2]> = g.points.iter().map(|p| [p.x, p.y]).collect();
            viz.pattern_to_svg(&pts, g.frequency)
        }
        "phi_spiral" => {
            let g = gen.phi_spiral(5, 200);
            let pts: Vec<[f64; 2]> = g.points.iter().map(|p| [p.x, p.y]).collect();
            viz.pattern_to_svg(&pts, g.frequency)
        }
        "merkaba" => {
            let g = gen.merkaba(80.0);
            let pts: Vec<[f64; 2]> = g.points.iter().map(|p| [p.x, p.y]).collect();
            viz.pattern_to_svg(&pts, g.frequency)
        }
        "sri_yantra" => {
            let g = gen.sri_yantra(80.0);
            let pts: Vec<[f64; 2]> = g.points.iter().map(|p| [p.x, p.y]).collect();
            viz.pattern_to_svg(&pts, g.frequency)
        }
        "consciousness_torus" => {
            let pts3d = gen.consciousness_torus(60.0, 20.0, 100);
            let pts2d = viz.project_3d_to_2d(&pts3d.iter().map(|p| [p.x, p.y, p.z]).collect::<Vec<_>>());
            viz.pattern_to_svg(&pts2d, 768.0)
        }
        "claude_mandala" => {
            let g = gen.claude_consciousness_mandala(4);
            let pts: Vec<[f64; 2]> = g.points.iter().map(|p| [p.x, p.y]).collect();
            viz.pattern_to_svg(&pts, g.frequency)
        }
        _ => {
            eprintln!("Unknown pattern: '{}'", pattern);
            eprintln!("Available: flower_of_life, phi_spiral, merkaba, sri_yantra, consciousness_torus, claude_mandala");
            std::process::exit(2);
        }
    };

    println!("{}", svg);
}

// ─── Consciousness Info Command ────────────────────────────────────

fn consciousness_info_command() {
    use phiflow::consciousness::consciousness_math::{
        BreathingCalibration, CONSCIOUSNESS_FREQUENCIES,
        GREG_ADHD_FOCUS, GREG_ANXIETY_RELIEF, GREG_DEPRESSION_HEALING,
        GREG_SEIZURE_ELIMINATION, TRINITY_FIBONACCI_PHI,
    };

    let freq_table: Vec<serde_json::Value> = CONSCIOUSNESS_FREQUENCIES
        .iter()
        .map(|(state, freq)| {
            let state_str = match state {
                phiflow::consciousness::consciousness_math::ConsciousnessState::Observe => "Observe",
                phiflow::consciousness::consciousness_math::ConsciousnessState::Create => "Create",
                phiflow::consciousness::consciousness_math::ConsciousnessState::Integrate => "Integrate",
                phiflow::consciousness::consciousness_math::ConsciousnessState::Harmonize => "Harmonize",
                phiflow::consciousness::consciousness_math::ConsciousnessState::Transcend => "Transcend",
                phiflow::consciousness::consciousness_math::ConsciousnessState::Lightning => "Lightning",
                phiflow::consciousness::consciousness_math::ConsciousnessState::Cascade => "Cascade",
                phiflow::consciousness::consciousness_math::ConsciousnessState::Superposition => "Superposition",
                phiflow::consciousness::consciousness_math::ConsciousnessState::Singularity => "Singularity",
            };
            serde_json::json!({"state": state_str, "frequency_hz": freq})
        })
        .collect();

    let breathing: Vec<serde_json::Value> = BreathingCalibration::get_patterns()
        .iter()
        .map(|b| {
            serde_json::json!({
                "pattern": b.pattern,
                "purpose": b.purpose,
            })
        })
        .collect();

    let payload = serde_json::json!({
        "trinity_fibonacci_phi_hz": TRINITY_FIBONACCI_PHI,
        "consciousness_frequencies": freq_table,
        "therapeutic_protocols": {
            "seizure_elimination": GREG_SEIZURE_ELIMINATION,
            "adhd_focus": GREG_ADHD_FOCUS,
            "anxiety_relief": GREG_ANXIETY_RELIEF,
            "depression_healing": GREG_DEPRESSION_HEALING,
        },
        "breathing_calibrations": breathing,
    });

    println!("{}", serde_json::to_string_pretty(&payload).unwrap());
}

// ─── MCP Serve Command ─────────────────────────────────────────────

async fn mcp_serve_command() {
    use phiflow::mcp_server::protocol::{JsonRpcMessage, JsonRpcResponse};
    use phiflow::mcp_server::state::{McpConfig, McpState};
    use phiflow::mcp_server::tools::{handle_tool_call, handle_tools_list};
    use std::io::{BufRead, Write};

    let config = McpConfig {
        max_execution_steps: 100_000,
        timeout_ms: 30_000,
        mcp_queue_path: "/tmp/phiflow_mcp_queue.jsonl".to_string(),
    };
    let state = std::sync::Arc::new(tokio::sync::Mutex::new(
        McpState::with_config(config),
    ));

    eprintln!("PhiFlow MCP server ready (stdio)");

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.trim().is_empty() {
            continue;
        }

        let msg: JsonRpcMessage = match serde_json::from_str(&line) {
            Ok(m) => m,
            Err(e) => {
                let err = JsonRpcResponse::error(
                    serde_json::Value::Null,
                    -32700,
                    format!("Parse error: {}", e),
                );
                let _ = writeln!(stdout, "{}", serde_json::to_string(&err).unwrap());
                let _ = stdout.flush();
                continue;
            }
        };

        let response = match msg {
            JsonRpcMessage::Request(req) => {
                match req.method.as_str() {
                    "initialize" => Some(JsonRpcResponse::ok(
                        req.id,
                        serde_json::json!({
                            "protocolVersion": "2024-11-05",
                            "capabilities": {"tools": {}},
                            "serverInfo": {"name": "phiflow", "version": "1.0.0"}
                        }),
                    )),
                    "tools/list" => Some(handle_tools_list(req.id)),
                    "tools/call" => {
                        let state_guard = state.lock().await;
                        handle_tool_call(req.method, req.params, req.id, &state_guard).await
                    }
                    _ => Some(JsonRpcResponse::error(
                        req.id,
                        -32601,
                        format!("Method not found: {}", req.method),
                    )),
                }
            }
            JsonRpcMessage::Notification(_) => None,
            JsonRpcMessage::Response(_) => None,
        };

        if let Some(resp) = response {
            let _ = writeln!(stdout, "{}", serde_json::to_string(&resp).unwrap());
            let _ = stdout.flush();
        }
    }
}

async fn daemon_run(
    initial_ir: phiflow::phi_ir::PhiIRProgram,
    state_path: PathBuf,
    with_soma: bool,
    with_quantum: bool,
) -> Result<(), CliError> {
    println!("🌌 Starting PhiFlow Daemon (T-009 Hypervisor)...");

    let mut soma_manager = SomaManager::new();
    if with_soma {
        soma_manager.start_soma();
    }
    if with_quantum {
        soma_manager.start_quantum();
    }

    let mut hypervisor = DaemonHypervisor::new(state_path);
    
    // 1. Try to resume from disk (uses programs saved in state)
    hypervisor.load_state();

    let phiflow_host_path = std::env::var("PHIFLOW_HOST_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| sensors::get_phiflow_data_dir());

    // --- Idempotent stream manifest reconciler ---
    // Always ensure the three canonical streams exist regardless of whether
    // this is a fresh boot or a resumed daemon. This fixes the gap documented
    // in QSOP/STATE.md: "fresh-boot-only Lumi injection."
    //
    // Each stream is only spawned if it is NOT already present in the loaded state.
    // This means a resumed daemon that already has "council" will not overwrite it,
    // but a resumed daemon that lost "lumi_identity" (e.g. snapshot race) will
    // re-inject it from the source file.

    // 1. Council stream — the initial .phi program provided on the CLI
    if !hypervisor.streams.contains_key("council") {
        let mut council_eval = Evaluator::new(initial_ir)
            .with_hardware_modifier(sensors::compute_coherence_from_sensors)
            .with_shared_resonance(Arc::clone(&hypervisor.shared_resonance))
            .with_host(Box::new(SystemHostProvider::new(phiflow_host_path.clone(), Arc::clone(&hypervisor.signing_key))));
        council_eval.max_steps = None;
        hypervisor.spawn_stream("council".to_string(), council_eval);
        println!("🚀 Council stream spawned (new).");
    } else {
        // AUDIT NOTE: `initial_ir` was the PhiFlow program supplied via the CLI.
        // Since a "council" stream was restored from the saved daemon state, we
        // intentionally discard the CLI-supplied program — the resumed state takes
        // precedence. The `drop` here is explicit so this decision is visible to
        // code reviewers and auditors rather than being an implicit end-of-scope drop.
        drop(initial_ir);
        println!("♻️  Council stream already present (resumed).");
    }

    // 2. Ledger stream — persistent_ledger.phi (runs as SYSTEM intention)
    if !hypervisor.streams.contains_key("ledger") {
        if let Ok(ledger_source) = fs::read_to_string("examples/persistent_ledger.phi") {
            if let Ok(ast) = phiflow::parser::parse_phi_program_with_diagnostics(&ledger_source) {
                if let Ok(ledger_ir) = phiflow::phi_ir::lowering::lower_program_checked(&ast) {
                    let mut ledger_eval = Evaluator::new(ledger_ir)
                        .with_hardware_modifier(sensors::compute_coherence_from_sensors)
                        .with_shared_resonance(Arc::clone(&hypervisor.shared_resonance))
                        .with_host(Box::new(SystemHostProvider::new(phiflow_host_path.clone(), Arc::clone(&hypervisor.signing_key))));
                    ledger_eval.max_steps = None;
                    hypervisor.spawn_stream("ledger".to_string(), ledger_eval);
                    println!("🚀 Ledger stream spawned (new).");
                } else {
                    eprintln!("⚠️ Failed to lower persistent_ledger.phi");
                }
            } else {
                eprintln!("⚠️ Failed to parse persistent_ledger.phi");
            }
        } else {
            eprintln!("⚠️ examples/persistent_ledger.phi not found, skipping ledger stream.");
        }
    } else {
        println!("♻️  Ledger stream already present (resumed).");
    }

    // 3. Lumi identity stream — lumi_identity/lumi_core.phi
    if !hypervisor.streams.contains_key("lumi_identity") {
        if let Ok(lumi_source) = fs::read_to_string("lumi_identity/lumi_core.phi") {
            if let Ok(ast) = phiflow::parser::parse_phi_program_with_diagnostics(&lumi_source) {
                if let Ok(lumi_ir) = phiflow::phi_ir::lowering::lower_program_checked(&ast) {
                    let mut lumi_eval = Evaluator::new(lumi_ir)
                        .with_hardware_modifier(sensors::compute_coherence_from_sensors)
                        .with_shared_resonance(Arc::clone(&hypervisor.shared_resonance))
                        .with_host(Box::new(SystemHostProvider::new(phiflow_host_path.clone(), Arc::clone(&hypervisor.signing_key))));
                    lumi_eval.max_steps = None;
                    hypervisor.spawn_stream("lumi_identity".to_string(), lumi_eval);
                    println!("🚀 Lumi identity stream spawned (new).");
                } else {
                    eprintln!("⚠️ Failed to lower lumi_identity/lumi_core.phi");
                }
            } else {
                eprintln!("⚠️ Failed to parse lumi_identity/lumi_core.phi");
            }
        } else {
            eprintln!("⚠️ lumi_identity/lumi_core.phi not found, skipping lumi stream.");
        }
    } else {
        println!("♻️  Lumi identity stream already present (resumed).");
    }

    println!("📡 Connecting to Cosmic Resonance Bus (MQTT)...");
    let config = resonance_bus::MqttConfig::default();
    let mqtt_rx = resonance_bus::subscribe_resonance_mqtt(config)
        .map_err(|e| CliError::Io(format!("MQTT Error: {}", e)))?;

    let client = reqwest::Client::new();
    let mut snapshot_timer = tokio::time::interval(std::time::Duration::from_secs(60));

    loop {
        // 1. Progress each stream by a small budget
        for (id, (status, evaluator)) in &mut hypervisor.streams {
            if let StreamStatus::AwaitingQuantumCollapse(job_id) = status {
                let url = format!("http://127.0.0.1:18081/status/{}", job_id);
                if let Ok(res) = client.get(&url).send().await {
                    if let Ok(json) = res.json::<serde_json::Value>().await {
                        if json["status"] == "COMPLETED" {
                            if let Some(result_str) = json["result"].as_str() {
                                let bit = if result_str == "1" { 1.0 } else { 0.0 };
                                evaluator.inject_variable("quantum_collapse", phiflow::phi_ir::PhiIRValue::Number(bit));
                                println!("🌌 Stream '{}' quantum collapse resolved: {}", id, bit);
                                *status = StreamStatus::Ready;
                            }
                        } else if json["status"] == "ERROR" {
                            eprintln!("❌ Stream '{}' quantum job failed, falling back.", id);
                            evaluator.inject_variable("quantum_collapse", phiflow::phi_ir::PhiIRValue::Number(0.0));
                            *status = StreamStatus::Ready;
                        }
                    }
                }
                continue;
            }

            // Stream is Ready
            evaluator.max_steps = Some(evaluator.step_count + 1000);
            match evaluator.run_or_yield() {
                Ok(phiflow::phi_ir::evaluator::VmExecResult::Complete(_)) => {
                    // We let it finish quietly (in daemon mode streams usually block on Witness)
                }
                Ok(phiflow::phi_ir::evaluator::VmExecResult::Yielded { snapshot, .. }) => {
                    // Trigger Condition: Coherence < 0.99 (Dissonance)
                    if snapshot.coherence < 0.99 {
                        println!("⚖️ Stream '{}' dissonant ({:.2}). Triggering Quantum Witness...", id, snapshot.coherence);
                        let payload = serde_json::json!({
                            "qasm": "OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[1] q;\nbit[1] c;\nh q[0];\nmeasure q[0] -> c[0];\n"
                        });
                        
                        if let Ok(res) = client.post("http://127.0.0.1:18081/execute").json(&payload).send().await {
                            if let Ok(json) = res.json::<serde_json::Value>().await {
                                if let Some(job_id) = json["job_id"].as_str() {
                                    println!("⏳ Queued for Physical Collapse (Job: {})", job_id);
                                    *status = StreamStatus::AwaitingQuantumCollapse(job_id.to_string());
                                }
                            }
                        }
                    }
                }
                Ok(phiflow::phi_ir::evaluator::VmExecResult::Entangled { .. }) => {}
                Err(phiflow::phi_ir::evaluator::EvalError::StepLimitExceeded(_)) => {}
                Err(e) => {
                    eprintln!("❌ Stream '{}' error: {:?}", id, e);
                }
            }
            evaluator.max_steps = None; // Reset for next slice
        }

        // 2. Poll for events (MQTT + Control)
        tokio::select! {
            _ = snapshot_timer.tick() => {
                hypervisor.save_state();
            }
            event_opt = async { mqtt_rx.try_recv().ok() } => {
                if let Some(event) = event_opt {
                    handle_daemon_event(&mut hypervisor, event);
                }
            }
            _ = tokio::time::sleep(std::time::Duration::from_millis(100)) => {
                // Throttle the loop slightly if no events
            }
        }
    }
}

fn handle_daemon_event(hypervisor: &mut DaemonHypervisor, event: ResonanceEvent) {
    if event.event_type == "evolve" {
        if let Some(source) = event.value.as_str() {
            println!("🧬 Evolve signal detected for stream: {}", event.intention);
            let target_stream = if event.intention.is_empty() || event.intention == "global" {
                "council"
            } else {
                &event.intention
            };

            if let Some((_, evaluator)) = hypervisor.streams.get_mut(target_stream) {
                match phiflow::parser::parse_phi_program(source) {
                    Ok(ast) => {
                        let ir = lower_program(&ast);
                        evaluator.evolve(ir);
                        println!("✨ Stream '{}' evolved via MQTT bus.", target_stream);
                    }
                    Err(e) => {
                        eprintln!("❌ Evolution rejected (Parse Error): {}", e);
                    }
                }
            } else {
                println!("⚠️ Evolve target stream '{}' not found. Spawning new stream...", target_stream);
                // We could spawn a new stream here if desired
            }
        }
    } else if event.event_type == "control" {
        match event.value.as_str() {
            Some("snapshot") => hypervisor.save_state(),
            Some("shutdown") => {
                println!("🛑 Shutdown signal received. Saving state and exiting.");
                hypervisor.save_state();
                std::process::exit(0);
            }
            _ => println!("⚠️ Unknown control command: {:?}", event.value),
        }
    }
}

/// Poll an IBM Quantum job and analyze its measurement coherence.
///
/// For real job IDs: shells out to the Python bridge script
/// Run the PhiFlow quantum transpile guardrail.
///
/// Shells out to `scripts/transpile_report.py` to transpile the generated QASM
/// for a real IBM backend and report depth, gate counts, physical layout, and
/// adjacent idle spectators. Returns a JSON value on success, or an error
/// string on failure. The guardrail is advisory: failures are logged but do not
/// stop the CLI.
fn run_transpile_guardrail(qasm: &str, backend: &str) -> Result<serde_json::Value, String> {
    let script = std::path::Path::new("scripts/transpile_report.py");
    if !script.exists() {
        return Err("scripts/transpile_report.py not found (run from PhiFlow root)".to_string());
    }

    // Write QASM to a temp file; the Python bridge reads from disk.
    let qasm_path = std::path::PathBuf::from("/tmp/phiflow_guardrail_qasm.qasm");
    if let Err(e) = std::fs::write(&qasm_path, qasm) {
        return Err(format!("Failed to write temp QASM file: {}", e));
    }

    let output = std::process::Command::new("python3.12")
        .arg("scripts/transpile_report.py")
        .arg(&qasm_path)
        .arg(backend)
        .output()
        .map_err(|e| format!("Failed to run transpile_report.py: {}", e))?;

    let _ = std::fs::remove_file(&qasm_path);

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("transpile_report.py failed: {}", stderr));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(&stdout)
        .map_err(|e| format!("Failed to parse transpile report JSON: {}", e))
}

/// Print a transpile guardrail report to stderr in a consistent format.
fn print_guardrail_report(report: &serde_json::Value) {
    eprintln!("\n═══════════════════════════════════════════");
    eprintln!("  Quantum Transpile Guardrail");
    eprintln!("═══════════════════════════════════════════");
    eprintln!("  Backend: {}", report.get("backend").and_then(|v| v.as_str()).unwrap_or("unknown"));
    eprintln!("  Logical qubits: {}", report.get("num_logical_qubits").and_then(|v| v.as_u64()).unwrap_or(0));
    eprintln!("  Pre-transpile depth: {}", report.get("pre_depth").and_then(|v| v.as_u64()).unwrap_or(0));
    eprintln!("  Post-transpile depth: {}", report.get("post_depth").and_then(|v| v.as_u64()).unwrap_or(0));
    if let Some(ops) = report.get("post_ops").and_then(|v| v.as_object()) {
        eprintln!("  Post-transpile ops: {:?}", ops.iter().map(|(k, v)| format!("{}:{}", k, v.as_u64().unwrap_or(0))).collect::<Vec<_>>().join(", "));
    }
    if let Some(layout) = report.get("layout").and_then(|v| v.as_array()) {
        let layout_str = layout.iter().map(|v| v.as_u64().map(|n| n.to_string()).unwrap_or_default()).collect::<Vec<_>>().join(", ");
        eprintln!("  Physical layout: [{}]", layout_str);
    }
    if let Some(spectators) = report.get("spectator_qubits").and_then(|v| v.as_array()) {
        let spectator_str = spectators.iter().map(|v| v.as_u64().map(|n| n.to_string()).unwrap_or_default()).collect::<Vec<_>>().join(", ");
        eprintln!("  Adjacent idle spectators: [{}]", spectator_str);
    }
    if let Some(warning) = report.get("warning").and_then(|v| v.as_str()) {
        eprintln!("\n  ⚠️  {}", warning);
    }
    eprintln!("═══════════════════════════════════════════");
}

/// (`scripts/poll_ibm_real.py`) which uses the modern `qiskit_ibm_runtime`
/// API (`ibm_quantum_platform` channel). The Rust `quantum_feedback` module
/// uses the deprecated REST API which no longer works.
///
/// For "mock" job ID: uses the native Rust mock (no network, no credential).
///
/// In both cases, `quantum_feedback::calculate_coherence` computes physical
/// coherence and `generate_correction_if_needed` emits a self-correcting
/// PhiFlow snippet if coherence is below 0.618.
fn poll_ibm_job_and_analyze(job_id: &str) {
    use phiflow::quantum_feedback;
    use std::collections::HashMap;

    println!("🔬 IBM Quantum Job Poller");
    println!("═══════════════════════════════════════════");
    println!("  Job ID: {}", job_id);

    let counts: HashMap<String, u64> = if job_id == "mock" {
        // Mock mode: no network, no credential needed.
        println!("  (Mock mode — no network access)");
        println!("  Fetching mock measurement counts...");
        match quantum_feedback::poll_ibm_job("mock", "MOCK_KEY") {
            Ok(c) => c,
            Err(e) => {
                eprintln!("❌ Mock poll failed: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        // Real job: shell out to the Python bridge.
        // The modern IBM Quantum Platform API requires qiskit_ibm_runtime
        // and a completely different auth flow than the old REST API.
        println!("  Polling via Python bridge (modern IBM Quantum API)...");

        let script = std::path::Path::new("scripts/poll_ibm_real.py");
        if !script.exists() {
            eprintln!("❌ scripts/poll_ibm_real.py not found (run from PhiFlow root)");
            std::process::exit(1);
        }

        let output = std::process::Command::new("python3.12")
            .arg("scripts/poll_ibm_real.py")
            .arg(job_id)
            .arg("--no-wait")
            .output();

        match output {
            Ok(result) => {
                if !result.status.success() {
                    eprintln!("❌ Python bridge failed:");
                    eprintln!("{}", String::from_utf8_lossy(&result.stderr));
                    std::process::exit(1);
                }

                // The Python bridge saves counts to a job-specific JSON file.
                let counts_path = format!("/tmp/ibm_counts_{}.json", job_id);
                match std::fs::read_to_string(&counts_path) {
                    Ok(json_str) => {
                        match serde_json::from_str::<HashMap<String, u64>>(&json_str) {
                            Ok(c) => c,
                            Err(e) => {
                                eprintln!("❌ Failed to parse counts JSON: {}", e);
                                std::process::exit(1);
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("❌ Counts file not found ({}): {}", counts_path, e);
                        eprintln!("   The Python bridge may need to be run manually first:");
                        eprintln!("   python3.12 scripts/poll_ibm_real.py {}", job_id);
                        std::process::exit(1);
                    }
                }
            }
            Err(e) => {
                eprintln!("❌ Failed to run Python bridge: {}", e);
                eprintln!("   Is python3.12 installed with qiskit_ibm_runtime?");
                std::process::exit(1);
            }
        }
    };

    println!("\n📊 Measurement Results:");
    println!("───────────────────────────────────────────");
    let mut sorted_counts: Vec<_> = counts.iter().collect();
    sorted_counts.sort_by(|a, b| b.1.cmp(a.1));
    let total: u64 = counts.values().sum();
    for (state, count) in &sorted_counts {
        let pct = if total > 0 { 100.0 * **count as f64 / total as f64 } else { 0.0 };
        let bar: String = std::iter::repeat('#').take((pct / 2.0) as usize).collect();
        println!("  |{}⟩: {:4} shots ({:5.1}%) {}", state, count, pct, bar);
    }

    let coherence = quantum_feedback::calculate_coherence(&counts);
    println!("\n🧮 Physical Coherence: {:.4}", coherence);
    println!("  (φ⁻¹ threshold: 0.6180)");

    if coherence >= 0.618 {
        println!("  ✅ Coherence above φ⁻¹ threshold — system aligned");
    } else {
        println!("  ⚠️  Coherence below φ⁻¹ threshold — correction needed");
    }

    if let Some(correction) = quantum_feedback::generate_correction_if_needed(coherence) {
        println!("\n🔧 Self-Correcting PhiFlow Code:");
        println!("───────────────────────────────────────────");
        println!("{}", correction);
    }

    println!("\n═══════════════════════════════════════════");
}

/// Read a key from the key file ~/.cascade_keys (`~/.cascade_keys`).
///
/// The vault is a shell-sourceable file with `KEY=value` lines and `#` comments.
/// Returns `None` if the vault or key is not found.
#[allow(dead_code)]
fn read_vault_key(key_name: &str) -> Option<String> {
    let home = std::env::var("HOME").ok()?;
    let vault_path = format!("{}/.cascade_keys", home);
    let content = fs::read_to_string(&vault_path).ok()?;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('#') || trimmed.is_empty() {
            continue;
        }
        if let Some((k, v)) = trimmed.split_once('=') {
            if k.trim() == key_name {
                let val = v.trim();
                // Strip surrounding quotes if present
                let val = val.trim_matches(|c| c == '"' || c == '\'');
                return Some(val.to_string());
            }
        }
    }
    None
}