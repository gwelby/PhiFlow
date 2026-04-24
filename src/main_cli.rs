use clap::Parser;
use phiflow::compile_to_openqasm_with_options;
use phiflow::parser::parse_phi_program_with_diagnostics;
use phiflow::phi_ir::evaluator::{Evaluator, VmExecResult};
use phiflow::phi_ir::lowering::{lower_program, lower_program_checked};
use phiflow::phi_ir::openqasm::OpenQasmEmitter;
use phiflow::phi_ir::quantum_codegen::compile_ir_to_quantum;
use phiflow::phi_ir::topology_transpiler::{RoutingStrategy, TopologyTranspileConfig};
use phiflow::quantum::ibm_quantum::IBMQuantumBackend;
use phiflow::quantum::{BackendTopologyProfile, QuantumBackend, QuantumConfig};
use phiflow::phi_ir::PhiIRValue;
use phiflow::resonance_bus::{self, ResonanceEvent};
use phiflow::sensors;
use phiflow::system_host::SystemHostProvider;
use phiflow::OpenQasmCompileOptions;
use phiflow::PhiDiagnostic;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// A command-line interpreter for the PhiFlow language.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The path to the .phi file to execute.
    #[arg(required = true)]
    file: PathBuf,

    /// Emit parse errors as a strict JSON array of PhiDiagnostic objects (for tooling).
    #[arg(long, default_value_t = false)]
    json_errors: bool,

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

    /// Run as a daemon, listening for evolve events.
    #[arg(long, default_value_t = false)]
    daemon: bool,

    /// Path to persist the daemon state to.
    #[arg(long, default_value = "D:\\CosmicFamily\\DAEMON_STATE.json")]
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
        let child = std::process::Command::new("python")
            .arg("D:/Projects/PhiHarmonic/SOMA/soma.py")
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

const APIKEY_PATH: &str = "apikey.json";

#[derive(Debug, Deserialize)]
struct IbmCredentials {
    apikey: String,
    service_crn: Option<String>,
    region: Option<String>,
}

struct RunReport {
    final_coherence: f64,
    resonance_events: Vec<(String, PhiIRValue)>,
    ended_streams: Vec<String>,
    _program: phiflow::phi_ir::PhiIRProgram,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    let json_errors = args.json_errors;

    match run(
        &args.file,
        json_errors,
        args.target.clone(),
        args.optimize_depth,
        args.topology_aware,
        args.topology_backend.clone(),
        args.daemon,
        args.state_path.clone(),
        args.handoff.clone(),
        args.with_soma,
        args.with_quantum,
        args.max_steps,
    )
    .await
    {
        Ok(Some(report)) => {
            if json_errors {
                println!("[]");
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
    target: Option<String>,
    optimize_depth: bool,
    topology_aware: bool,
    topology_backend: String,
    daemon: bool,
    state_path: PathBuf,
    handoff: Option<String>,
    with_soma: bool,
    with_quantum: bool,
    max_steps: usize,
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

    if json_errors {
        return Ok(Some(RunReport {
            final_coherence: 0.0,
            resonance_events: Vec::new(),
            ended_streams: Vec::new(),
            _program: ir_program,
        }));
    }

    if let Some(t) = target {
        match t.as_str() {
            "quantum" => {
                println!("Routing to Quantum Codegen Backend...");
                let circuit = compile_ir_to_quantum(&ir_program);
                println!("Generates Quantum Circuit:");
                println!("{:#?}", circuit);
                return Ok(None);
            }
            "openqasm" => {
                if topology_aware {
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
                    };
                    let qasm = compile_to_openqasm_with_options(&source, &options)
                        .map_err(CliError::Eval)?;
                    print!("{}", qasm);
                    return Ok(None);
                }

                let mut emitter = OpenQasmEmitter::new();
                emitter.optimize_depth = optimize_depth;
                let qasm = emitter
                    .emit(&ir_program)
                    .map_err(|e| CliError::Eval(e.to_string()))?;
                print!("{}", qasm);
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

    let mut evaluator = Evaluator::new(ir_program.clone())
        .with_hardware_modifier(sensors::compute_coherence_from_sensors);
    
    if max_steps > 0 {
        evaluator.max_steps = Some(max_steps);
    } else {
        evaluator.max_steps = Some(1_000_000_000);
    }

    let _result = evaluator.run().map_err(|e| CliError::Eval(e.to_string()))?;

    Ok(Some(RunReport {
        final_coherence: evaluator.resolved_coherence(),
        resonance_events: evaluator.resonance_events().to_vec(),
        ended_streams: evaluator.ended_streams().to_vec(),
        _program: ir_program,
    }))
}

fn load_ibm_quantum_config(backend_name: &str) -> Result<QuantumConfig, CliError> {
    let credentials_json = fs::read_to_string(APIKEY_PATH).map_err(|e| {
        CliError::Io(format!(
            "Failed to read `{APIKEY_PATH}` for topology-aware OpenQASM: {e}"
        ))
    })?;
    let credentials: IbmCredentials = serde_json::from_str(&credentials_json).map_err(|e| {
        CliError::Io(format!(
            "Failed to parse `{APIKEY_PATH}` for topology-aware OpenQASM: {e}"
        ))
    })?;
    let service_crn = credentials.service_crn.clone().ok_or_else(|| {
        CliError::Io(
            "Topology-aware IBM compilation requires `service_crn` in apikey.json".to_string(),
        )
    })?;

    Ok(QuantumConfig {
        backend_name: backend_name.to_string(),
        api_token: Some(credentials.apikey),
        service_crn: Some(service_crn),
        region: credentials.region,
        hub: None,
        group: None,
        project: None,
        max_qubits: 156,
        shots: 1024,
        timeout_seconds: 300,
    })
}

async fn fetch_live_topology_profile(
    backend_name: &str,
) -> Result<BackendTopologyProfile, CliError> {
    let config = load_ibm_quantum_config(backend_name)?;
    let mut backend = IBMQuantumBackend::with_backend(backend_name.to_string());
    backend
        .initialize(config)
        .await
        .map_err(|e| CliError::Eval(format!("Failed to initialize IBM backend: {e}")))?;
    backend
        .fetch_topology_profile()
        .await
        .map_err(|e| CliError::Eval(format!("Failed to fetch backend topology profile: {e}")))
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
                        .with_host(Box::new(SystemHostProvider::new(PathBuf::from("D:\\CosmicFamily"), Arc::clone(&self.signing_key))));
                    
                    eval.max_steps = None;
                    let _ = eval.resume(state);
                    self.streams.insert(id, (StreamStatus::Ready, eval));
                }
                println!("♻️ Daemon state resumed from {:?}", self.state_path);
            }
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
            .with_host(Box::new(SystemHostProvider::new(PathBuf::from("D:\\CosmicFamily"), Arc::clone(&hypervisor.signing_key))));
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
                        .with_host(Box::new(SystemHostProvider::new(PathBuf::from("D:\\CosmicFamily"), Arc::clone(&hypervisor.signing_key))));
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
                        .with_host(Box::new(SystemHostProvider::new(PathBuf::from("D:\\CosmicFamily"), Arc::clone(&hypervisor.signing_key))));
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
