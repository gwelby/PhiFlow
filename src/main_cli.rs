use clap::Parser;
use phiflow::parser::parse_phi_program_with_diagnostics;
use phiflow::phi_ir::evaluator::{Evaluator, VmExecResult};
use phiflow::phi_ir::lowering::{lower_program, lower_program_checked};
use phiflow::phi_ir::openqasm::OpenQasmEmitter;
use phiflow::phi_ir::quantum_codegen::compile_ir_to_quantum;
use phiflow::phi_ir::PhiIRValue;
use phiflow::resonance_bus::{self, ResonanceEvent};
use phiflow::sensors;
use phiflow::system_host::SystemHostProvider;
use phiflow::PhiDiagnostic;
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
}

struct SomaManager {
    child: Option<std::process::Child>,
}

impl SomaManager {
    fn new() -> Self {
        Self { child: None }
    }

    fn start(&mut self) {
        println!("🌀 Starting SOMA Sensor Suite (Python)...");
        let child = std::process::Command::new("python")
            .arg("D:/Projects/PhiHarmonic/SOMA/soma.py")
            .arg("--phiflow")
            .arg("--headless")
            .spawn();

        match child {
            Ok(c) => {
                println!("✅ SOMA Process started (PID: {})", c.id());
                self.child = Some(c);
            }
            Err(e) => {
                eprintln!("❌ Failed to start SOMA: {}", e);
            }
        }
    }

    fn stop(&mut self) {
        if let Some(mut child) = self.child.take() {
            println!("🛑 Stopping SOMA Process...");
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
    program: phiflow::phi_ir::PhiIRProgram,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    match run(
        &args.file,
        args.json_errors,
        args.target,
        args.optimize_depth,
        args.daemon,
        args.state_path,
        args.handoff,
        args.with_soma,
        args.max_steps,
    )
    .await
    {
        Ok(Some(report)) => {
            if args.json_errors {
                println!("[]");
                std::process::exit(0);
            }

            for (_scope, value) in &report.resonance_events {
                match value {
                    PhiIRValue::Number(n) => {
                        println!("🔔 Resonating Field: {:.4}Hz", n);
                    }
                    PhiIRValue::String(idx) => {
                        let s = report.program.string_table.get(*idx as usize).cloned().unwrap_or_else(|| format!("_str_{}", idx));
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
            if args.json_errors {
                let payload = vec![diag];
                let _ = serde_json::to_string(&payload).map(|json| println!("{}", json));
                std::process::exit(2);
            }
            eprintln!("{}", diag);
            std::process::exit(2);
        }
        Err(CliError::Io(msg)) => {
            if args.json_errors {
                println!("[]");
            }
            eprintln!("Error: {}", msg);
            std::process::exit(1);
        }
        Err(CliError::Eval(msg)) => {
            if args.json_errors {
                println!("[]");
            }
            eprintln!("Runtime error: {}", msg);
            std::process::exit(1);
        }
        Err(CliError::Lower(msg)) => {
            if args.json_errors {
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
    daemon: bool,
    state_path: PathBuf,
    handoff: Option<String>,
    with_soma: bool,
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
            program: ir_program,
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
        daemon_run(ir_program, state_path, with_soma).await?;
        return Ok(None);
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
        program: ir_program,
    }))
}

struct DaemonHypervisor<'a> {
    streams: HashMap<String, Evaluator<'a>>,
    shared_resonance: Arc<Mutex<HashMap<String, Vec<PhiIRValue>>>>,
    state_path: PathBuf,
}

impl<'a> DaemonHypervisor<'a> {
    fn new(state_path: PathBuf) -> Self {
        Self {
            streams: HashMap::new(),
            shared_resonance: Arc::new(Mutex::new(HashMap::new())),
            state_path,
        }
    }

    fn spawn_stream(&mut self, id: String, evaluator: Evaluator<'a>) {
        self.streams.insert(id, evaluator);
    }

    fn save_state(&self) {
        let mut states = HashMap::new();
        for (id, eval) in &self.streams {
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
                        .with_host(Box::new(SystemHostProvider::new(PathBuf::from("D:\\CosmicFamily"))));
                    
                    eval.max_steps = None;
                    let _ = eval.resume(state);
                    self.streams.insert(id, eval);
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
) -> Result<(), CliError> {
    println!("🌌 Starting PhiFlow Daemon (T-009 Hypervisor)...");

    let mut soma_manager = SomaManager::new();
    if with_soma {
        soma_manager.start();
    }

    let mut hypervisor = DaemonHypervisor::new(state_path);
    
    // 1. Try to resume from disk (uses programs saved in state)
    hypervisor.load_state();

    // 2. If no streams resumed, create the initial "Council" stream from provided IR
    if hypervisor.streams.is_empty() {
        let mut council_eval = Evaluator::new(initial_ir)
            .with_hardware_modifier(sensors::compute_coherence_from_sensors)
            .with_shared_resonance(Arc::clone(&hypervisor.shared_resonance))
            .with_host(Box::new(SystemHostProvider::new(PathBuf::from("D:\\CosmicFamily"))));
        
        // Daemon mode has no step limit
        council_eval.max_steps = None;
        hypervisor.spawn_stream("council".to_string(), council_eval);
        println!("🚀 Initial Council stream spawned.");

        // Spawn the persistent ledger stream
        if let Ok(ledger_source) = fs::read_to_string("examples/persistent_ledger.phi") {
            if let Ok(ast) = phiflow::parser::parse_phi_program_with_diagnostics(&ledger_source) {
                if let Ok(ledger_ir) = phiflow::phi_ir::lowering::lower_program_checked(&ast) {
                    let mut ledger_eval = Evaluator::new(ledger_ir)
                        .with_hardware_modifier(sensors::compute_coherence_from_sensors)
                        .with_shared_resonance(Arc::clone(&hypervisor.shared_resonance))
                        .with_host(Box::new(SystemHostProvider::new(PathBuf::from("D:\\CosmicFamily"))));
                    ledger_eval.max_steps = None;
                    hypervisor.spawn_stream("ledger".to_string(), ledger_eval);
                    println!("🚀 Initial Ledger stream spawned.");
                } else {
                    eprintln!("⚠️ Failed to lower persistent_ledger.phi");
                }
            } else {
                eprintln!("⚠️ Failed to parse persistent_ledger.phi");
            }
        } else {
            eprintln!("⚠️ examples/persistent_ledger.phi not found, skipping ledger stream.");
        }
    }

    println!("📡 Connecting to Cosmic Resonance Bus (MQTT)...");
    let config = resonance_bus::MqttConfig::default();
    let mqtt_rx = resonance_bus::subscribe_resonance_mqtt(config)
        .map_err(|e| CliError::Io(format!("MQTT Error: {}", e)))?;

    let mut snapshot_timer = tokio::time::interval(std::time::Duration::from_secs(60));

    loop {
        // 1. Progress each stream by a small budget
        for (id, evaluator) in &mut hypervisor.streams {
            // Run a small slice of instructions to remain responsive
            evaluator.max_steps = Some(evaluator.step_count + 1000);
            match evaluator.run_or_yield() {
                Ok(VmExecResult::Complete(_)) => {
                    println!("🌊 Stream '{}' completed normally.", id);
                }
                Ok(VmExecResult::Yielded { .. }) | Ok(VmExecResult::Entangled { .. }) => {
                    // Normal yield, continue next loop
                }
                Err(phiflow::phi_ir::evaluator::EvalError::StepLimitExceeded(_)) => {
                    // Budget exhausted, normal for daemon
                }
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

            if let Some(evaluator) = hypervisor.streams.get_mut(target_stream) {
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
