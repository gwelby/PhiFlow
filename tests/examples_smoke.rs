//! Smoke test: every `examples/*.phi` must run under `phic` without error.
//!
//! Added 2026-11-11 after three examples were found broken by hand
//! (`field_guardian_v2.phi` never stopped, `memory_demo.phi` compared Void,
//! and `cognitive_dissonance.phi` is a quantum-target fixture that plain `phic`
//! cannot run). Nothing ran the examples automatically before this.
//!
//! - Finite examples must exit 0 within `FINITE_LIMIT`.
//! - Long-running examples (daemons and long paced loops, see
//!   examples/README.md) must not fail during `DAEMON_WINDOW`; still running
//!   at the end of the window counts as a pass and the process is killed.
//!
//! Runs with no live SOMA (state path points at a missing file) and private
//! data/state paths in a temp dir, so it never touches the real daemon state.
//! Examples run in parallel, `PARALLEL` at a time.

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const FINITE_LIMIT: Duration = Duration::from_secs(45);
const DAEMON_WINDOW: Duration = Duration::from_secs(4);
const PARALLEL: usize = 4;

/// Examples that legitimately run longer than `FINITE_LIMIT` (or forever).
/// Keep in sync with the "Long-running programs" table in examples/README.md.
const LONG_RUNNING: &[&str] = &[
    "CODEX_BEAUTIFUL_ABSURDITY",
    "p2_quantum_witness",
    "council_daemon",
    "council_engage",
    "persistent_ledger",
    "p1_soma_bridge",
    "soma_reality_bridge",
    "type4_r5_probe",
    "type4_soma_daemon",
    "type4_soma_expressive",
    "type4_soma_motor",
    "type4_esig_clock_trace",
    "type4_esig_trace",
];

/// Quantum-target fixtures: valid only on the quantum overlay path (e.g. 2-arg
/// `coherence(a, b)`), so plain `phic` cannot run them. Covered by
/// tests/topology_transpiler_tests.rs and tests/ibm_hardware_runner.rs instead.
const QUANTUM_FIXTURES: &[&str] = &["cognitive_dissonance"];

#[derive(Debug)]
enum Outcome {
    Exited { code: Option<i32>, secs: f64 },
    StillRunning,
    TimedOut,
}

fn wait_with_limit(child: &mut Child, limit: Duration) -> Outcome {
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                return Outcome::Exited {
                    code: status.code(),
                    secs: start.elapsed().as_secs_f64(),
                }
            }
            Ok(None) if start.elapsed() >= limit => {
                let _ = child.kill();
                let _ = child.wait();
                return Outcome::TimedOut;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(50)),
            Err(e) => panic!("wait failed: {e}"),
        }
    }
}

fn run_one(phic: &Path, example: &Path, work: &Path) -> Result<(), String> {
    let stem = example.file_stem().unwrap().to_string_lossy().to_string();
    let long = LONG_RUNNING.contains(&stem.as_str());
    let out = std::fs::File::create(work.join(format!("{stem}.out"))).unwrap();
    let err_path = work.join(format!("{stem}.err"));
    let err = std::fs::File::create(&err_path).unwrap();
    let mut child = Command::new(phic)
        .arg(example)
        .env("SOMA_STATE_PATH", work.join("absent_soma.json"))
        .env("XDG_DATA_HOME", work.join("xdg"))
        .env("PHIFLOW_DAEMON_STATE_PATH", work.join(format!("{stem}_daemon.json")))
        .env("PHIFLOW_QUANTUM_STATE_PATH", work.join("absent_quantum.json"))
        .env("PHIFLOW_CLAIMS_STATE_PATH", work.join("absent_claims.json"))
        .env_remove("PHIFLOW_STRICT_SENSORS")
        .env("CUDA_VISIBLE_DEVICES", "")
        .stdin(Stdio::null())
        .stdout(out)
        .stderr(err)
        .spawn()
        .map_err(|e| format!("{stem}: spawn failed: {e}"))?;

    let limit = if long { DAEMON_WINDOW } else { FINITE_LIMIT };
    let outcome = match wait_with_limit(&mut child, limit) {
        Outcome::TimedOut if long => Outcome::StillRunning,
        other => other,
    };
    let tail = || {
        let text = std::fs::read_to_string(&err_path).unwrap_or_default();
        let lines: Vec<&str> = text.lines().collect();
        lines[lines.len().saturating_sub(5)..].join(" | ")
    };
    match outcome {
        Outcome::Exited { code: Some(0), .. } | Outcome::StillRunning => Ok(()),
        Outcome::Exited { code, secs } => Err(format!(
            "{stem}: exited {code:?} after {secs:.1}s; stderr tail: {}",
            tail()
        )),
        Outcome::TimedOut => Err(format!(
            "{stem}: still running after {}s (finite example). If it is a daemon by \
             design, add it to LONG_RUNNING and to examples/README.md",
            FINITE_LIMIT.as_secs()
        )),
    }
}

#[test]
fn every_example_runs() {
    let phic = PathBuf::from(env!("CARGO_BIN_EXE_phic"));
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut examples: Vec<PathBuf> = std::fs::read_dir(root.join("examples"))
        .expect("examples dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map_or(false, |x| x == "phi"))
        .collect();
    examples.sort();
    assert!(examples.len() > 50, "found only {} examples", examples.len());

    examples.retain(|p| {
        !QUANTUM_FIXTURES.contains(&p.file_stem().unwrap().to_string_lossy().as_ref())
    });
    for name in LONG_RUNNING.iter().chain(QUANTUM_FIXTURES) {
        let path = root.join("examples").join(format!("{name}.phi"));
        assert!(path.exists(), "{name}.phi is listed in the smoke test but does not exist");
    }
    for name in LONG_RUNNING {
        assert!(
            examples.iter().any(|p| p.file_stem().unwrap() == *name),
            "LONG_RUNNING lists {name}, which no longer exists"
        );
    }

    let work = tempfile::tempdir().expect("tempdir");
    let queue = Arc::new(Mutex::new(examples.clone()));
    let failures = Arc::new(Mutex::new(Vec::<String>::new()));
    let workers: Vec<_> = (0..PARALLEL)
        .map(|_| {
            let queue = Arc::clone(&queue);
            let failures = Arc::clone(&failures);
            let phic = phic.clone();
            let work = work.path().to_path_buf();
            std::thread::spawn(move || loop {
                let next = queue.lock().unwrap().pop();
                let Some(example) = next else { break };
                if let Err(e) = run_one(&phic, &example, &work) {
                    failures.lock().unwrap().push(e);
                }
            })
        })
        .collect();
    for w in workers {
        w.join().expect("worker panicked");
    }

    let mut failures = failures.lock().unwrap().clone();
    failures.sort();
    assert!(
        failures.is_empty(),
        "{} of {} examples failed:\n  {}",
        failures.len(),
        examples.len(),
        failures.join("\n  ")
    );
}
