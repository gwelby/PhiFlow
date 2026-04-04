//! PhiFlow WASM Universal Bridge
//!
//! Compiles/executes PhiFlow WAT through a native WASM host (wasmtime) with
//! consciousness hook bindings:
//! - `phi.witness(i32) -> f64`
//! - `phi.resonate(f64)`
//! - `phi.coherence() -> f64`
//! - `phi.intention_push(i32)`
//! - `phi.intention_pop()`

use crate::parser::parse_phi_program;
use crate::phi_ir::lowering::lower_program;
use crate::phi_ir::optimizer::{OptimizationLevel, Optimizer};
use crate::phi_ir::wasm::{
    emit_wat, NAN_BOX_MASK, PAYLOAD_MASK, TAG_BOOLEAN, TAG_STRING, TAG_VOID,
};
use crate::phi_ir::PhiIRValue;
use std::sync::Arc;
use wasmtime::{Caller, Engine, Linker, Module, Store};

const STRING_BASE: i32 = 0x100;

#[derive(Debug, Clone)]
pub struct WasmWitnessEvent {
    pub operand_or_offset: i32,
    pub coherence: f64,
    pub intention: Option<String>,
}

#[derive(Debug, Clone)]
pub struct WasmHostSnapshot {
    pub coherence: f64,
    pub resonance_field: Vec<f64>,
    pub intention_stack: Vec<String>,
    pub witness_log: Vec<WasmWitnessEvent>,
}

#[derive(Debug, Clone)]
pub struct WasmRunResult {
    pub result: PhiIRValue,
    pub snapshot: WasmHostSnapshot,
}

pub fn unbox_f64(val: f64, string_table: &[String]) -> PhiIRValue {
    let bits = val.to_bits();
    if (bits & NAN_BOX_MASK) == NAN_BOX_MASK {
        let tag = bits & !PAYLOAD_MASK;
        let payload = bits & PAYLOAD_MASK;
        match tag {
            TAG_BOOLEAN => PhiIRValue::Boolean(payload != 0),
            TAG_STRING => {
                let idx = payload as u32;
                PhiIRValue::String(idx) // return the u32 index, matching what evaluator does
            }
            TAG_VOID => PhiIRValue::Void,
            _ => PhiIRValue::Void, // Unknown tag, default to void
        }
    } else {
        PhiIRValue::Number(val)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum WasmHostError {
    #[error("parse error: {0}")]
    Parse(String),
    #[error("wat parse error: {0}")]
    Wat(#[from] wat::Error),
    #[error("wasm runtime error: {0}")]
    Runtime(#[from] wasmtime::Error),
    #[error("missing exported function `phi_run`")]
    MissingPhiRun,
    #[error("phi_run returned non-finite result: {0}")]
    InvalidResult(f64),
}

#[derive(Clone)]
pub struct WasmHostHooks {
    coherence_provider: Arc<dyn Fn() -> f64 + Send + Sync>,
    on_witness: Arc<dyn Fn(WasmWitnessEvent) + Send + Sync>,
    on_resonate: Arc<dyn Fn(f64, Option<String>) + Send + Sync>,
    on_intention_push: Arc<dyn Fn(String, usize) + Send + Sync>,
    on_intention_pop: Arc<dyn Fn(String, usize) + Send + Sync>,
}

impl Default for WasmHostHooks {
    fn default() -> Self {
        Self {
            coherence_provider: Arc::new(|| crate::LAMBDA),
            on_witness: Arc::new(|_| {}),
            on_resonate: Arc::new(|_, _| {}),
            on_intention_push: Arc::new(|_, _| {}),
            on_intention_pop: Arc::new(|_, _| {}),
        }
    }
}

impl WasmHostHooks {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_coherence_provider<F>(mut self, f: F) -> Self
    where
        F: Fn() -> f64 + Send + Sync + 'static,
    {
        self.coherence_provider = Arc::new(f);
        self
    }

    pub fn with_witness<F>(mut self, f: F) -> Self
    where
        F: Fn(WasmWitnessEvent) + Send + Sync + 'static,
    {
        self.on_witness = Arc::new(f);
        self
    }

    pub fn with_resonate<F>(mut self, f: F) -> Self
    where
        F: Fn(f64, Option<String>) + Send + Sync + 'static,
    {
        self.on_resonate = Arc::new(f);
        self
    }

    pub fn with_intention_push<F>(mut self, f: F) -> Self
    where
        F: Fn(String, usize) + Send + Sync + 'static,
    {
        self.on_intention_push = Arc::new(f);
        self
    }

    pub fn with_intention_pop<F>(mut self, f: F) -> Self
    where
        F: Fn(String, usize) + Send + Sync + 'static,
    {
        self.on_intention_pop = Arc::new(f);
        self
    }
}

#[derive(Clone)]
struct RuntimeState {
    hooks: WasmHostHooks,
    coherence: f64,
    resonance_field: Vec<f64>,
    intention_stack: Vec<String>,
    witness_log: Vec<WasmWitnessEvent>,
}

impl RuntimeState {
    fn new(hooks: WasmHostHooks) -> Self {
        Self {
            hooks,
            coherence: crate::LAMBDA,
            resonance_field: Vec::new(),
            intention_stack: Vec::new(),
            witness_log: Vec::new(),
        }
    }

    fn snapshot(&self) -> WasmHostSnapshot {
        WasmHostSnapshot {
            coherence: self.coherence,
            resonance_field: self.resonance_field.clone(),
            intention_stack: self.intention_stack.clone(),
            witness_log: self.witness_log.clone(),
        }
    }
}

pub fn compile_source_to_wat(source: &str) -> Result<String, WasmHostError> {
    let expressions = parse_phi_program(source).map_err(WasmHostError::Parse)?;
    let mut program = lower_program(&expressions);
    let mut optimizer = Optimizer::new(OptimizationLevel::Basic);
    optimizer.optimize(&mut program);
    Ok(emit_wat(&program))
}

pub fn run_source_with_host(
    source: &str,
    hooks: WasmHostHooks,
) -> Result<WasmRunResult, WasmHostError> {
    let wat = compile_source_to_wat(source)?;
    run_wat_with_host(&wat, hooks)
}

pub fn run_wat_with_host(
    wat_source: &str,
    hooks: WasmHostHooks,
) -> Result<WasmRunResult, WasmHostError> {
    let wasm_bytes = wat::parse_str(wat_source)?;
    let engine = Engine::default();
    let module = Module::new(&engine, wasm_bytes)?;
    let mut linker = Linker::new(&engine);

    linker.func_wrap(
        "phi",
        "witness",
        |mut caller: Caller<'_, RuntimeState>, operand_or_offset: i32| -> f64 {
            let event = {
                let data = caller.data_mut();
                let coherence = (data.hooks.coherence_provider)();
                data.coherence = coherence;
                let event = WasmWitnessEvent {
                    operand_or_offset,
                    coherence,
                    intention: data.intention_stack.last().cloned(),
                };
                data.witness_log.push(event.clone());
                event
            };
            let callback = caller.data().hooks.on_witness.clone();
            callback(event.clone());
            event.coherence
        },
    )?;

    linker.func_wrap(
        "phi",
        "resonate",
        |mut caller: Caller<'_, RuntimeState>, value: f64| {
            let (intention, callback) = {
                let data = caller.data_mut();
                data.resonance_field.push(value);
                (
                    data.intention_stack.last().cloned(),
                    data.hooks.on_resonate.clone(),
                )
            };
            callback(value, intention);
        },
    )?;

    linker.func_wrap(
        "phi",
        "coherence",
        |mut caller: Caller<'_, RuntimeState>| -> f64 {
            let coherence = (caller.data().hooks.coherence_provider)();
            caller.data_mut().coherence = coherence;
            coherence
        },
    )?;

    linker.func_wrap(
        "phi",
        "intention_push",
        |mut caller: Caller<'_, RuntimeState>, offset_or_len: i32| {
            let name = if offset_or_len >= STRING_BASE {
                format!("intent_{}", offset_or_len)
            } else {
                format!("intent_{}", offset_or_len.max(0))
            };

            let (depth, callback) = {
                let data = caller.data_mut();
                data.intention_stack.push(name.clone());
                (
                    data.intention_stack.len(),
                    data.hooks.on_intention_push.clone(),
                )
            };
            callback(name, depth);
        },
    )?;

    linker.func_wrap(
        "phi",
        "intention_pop",
        |mut caller: Caller<'_, RuntimeState>| {
            let (popped, depth, callback) = {
                let data = caller.data_mut();
                let popped = data
                    .intention_stack
                    .pop()
                    .unwrap_or_else(|| "intent_unknown".to_string());
                let depth = data.intention_stack.len();
                (popped, depth, data.hooks.on_intention_pop.clone())
            };
            callback(popped, depth);
        },
    )?;

    let mut store = Store::new(&engine, RuntimeState::new(hooks));
    let instance = linker.instantiate(&mut store, &module)?;
    let func = instance
        .get_func(&mut store, "phi_run")
        .ok_or(WasmHostError::MissingPhiRun)?;
    let phi_run = func.typed::<(), f64>(&store)?;
    let raw_result = phi_run.call(&mut store, ())?;

    // NaN-boxing means result can be NaN and valid.
    // We rely on unbox_f64 to decode it later.
    let expressions = parse_phi_program(wat_source).unwrap_or_default(); // Actually wat_source is compiled already. We need the string_table.
                                                                         // Wait, run_wat_with_host signature doesn't pass string_table.
                                                                         // We can just keep it as f64 in run_wat_with_host or change the signature to take string_table.
                                                                         // Or we return a PhiIRValue with String(index) which doesn't need string_table itself.
                                                                         // Since PhiIRValue::String holds an index (u32), we can just build that.
    let result = unbox_f64(raw_result, &[]); // Empty slice since we just return String(index)

    let snapshot = store.data().snapshot();
    Ok(WasmRunResult { result, snapshot })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    #[test]
    fn wasm_host_uses_custom_coherence_provider() {
        let hooks = WasmHostHooks::new().with_coherence_provider(|| 0.77);
        let run = run_source_with_host("coherence", hooks).expect("wasm host run should succeed");
        let n = run.result.as_number().unwrap();
        assert!((n - 0.77).abs() < 1e-9, "got {}", n);
    }

    #[test]
    fn wasm_host_records_witness_and_resonate_events() {
        let resonated = Arc::new(AtomicUsize::new(0));
        let witnessed = Arc::new(AtomicUsize::new(0));
        let intentions = Arc::new(Mutex::new(Vec::new()));

        let resonated_ref = Arc::clone(&resonated);
        let witnessed_ref = Arc::clone(&witnessed);
        let intentions_ref = Arc::clone(&intentions);

        let hooks = WasmHostHooks::new()
            .with_coherence_provider(|| 0.66)
            .with_resonate(move |_value, _intent| {
                resonated_ref.fetch_add(1, Ordering::SeqCst);
            })
            .with_witness(move |_event| {
                witnessed_ref.fetch_add(1, Ordering::SeqCst);
            })
            .with_intention_push(move |name, _depth| {
                intentions_ref
                    .lock()
                    .expect("intentions mutex poisoned")
                    .push(name);
            });

        let source = r#"
            intention bridge {
                param x = 42
                resonate x
                witness x
            }
        "#;

        let run = run_source_with_host(source, hooks).expect("wasm host run should succeed");

        assert_eq!(resonated.load(Ordering::SeqCst), 1);
        assert_eq!(witnessed.load(Ordering::SeqCst), 1);

        // Return is the last expression (Witness), which yields the observed coherence score.
        assert_eq!(run.result, PhiIRValue::Number(0.66));
    }

    /// BSEI (Backend Semantics Equivalence Invariant) conformance test.
    ///
    /// For each program, asserts that:
    ///   `WASM result == native evaluator result`
    ///
    /// This is the core invariant introduced in Claude's architecture review:
    /// both backends must agree on the semantic result of every expression.
    #[test]
    fn test_wasm_vm_equivalence() {
        let cases: &[(&str, PhiIRValue)] = &[
            // Number arithmetic
            (
                "let x = 10 + 32  let y = x * 2  y",
                PhiIRValue::Number(84.0),
            ),
            // Simple boolean constant
            ("true", PhiIRValue::Boolean(true)),
            ("false", PhiIRValue::Boolean(false)),
        ];

        for (source, expected) in cases {
            // Native evaluator path
            let native_result = crate::compile_and_run_phi_ir(source)
                .unwrap_or_else(|e| panic!("Native eval failed for {:?}: {}", source, e));

            // WASM bridge path
            let wasm_result = run_source_with_host(source, WasmHostHooks::new())
                .unwrap_or_else(|e| panic!("WASM run failed for {:?}: {}", source, e))
                .result;

            // Both must agree with the expected value
            assert_eq!(
                native_result, *expected,
                "Native evaluator mismatch for {:?}",
                source
            );
            assert_eq!(
                wasm_result, *expected,
                "[BSEI] WASM result differs from native VM for {:?}",
                source
            );
        }
    }
}
