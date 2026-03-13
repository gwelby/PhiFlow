use crate::mcp_server::state::{McpState, StreamContext};
use crate::mcp_server::protocol::{JsonRpcError, JsonRpcResponse};
use crate::phi_ir::evaluator::{EvalExecResult, Evaluator};
use crate::phi_ir::PhiIRValue;
use serde_json::{json, Value};
use uuid::Uuid;
use std::sync::Arc;
use crate::parser::{PhiLexer, PhiParser};
use crate::phi_ir::lowering::lower_program;
use crate::mcp_server::state::McpHostProvider;
use std::time::Duration;

fn snapshot_shared_resonance(state: &McpState) -> std::collections::HashMap<String, Vec<PhiIRValue>> {
    state.shared_resonance.lock().unwrap().clone()
}

pub async fn handle_tool_call(
    method: String,
    params: Value,
    id: Value,
    state: &McpState,
) -> Option<JsonRpcResponse> {
    if method != "tools/call" {
        return None;
    }

    let name = params.get("name").and_then(|n| n.as_str()).unwrap_or("");
    let args = params.get("arguments").cloned().unwrap_or(json!({}));

    let result = match name {
        "spawn_phi_stream" => spawn_phi_stream(args, state).await,
        "read_resonance_field" => read_resonance_field(args, state).await,
        "resume_phi_stream" => resume_phi_stream(args, state).await,
        "resume_entangled_streams" => resume_entangled_streams(args, state).await,
        _ => Err(JsonRpcError {
            code: -32601,
            message: format!("Unknown tool: {}", name),
            data: None,
        }),
    };

    Some(match result {
        Ok(v) => JsonRpcResponse::ok(id, v),
        Err(e) => JsonRpcResponse::error(id, e.code, e.message.clone()),
    })
}

pub fn handle_tools_list(id: Value) -> JsonRpcResponse {
    let tools = json!({
        "tools": [
            {
                "name": "spawn_phi_stream",
                "description": "Compile and run a PhiFlow program in the background.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "source_code": {
                            "type": "string",
                            "description": "The PhiFlow (.phi) source code to execute"
                        }
                    },
                    "required": ["source_code"]
                }
            },
            {
                "name": "read_resonance_field",
                "description": "Read the state and resonance field of a running PhiFlow stream",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "stream_id": {
                            "type": "string",
                            "description": "The ID of the stream to read"
                        }
                    },
                    "required": ["stream_id"]
                }
            },
            {
                "name": "resume_phi_stream",
                "description": "Resume a PhiFlow stream that yielded at a witness statement",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "stream_id": {
                            "type": "string",
                            "description": "The ID of the stream to resume"
                        }
                    },
                    "required": ["stream_id"]
                }
            },
            {
                "name": "resume_entangled_streams",
                "description": "Resume all PhiFlow streams phase-locked to a specific resonant frequency",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "frequency": {
                            "type": "number",
                            "description": "The entangle frequency to lock onto (e.g., 432.0)"
                        }
                    },
                    "required": ["frequency"]
                }
            }
        ]
    });
    JsonRpcResponse::ok(id, tools)
}

async fn spawn_phi_stream(args: Value, state: &McpState) -> Result<Value, JsonRpcError> {
    let source_code = args.get("source_code").and_then(|s| s.as_str()).unwrap_or("");
    if source_code.is_empty() {
        return Err(JsonRpcError {
            code: -32602,
            message: "Missing source_code parameter".to_string(),
            data: None,
        });
    }

    let stream_id = Uuid::new_v4().to_string();
    let state_clone = state.clone();
    let id_clone = stream_id.clone();
    
    // Parse the code immediately to catch syntax errors before spawning
    let mut lexer = PhiLexer::new(source_code);
    let tokens = match lexer.tokenize() {
        Ok(t) => t,
        Err(e) => return Err(JsonRpcError {
            code: -32000,
            message: format!("Lexer error: {:?}", e),
            data: None,
        }),
    };
    let mut parser = PhiParser::new(tokens);
    let program = match parser.parse() {
        Ok(p) => p,
        Err(e) => return Err(JsonRpcError {
            code: -32000,
            message: format!("Parse error: {:?}", e),
            data: None,
        }),
    };

    let ir_program = lower_program(&program);

    // Store initial status
    {
        let mut streams = state.streams.lock().unwrap();
        streams.insert(stream_id.clone(), StreamContext {
            id: stream_id.clone(),
            status: "running".to_string(),
            frozen_state: None,
            ir_program: Some(ir_program.clone()),
            last_witness: None,
            resonance_field: snapshot_shared_resonance(state),
            result: None,
        });
    }

    // Spawn blocking task for evaluation
    let program_for_task = ir_program;
    let timeout_duration = Duration::from_millis(state_clone.config.timeout_ms);
    
    let eval_handle = tokio::task::spawn_blocking(move || {
        let shared_resonance = Arc::clone(&state_clone.shared_resonance);
        let mut evaluator = Evaluator::new(&program_for_task)
            .with_shared_resonance(shared_resonance)
            .with_host(Box::new(McpHostProvider { config: Arc::clone(&state_clone.config) }))
            .with_max_steps(state_clone.config.max_execution_steps);
        let result = evaluator.run_or_yield();

        
        let mut streams = state_clone.streams.lock().unwrap();
        if let Some(ctx) = streams.get_mut(&id_clone) {
            match result {
                Ok(EvalExecResult::Complete(val)) => {
                    ctx.status = "completed".to_string();
                    ctx.result = Some(format!("{:?}", val));
                    ctx.resonance_field = snapshot_shared_resonance(&state_clone);
                }
                Ok(EvalExecResult::Yielded { snapshot, frozen_state }) => {
                    ctx.status = "yielded".to_string();
                    ctx.last_witness = Some(snapshot);
                    ctx.frozen_state = Some(frozen_state);
                    ctx.resonance_field = snapshot_shared_resonance(&state_clone);
                }
                Ok(EvalExecResult::Entangled { frequency, frozen_state }) => {
                    ctx.status = format!("entangled_{}", frequency);
                    ctx.frozen_state = Some(frozen_state);
                    ctx.resonance_field = snapshot_shared_resonance(&state_clone);

                    // Add to entanglement queue
                    // Convert f64 to bits for exact hashing
                    let freq_key = frequency.to_bits();

                    let should_resume = {
                        let mut queue = state_clone.entanglement_queue.lock().unwrap();
                        let waiting = queue.entry(freq_key).or_insert_with(Vec::new);
                        waiting.push(id_clone.clone());
                        waiting.len() >= 2 // Phase-lock threshold
                    };

                    if should_resume {
                        let s = Arc::new(state_clone.clone());
                        tokio::task::spawn_blocking(move || {
                            let _ = resume_entangled_streams_internal(frequency, s);
                        });
                    }
                }
                Err(e) => {
                    ctx.status = "failed".to_string();
                    ctx.result = Some(format!("Error: {:?}", e));
                }
            }
        }
    });

    // Enforce timeout
    if let Err(_) = tokio::time::timeout(timeout_duration, eval_handle).await {
        let mut streams = state.streams.lock().unwrap();
        if let Some(ctx) = streams.get_mut(&stream_id) {
            ctx.status = "failed".to_string();
            ctx.result = Some("Error: Execution timed out".to_string());
        }
    }

    Ok(json!({
        "content": [{
            "type": "text",
            "text": format!("Spawned stream {}", stream_id)
        }]
    }))
}

async fn read_resonance_field(args: Value, state: &McpState) -> Result<Value, JsonRpcError> {
    let stream_id = args.get("stream_id").and_then(|s| s.as_str()).unwrap_or("");

    let (status, witness, result) = {
        let streams = state.streams.lock().unwrap();
        let ctx = streams.get(stream_id).ok_or_else(|| JsonRpcError {
            code: -32602,
            message: format!("Stream not found: {}", stream_id),
            data: None,
        })?;
        (ctx.status.clone(), ctx.last_witness.clone(), ctx.result.clone())
    };

    let shared_resonance = snapshot_shared_resonance(state);
    let mut resonance_map = serde_json::Map::new();
    for (intent, values) in &shared_resonance {
        let val_strings: Vec<String> = values.iter().map(|v| format!("{:?}", v)).collect();
        resonance_map.insert(intent.clone(), json!(val_strings));
    }

    let mut output = json!({
        "status": status,
        "resonance_field": resonance_map,
    });

    if let Some(w) = &witness {
        output["coherence"] = json!(w.coherence);
        output["intention_stack"] = json!(w.intention_stack);
        output["observed_value"] = w
            .observed_value
            .as_ref()
            .map(|v| json!(v))
            .unwrap_or(Value::Null);
    }

    if let Some(res) = &result {
        output["result"] = json!(res);
    }

    Ok(json!({
        "content": [{
            "type": "text",
            "text": serde_json::to_string_pretty(&output).unwrap()
        }]
    }))
}

fn resume_entangled_streams_internal(frequency: f64, state: Arc<McpState>) -> Result<usize, JsonRpcError> {
    let freq_key = frequency.to_bits();
    
    // 1. Pop all waiting streams for this frequency
    let streams_to_resume = {
        let mut queue = state.entanglement_queue.lock().unwrap();
        queue.remove(&freq_key).unwrap_or_default()
    };

    if streams_to_resume.is_empty() {
        return Ok(0);
    }

    let mut resumed_count = 0;

    // 2. Resume them all
    for stream_id in &streams_to_resume {
        let (frozen_state, ir_program) = {
            let mut streams = state.streams.lock().unwrap();
            let ctx = if let Some(c) = streams.get_mut(stream_id) { c } else { continue; };
            if !ctx.status.starts_with("entangled_") { continue; }

            let fs = if let Some(f) = ctx.frozen_state.take() { f } else { continue; };
            let prog = if let Some(p) = ctx.ir_program.clone() { p } else { continue; };
            
            ctx.status = "running".to_string();
            (fs, prog)
        };

        let state_clone = (*state).clone();
        let id_clone = stream_id.clone();
        let timeout_duration = Duration::from_millis(state_clone.config.timeout_ms);
        
        let eval_handle = tokio::task::spawn_blocking(move || {
            let shared_resonance = Arc::clone(&state_clone.shared_resonance);
            let mut evaluator = Evaluator::new(&ir_program)
                .with_shared_resonance(shared_resonance)
                .with_host(Box::new(McpHostProvider { config: Arc::clone(&state_clone.config) }))
                .with_max_steps(state_clone.config.max_execution_steps);
            let result = evaluator.resume(frozen_state);

            let mut streams = state_clone.streams.lock().unwrap();
            if let Some(ctx) = streams.get_mut(&id_clone) {
                match result {
                    Ok(EvalExecResult::Complete(val)) => {
                        ctx.status = "completed".to_string();
                        ctx.result = Some(format!("{:?}", val));
                        ctx.resonance_field = snapshot_shared_resonance(&state_clone);
                    }
                    Ok(EvalExecResult::Yielded { snapshot, frozen_state }) => {
                        ctx.status = "yielded".to_string();
                        ctx.last_witness = Some(snapshot);
                        ctx.frozen_state = Some(frozen_state);
                        ctx.resonance_field = snapshot_shared_resonance(&state_clone);
                    }
                    Ok(EvalExecResult::Entangled { frequency: next_freq, frozen_state }) => {
                        ctx.status = format!("entangled_{}", next_freq);
                        ctx.frozen_state = Some(frozen_state);
                        ctx.resonance_field = snapshot_shared_resonance(&state_clone);

                        let next_freq_key = next_freq.to_bits();
                        let should_resume = {
                            let mut queue = state_clone.entanglement_queue.lock().unwrap();
                            let waiting = queue.entry(next_freq_key).or_insert_with(Vec::new);
                            waiting.push(id_clone.clone());
                            waiting.len() >= 2
                        };

                        if should_resume {
                            let s = Arc::new(state_clone.clone());
                            // Since this is a synchronous callback context, spawn it correctly
                            tokio::task::spawn_blocking(move || {
                                let _ = resume_entangled_streams_internal(next_freq, s);
                            });
                        }
                    }
                    Err(e) => {
                        ctx.status = "failed".to_string();
                        ctx.result = Some(format!("Error: {:?}", e));
                    }
                }
            }
        });

        // Spawn a background monitor for the timeout (since we're resuming multiple blindly here)
        let sid = stream_id.clone();
        let state_ref = (*state).clone();
        tokio::spawn(async move {
            if let Err(_) = tokio::time::timeout(timeout_duration, eval_handle).await {
                let mut streams = state_ref.streams.lock().unwrap();
                if let Some(ctx) = streams.get_mut(&sid) {
                    ctx.status = "failed".to_string();
                    ctx.result = Some("Error: Execution timed out".to_string());
                }
            }
        });

        resumed_count += 1;
    }

    Ok(resumed_count)
}

async fn resume_entangled_streams(args: Value, state: &McpState) -> Result<Value, JsonRpcError> {
    let frequency = args.get("frequency").and_then(|f| f.as_f64()).ok_or_else(|| JsonRpcError {
        code: -32602,
        message: "Missing or invalid frequency parameter".to_string(),
        data: None,
    })?;

    let state_arc = Arc::new(state.clone());
    let resumed_count = resume_entangled_streams_internal(frequency, state_arc)?;

    if resumed_count == 0 {
        return Ok(json!({
            "content": [{
                "type": "text",
                "text": format!("No streams waiting on frequency {}", frequency)
            }]
        }));
    }

    Ok(json!({
        "content": [{
            "type": "text",
            "text": format!("Resumed {} entangled streams on frequency {}", resumed_count, frequency)
        }]
    }))
}

async fn resume_phi_stream(args: Value, state: &McpState) -> Result<Value, JsonRpcError> {
    let stream_id = args.get("stream_id").and_then(|s| s.as_str()).unwrap_or("");
    
    let (frozen_state, ir_program) = {
        let mut streams = state.streams.lock().unwrap();
        if let Some(ctx) = streams.get_mut(stream_id) {
            if ctx.status != "yielded" {
                return Err(JsonRpcError {
                    code: -32000,
                    message: format!("Stream {} is not yielded, current status: {}", stream_id, ctx.status),
                    data: None,
                });
            }
            
            let fs = ctx.frozen_state.take().ok_or_else(|| JsonRpcError {
                code: -32000,
                message: "No frozen state available for resumption.".to_string(),
                data: None,
            })?;
            
            let prog = ctx.ir_program.clone().ok_or_else(|| JsonRpcError {
                code: -32000,
                message: "No IR program available for resumption.".to_string(),
                data: None,
            })?;
            
            ctx.status = "running".to_string();
            (fs, prog)
        } else {
            return Err(JsonRpcError {
                code: -32602,
                message: format!("Stream not found: {}", stream_id),
                data: None,
            });
        }
    };

    let state_clone = state.clone();
    let id_clone = stream_id.to_string();
    let timeout_duration = Duration::from_millis(state_clone.config.timeout_ms);
    
    // Spawn blocking task for resumed evaluation
    let eval_handle = tokio::task::spawn_blocking(move || {
        let shared_resonance = Arc::clone(&state_clone.shared_resonance);
        let mut evaluator = Evaluator::new(&ir_program)
            .with_shared_resonance(shared_resonance)
            .with_host(Box::new(McpHostProvider { config: Arc::clone(&state_clone.config) }))
            .with_max_steps(state_clone.config.max_execution_steps);
        let result = evaluator.resume(frozen_state);

        let mut streams = state_clone.streams.lock().unwrap();
        if let Some(ctx) = streams.get_mut(&id_clone) {
            match result {
                Ok(EvalExecResult::Complete(val)) => {
                    ctx.status = "completed".to_string();
                    ctx.result = Some(format!("{:?}", val));
                    ctx.resonance_field = snapshot_shared_resonance(&state_clone);
                }
                Ok(EvalExecResult::Yielded { snapshot, frozen_state }) => {
                    ctx.status = "yielded".to_string();
                    ctx.last_witness = Some(snapshot);
                    ctx.frozen_state = Some(frozen_state);
                    ctx.resonance_field = snapshot_shared_resonance(&state_clone);
                }
                Ok(EvalExecResult::Entangled { frequency, frozen_state }) => {
                    ctx.status = format!("entangled_{}", frequency);
                    ctx.frozen_state = Some(frozen_state);
                    ctx.resonance_field = snapshot_shared_resonance(&state_clone);

                    let next_freq_key = frequency.to_bits();
                    let should_resume = {
                        let mut queue = state_clone.entanglement_queue.lock().unwrap();
                        let waiting = queue.entry(next_freq_key).or_insert_with(Vec::new);
                        waiting.push(id_clone.clone());
                        waiting.len() >= 2
                    };

                    if should_resume {
                        let s = Arc::new(state_clone.clone());
                        tokio::task::spawn_blocking(move || {
                            let _ = resume_entangled_streams_internal(frequency, s);
                        });
                    }
                }
                Err(e) => {
                    ctx.status = "failed".to_string();
                    ctx.result = Some(format!("Error: {:?}", e));
                }
            }
        }
    });

    // Enforce timeout
    if let Err(_) = tokio::time::timeout(timeout_duration, eval_handle).await {
        let mut streams = state.streams.lock().unwrap();
        if let Some(ctx) = streams.get_mut(stream_id) {
            ctx.status = "failed".to_string();
            ctx.result = Some("Error: Execution timed out".to_string());
        }
    }

    Ok(json!({
        "content": [{
            "type": "text",
            "text": format!("Resumed stream {}", stream_id)
        }]
    }))
}
