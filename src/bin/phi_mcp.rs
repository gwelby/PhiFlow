use phiflow::mcp_server::protocol::{JsonRpcMessage, JsonRpcResponse};
use phiflow::mcp_server::state::{McpConfig, McpState};
use phiflow::mcp_server::tools::{handle_tool_call, handle_tools_list};
use serde_json::json;
use std::io::{self, BufRead, Write};
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut config = McpConfig::default();
    if let Ok(steps) = std::env::var("PHI_MAX_STEPS") {
        if let Ok(val) = steps.parse() {
            config.max_execution_steps = val;
        }
    }
    if let Ok(timeout) = std::env::var("PHI_TIMEOUT_MS") {
        if let Ok(val) = timeout.parse() {
            config.timeout_ms = val;
        }
    }
    if let Ok(queue_path) = std::env::var("MCP_QUEUE_PATH") {
        config.mcp_queue_path = queue_path;
    }

    let state = McpState::with_config(config);
    let stdin = io::stdin();
    let stdout = io::stdout();

    // Standard MCP JSON-RPC 2.0 loop over stdin
    for line_result in stdin.lock().lines() {
        let line = match line_result {
            Ok(l) => l,
            Err(_) => break, // EOF or error
        };

        if line.trim().is_empty() {
            continue;
        }

        let msg: JsonRpcMessage = match serde_json::from_str(&line) {
            Ok(m) => m,
            Err(e) => {
                let err_resp = JsonRpcResponse::error(
                    serde_json::Value::Null,
                    -32700,
                    format!("Parse error: {}", e),
                );
                output_message(&err_resp, &stdout);
                continue;
            }
        };

        match msg {
            JsonRpcMessage::Request(req) => {
                let id = req.id.clone();
                let method = req.method.as_str();

                let response = match method {
                    "initialize" => handle_initialize(id, req.params),
                    "ping" => JsonRpcResponse::ok(id, json!({})),
                    "tools/list" => handle_tools_list(id),
                    "tools/call" => {
                        if let Some(resp) =
                            handle_tool_call(req.method, req.params, id.clone(), &state).await
                        {
                            resp
                        } else {
                            JsonRpcResponse::error(id, -32601, "Method not found")
                        }
                    }
                    _ => JsonRpcResponse::error(id, -32601, "Method not found"),
                };

                output_message(&response, &stdout);
            }
            JsonRpcMessage::Notification(_) => {
                // Ignore notifications like initialized
            }
            JsonRpcMessage::Response(_) => {
                // Ignore responses
            }
        }
    }

    Ok(())
}

fn handle_initialize(id: serde_json::Value, params: serde_json::Value) -> JsonRpcResponse {
    let requested_protocol = params
        .get("protocolVersion")
        .and_then(|v| v.as_str())
        .unwrap_or("2024-11-05");

    JsonRpcResponse::ok(
        id,
        json!({
            "protocolVersion": requested_protocol,
            "capabilities": {
                "tools": {
                    "listChanged": false
                }
            },
            "serverInfo": {
                "name": "phiflow-mcp",
                "version": env!("CARGO_PKG_VERSION")
            }
        }),
    )
}

fn output_message(resp: &JsonRpcResponse, mut stdout: &std::io::Stdout) {
    if let Ok(json_str) = serde_json::to_string(resp) {
        let _ = writeln!(stdout, "{}", json_str);
        let _ = stdout.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::handle_initialize;
    use serde_json::json;

    #[test]
    fn initialize_returns_tools_capability() {
        let response = handle_initialize(json!(1), json!({ "protocolVersion": "2024-11-05" }));
        let result = response.result.expect("initialize should return result");

        assert_eq!(result["protocolVersion"], "2024-11-05");
        assert!(result["capabilities"]["tools"].is_object());
        assert_eq!(result["serverInfo"]["name"], "phiflow-mcp");
    }
}
