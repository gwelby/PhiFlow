use phiflow::mcp_server::state::McpState;
use phiflow::mcp_server::tools::handle_tool_call;
use serde_json::{json, Value};
use tokio;

async fn call_tool(state: &McpState, id: i64, name: &str, arguments: Value) -> Value {
    let response = handle_tool_call(
        "tools/call".into(),
        json!({
            "name": name,
            "arguments": arguments
        }),
        json!(id),
        state,
    )
    .await
    .expect("tool call should return JSON-RPC response");
    serde_json::to_value(response).expect("response should serialize to JSON value")
}

fn content_text(response: &Value) -> String {
    response["result"]["content"][0]["text"]
        .as_str()
        .expect("response content text should exist")
        .to_string()
}

async fn spawn_stream(state: &McpState, source_code: &str, id: i64) -> String {
    let response = call_tool(
        state,
        id,
        "spawn_phi_stream",
        json!({
            "source_code": source_code
        }),
    )
    .await;
    content_text(&response).replace("Spawned stream ", "")
}

async fn read_stream(state: &McpState, stream_id: &str, id: i64) -> Value {
    let response = call_tool(
        state,
        id,
        "read_resonance_field",
        json!({
            "stream_id": stream_id
        }),
    )
    .await;
    let text = content_text(&response);
    serde_json::from_str(&text).expect("read_resonance_field text should be valid JSON")
}

async fn wait_for_status(state: &McpState, stream_id: &str, expected: &str) -> Value {
    for attempt in 0..30 {
        let payload = read_stream(state, stream_id, 10_000 + attempt).await;
        if payload["status"] == expected {
            return payload;
        }
        tokio::time::sleep(tokio::time::Duration::from_millis(20)).await;
    }
    panic!("stream {} did not reach status {}", stream_id, expected);
}

#[tokio::test]
async fn test_mcp_spawn_and_read() {
    let state = McpState::new();

    let source_code = r#"
        param a = 1
        intention healing {
            param b = a + 2
            witness a
        }
    "#;

    let stream_id = spawn_stream(&state, source_code, 1).await;
    let read_content = wait_for_status(&state, &stream_id, "yielded").await;

    assert_eq!(read_content["status"], "yielded");
    assert_eq!(read_content["intention_stack"][0], "healing");
    assert_eq!(read_content["observed_value"], "1.0");

    let resume_response = call_tool(
        &state,
        3,
        "resume_phi_stream",
        json!({
            "stream_id": stream_id
        }),
    )
    .await;
    let content = content_text(&resume_response);
    assert_eq!(content, format!("Resumed stream {}", stream_id));

    let read_content_2 = wait_for_status(&state, &stream_id, "completed").await;
    assert_eq!(read_content_2["status"], "completed");
}

#[tokio::test]
async fn test_mcp_shared_resonance_visible_across_streams() {
    let state = McpState::new();

    let source_one = r#"
        intention global_peace {
            param group = 1
            resonate group
        }
    "#;

    let source_two = r#"
        intention global_peace {
            param group = 2
            resonate group
        }
    "#;

    let stream_one = spawn_stream(&state, source_one, 101).await;
    let stream_two = spawn_stream(&state, source_two, 102).await;

    let _completed_one = wait_for_status(&state, &stream_one, "completed").await;
    let completed_two = wait_for_status(&state, &stream_two, "completed").await;

    let values = completed_two["resonance_field"]["global_peace"]
        .as_array()
        .expect("global_peace resonance field should exist");
    assert_eq!(values.len(), 2, "shared resonance should collect both streams");

    let mut value_strings: Vec<String> = values
        .iter()
        .map(|v| v.as_str().expect("value should be string").to_string())
        .collect();
    value_strings.sort();

    assert_eq!(
        value_strings,
        vec!["Number(1.0)".to_string(), "Number(2.0)".to_string()]
    );

    assert_eq!(completed_two["status"], "completed");
}
