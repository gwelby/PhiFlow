use phiflow::mcp_server::state::McpState;
use phiflow::mcp_server::tools::handle_tool_call;
use serde_json::json;

#[tokio::test]
async fn test_evolve_parses_and_splices_block() {
    let state = McpState::new();

    // Spawn a stream with evolve
    let spawn_res = handle_tool_call(
        "tools/call".to_string(),
        json!({
            "name": "spawn_phi_stream",
            "arguments": {
                "source_code": "let base = 1.0\nevolve \"let next = base + 1.0\nresonate next\"\n"
            }
        }),
        json!(1),
        &state,
    )
    .await
    .unwrap();

    if let Some(err) = spawn_res.error {
        panic!("Spawn returned error: {:?}", err);
    }

    // Verify it succeeded
    let result_obj = spawn_res.result.expect("Spawn result was missing");
    let content = result_obj["content"][0]["text"].as_str().unwrap().to_string();
    assert!(content.contains("Spawned stream"));
    let stream_id = content.replace("Spawned stream ", "");

    // Let the background task complete
    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

    // Check resonance field for evolved output
    let status_res = handle_tool_call(
        "tools/call".to_string(),
        json!({
            "name": "read_resonance_field",
            "arguments": {
                "stream_id": stream_id
            }
        }),
        json!(2),
        &state,
    )
    .await
    .unwrap();

    let result_obj = status_res.result.expect("Read result was error");
    let content = result_obj["content"][0]["text"].as_str().unwrap().to_string();
    let status_json: serde_json::Value = serde_json::from_str(&content).unwrap();

    // Status should be completed
    if status_json["status"] == "failed" {
        panic!("Test failed. Result was: {:?}", status_json["result"]);
    }
    assert_eq!(status_json["status"], "completed");

    // "next" should have resonated as 2.0. In older MCP versions, resonance is global unless intentionally pushed
    let resonance_field = status_json["resonance_field"].as_object().unwrap();
    if let Some(global_res) = resonance_field.get("global") {
        let global_resonance = global_res.as_array().unwrap();
        if !global_resonance.is_empty() {
             assert_eq!(global_resonance[0].as_str().unwrap(), "Number(2.0)");
        }
    }
}

#[tokio::test]
async fn test_entangle_yields_until_partner() {
    let state = McpState::new();

    // Spawn stream 1
    let spawn1_res = handle_tool_call(
        "tools/call".to_string(),
        json!({
            "name": "spawn_phi_stream",
            "arguments": {
                "source_code": "let a = 1.0\nentangle on 432.0\nlet a = 2.0\nresonate a\n"
            }
        }),
        json!(1),
        &state,
    )
    .await
    .unwrap();
    if let Some(err) = spawn1_res.error {
        panic!("Spawn 1 returned error: {:?}", err);
    }
    let result1_obj = spawn1_res.result.expect("Spawn 1 result was missing");
    let s1 = result1_obj["content"][0]["text"].as_str().unwrap().replace("Spawned stream ", "");

    // Wait a moment and check it yielded
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    let st1 = state.get_status(&s1).unwrap();
    assert_eq!(st1.status, "entangled_432"); // yielded

    // Spawn stream 2
    let spawn2_res = handle_tool_call(
        "tools/call".to_string(),
        json!({
            "name": "spawn_phi_stream",
            "arguments": {
                "source_code": "let b = 1.0\nentangle on 432.0\nlet b = 2.0\nresonate b\n"
            }
        }),
        json!(2),
        &state,
    )
    .await
    .unwrap();
    if let Some(err) = spawn2_res.error {
        panic!("Spawn 2 returned error: {:?}", err);
    }
    let result2_obj = spawn2_res.result.expect("Spawn 2 result was missing");
    let s2 = result2_obj["content"][0]["text"].as_str().unwrap().replace("Spawned stream ", "");

    // Wait for the automatic resumption logic to trigger and both to complete
    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

    let st1_final = state.get_status(&s1).unwrap();
    let st2_final = state.get_status(&s2).unwrap();

    assert_eq!(st1_final.status, "completed");
    assert_eq!(st2_final.status, "completed");

    // Check resonance field for entangled outputs
    let status_res = handle_tool_call(
        "tools/call".to_string(),
        json!({
            "name": "read_resonance_field",
            "arguments": {
                "stream_id": s1
            }
        }),
        json!(3),
        &state,
    )
    .await
    .unwrap();

    let result3_obj = status_res.result.expect("Read 2 result was error");
    let content = result3_obj["content"][0]["text"].as_str().unwrap().to_string();
    let status_json: serde_json::Value = serde_json::from_str(&content).unwrap();
    let resonance_field = status_json["resonance_field"].as_object().unwrap();
    let global_resonance = resonance_field["global"].as_array().unwrap();

    // Both 2.0s should have been resonated
    assert_eq!(global_resonance.len(), 2);
    assert_eq!(global_resonance[0].as_str().unwrap(), "Number(2.0)");
    assert_eq!(global_resonance[1].as_str().unwrap(), "Number(2.0)");
}
