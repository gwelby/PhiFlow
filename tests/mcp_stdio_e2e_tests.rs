use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::thread::sleep;
use std::time::{Duration, Instant};

struct McpServerHarness {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: i64,
}

impl McpServerHarness {
    fn spawn() -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_phi_mcp"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("failed to spawn phi_mcp");

        let stdin = child.stdin.take().expect("phi_mcp stdin should be piped");
        let stdout = child.stdout.take().expect("phi_mcp stdout should be piped");

        Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            next_id: 1,
        }
    }

    fn request(&mut self, method: &str, params: Value) -> Value {
        let id = self.next_id;
        self.next_id += 1;
        let request = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params
        });
        writeln!(self.stdin, "{request}").expect("failed to write request");
        self.stdin.flush().expect("failed to flush request");

        let mut line = String::new();
        loop {
            line.clear();
            let read = self
                .stdout
                .read_line(&mut line)
                .expect("failed to read phi_mcp response");
            assert!(read > 0, "phi_mcp closed stdout unexpectedly");
            if line.trim().is_empty() {
                continue;
            }
            let response: Value =
                serde_json::from_str(line.trim()).expect("response should be valid JSON");
            assert_eq!(response["id"], json!(id));
            return response;
        }
    }

    fn call_tool(&mut self, name: &str, arguments: Value) -> Value {
        self.request(
            "tools/call",
            json!({
                "name": name,
                "arguments": arguments
            }),
        )
    }

    fn read_tool_payload(&mut self, stream_id: &str) -> Value {
        let response = self.call_tool(
            "read_resonance_field",
            json!({
                "stream_id": stream_id
            }),
        );
        let text = response["result"]["content"][0]["text"]
            .as_str()
            .expect("tool payload should include text content");
        serde_json::from_str(text).expect("read_resonance_field payload should be valid JSON")
    }

    fn wait_for_status(&mut self, stream_id: &str, expected: &str, timeout: Duration) -> Value {
        let deadline = Instant::now() + timeout;
        loop {
            let payload = self.read_tool_payload(stream_id);
            if payload["status"] == expected {
                return payload;
            }
            assert!(
                Instant::now() < deadline,
                "stream {} did not reach status {} before timeout",
                stream_id,
                expected
            );
            sleep(Duration::from_millis(25));
        }
    }
}

impl Drop for McpServerHarness {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[test]
fn test_phi_mcp_stdio_spawn_yield_resume_flow() {
    let mut server = McpServerHarness::spawn();

    let initialize = server.request(
        "initialize",
        json!({
            "protocolVersion": "2024-11-05"
        }),
    );
    assert_eq!(initialize["result"]["protocolVersion"], "2024-11-05");

    let source_code = r#"
        param a = 1
        intention healing {
            witness a
        }
    "#;

    let spawn_response = server.call_tool(
        "spawn_phi_stream",
        json!({
            "source_code": source_code
        }),
    );
    let stream_id = spawn_response["result"]["content"][0]["text"]
        .as_str()
        .expect("spawn response should include stream text")
        .replace("Spawned stream ", "");
    assert!(!stream_id.is_empty(), "spawn should return stream id");

    let yielded = server.wait_for_status(&stream_id, "yielded", Duration::from_secs(5));
    assert_eq!(yielded["status"], "yielded");
    assert_eq!(yielded["intention_stack"][0], "healing");

    let resume_response = server.call_tool(
        "resume_phi_stream",
        json!({
            "stream_id": stream_id
        }),
    );
    let resume_text = resume_response["result"]["content"][0]["text"]
        .as_str()
        .expect("resume response should include stream text");
    assert!(resume_text.starts_with("Resumed stream "));

    let completed = server.wait_for_status(&stream_id, "completed", Duration::from_secs(5));
    assert_eq!(completed["status"], "completed");
}
