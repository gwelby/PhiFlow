const { spawn } = require('node:child_process');

console.log("Starting MCP Guardrails Test...");

const phiProcess = spawn('cargo', ['run', '--bin', 'phi_mcp'], {
    env: {
        ...process.env,
        PHI_MAX_STEPS: '50',
        PHI_TIMEOUT_MS: '2000'
    }
});

let outputBuffer = '';
let streamId = null;
let pollInterval = null;

phiProcess.stdout.on('data', (data) => {
    const lines = data.toString().split('\n');

    for (const line of lines) {
        if (!line.trim()) continue;
        console.log(`[STDOUT] ${line}`);

        try {
            const resp = JSON.parse(line);
            if (resp.id === "spawn_req" && resp.result?.content?.[0]?.text) {
                const text = resp.result.content[0].text;
                const match = text.match(/Spawned stream ([a-f0-9\-]+)/);
                if (match) {
                    streamId = match[1];
                    console.log(`✅ Extracted Stream ID: ${streamId}`);
                    // Start polling
                    pollInterval = setInterval(pollStatus, 500);
                }
            }

            if (resp.id === "poll_req" && resp.result?.content?.[0]?.text) {
                const text = resp.result.content[0].text;
                const state = JSON.parse(text);
                console.log(`Current Status: ${state.status}`);
                if (state.status === "failed") {
                    console.log(`✅ SUCCESS: Guardrail caught the runaway process! Result: ${state.result}`);
                    clearInterval(pollInterval);
                    try { require('child_process').execSync('taskkill /F /IM phi_mcp.exe'); } catch (e) { }
                    process.exit(0);
                }
            }
        } catch (e) {
            // not json
        }
    }
});

phiProcess.stderr.on('data', (data) => {
    console.error(`[STDERR] ${data.toString().trim()}`);
});

function pollStatus() {
    if (!streamId) return;
    const req = {
        jsonrpc: "2.0",
        id: "poll_req",
        method: "tools/call",
        params: {
            name: "read_resonance_field",
            arguments: {
                stream_id: streamId
            }
        }
    };
    phiProcess.stdin.write(JSON.stringify(req) + "\n");
}

// Give cargo time to build
setTimeout(() => {
    console.log("Sending runaway script...");
    const req = {
        jsonrpc: "2.0",
        id: "spawn_req",
        method: "tools/call",
        params: {
            name: "spawn_phi_stream",
            arguments: {
                source_code: `
                    var i = 0.0
                    while true {
                        i = i + 1.0
                    }
                `
            }
        }
    };
    phiProcess.stdin.write(JSON.stringify(req) + "\n");
}, 15000);

// Global timeout
setTimeout(() => {
    console.error("❌ FAILED: Test timed out.");
    try { require('child_process').execSync('taskkill /F /IM phi_mcp.exe'); } catch (e) { }
    process.exit(1);
}, 30000);
