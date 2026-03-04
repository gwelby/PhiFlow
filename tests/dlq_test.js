#!/usr/bin/env node
/**
 * Dead-Letter Queue (DLQ) Auto-Escalation Test
 * 
 * This script:
 * 1. Starts the MCP Message Bus server
 * 2. Sends a message with a 2-second TTL
 * 3. Waits 3 seconds (letting it expire)
 * 4. Triggers the `sweep_queue` MCP tool
 * 5. Verifies the message is marked 'timeout' in queue.json
 * 6. Verifies the message was written to /QSOP/mail/dead_letter/<id>.json
 * 7. Verifies the UNRECONCILED entry in CHANGELOG.md
 */

import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const QUEUE_PATH = path.resolve(__dirname, '../../mcp-message-bus/queue.json');
const SERVER_PATH = path.resolve(__dirname, '../../mcp-message-bus/server.js');
const PAYLOAD_PATH = path.resolve(__dirname, '../QSOP/mail/payloads/OBJ-20260228-001.md');
const DLQ_PATH = path.resolve(__dirname, '../QSOP/mail/dead_letter');
const CHANGELOG_PATH = path.resolve(__dirname, '../QSOP/CHANGELOG.md');

console.log('[DLQ_TEST] Starting DLQ Auto-Escalation Test...');

// ── Send message via MCP server ──────────────────────────────────────────────

const bus = spawn('node', [SERVER_PATH], { cwd: path.dirname(SERVER_PATH) });

let messageId = null;
let testPassed = false;

bus.stderr.on('data', (d) => console.error(`[BUS STDERR] ${d.toString().trim()}`));

bus.stdout.on('data', (data) => {
    const output = data.toString();
    const lines = output.split('\n');

    for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;

        try {
            const resp = JSON.parse(trimmed);

            if (resp.id === 'send_req' && resp.result?.content?.[0]?.text) {
                const parsed = JSON.parse(resp.result.content[0].text);
                if (parsed.status === 'queued') {
                    messageId = parsed.message_id;
                    console.log(`✅ [DLQ_TEST] Message queued with 2s TTL. ID: ${messageId}`);

                    console.log(`⏳ [DLQ_TEST] Waiting 3 seconds for it to expire...`);
                    setTimeout(() => {
                        console.log(`🧹 [DLQ_TEST] Triggering manual sweep_queue...`);
                        const sweepReq = JSON.stringify({
                            jsonrpc: '2.0', id: 'sweep_req', method: 'tools/call',
                            params: { name: 'sweep_queue', arguments: {} }
                        });
                        bus.stdin.write(sweepReq + '\n');
                    }, 3000);
                }
            }

            if (resp.id === 'sweep_req' && resp.result?.content?.[0]?.text) {
                const parsed = JSON.parse(resp.result.content[0].text);
                if (parsed.status === 'success') {
                    console.log(`✅ [DLQ_TEST] Sweep complete. Swept count: ${parsed.swept_count}`);

                    if (parsed.swept_count !== 1) {
                        console.error(`❌ [DLQ_TEST] Expected 1 message swept, got ${parsed.swept_count}`);
                        process.exit(1);
                    }

                    // 1. Verify queue.json
                    const queue = JSON.parse(fs.readFileSync(QUEUE_PATH, 'utf8'));
                    const msg = queue.find(m => m.id === messageId);
                    if (msg?.status === 'timeout') {
                        console.log(`✅ [DLQ_TEST] queue.json updated to "timeout".`);
                    } else {
                        console.error('❌ [DLQ_TEST] queue.json status is NOT timeout!');
                        process.exit(1);
                    }

                    // 2. Verify DLQ file
                    const dlqFile = path.join(DLQ_PATH, `${messageId}.json`);
                    if (fs.existsSync(dlqFile)) {
                        console.log(`✅ [DLQ_TEST] DLQ file successfully written: ${dlqFile}`);
                    } else {
                        console.error('❌ [DLQ_TEST] DLQ file not found!');
                        process.exit(1);
                    }

                    // 3. Verify CHANGELOG
                    const changelog = fs.readFileSync(CHANGELOG_PATH, 'utf8');
                    if (changelog.includes(messageId) && changelog.includes('UNRECONCILED')) {
                        console.log(`✅ [DLQ_TEST] CHANGELOG entry confirmed.`);
                    } else {
                        console.error('❌ [DLQ_TEST] CHANGELOG entry missing!');
                        process.exit(1);
                    }

                    testPassed = true;
                    console.log('🎉 [DLQ_TEST] All assertions passed. DLQ mechanism works.');
                    bus.kill();
                    process.exit(0);
                }
            }
        } catch (e) { /* not JSON */ }
    }
});

// Boot and send after a short delay
setTimeout(() => {
    const req = JSON.stringify({
        jsonrpc: '2.0', id: 'init', method: 'initialize',
        params: { protocolVersion: '2024-11-05', capabilities: {}, clientInfo: { name: 'antigravity-test', version: '0.1' } }
    });
    bus.stdin.write(req + '\n');

    setTimeout(() => {
        // Send a message with ONLY a 2 second TTL
        const sendReq = JSON.stringify({
            jsonrpc: '2.0', id: 'send_req', method: 'tools/call',
            params: {
                name: 'send_message', arguments: {
                    from: 'antigravity',
                    to: 'codex',
                    intent: 'dlq_timeout_test',
                    payload_ref: PAYLOAD_PATH,
                    requires_ack: true,
                    ttl_s: 2
                }
            }
        });
        bus.stdin.write(sendReq + '\n');
    }, 300);
}, 500);

// Hard Timeout
setTimeout(() => {
    if (!testPassed) {
        console.error('❌ [DLQ_TEST] Script timed out.');
    }
    bus.kill();
    process.exit(1);
}, 8000);
