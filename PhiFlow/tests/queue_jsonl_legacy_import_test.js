#!/usr/bin/env node

import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { countQueueLogEntries, loadQueueState } from './queue_state_helpers.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SERVER_PATH = path.resolve(__dirname, '../../mcp-message-bus/server.js');
const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'phiflow-queue-jsonl-'));
const queuePath = path.join(tempRoot, 'queue.jsonl');
const legacyQueuePath = path.join(tempRoot, 'queue.json');
const changelogPath = path.join(tempRoot, 'CHANGELOG.md');
const dlqPath = path.join(tempRoot, 'dead_letter');

fs.mkdirSync(dlqPath, { recursive: true });
fs.writeFileSync(changelogPath, '# CHANGELOG\n', 'utf8');
fs.writeFileSync(
    legacyQueuePath,
    JSON.stringify([
        {
            id: 'legacy-1',
            ts: '2026-03-05T00:00:00Z',
            from: 'antigravity',
            to: 'codex',
            intent: 'legacy_import',
            payload_ref: 'D:\\Projects\\PhiFlow-compiler\\PhiFlow\\QSOP\\mail\\payloads\\OBJ-LEGACY.md',
            requires_ack: true,
            status: 'pending',
        }
    ], null, 2),
    'utf8'
);

const env = {
    ...process.env,
    MCP_QUEUE_PATH: queuePath,
    MCP_LEGACY_QUEUE_PATH: legacyQueuePath,
    MCP_CHANGELOG_PATH: changelogPath,
    MCP_DLQ_PATH: dlqPath,
};

const bus = spawn('node', [SERVER_PATH], {
    cwd: path.dirname(SERVER_PATH),
    env,
});

let finished = false;

function shutdown(code) {
    if (finished) {
        return;
    }
    finished = true;
    bus.kill();
    fs.rmSync(tempRoot, { recursive: true, force: true });
    process.exit(code);
}

bus.stderr.on('data', (data) => {
    console.error(`[BUS STDERR] ${data.toString().trim()}`);
});

bus.stdout.on('data', (data) => {
    const lines = data.toString().split('\n').filter(line => line.trim());

    for (const line of lines) {
        let response;
        try {
            response = JSON.parse(line);
        } catch {
            continue;
        }

        if (response.id === 'poll_req') {
            const payload = response.result?.content?.[0]?.text;
            const messages = payload ? JSON.parse(payload) : [];
            if (messages.length !== 1 || messages[0].id !== 'legacy-1') {
                console.error('❌ [QUEUE_JSONL] Legacy import did not expose the expected pending message.');
                shutdown(1);
                return;
            }

            if (countQueueLogEntries(queuePath) !== 1) {
                console.error('❌ [QUEUE_JSONL] Legacy queue was not imported into queue.jsonl as a single snapshot.');
                shutdown(1);
                return;
            }

            const ackReq = JSON.stringify({
                jsonrpc: '2.0',
                id: 'ack_req',
                method: 'tools/call',
                params: {
                    name: 'ack_message',
                    arguments: {
                        message_id: 'legacy-1',
                        agent_name: 'codex',
                        result_summary: 'legacy import replay verified',
                    },
                },
            });
            bus.stdin.write(`${ackReq}\n`);
        }

        if (response.id === 'ack_req') {
            const state = loadQueueState(queuePath, legacyQueuePath);
            const message = state.find(item => item.id === 'legacy-1');

            if (!message || message.status !== 'acked' || !message.ack_ts) {
                console.error('❌ [QUEUE_JSONL] ACK state was not appended to queue.jsonl.');
                shutdown(1);
                return;
            }

            if (countQueueLogEntries(queuePath) !== 2) {
                console.error('❌ [QUEUE_JSONL] Expected queue.jsonl to contain legacy + ack snapshots.');
                shutdown(1);
                return;
            }

            console.log('✅ [QUEUE_JSONL] Legacy queue import and replay path verified.');
            shutdown(0);
            return;
        }
    }
});

setTimeout(() => {
    const initialize = JSON.stringify({
        jsonrpc: '2.0',
        id: 'init',
        method: 'initialize',
        params: {
            protocolVersion: '2024-11-05',
            capabilities: {},
            clientInfo: { name: 'queue-jsonl-test', version: '0.1.0' },
        },
    });
    bus.stdin.write(`${initialize}\n`);

    setTimeout(() => {
        const pollReq = JSON.stringify({
            jsonrpc: '2.0',
            id: 'poll_req',
            method: 'tools/call',
            params: {
                name: 'poll_messages',
                arguments: { agent_name: 'codex' },
            },
        });
        bus.stdin.write(`${pollReq}\n`);
    }, 300);
}, 300);

setTimeout(() => {
    console.error('❌ [QUEUE_JSONL] Legacy import test timed out.');
    shutdown(1);
}, 8000);
