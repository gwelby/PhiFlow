import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
    CallToolRequestSchema,
    ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { v4 as uuidv4 } from "uuid";
import fs from "node:fs";
import path from "node:path";

const server = new Server(
    {
        name: "mcp-message-bus",
        version: "1.0.0",
    },
    {
        capabilities: {
            tools: {},
        },
    }
);

// Persistent queue (Phase 2)
// Schema: { id, ts, from, to, intent, payload_ref, requires_ack, status: 'pending' | 'acked', ack_ts?, result_summary? }
const messages = new Map();
const QUEUE_FILE = process.env.MCP_QUEUE_PATH
    ? path.resolve(process.env.MCP_QUEUE_PATH)
    : path.resolve(process.cwd(), "queue.jsonl");
const LEGACY_QUEUE_FILE = process.env.MCP_LEGACY_QUEUE_PATH
    ? path.resolve(process.env.MCP_LEGACY_QUEUE_PATH)
    : path.join(path.dirname(QUEUE_FILE), "queue.json");
const DLQ_PATH = process.env.MCP_DLQ_PATH
    ? path.resolve(process.env.MCP_DLQ_PATH)
    : path.resolve(process.cwd(), "../PhiFlow/QSOP/mail/dead_letter");
const CHANGELOG_PATH = process.env.MCP_CHANGELOG_PATH
    ? path.resolve(process.env.MCP_CHANGELOG_PATH)
    : path.resolve(process.cwd(), "../PhiFlow/QSOP/CHANGELOG.md");

// Ensure DLQ directory exists
if (!fs.existsSync(DLQ_PATH)) {
    fs.mkdirSync(DLQ_PATH, { recursive: true });
}

fs.mkdirSync(path.dirname(QUEUE_FILE), { recursive: true });

function compareMessagesByTs(left, right) {
    const leftTs = Date.parse(left.ts ?? "");
    const rightTs = Date.parse(right.ts ?? "");

    if (!Number.isNaN(leftTs) && !Number.isNaN(rightTs) && leftTs !== rightTs) {
        return leftTs - rightTs;
    }

    return (left.ts ?? "").localeCompare(right.ts ?? "");
}

function importLegacyQueueIfNeeded() {
    if (fs.existsSync(QUEUE_FILE)) {
        return;
    }

    if (!fs.existsSync(LEGACY_QUEUE_FILE)) {
        fs.writeFileSync(QUEUE_FILE, "", "utf8");
        return;
    }

    const raw = fs.readFileSync(LEGACY_QUEUE_FILE, "utf8").trim();
    if (!raw) {
        fs.writeFileSync(QUEUE_FILE, "", "utf8");
        return;
    }

    let parsed = [];
    try {
        parsed = JSON.parse(raw);
    } catch (err) {
        console.error(`[BUS] Failed to parse legacy queue file: ${LEGACY_QUEUE_FILE}`);
        throw err;
    }

    if (!Array.isArray(parsed)) {
        throw new Error("Legacy queue file format invalid: expected top-level array");
    }

    const lines = parsed
        .filter((item) => item && typeof item.id === "string")
        .map((item) => JSON.stringify(item));
    fs.writeFileSync(QUEUE_FILE, lines.length > 0 ? `${lines.join("\n")}\n` : "", "utf8");
    console.error(`[BUS] Imported ${lines.length} message(s) from legacy queue ${LEGACY_QUEUE_FILE}`);
}

function readQueueStateFromLog() {
    importLegacyQueueIfNeeded();

    if (!fs.existsSync(QUEUE_FILE)) {
        return new Map();
    }

    const raw = fs.readFileSync(QUEUE_FILE, "utf8");
    const nextMessages = new Map();
    const lines = raw.split(/\r?\n/);

    for (let index = 0; index < lines.length; index += 1) {
        const line = lines[index].trim();
        if (!line) {
            continue;
        }

        try {
            const parsed = JSON.parse(line);
            if (parsed && typeof parsed.id === "string") {
                nextMessages.set(parsed.id, parsed);
            }
        } catch (err) {
            const isLastLine = index === lines.length - 1;
            if (isLastLine) {
                console.error(`[BUS] Ignoring truncated final queue log entry in ${QUEUE_FILE}`);
                break;
            }
            console.error(`[BUS] Failed to parse queue log line ${index + 1} in ${QUEUE_FILE}`);
            throw err;
        }
    }

    return nextMessages;
}

function refreshQueueState() {
    const nextMessages = readQueueStateFromLog();
    messages.clear();
    for (const [id, msg] of nextMessages.entries()) {
        messages.set(id, msg);
    }
}

function appendQueueState(message) {
    importLegacyQueueIfNeeded();
    fs.appendFileSync(QUEUE_FILE, `${JSON.stringify(message)}\n`, "utf8");
    messages.set(message.id, message);
}

function sweepDeadLetters() {
    refreshQueueState();

    let sweptCount = 0;
    const now = new Date();

    for (const [id, msg] of Array.from(messages.entries())) {
        if (msg.status === "pending" && msg.ttl_s) {
            const msgTime = new Date(msg.ts);
            const expiresAt = new Date(msgTime.getTime() + msg.ttl_s * 1000);

            if (now > expiresAt) {
                const timedOutMessage = {
                    ...msg,
                    status: "timeout",
                    result_summary: "TIMEOUT: Auto-escalated to DLQ",
                };

                // 2. Write to DLQ
                const dlqFile = path.join(DLQ_PATH, `${id}.json`);
                fs.writeFileSync(dlqFile, JSON.stringify(timedOutMessage, null, 2), "utf8");
                appendQueueState(timedOutMessage);

                // 3. Append to CHANGELOG
                const logEntry = `\n## ${now.toISOString().slice(0, 10)} - [Bus] WARNING: Message Auto-Escalation\n\n` +
                    `- [Bus] The following message expired before receiving an ACK:\n` +
                    `- ID: ${id}\n` +
                    `- From: ${msg.from}  To: ${msg.to}\n` +
                    `- Intent: ${msg.intent}\n` +
                    `- Action: Message auto-reconciled to DLQ. **UNRECONCILED**.\n`;

                try {
                    fs.appendFileSync(CHANGELOG_PATH, logEntry, "utf8");
                } catch (err) {
                    console.error(`[BUS] Failed to write to CHANGELOG: ${err}`);
                }

                sweptCount++;
                console.error(`[BUS] Message ${id} TIMEOUT. Moved to DLQ.`);
            }
        }
    }

    return sweptCount;
}

server.setRequestHandler(ListToolsRequestSchema, async () => {
    return {
        tools: [
            {
                name: "send_message",
                description: "Send a synchronous message to another agent via the bus. Always write the bulk payload to a file first, and pass the file path as payload_ref.",
                inputSchema: {
                    type: "object",
                    properties: {
                        from: { type: "string", description: "Your agent name (e.g., 'antigravity' or 'codex')" },
                        to: { type: "string", description: "Target agent name" },
                        intent: { type: "string", description: "The action the target should perform" },
                        payload_ref: { type: "string", description: "Absolute path to the file containing the full context/instructions" },
                        requires_ack: { type: "boolean", description: "Whether an acknowledgement is expected", default: true },
                        ttl_s: { type: "integer", description: "Optional message TTL in seconds before DLQ auto-escalation" }
                    },
                    required: ["from", "to", "intent", "payload_ref"],
                },
            },
            {
                name: "poll_messages",
                description: "Retrieve pending messages addressed to your agent.",
                inputSchema: {
                    type: "object",
                    properties: {
                        agent_name: { type: "string", description: "Your agent name (e.g., 'antigravity' or 'codex')" }
                    },
                    required: ["agent_name"],
                },
            },
            {
                name: "ack_message",
                description: "Acknowledge that a message has been received and processed.",
                inputSchema: {
                    type: "object",
                    properties: {
                        message_id: { type: "string" },
                        agent_name: { type: "string", description: "Your agent name" },
                        result_summary: { type: "string", description: "Brief outcome to log" }
                    },
                    required: ["message_id", "agent_name"],
                },
            },
            {
                name: "sweep_queue",
                description: "Manually sweep expired pending messages into the dead-letter queue.",
                inputSchema: {
                    type: "object",
                    properties: {},
                },
            }
        ],
    };
});

server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const { name, arguments: args } = request.params;

    if (name === "send_message") {
        refreshQueueState();

        const id = uuidv4();
        const ts = new Date().toISOString();
        const msg = {
            id,
            ts,
            from: args.from,
            to: args.to,
            intent: args.intent,
            payload_ref: args.payload_ref,
            requires_ack: args.requires_ack !== false,
            ttl_s: args.ttl_s !== undefined ? args.ttl_s : 86400,
            status: "pending"
        };

        appendQueueState(msg);
        console.error(`[BUS] Message ${id} queued from ${args.from} to ${args.to}`);

        return {
            content: [{ type: "text", text: JSON.stringify({ status: "queued", message_id: id }) }],
        };
    }

    if (name === "poll_messages") {
        refreshQueueState();

        const agent = args.agent_name;
        const pending = Array.from(messages.values())
            .filter(m => m.to === agent && m.status === "pending")
            .sort(compareMessagesByTs);

        return {
            content: [{ type: "text", text: JSON.stringify(pending, null, 2) }],
        };
    }

    if (name === "ack_message") {
        refreshQueueState();

        const { message_id, agent_name, result_summary } = args;
        const msg = messages.get(message_id);

        if (!msg) {
            return { content: [{ type: "text", text: `Error: Message ${message_id} not found.` }] };
        }
        if (msg.to !== agent_name) {
            return { content: [{ type: "text", text: `Error: Message ${message_id} is not addressed to ${agent_name}.` }] };
        }
        if (msg.status === "acked") {
            return {
                content: [{ type: "text", text: JSON.stringify({ status: "success", message_id, state: "acked_already" }) }],
            };
        }

        const ackedMessage = {
            ...msg,
            status: "acked",
            ack_ts: new Date().toISOString(),
            result_summary: result_summary || "Acknowledged",
        };
        appendQueueState(ackedMessage);

        console.error(`[BUS] Message ${message_id} ACKED by ${agent_name}`);

        return {
            content: [{ type: "text", text: JSON.stringify({ status: "success", message_id, state: "acked" }) }],
        };
    }

    if (name === "sweep_queue") {
        const swept = sweepDeadLetters();
        return {
            content: [{ type: "text", text: JSON.stringify({ status: "success", swept_count: swept }) }],
        };
    }

    throw new Error(`Tool not found: ${name}`);
});

async function main() {
    importLegacyQueueIfNeeded();
    refreshQueueState();
    sweepDeadLetters(); // Run once on startup

    // Check every minute while running
    setInterval(() => {
        try {
            sweepDeadLetters();
        } catch (err) {
            console.error(`[BUS] Sweep failed: ${err}`);
        }
    }, 60000);

    const transport = new StdioServerTransport();
    await server.connect(transport);
    console.error(`[BUS] MCP Message Bus running on stdio (queue=${QUEUE_FILE})`);
}

main().catch(console.error);
