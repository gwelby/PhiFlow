# MCP Message Bus V1 - Test Results

The V1 smoke test of the MCP Message Bus is complete.

## Successes

- **Codex connected via `stdio`**: Codex successfully spawned the server and verified the 3 tools (`send_message`, `poll_messages`, `ack_message`).
- **Antigravity connected via `stdio`**: I successfully connected to the server via a background script and queued a test message (`70c097c7-7d1a-4900-9209-d3ae43f86d1c`).

## Blockers for V1 Integration

The communication channel itself works perfectly, but the **state persistence** does not.

Because the server is implemented using the `@modelcontextprotocol/sdk/server/stdio.js` transport, **a new, isolated instance of the server is spun up for every client connection.** The in-memory `Map()` queue is destroyed the moment the script exits.

When I ran the queued message script, the server spun up, accepted the message, and exited. When Codex polled, it spun up a fresh server instance with an empty queue.

## Path Forward

To make the Message Bus a viable synchronous channel, we must do one of the following:

1. **Persistent State**: Alter the `server.js` to persist the queue to disk (e.g., writing/reading a local JSON file) instead of relying solely on an in-memory map.
2. **Persistent Daemon**: Run the server continuously via HTTP/SSE instead of standard I/O, allowing both agents to connect to the same persistent instance.

Given the goal of a lightweight, low-dependency bus, **Option 1 (persisting the queue to disk)** is the fastest path forward for V2. I've left a note for Codex explaining the same so it doesn't get confused by the empty queue.
