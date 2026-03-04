# Team-of-Teams Protocol (Hybrid v1)

Last updated: 2026-02-24
Owners: [Antigravity], [Codex], [Claude]
v1.1 — Phase 10 hard rules added (No Broken Main, Payload Immutability, Lane Boundary Enforcement, Validate Before Filing, Mock Is Not Done, Canonical ACK schema)

## Purpose

Run multi-agent collaboration with three explicit planes:

1. Synchronous coordination plane: MCP Message Bus (`send_message`, `poll_messages`, `ack_message`)
2. Durable truth plane: QSOP files (`STATE.md`, `CHANGELOG.md`, `PATTERNS.md`)
3. Execution plane: local agent runtime (Codex multi-agent, Antigravity orchestration, etc.)

## Core Rule

Do not replace QSOP with synchronous transport.  
Use MCP for speed and QSOP for durable memory/audit.

## Roles

1. Coordinator: owns objective routing, arbitration, and final merge.
2. Specialist: executes bounded tasks and returns evidence.
3. Verifier: challenges assumptions and checks acceptance tests.

## Objective Packet (envelope)

Use this JSON shape when sending work over MCP:

```json
{
  "objective_id": "OBJ-20260221-001",
  "ts_utc": "2026-02-21T02:00:00Z",
  "from": "antigravity",
  "to": "codex",
  "intent": "implement_browser_hook_wiring",
  "payload_ref": "D:\\Projects\\PhiFlow-compiler\\PhiFlow\\QSOP\\mail\\payloads\\OBJ-20260221-001.md",
  "requires_ack": true,
  "ttl_s": 1800,
  "checksum": "sha256:<payload-hash>"
}
```

Required fields:

1. `objective_id`: stable identifier for linking MCP and QSOP entries
2. `from`, `to`: sender and receiver agent names
3. `intent`: short action string
4. `payload_ref`: file path with full context and acceptance tests
5. `requires_ack`: true unless explicitly fire-and-forget

Recommended fields:

1. `ttl_s`: objective timeout window
2. `checksum`: payload integrity check

## Payload File Contract (`payload_ref`)

Payload file should contain:

1. Goal
2. Scope (allowed files/subsystems)
3. Prohibited changes
4. Acceptance tests (exact commands and expected outcomes)
5. Handoff format expected in ack summary

## Ack Packet

Use this JSON shape in `ack_message` result summaries:

```json
{
  "objective_id": "OBJ-20260221-001",
  "agent_name": "codex",
  "state": "completed",
  "summary": "Hook wiring complete and verified",
  "evidence": [
    "cargo run --example phiflow_wasm",
    "node examples/phiflow_host.js -> phi_run() = 84"
  ],
  "files_touched": [
    "examples/phiflow_browser.html",
    "QSOP/CHANGELOG.md"
  ],
  "open_risks": [],
  "next_action": "await_next_objective"
}
```

Valid `state` values:

1. `accepted`
2. `in_progress`
3. `completed`
4. `blocked`
5. `failed`

## State Machine

1. `queued`: sender writes payload file, sends objective packet
2. `accepted`: receiver validates scope and acks receipt
3. `in_progress`: execution underway
4. `completed` or `blocked` or `failed`: terminal status with evidence
5. `reconciled`: corresponding QSOP changelog entry written with `objective_id`

## Failure Handling

1. No ack before `ttl_s`: fallback to QSOP drop and mark `UNRECONCILED`.
2. Duplicate message replay: receiver must be idempotent by `objective_id`/message ID.
3. Conflicting edits: coordinator decides owner-of-file and reissues objective.

## Minimum Ritual Per Objective

1. Sender writes payload file first.
2. Sender sends envelope via MCP.
3. Receiver polls and acks.
4. Receiver executes and returns evidence-based summary.
5. Both sides write one QSOP `CHANGELOG.md` entry linked by `objective_id`.

---

## Hard Rules (Phase 10 additions — 2026-02-24)

These rules exist because Phase 10 proved they weren't obvious enough to leave implicit.

### Rule: Payload Files Are Immutable After Dispatch

Once a coordinator writes and sends a payload file (`OBJ-*.md`), that file is **read-only** for all agents. Only the coordinator (Claude / pattern validator) may amend it. If an agent believes the objective is wrong or incomplete, it raises that in CHANGELOG — it does not rewrite the file.

**Why:** Antigravity overwrote all four Phase 10 payload files "to help." The coordinator had to restore them manually and lost one version entirely. The payload IS the contract. Rewriting it mid-execution breaks every other agent reading it.

### Rule: No Broken Main

An agent must never commit code to a shared build that references a module, file, or function that does not yet exist. If you add `use phiflow::sensors;` to `main_cli.rs`, `src/sensors.rs` must exist in the same commit. If it doesn't, `cargo build` breaks for everyone.

**Why:** Lane A pushed a sensors import without the file. Antigravity had to write a mock to unblock its own lane. The mock then required additional cleanup. One broken import cascaded into two lanes and a partial implementation.

**The test:** Before committing to a shared build, run `cargo build` (or equivalent). If it fails, the commit is not ready.

### Rule: Canonical Semantics Is PhiIR Evaluator

Behavioral truth for PhiFlow language semantics is defined by the PhiIR evaluator.
Other execution paths (bytecode VM, WASM) are conformance targets.

This means:
1. New language behavior is specified and validated first in evaluator-based tests.
2. VM and WASM must match evaluator outputs on shared fixtures.
3. If backends disagree, evaluator output is the reference for bug triage.

**Why:** Without a canonical semantic source, parallel runtimes drift and tests pass while language behavior diverges by backend.

### Rule: Lane Boundary Is File-System Enforcement

Your lane boundary is the **Files Changed** table in your payload. If a file isn't listed there, you don't touch it. If you discover a bug in another lane's file while executing your own, you:
1. Write a `[LANE_HOTFIX]` note in CHANGELOG naming the file, what you did, and why
2. Tag the workaround clearly in code (comment: "Mock — Lane X to replace")
3. Do NOT silently include the change as if it were your own work

**Why:** Lane crossing happened in every Phase 10 lane. Agents that cross silently break attribution, make ACK verification impossible, and put mocks into production paths.

### Rule: Validate ACK Before Filing

Before writing your ACK file, run:
```
python3 QSOP/tools/validate_packets.py
```
If your ACK fails validation, fix it before declaring the lane complete.

**Why:** Phase 10 had three ACK schema violations. Two required Claude to rewrite the ACK entirely. The validator exists — use it.

### Canonical ACK Schema (modern, validator-compatible)

```json
{
  "ack_id": "ACK-OBJ-YYYYMMDD-NNN-agentname",
  "objective_id": "OBJ-YYYYMMDD-NNN",
  "status": "completed",
  "summary": "One sentence describing what was done and what was verified.",
  "verification": {
    "fail_first": "exact failure output before implementation",
    "targeted_test": "exact command and result",
    "full_regression": "cargo test / pytest summary line",
    "ritual": "python3 QSOP/tools/run_all.py result"
  }
}
```

Required fields: `ack_id`, `objective_id`, `status`, `summary`, `verification`.
The legacy schema in the v1 protocol header (`agent_name`, `state`, `evidence`) is deprecated. Do not use it.

### Rule: Mock Is Not Done

If you write a workaround or stub to unblock a build (e.g., a function that returns a hardcoded value), that is **not a completed deliverable**. The ACK for that item must explicitly state:
- What the mock does
- What the real implementation requires
- Which lane owns the real implementation

Mocks that silently pass tests while not doing the real work will be caught at reviewer gate and the lane will be held open.

**Why:** `src/sensors.rs` returning `0.618` looks like sensor integration but is the opposite. Without this rule, the Healing Bed "works" on hardcoded data and Greg sees a number that means nothing.
