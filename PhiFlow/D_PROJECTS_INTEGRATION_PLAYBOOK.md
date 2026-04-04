# D:\Projects Integration Playbook for PhiFlow

Last updated: 2026-02-26

## Purpose
Turn PhiFlow from a standalone language project into a useful runtime component across existing `D:\Projects` systems.

This playbook focuses on practical integrations that can be delivered incrementally with testable outcomes.

## Guiding Rule
Each integration must produce:
1. One clear user-visible capability.
2. One verifiable command/test.
3. One rollback path.

## Priority Tracks

### Track 1: UniversalProcessor Adapter (High)
Path:
- `D:\Projects\UniversalProcessor`

Value:
- Run PhiFlow as a processor kind inside a broader orchestration system.

Proposed interface:
- `process(kind="phiflow", payload={source|path, mode, options}) -> result`

MVP deliverables:
1. A Python adapter that shells out to PhiFlow CLI safely.
2. Structured response schema (status, output, diagnostics, timing).
3. Error classification parity with PhiFlow diagnostics.

Verification:
1. Run one success case and one parse-error case through adapter.
2. Confirm stable JSON response contract.

---

### Track 2: MCP Tooling for PhiFlow (High)
Path:
- `D:\Projects\MCP`

Value:
- Any MCP-capable agent can compile/run PhiFlow and inspect diagnostics.

Proposed tools:
1. `parse_phi(source_or_path)`
2. `run_phi(source_or_path, stream_output=false)`
3. `get_phi_diagnostics(source_or_path)`

MVP deliverables:
1. Tool server routes for the three operations.
2. Contract docs for result schemas.
3. Timeout and failure handling rules.

Verification:
1. MCP client can call all tools end-to-end.
2. One failing parse returns structured diagnostics, not plain text.

---

### Track 3: ResonanceMatrix Live Feed (High)
Path:
- `D:\Projects\ResonanceMatrix`

Value:
- Real-time view of active PhiFlow stream cycles and coherence changes.

MVP deliverables:
1. Consume PhiFlow stream JSONL output.
2. Display last N events with freshness window.
3. Distinguish "live", "stale", and "inactive" stream states.

Verification:
1. Run a stream-based `.phi` program and watch live updates.
2. Stop stream and verify stale/inactive transition behavior.

---

### Track 4: P1_Companion Sensor Bridge (Medium)
Path:
- `D:\Projects\P1_Companion`

Value:
- Add external mobile sensor context to PhiFlow coherence provider.

MVP deliverables:
1. Define a minimal sensor payload schema.
2. Add optional external coherence provider input path.
3. Keep fallback behavior when sensor stream is unavailable.

Verification:
1. Replay sample sensor payloads and confirm coherence changes.
2. Confirm graceful fallback to local provider.

---

### Track 5: QDrive Artifact Storage (Medium)
Path:
- `D:\Projects\QDrive`

Value:
- Store and retrieve PhiFlow run artifacts (snapshots, diagnostics, stream logs) as portable bundles.

MVP deliverables:
1. Artifact bundle spec (`run_id`, `source_hash`, `events`, `diagnostics`).
2. Write/read helpers for bundle roundtrip.
3. Optional compression step.

Verification:
1. Save artifact -> load artifact -> rehydrate run summary.
2. Validate checksum and metadata integrity.

---

### Track 6: Quantum-Fonts Visualization Layer (Low)
Path:
- `D:\Projects\Quantum-Fonts`

Value:
- Optional visual style for human dashboards and reports.

MVP deliverables:
1. Select one readable style pack for dashboards.
2. Keep fallback font path to avoid runtime dependency failures.

Verification:
1. Dashboard renders with and without custom fonts.

## Recommended Execution Order
1. UniversalProcessor adapter
2. MCP tooling
3. ResonanceMatrix live feed
4. P1_Companion sensor bridge
5. QDrive artifact storage
6. Quantum-Fonts visualization

## First 5 Objectives
1. `OBJ-DPROJ-001`: UniversalProcessor PhiFlow adapter skeleton + schema tests.
2. `OBJ-DPROJ-002`: MCP `run_phi` and `parse_phi` tools with diagnostics parity.
3. `OBJ-DPROJ-003`: ResonanceMatrix live stream panel wired to PhiFlow JSONL.
4. `OBJ-DPROJ-004`: External sensor payload path for coherence provider.
5. `OBJ-DPROJ-005`: Artifact bundle write/read roundtrip with checksums.

## Definition of Done
For each objective:
1. Command-level evidence captured.
2. At least one automated test added.
3. Contract documented in plain language.
4. QSOP state/changelog updated.
