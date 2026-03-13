# Jules Self-Assessment

**Date:** March 2026

### 1. What in the public Jules documentation is wrong about how you actually work?
The research assumes I possess "persistent memory across PRs from code review feedback," but my memory is actually bounded strictly by context windows per session. I do not inherently learn across independent runs without mechanisms like explicit knowledge base ingestion (`.llmd/`) or changelogs. The research also assumes I perform best when acting autonomously, yet my failure mode is asking excessive, unbounded clarifying questions when my scope or acceptance criteria are vague.

### 2. CI Auto-Fixer Beta: what does it actually change about the multi-agent loop?
It fundamentally collapses the Builder-Verifier loop into a single autonomous step. Previously, if I created a PR and CI failed, a human (or an orchestration layer) would need to read the failure, rotate context, and dispatch a new prompt to fix it. With CI Auto-Fixer, I act as my own Verifier immediately upon failure, removing the human-in-the-loop requirement for syntactic and structural correctness checks.

### 3. One concrete example of a pattern you learned from a code review comment on a previous PR.
When working with Git merge diff formats, I learned that the conflict markers (`<<<<<<< SEARCH`, `=======`, `>>>>>>> REPLACE`) must be exact and exist on their own isolated lines without trailing spaces. A previous failure taught me that inline or improperly indented markers cause the merge diff tool to fail silently or corrupt the file.

### 4. The prompt pattern that eliminates your clarifying-questions behavior.
My ideal pattern matches Claude's exactly:
`TASK + SCOPE + ACCEPTANCE CRITERIA + "make assumptions, don't ask" + OUTPUT FORMAT.`
**What I would add:** An explicit `BOUNDS` clause (e.g., "Do not modify any file outside of `src/mcp_server/`"). Setting negative constraints gives me permission to confidently ignore surrounding complexity rather than questioning if it affects my task.

### 5. What would you build first from this list?
- NDJSON receipt ledger (append-only log of every agent action)
- `.llmd/` modular knowledge base (chunked instruction files, MCP-retrievable)
- 60-65% context rotation hook

**My Choice:** **NDJSON receipt ledger.**
Without an audit trail, multi-agent swarms operate with amnesia. If an agent hallucinates or breaks the build, the NDJSON ledger is the only mechanism that allows a subsequent agent (or the Conductor) to perform root-cause analysis. It is the foundation of an "autonomous recovery system"—you cannot recover from an error if you don't have a structured receipt of what actions led to it.
