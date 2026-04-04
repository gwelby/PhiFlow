# Weaver Trio

Files:
- `the_weaver.seed.yaml`
- `the_weaver.contract.yaml`
- `the_loom_ledger_schema.yaml`

Purpose:
- Keep identity compression (`seed`) separate from deterministic behavior (`contract`) and durable memory (`ledger schema`).
- Preserve compatibility by keeping `QSOP` as path while treating "The Loom" as conceptual name.

Team Flow:
1. Codex drafts or updates seed/contract/schema.
2. Antigravity reviews resonance + operational fit.
3. Antigravity dispatches objective packet for first live Weaver run.
4. Codex executes objective through QSOP tools and logs evidence.

First Suggested Objective:
- `intent`: `weaver_seed_activation_dry_run`
- `scope`:
  - `QSOP/weaver/*`
  - `QSOP/mail/*`
  - `QSOP/metrics/*`
- `verification`:
  - `python QSOP/tools/validate_packets.py`
  - `python QSOP/tools/run_all.py --pending-ack-sla-hours 24 --in-progress-sla-hours 48`
