# Weaver Staging

This directory is the review surface for objective `OBJ-20260221-003`.

Files:
- `the_weaver.seed.yaml`
- `the_weaver.contract.yaml`
- `the_loom_ledger_schema.yaml`

Reference masters used:
- `QSOP/weaver/references/cascade_reference.yaml`
- `QSOP/weaver/references/tesla_reference.yaml`

Review flow:
1. Antigravity reviews staging trio for resonance and outside-zero fidelity.
2. Codex applies approved changes to primary Weaver trio in `QSOP/weaver/`.
3. Codex writes completion ACK for `OBJ-20260221-003` with evidence.

Verification commands:
- `python QSOP/tools/validate_packets.py`
- `python QSOP/tools/run_all.py --pending-ack-sla-hours 24 --in-progress-sla-hours 48`
