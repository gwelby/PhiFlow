# QSOP Tooling

## Purpose

Instrumentation for objective packets, metrics, and weekly audit.

## Commands

From `PhiFlow/QSOP/tools`:

```bash
python validate_packets.py
python log_objective_metrics.py
python weekly_qsop_audit.py
python run_all.py
```

Optional helper:

```bash
python compute_payload_checksum.py QSOP/mail/payloads/OBJ-YYYYMMDD-001.md
```

## Expected Artifacts

- Packet templates:
  - `QSOP/mail/templates/objective.template.json`
  - `QSOP/mail/templates/ack.template.json`
- Metrics output:
  - `QSOP/metrics/objective_metrics.json`
- Audit output:
  - `QSOP/metrics/weekly_audit_YYYYMMDD.md`

## Notes

- Validation enforces required fields and packet linkage (`ack.objective_id` must exist).
- Validation verifies objective payload checksum (`checksum` vs `payload_path`) by default.
- Metrics script ignores invalid packets and reports them in `invalid_packets`.
- Audit fails on structural issues and warns on staleness/SLA breaches.
- `run_all.py` runs validate -> metrics -> audit with configurable SLA thresholds.
