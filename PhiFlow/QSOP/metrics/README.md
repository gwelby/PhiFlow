# QSOP Metrics

Generated artifacts:

- `objective_metrics.json`: latest computed objective metrics snapshot.
- `weekly_audit_YYYYMMDD.md`: weekly audit report output.

Regenerate with:

```bash
python QSOP/tools/log_objective_metrics.py
python QSOP/tools/weekly_qsop_audit.py
python QSOP/tools/run_all.py
```
