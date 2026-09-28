# run_phi Diagnostics Pipeline

`p1_host/run_phi.py` executes a two-stage flow for `.phi` sources:

1. **Diagnostics stage** (`phic --json-errors`)
   - Exit `0`: parse success, continue
   - Exit `2`: parse diagnostics on stdout (strict schema v1), stop
   - Exit `1`: IO/runtime failure, stop

2. **Execution stage** (existing WAT/WASM host path)
   - Compile/load module source
   - Execute through `P1Host`
   - Emit final snapshot JSON

## Diagnostics Schema v1 (strict)
Each diagnostic object must contain exactly these fields:
- `error_code`
- `line`
- `column`
- `found`
- `expected`
- `hint`
- `example_fix`

The parser rejects payloads with missing fields, extra fields, non-array top-level JSON, or mixed prose+JSON output.
