# QSOP Changelog

*Last updated: 2026-07-30*

## 2026-07-30
- Archived ~8,100 lines of speculative modules (cuda, bio_compute, hardware, ir) to `src/_archive/speculative/`
- Ran semantic coherence experiment on IBM Heron hardware (4 programs, 4 jobs)
- Archived 82 stale root .md files to `docs/archive/`
- Archived 118 stale docs/*.md files to `docs/archive/`
- Archived QSOP process artifacts (mail, evidence, dispatches) to `docs/archive/QSOP/`
- Created vision-to-reality audit and Codex audit request

## 2026-07-19
- Ceremony engine: OSC input + blocking listen + WebSocket remote verified

## 2026-07-14
- Three-backend equivalence restored (WASM runner missing 8 of 14 phi imports — fixed in commit `66f6e2a`)

## 2026-07-10
- GHZ scaling experiment on ibm_marrakesh (n=4 through n=8, 4096 shots each)

## 2026-04-14
- First IBM Quantum hardware execution (job `d7euddh5a5qc73drdosg` on ibm_fez)

## 2026-03-24
- Windows release build fixed (lto = "thin", codegen-units = 4)

## 2026-02-19
- First compiler build — `Number(84.0)` matching evaluator
