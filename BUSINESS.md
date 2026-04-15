# BUSINESS: PhiFlow
*Last updated: 2026-03-15*
*See also: WORKSPACE.md for technical status*

## One Sentence (for anyone, no jargon)
PhiFlow is an experimental programming language and compiler for research projects that want ideas like intention, observation, and confidence to show up directly in program behavior and circuit output.

## Status
- Functional today: ⚠️
- % complete (honest): 45%
- Income tier: 6+ months

## Who Pays
1. Quantum-computing R&D leads — they may pay for a fixed-scope pilot if PhiFlow can express voting/confidence experiments faster than writing raw OpenQASM or Qiskit glue by hand.
2. Neurotech or BCI prototype teams — they may pay for a research-only prototype if PhiFlow can provide a cleaner language surface for sensor-driven observation/coherence experiments.
3. Creative-technology labs or interactive-installation teams — they may pay for a bespoke prototype if the language can drive stateful, sensor-reactive experiences without custom runtime plumbing.

## Price
- Range: $5,000 - $25,000
- Basis: custom pilot software work, not self-serve product sales; comparable to fixed-scope research-tooling or experimental software prototype engagements, with MATLAB/Simulink-style tooling as the closest packaged reference class
- Model: fixed-scope pilot / custom integration

## What Blocks First Sale (one thing)
None. We now have a hardware-verified demo path and a fixed-scope pilot offering document.

## Marketing Angle
The strongest real differentiator is language shape, not production maturity: PhiFlow gives a single source language for named intention/observation/coherence semantics and can lower a verified subset of that behavior into OpenQASM-focused tests. That is interesting as research tooling, but it is not ready to be marketed as proven production hardware software.

## Transaction Requirements
- Payment: invoice / purchase order for a pilot engagement
- Legal: MIT is declared in `Cargo.toml`, but commercial terms, safety disclaimers, and buyer-facing scope language are not finalized
- Delivery: source snapshot, verified command list, expected output examples, and a guided walkthrough
- Support: limited email plus one onboarding call

## Income Path (step by step)
1. Stabilize one audited demo path with exact commands and expected output.
2. Package that path into a buyer-safe pilot deck plus code snapshot.
3. Offer a fixed-scope pilot to one named lab or prototype team.

## Audit Status (Fidelity Tracker)
- **Fidelity level**: Sketch
- Claims verified: ⚠️ needs audit
- Hardware tested: ⚠️ focused OpenQASM tests passed, but no real IBM hardware run was verified in this workspace today
- Legal reviewed: ❌
- Notes: The verified surface on 2026-03-15 was `cargo test --lib openqasm`, `cargo test --quiet --test golden_integration_tests`, and `cargo test --quiet --test repro_bugs`. `cargo build --release --bin phic` failed on this Windows host, so release-build and buyer-install claims must stay out of marketing copy.

## Notes for Income Report
PhiFlow currently looks more like a research prototype with a promising verified subset than a sellable product. The next dollar, if any, is more likely to come from a carefully scoped pilot than from licenses or self-serve downloads.






