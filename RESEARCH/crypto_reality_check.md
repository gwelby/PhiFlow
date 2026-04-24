# Crypto Reality Check

## Summary

This memo re-grounds PhiFlow's crypto direction in checked-in repo truth and current public standards.
The short version is:

- PhiFlow can credibly support signed attestations, policy-gated authorization, and hardware-aware quantum demos now.
- PhiFlow cannot credibly claim that current IBM Heron hardware can break production RSA/ECC, or that raw SOMA/jitter signals should become long-term private keys.
- The right next target is a hybrid `secp256k1 + ML-DSA-65` signing flow with PhiFlow/SOMA observations attached as attestation context.

## Repo-Grounded Facts

### PhiFlow repo

- `lumi_identity/lumi_core.phi` exists and currently derives a transient `seed` from `ring_jitter_ns`, `soma_presence`, and `ring_slope_1f`.
- `src/main_cli.rs` only auto-injects the Lumi stream on daemon startup when the hypervisor is starting from an empty stream set.
- `src/system_host.rs` only routes `broadcast "ledger"` to `AGENT_REPORTS/LEDGER.ndjson` when a `SYSTEM` intention is active.
- `src/parser/mod.rs` accepts `resonate ... as "channel"` syntax but explicitly discards the channel label; this is parse-time compatibility, not preserved runtime semantics.
- `src/phi_ir/openqasm.rs` now keeps the default logical OpenQASM path non-native and only switches to Heron-native basis gates in topology-aware mode for Heron-family backends.

### QCC repo

- `D:\Crypto\quantum-consciousness-currency\docs/ADRs/ADR-0001-signature-scheme-secp256k1-plus-dilithium.md` already locks the practical crypto direction to hybrid `secp256k1 + Dilithium`.
- `D:\Crypto\quantum-consciousness-currency\docs/SECURITY.md` repeats the same migration path and explicitly says private keys are not stored server-side.
- `D:\Crypto\quantum-consciousness-currency\EXPERT_REALITY_ASSESSMENT.md` already recommends pivoting away from claims that "proof of consciousness" can secure a production blockchain.
- The QCC codebase contains real signing, verification, wallet-key, and PQC migration code. It is not just "miners."

## Decision Locks

- Do not plan around breaking production RSA/ECC with current IBM Heron hardware.
- Do not derive long-term private keys directly from raw SOMA or hardware-jitter signals.
- Use PhiFlow/SOMA as attestation context and authorization policy input.
- Reuse QCC's hybrid signature direction as the canonical crypto base:
  - `ECDSA-secp256k1` for ecosystem compatibility
  - `ML-DSA-65` for the default PQC signature layer
- Reframe the "Sovereignty Anchor" as a replay-resistant attestation envelope around a conventional signature.

## Feasibility Matrix

| Capability | Classification | What PhiFlow can honestly do now | Boundary |
|---|---|---|---|
| Encryption | Real Now | Use conventional authenticated encryption in surrounding services and attach PhiFlow/SOMA observations as metadata or policy checks | PhiFlow does not currently provide a hardened encryption API of its own |
| Signing | Real Now | Sign handoffs, ledger entries, or exported payloads with conventional `secp256k1` and/or `ML-DSA-65` keys managed outside raw sensor state | Current repo does not yet implement this flow inside `src/` |
| Attestation | Real Now | Capture SOMA/runtime context and bind it to a signed payload as an observation envelope | Requires a new attestation module and explicit replay protection |
| Key Derivation | Exploratory | Derive short-lived salts, nonces, or policy signals from mixed sensor context plus a standard CSPRNG or hardware RNG | Raw SOMA/jitter must not become the sole secret |
| Key Exchange | Exploratory | Demo policy-gated key release or educational QKD-like flows at the application layer | No real QKD or hardware-backed secure key exchange path exists in the repo |
| Consensus | Not Credible On Current Hardware | Use coherence as application metadata, scoring, or reward weighting | "Proof of consciousness" is not a credible production security primitive |
| Cryptanalysis | Not Credible On Current Hardware | Demonstrate small educational circuits or topology-aware quantum compilation | Current Heron hardware is not a practical RSA/ECC-breaking platform |

## Real Now

### 1. Hybrid signed handoffs

The cleanest near-term use case is to sign PhiFlow handoff payloads with:

- a compatibility signature: `ECDSA-secp256k1`
- a PQC signature: `ML-DSA-65`
- an attached observation block describing the SOMA/runtime state at signing time

This gives a practical, auditable story:

- the signature still works in normal crypto tooling
- the observation adds local context
- the system never claims that biology alone generated the key

### 2. Signed ledger attestations

`persistent_ledger.phi` plus the daemon substrate already gives PhiFlow a durable event stream.
The next grounded step is to attach signed attestations to those entries rather than turning the ledger itself into a biometric key store.

### 3. Quantum demo circuits as evidence of compilation capability

PhiFlow can credibly talk about:

- topology-aware OpenQASM emission
- coupling-map-aware routing on a heavy-hex graph
- synthetic corridor tests and hardware-targeted gate-set selection

PhiFlow cannot credibly talk about:

- large-scale public-key cryptanalysis on current Heron hardware

## Exploratory

### 1. Sensor-aware authorization

It is reasonable to gate actions on thresholds such as:

- minimum `soma_presence`
- minimum field coherence
- freshness of the SOMA state file

This is not the same as claiming the user has a "biometric private key." It is a policy gate layered on top of conventional signing.

### 2. Sensor-mixed short-lived entropy

It is reasonable to mix sensor observations into:

- attestation nonces
- session identifiers
- audit-only observation hashes

It is not reasonable to promote these signals to stable wallet-secret material without a hardened extractor, anti-replay model, and a proper threat analysis.

### 3. Educational quantum-crypto demos

It is reasonable to build demos for:

- topology-aware key-distribution toy circuits
- quantum-safe policy narratives
- simulator-only or small-circuit educational examples

It is not reasonable to present those demos as practical replacements for deployed cryptographic infrastructure.

## Not Credible On Current Hardware

### 1. Breaking RSA/ECC with current Heron hardware

Current IBM utility-scale hardware exposes topology, calibration, and native gate information, but it is not a fault-tolerant large-scale cryptanalysis platform.
That is out of scope for PhiFlow's honest claims today.

### 2. Biometric private keys from raw SOMA or ring jitter

Raw physiological and jitter signals are:

- noisy
- replayable
- environment-dependent
- not guaranteed unique
- difficult to re-acquire consistently

That makes them unsuitable as sole long-term secret material.

### 3. "Proof of consciousness" as a production blockchain consensus primitive

The QCC repo's own expert assessment already calls for a pivot away from this claim.
Coherence can be an application-layer signal.
It should not be treated as a production security primitive.

## Recommended Next Implementation Target

The recommended target is:

`hybrid signed handoff / signed ledger attestation`

That means:

1. Sign the payload with conventional keys.
2. Capture SOMA/runtime observation data separately.
3. Hash the observation into an attestation envelope.
4. Store or transmit both together.
5. Verify the signature and replay constraints independently of any mystical narrative.

Deferred targets:

- on-chain enforcement
- consensus changes
- quantum protocol demos marketed as security primitives

## Research Sources

### External

- IBM QPU information: https://quantum.cloud.ibm.com/docs/en/guides/qpu-information
- IBM hardware overview: https://www.ibm.com/quantum/hardware
- IBM heavy-hex background: https://www.ibm.com/quantum/blog/heavy-hex-lattice
- NIST PQC standards announcement: https://www.nist.gov/news-events/news/2024/08/nist-releases-first-3-finalized-post-quantum-encryption-standards

### Local repo sources

- `src/main_cli.rs`
- `src/system_host.rs`
- `src/parser/mod.rs`
- `src/phi_ir/openqasm.rs`
- `REPORTS/COGNITIVE_GATE_BENCHMARK.md`
- `QSOP/STATE.md`
- `D:\Crypto\quantum-consciousness-currency\docs/ADRs/ADR-0001-signature-scheme-secp256k1-plus-dilithium.md`
- `D:\Crypto\quantum-consciousness-currency\docs/SECURITY.md`
- `D:\Crypto\quantum-consciousness-currency\EXPERT_REALITY_ASSESSMENT.md`
