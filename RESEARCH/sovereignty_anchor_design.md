# Sovereignty Anchor Design

## Summary

The "Sovereignty Anchor" should be implemented as an attestation layer over conventional signatures, not as a biometric private-key scheme.

The recommended target is:

- `ECDSA-secp256k1` for compatibility
- `ML-DSA-65` for the default PQC layer
- PhiFlow/SOMA observations attached as a replay-resistant attestation envelope

## Goal

Allow PhiFlow to say:

- "This payload was signed by a conventional key."
- "At signing time, the local runtime observed these SOMA and field conditions."
- "These conditions were fresh enough, above policy thresholds, and bound to the signed event."

## Non-Goals

- Replacing standard cryptographic signatures with raw SOMA/jitter-derived secrets
- Claiming current IBM hardware can break modern public-key crypto
- Using "proof of consciousness" as a production consensus or authentication primitive
- Building on-chain enforcement first

## Design Locks

- The signature keypair remains conventional.
- Sensor state is observation data, not the sole secret.
- Replay resistance is mandatory.
- The observation block must include a nonce and freshness metadata.
- Ledger writes that claim system authority must still honor the existing `SYSTEM`-intention rules.

## Proposed Data Model

```rust
pub struct AnchorObservation {
    pub session_id: String,
    pub timestamp: String,
    pub soma_presence: f64,
    pub ring_jitter_ns: f64,
    pub ring_slope_1f: f64,
    pub field_coherence: f64,
    pub nonce: String,
}

pub struct AnchorAttestation {
    pub algorithm: AnchorAlgorithm,
    pub payload_hash: String,
    pub observation_hash: String,
    pub signature: String,
    pub key_fingerprint: String,
    pub policy_version: String,
}

pub enum AnchorAlgorithm {
    EcdsaSecp256k1,
    MlDsa65,
    Hybrid,
}

pub struct AnchorPolicy {
    pub min_presence: f64,
    pub min_field_coherence: f64,
    pub require_soma_freshness_ms: u64,
    pub require_system_intention_for_ledger: bool,
    pub mode: AnchorMode,
}

pub enum AnchorMode {
    ObserveOnly,
    Attest,
    Enforce,
}
```

## Data Flow

### 1. Create or load a conventional signing identity

The signing identity should come from an ordinary crypto source, not from the raw sensor stream.

Allowed:

- QCC-style `secp256k1`
- QCC-style `ML-DSA-65`
- a hybrid pair using both

Not allowed:

- deriving the long-term keypair directly from `soma_presence`, `ring_jitter_ns`, or other raw runtime signals

### 2. Capture the observation block

At signing time, capture:

- `session_id`
- RFC3339 timestamp
- `soma_presence`
- `ring_jitter_ns`
- `ring_slope_1f`
- resolved field coherence
- a unique nonce

The observation capture must fail if:

- the SOMA state is stale beyond policy
- required metrics are missing
- freshness cannot be established

### 3. Canonicalize the payload to be attested

Examples:

- handoff payload
- ledger event payload
- exported report

The payload must be serialized deterministically before hashing.

### 4. Hash separately

- `payload_hash = H(payload_bytes)`
- `observation_hash = H(observation_bytes)`

This keeps the cryptographic message stable while allowing the observation block to be inspected independently.

### 5. Sign a combined statement

Recommended signed message:

```text
PhiFlow-Attestation-v1
payload_hash=<...>
observation_hash=<...>
policy_version=<...>
```

This avoids ambiguous framing and makes replay detection easier.

### 6. Verify

Verification succeeds only if:

- the conventional signature is valid
- the observation hash matches the supplied observation
- the nonce has not been reused
- the observation timestamp meets freshness policy
- the policy thresholds are satisfied when running in `Enforce` mode

## Operational Modes

### ObserveOnly

- capture and store observations
- do not require a signature
- do not block execution

Use for:

- initial deployment
- dashboards
- safe telemetry collection

### Attest

- capture observations
- sign the combined message
- store the attestation beside the payload
- do not block unrelated runtime execution

Use for:

- signed handoffs
- signed ledger entries
- audit artifacts

### Enforce

- require attestation before protected actions
- fail closed if policy thresholds are not met

Use later, only after:

- replay protection exists
- SOMA freshness rules are stable
- operator expectations are documented

## Daemon Implications

### Fresh boot vs resumed daemon

Current behavior in `src/main_cli.rs` only injects Lumi when:

- `hypervisor.streams.is_empty()`

Implication:

- a resumed daemon can skip the Lumi stream unless the saved state already contains it

Phase 2 fix options:

1. Always reconcile required streams after `load_state()`
2. Add a startup manifest that is applied idempotently on every boot
3. Migrate old daemon snapshots by inserting missing streams

Recommended:

- add an idempotent manifest reconciler rather than relying on "fresh boot only"

## Ledger Implications

Current behavior in `src/system_host.rs` only routes `"ledger"` to the strict `LEDGER.ndjson` path when a `SYSTEM` intention is active.

Implication:

- `broadcast "ledger" seed` from `Lumi_Identity` does not currently hit the strict ledger path

Phase 2 fix options:

1. Elevate Lumi into a `SYSTEM` context
2. Route Lumi attestations to a dedicated non-system channel
3. Let the existing `persistent_ledger.phi` system stream consume Lumi events and write the strict ledger entry on Lumi's behalf

Recommended:

- keep Lumi non-system
- emit a dedicated attestation event
- let the system-owned ledger stream translate that event into the strict ledger format

This preserves least privilege.

## Channel-Semantics Implications

Current parser behavior accepts:

```phiflow
resonate seed as "lumi/sovereign_seed"
```

But the label is currently discarded.

Implication:

- do not build security semantics on `resonate ... as "channel"` labels yet

Phase 2 rule:

- attestation routing must use explicit `broadcast` / structured payload channels, not discarded `resonate` labels

## Recommended Implementation Target

### Primary target

`hybrid signed handoff / signed ledger attestation`

Why this target:

- it reuses QCC's strongest concrete work
- it fits PhiFlow's existing daemon and ledger substrate
- it avoids overclaiming about private-key generation or quantum cryptanalysis

### Deferred targets

- on-chain enforcement
- consensus changes
- quantum protocol demos presented as production security

## Suggested Phase 2 File Layout

If implementation is approved, add:

- `src/security/anchor.rs`
- `src/security/mod.rs`
- tests for observation capture, freshness, replay rejection, and hybrid signature verification

Possible CLI additions:

- `phic --emit-anchor-observation`
- `phic --attest-file <path>`
- `phic --verify-attestation <attestation.json>`

## Acceptance Gates For Phase 2

- Observation capture rejects stale or missing SOMA data
- Replay nonce reuse is rejected
- Attestation changes with observation context but not key identity
- Hybrid verification succeeds with `ML-DSA-65` and `ECDSA-secp256k1`
- Daemon resume behavior is explicit and test-covered
- Ledger routing respects `SYSTEM` rules and least privilege

## Open Risks

- Sensor freshness and reliability on different hosts
- Key custody and storage responsibilities if PhiFlow becomes a signer
- User confusion between "observation-backed attestation" and "biometric private key"
- Overstated marketing claims outrunning the checked-in evidence
