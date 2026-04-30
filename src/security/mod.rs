/// PhiFlow Security Module
///
/// Provides attestation infrastructure for signing handoffs and ledger entries
/// with SOMA runtime observations attached as context. This is NOT a cryptographic
/// identity system — raw sensor values are never used as secrets.
///
/// Architecture (per RESEARCH/sovereignty_anchor_design.md):
///   - Conventional keys sign the payload (Phase 2: secp256k1 + ML-DSA-65)
///   - SOMA/runtime observations are attached as an attestation envelope
///   - Replay resistance via nonce + freshness check
///   - Least privilege: attestation channel is separate from SYSTEM ledger channel
pub mod anchor;
pub mod entropy_buffer;
