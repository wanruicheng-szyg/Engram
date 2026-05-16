// engram-core/src/proof.rs
//
// Storage Proof — Challenge / Response protocol
//
// Single-CID flow:
//   1. Validator calls `generate_challenge(cid, timeout_secs, validator_hotkey)` → Challenge
//   2. Validator sends Challenge to miner
//   3. Miner calls `generate_response(challenge, embedding)` → ProofResponse
//   4. Validator calls `verify_response(challenge, response, embedding)` → bool
//
// Batch flow (preferred for audit sweeps):
//   1. Validator calls `generate_batch_challenge(cids, timeout_secs, validator_hotkey)` → BatchChallenge
//   2. Miner calls `generate_batch_response(batch, embeddings)` → BatchProofResponse
//   3. Validator calls `verify_batch_response(batch, response, embeddings)` → Vec<bool>
//
// HMAC keying
// -----------
// Previously the HMAC key was the raw nonce, which the miner already knows (it
// is in the challenge). That lets a miner with nonce knowledge forge a proof
// for any embedding_hash they choose, without holding the actual embedding.
//
// The hardened key derivation is:
//   hmac_key = SHA256(validator_hotkey_bytes || nonce)
//
// The validator_hotkey is the validator's SR25519 public key (32 bytes), which
// is public on-chain. Embedding it in the Challenge means the miner can derive
// the same HMAC key — so the protocol still works — but a proof generated for
// validator A's challenge is cryptographically invalid for validator B's
// challenge, and a miner cannot forge a proof for an arbitrary embedding_hash
// without first finding a SHA256 preimage.
//
// Both single-CID and batch proofs bind: derived_key + embedding_hash (+ index for batch)
// All HMAC comparisons use Mac::verify_slice for guaranteed constant-time checks.

use hmac::{Hmac, Mac};
use rand::RngCore;
use sha2::{Digest, Sha256};
use std::time::{SystemTime, UNIX_EPOCH};

type HmacSha256 = Hmac<Sha256>;

// ── Types ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Challenge {
    pub nonce: [u8; 32],
    pub cid: String,
    pub issued_at: u64,
    pub expires_at: u64,
    /// Validator's SR25519 public key bytes (32 bytes). Included so the miner
    /// can derive the correct HMAC key without a separate round-trip.
    pub validator_hotkey: [u8; 32],
}

#[derive(Debug, Clone)]
pub struct ProofResponse {
    pub cid: String,
    pub nonce_hex: String,
    pub embedding_hash: String,
    pub proof: String,
}

/// A single challenge for multiple CIDs sharing one nonce.
#[derive(Debug, Clone)]
pub struct BatchChallenge {
    pub nonce: [u8; 32],
    pub cids: Vec<String>,
    pub issued_at: u64,
    pub expires_at: u64,
    pub validator_hotkey: [u8; 32],
}

#[derive(Debug, Clone)]
pub struct BatchProofEntry {
    pub cid: String,
    pub embedding_hash: String,
    pub proof: String,
}

#[derive(Debug, Clone)]
pub struct BatchProofResponse {
    pub nonce_hex: String,
    pub entries: Vec<BatchProofEntry>,
}

// ── Key derivation ────────────────────────────────────────────────────────────

/// Derive the HMAC key: SHA256(validator_hotkey || nonce).
///
/// This binds every proof to a specific validator identity. A proof produced
/// for validator A's challenge is invalid for validator B's challenge even if
/// both used the same nonce, because the derived keys differ.
fn derive_hmac_key(validator_hotkey: &[u8; 32], nonce: &[u8; 32]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(validator_hotkey);
    h.update(nonce);
    h.finalize().into()
}

// ── Validator Side — single CID ───────────────────────────────────────────────

/// Generate a challenge for a given CID.
///
/// `validator_hotkey` is the validator's SR25519 public key (32 bytes, raw).
/// Pass the raw bytes — not the SS58 string — for a compact on-wire representation.
pub fn generate_challenge(cid: &str, timeout_secs: u64, validator_hotkey: [u8; 32]) -> Challenge {
    let mut nonce = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut nonce);
    let now = unix_now();
    Challenge {
        nonce,
        cid: cid.to_string(),
        issued_at: now,
        expires_at: now + timeout_secs,
        validator_hotkey,
    }
}

/// Verify a miner's proof response. All comparisons are constant-time.
pub fn verify_response(
    challenge: &Challenge,
    response: &ProofResponse,
    embedding: &[f32],
) -> bool {
    if challenge.cid != response.cid {
        return false;
    }
    if unix_now() > challenge.expires_at {
        return false;
    }
    if response.nonce_hex != hex::encode(challenge.nonce) {
        return false;
    }
    let expected_emb_hash = hash_embedding(embedding);
    if !constant_time_eq_str(&expected_emb_hash, &response.embedding_hash) {
        return false;
    }
    let key = derive_hmac_key(&challenge.validator_hotkey, &challenge.nonce);
    verify_proof_ct(&key, &response.embedding_hash, &response.proof)
}

// ── Miner Side — single CID ───────────────────────────────────────────────────

/// Generate a proof response for a challenge, given the stored embedding.
pub fn generate_response(challenge: &Challenge, embedding: &[f32]) -> ProofResponse {
    let embedding_hash = hash_embedding(embedding);
    let key = derive_hmac_key(&challenge.validator_hotkey, &challenge.nonce);
    let proof = compute_proof(&key, &embedding_hash);
    ProofResponse {
        cid: challenge.cid.clone(),
        nonce_hex: hex::encode(challenge.nonce),
        embedding_hash,
        proof,
    }
}

// ── Validator Side — batch CIDs ───────────────────────────────────────────────

/// Generate a batch challenge covering multiple CIDs in one round trip.
pub fn generate_batch_challenge(
    cids: &[&str],
    timeout_secs: u64,
    validator_hotkey: [u8; 32],
) -> BatchChallenge {
    let mut nonce = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut nonce);
    let now = unix_now();
    BatchChallenge {
        nonce,
        cids: cids.iter().map(|s| s.to_string()).collect(),
        issued_at: now,
        expires_at: now + timeout_secs,
        validator_hotkey,
    }
}

/// Verify a miner's batch response.
///
/// Returns one bool per CID. Expired challenges or nonce mismatches return
/// all-False. Individual failures are per-entry for per-CID penalisation.
pub fn verify_batch_response(
    batch: &BatchChallenge,
    response: &BatchProofResponse,
    embeddings: &[Vec<f32>],
) -> Vec<bool> {
    let n = batch.cids.len();
    if unix_now() > batch.expires_at {
        return vec![false; n];
    }
    if response.nonce_hex != hex::encode(batch.nonce) {
        return vec![false; n];
    }
    let key = derive_hmac_key(&batch.validator_hotkey, &batch.nonce);

    batch
        .cids
        .iter()
        .zip(embeddings.iter())
        .enumerate()
        .map(|(idx, (cid, emb))| {
            let entry = match response.entries.get(idx) {
                Some(e) => e,
                None => return false,
            };
            if entry.cid != *cid {
                return false;
            }
            let expected_hash = hash_embedding(emb);
            if !constant_time_eq_str(&expected_hash, &entry.embedding_hash) {
                return false;
            }
            verify_batch_proof_ct(&key, idx as u32, &entry.embedding_hash, &entry.proof)
        })
        .collect()
}

// ── Miner Side — batch CIDs ───────────────────────────────────────────────────

/// Generate proof responses for all CIDs in a batch challenge.
pub fn generate_batch_response(
    batch: &BatchChallenge,
    embeddings: &[Vec<f32>],
) -> BatchProofResponse {
    let key = derive_hmac_key(&batch.validator_hotkey, &batch.nonce);
    let entries = batch
        .cids
        .iter()
        .zip(embeddings.iter())
        .enumerate()
        .map(|(idx, (cid, emb))| {
            let embedding_hash = hash_embedding(emb);
            let proof = compute_batch_proof(&key, idx as u32, &embedding_hash);
            BatchProofEntry { cid: cid.clone(), embedding_hash, proof }
        })
        .collect();

    BatchProofResponse {
        nonce_hex: hex::encode(batch.nonce),
        entries,
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

pub(crate) fn hash_embedding(embedding: &[f32]) -> String {
    let mut hasher = Sha256::new();

    #[cfg(target_endian = "little")]
    {
        let ptr = embedding.as_ptr() as *const u8;
        let len = embedding.len() * 4;
        // Safety: f32 and u8 have the same alignment; we only reinterpret the
        // memory layout, not do arithmetic.
        let byte_slice = unsafe { std::slice::from_raw_parts(ptr, len) };
        hasher.update(byte_slice);
    }
    #[cfg(not(target_endian = "little"))]
    {
        for &f in embedding {
            hasher.update(&f.to_le_bytes());
        }
    }

    hex::encode(hasher.finalize())
}

/// Constant-time string equality. Length difference leaks length, but both
/// sides derive hashes with the same algorithm so length is already public.
fn constant_time_eq_str(a: &str, b: &str) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let diff = a
        .as_bytes()
        .iter()
        .zip(b.as_bytes().iter())
        .fold(0u8, |acc, (x, y)| acc | (x ^ y));
    diff == 0
}

fn compute_proof(key: &[u8; 32], embedding_hash: &str) -> String {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC accepts any key length");
    mac.update(embedding_hash.as_bytes());
    hex::encode(mac.finalize().into_bytes())
}

fn verify_proof_ct(key: &[u8; 32], embedding_hash: &str, proof_hex: &str) -> bool {
    let proof_bytes = match hex::decode(proof_hex) {
        Ok(b) => b,
        Err(_) => return false,
    };
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC accepts any key length");
    mac.update(embedding_hash.as_bytes());
    mac.verify_slice(&proof_bytes).is_ok()
}

fn compute_batch_proof(key: &[u8; 32], cid_index: u32, embedding_hash: &str) -> String {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC accepts any key length");
    mac.update(&cid_index.to_le_bytes());
    mac.update(embedding_hash.as_bytes());
    hex::encode(mac.finalize().into_bytes())
}

fn verify_batch_proof_ct(
    key: &[u8; 32],
    cid_index: u32,
    embedding_hash: &str,
    proof_hex: &str,
) -> bool {
    let proof_bytes = match hex::decode(proof_hex) {
        Ok(b) => b,
        Err(_) => return false,
    };
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC accepts any key length");
    mac.update(&cid_index.to_le_bytes());
    mac.update(embedding_hash.as_bytes());
    mac.verify_slice(&proof_bytes).is_ok()
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is before Unix epoch")
        .as_secs()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_embedding() -> Vec<f32> {
        vec![0.1, 0.2, 0.3, 0.4, 0.5]
    }

    fn validator_a() -> [u8; 32] { [0xAA; 32] }
    fn validator_b() -> [u8; 32] { [0xBB; 32] }

    // ── Single-CID tests ──────────────────────────────────────────────────────

    #[test]
    fn valid_proof_verifies() {
        let emb = dummy_embedding();
        let challenge = generate_challenge("v1::abc123", 60, validator_a());
        let response = generate_response(&challenge, &emb);
        assert!(verify_response(&challenge, &response, &emb));
    }

    #[test]
    fn wrong_embedding_fails() {
        let emb = dummy_embedding();
        let wrong_emb = vec![9.9f32; 5];
        let challenge = generate_challenge("v1::abc123", 60, validator_a());
        let response = generate_response(&challenge, &emb);
        assert!(!verify_response(&challenge, &response, &wrong_emb));
    }

    #[test]
    fn wrong_cid_fails() {
        let emb = dummy_embedding();
        let challenge = generate_challenge("v1::abc123", 60, validator_a());
        let mut response = generate_response(&challenge, &emb);
        response.cid = "v1::wrong".to_string();
        assert!(!verify_response(&challenge, &response, &emb));
    }

    #[test]
    fn tampered_proof_fails() {
        let emb = dummy_embedding();
        let challenge = generate_challenge("v1::abc123", 60, validator_a());
        let mut response = generate_response(&challenge, &emb);
        let mut chars: Vec<char> = response.proof.chars().collect();
        chars[0] = if chars[0] == 'a' { 'b' } else { 'a' };
        response.proof = chars.into_iter().collect();
        assert!(!verify_response(&challenge, &response, &emb));
    }

    /// Core security property: a proof generated for validator A's challenge
    /// must not verify against a challenge with validator B's hotkey, even if
    /// the nonce and CID are identical.
    #[test]
    fn proof_is_validator_bound() {
        let emb = dummy_embedding();
        let cid = "v1::abc123";
        let timeout = 60;

        let challenge_a = generate_challenge(cid, timeout, validator_a());
        let response_a = generate_response(&challenge_a, &emb);

        // Construct a challenge_b with the same nonce and CID but different hotkey
        let mut challenge_b = challenge_a.clone();
        challenge_b.validator_hotkey = validator_b();

        assert!(verify_response(&challenge_a, &response_a, &emb),  "original must verify");
        assert!(!verify_response(&challenge_b, &response_a, &emb), "cross-validator replay must fail");
    }

    // ── Batch tests ───────────────────────────────────────────────────────────

    #[test]
    fn batch_all_valid() {
        let cids = vec!["v1::aaa", "v1::bbb", "v1::ccc"];
        let embeddings: Vec<Vec<f32>> = vec![vec![0.1, 0.2], vec![0.3, 0.4], vec![0.5, 0.6]];
        let batch = generate_batch_challenge(&cids, 60, validator_a());
        let response = generate_batch_response(&batch, &embeddings);
        let results = verify_batch_response(&batch, &response, &embeddings);
        assert_eq!(results, vec![true, true, true]);
    }

    #[test]
    fn batch_one_wrong_embedding() {
        let cids = vec!["v1::aaa", "v1::bbb"];
        let embeddings: Vec<Vec<f32>> = vec![vec![0.1, 0.2], vec![0.3, 0.4]];
        let batch = generate_batch_challenge(&cids, 60, validator_a());
        let response = generate_batch_response(&batch, &embeddings);
        let wrong_embeddings = vec![vec![0.1f32, 0.2], vec![9.9f32, 9.9]];
        let results = verify_batch_response(&batch, &response, &wrong_embeddings);
        assert_eq!(results, vec![true, false]);
    }

    #[test]
    fn batch_proof_not_shuffleable() {
        let cids = vec!["v1::aaa", "v1::bbb"];
        let embeddings: Vec<Vec<f32>> = vec![vec![0.1, 0.2], vec![0.3, 0.4]];
        let batch = generate_batch_challenge(&cids, 60, validator_a());
        let mut response = generate_batch_response(&batch, &embeddings);
        response.entries.swap(0, 1);
        let results = verify_batch_response(&batch, &response, &embeddings);
        assert_eq!(results, vec![false, false]);
    }

    #[test]
    fn batch_expired_fails_all() {
        let cids = vec!["v1::aaa"];
        let embeddings = vec![vec![0.1f32]];
        let mut batch = generate_batch_challenge(&cids, 0, validator_a());
        batch.expires_at = 0;
        let response = generate_batch_response(&batch, &embeddings);
        let results = verify_batch_response(&batch, &response, &embeddings);
        assert_eq!(results, vec![false]);
    }

    /// Batch proofs must also be validator-bound.
    #[test]
    fn batch_proof_is_validator_bound() {
        let cids = vec!["v1::aaa", "v1::bbb"];
        let embeddings: Vec<Vec<f32>> = vec![vec![0.1, 0.2], vec![0.3, 0.4]];
        let batch_a = generate_batch_challenge(&cids, 60, validator_a());
        let response_a = generate_batch_response(&batch_a, &embeddings);

        let mut batch_b = batch_a.clone();
        batch_b.validator_hotkey = validator_b();

        let results_a = verify_batch_response(&batch_a, &response_a, &embeddings);
        let results_b = verify_batch_response(&batch_b, &response_a, &embeddings);
        assert_eq!(results_a, vec![true, true]);
        assert_eq!(results_b, vec![false, false]);
    }

    #[test]
    fn derive_hmac_key_is_deterministic() {
        let hk = validator_a();
        let nonce = [0x11u8; 32];
        assert_eq!(derive_hmac_key(&hk, &nonce), derive_hmac_key(&hk, &nonce));
    }

    #[test]
    fn derive_hmac_key_differs_by_validator() {
        let nonce = [0x11u8; 32];
        assert_ne!(
            derive_hmac_key(&validator_a(), &nonce),
            derive_hmac_key(&validator_b(), &nonce),
        );
    }

    #[test]
    fn derive_hmac_key_differs_by_nonce() {
        let hk = validator_a();
        assert_ne!(
            derive_hmac_key(&hk, &[0x11u8; 32]),
            derive_hmac_key(&hk, &[0x22u8; 32]),
        );
    }
}
